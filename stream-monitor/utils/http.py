"""
Small HTTP helpers shared by the platform modules.

Error messages produced here never contain request URLs, headers or bodies,
so credentials cannot leak into terminal output.
"""

import time
from email.utils import parsedate_to_datetime

import requests

USER_AGENT = "stream-monitor/1.0 (+python-requests)"
RETRYABLE_STATUS = {500, 502, 503, 504}


class ApiError(Exception):
    """A request failed. `status` is the HTTP status (None for network errors)."""

    def __init__(self, message, status=None, data=None, headers=None):
        super().__init__(message)
        self.status = status
        self.data = data
        self.headers = headers or {}


class RateLimitError(ApiError):
    """The API asked us to slow down. `retry_after` is in seconds."""

    def __init__(self, message, retry_after, status=429, data=None, headers=None):
        super().__init__(message, status, data, headers)
        self.retry_after = max(1, int(retry_after))


def new_session():
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def parse_retry_after(headers, default=60):
    """Read a Retry-After header (seconds or HTTP date)."""
    value = headers.get("Retry-After")
    if not value:
        return default
    try:
        return max(1, int(float(value)))
    except ValueError:
        pass
    try:
        return max(1, int(parsedate_to_datetime(value).timestamp() - time.time()))
    except (TypeError, ValueError):
        return default


def _error_text(data):
    """Pull a short, human-readable message out of a JSON error body."""
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            text = err.get("message")
        else:
            text = data.get("message") or data.get("error_description") or err
        if text:
            return str(text)[:150]
    return ""


def request_json(session, method, url, *, timeout, retries=1, **kwargs):
    """
    Send a request and return (response, parsed_json).

    - Network errors and 5xx responses are retried `retries` times.
    - 429 raises RateLimitError (with Retry-After when the API sends it).
    - Any other non-2xx status raises ApiError carrying the parsed body,
      so platform code can inspect platform-specific error reasons.
    """
    for attempt in range(retries + 1):
        last_try = attempt == retries
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
        except requests.Timeout:
            if not last_try:
                time.sleep(1 + attempt)
                continue
            raise ApiError(f"request timed out after {timeout}s") from None
        except requests.RequestException as exc:
            if not last_try:
                time.sleep(1 + attempt)
                continue
            # Only the exception class name: str(exc) can include the URL.
            raise ApiError(f"network error ({type(exc).__name__})") from None

        try:
            data = resp.json() if resp.content else None
        except ValueError:
            data = None

        if resp.status_code == 429:
            raise RateLimitError(
                "rate limited (HTTP 429)",
                retry_after=parse_retry_after(resp.headers),
                data=data,
                headers=resp.headers,
            )
        if resp.status_code in RETRYABLE_STATUS and not last_try:
            time.sleep(1 + attempt)
            continue
        if not resp.ok:
            detail = _error_text(data)
            message = f"HTTP {resp.status_code}" + (f": {detail}" if detail else "")
            raise ApiError(message, resp.status_code, data, resp.headers)
        if data is None:
            raise ApiError("invalid JSON in API response", resp.status_code)
        return resp, data

    raise ApiError("request failed")  # not reached


class AppAccessToken:
    """
    OAuth 2 client-credentials token (used by Twitch and Kick).

    The token is fetched lazily, reused until shortly before it expires,
    and can be dropped with invalidate() after a 401 so the next call
    fetches a fresh one.
    """

    EXPIRY_MARGIN = 300  # renew 5 minutes before expiry

    def __init__(self, session, token_url, client_id, client_secret, timeout):
        self.session = session
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout
        self._token = None
        self._expires_at = 0.0

    def get(self):
        if self._token and time.time() < self._expires_at - self.EXPIRY_MARGIN:
            return self._token
        try:
            _, data = request_json(
                self.session,
                "POST",
                self.token_url,
                timeout=self.timeout,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                },
            )
        except RateLimitError:
            raise
        except ApiError as exc:
            if exc.status in (400, 401, 403):
                raise ApiError(
                    f"authentication failed (HTTP {exc.status}); check the client ID/secret in .env",
                    exc.status,
                ) from None
            raise ApiError(f"could not get access token: {exc}", exc.status) from None

        token = data.get("access_token") if isinstance(data, dict) else None
        if not token:
            raise ApiError("token response did not contain an access_token")
        try:
            expires_in = int(float(data.get("expires_in") or 3600))
        except (TypeError, ValueError):
            expires_in = 3600
        self._token = token
        self._expires_at = time.time() + expires_in
        return token

    def invalidate(self):
        self._token = None
        self._expires_at = 0.0
