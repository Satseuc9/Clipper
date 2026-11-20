"""
Twitch live status via the official Helix API.

Auth:   App access token, OAuth client-credentials flow
        POST https://id.twitch.tv/oauth2/token
Data:   GET https://api.twitch.tv/helix/users?login=...    (up to 100 logins)
        GET https://api.twitch.tv/helix/streams?user_login=...&first=100
            (up to 100 logins; only channels that are live are returned)

Get Streams returns nothing for a channel that does not exist, so logins are
first checked with Get Users. A login that Twitch does not know about is
reported UNKNOWN ("user not found"), never OFFLINE.

Rate limit: Helix uses a points bucket reported in the Ratelimit-Remaining /
Ratelimit-Reset headers. We stop until the reset time if it reaches zero
or a 429 is returned.

Credentials (.env): TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET
"""

import os
import time

from platforms.base import LIVE, OFFLINE, PlatformMonitor, StreamResult
from utils.http import ApiError, AppAccessToken, RateLimitError, request_json

TOKEN_URL = "https://id.twitch.tv/oauth2/token"
VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
USERS_URL = "https://api.twitch.tv/helix/users"
STREAMS_URL = "https://api.twitch.tv/helix/streams"

BATCH_SIZE = 100
VALIDATE_EVERY = 3600       # Twitch asks apps to validate tokens hourly
RECHECK_MISSING_AFTER = 3600


class TwitchMonitor(PlatformMonitor):
    key = "twitch"
    title = "TWITCH"

    def __init__(self, timeout=10, min_interval=10):
        super().__init__(timeout, min_interval)
        client_id = os.getenv("TWITCH_CLIENT_ID", "").strip()
        client_secret = os.getenv("TWITCH_CLIENT_SECRET", "").strip()
        self.configured = bool(client_id and client_secret)
        self.client_id = client_id
        self.token = AppAccessToken(self.session, TOKEN_URL, client_id, client_secret, timeout)
        self._last_validated = 0.0
        self._users = {}       # login -> display name, for logins that exist
        self._user_ids = {}    # login -> Twitch user ID
        self._missing = {}     # login -> time we last found it missing

    # ---- API plumbing ---------------------------------------------------------

    def _validate_token(self):
        """Hourly check that the app token is still valid (Twitch requirement)."""
        if time.time() - self._last_validated < VALIDATE_EVERY:
            return
        token = self.token.get()
        try:
            request_json(
                self.session, "GET", VALIDATE_URL, timeout=self.timeout, retries=0,
                headers={"Authorization": f"OAuth {token}"},
            )
        except ApiError as exc:
            if exc.status != 401:
                return                    # could not validate right now; retry next poll
            self.token.invalidate()       # token revoked or expired: get a new one
            self.token.get()
        self._last_validated = time.time()

    def _get(self, url, params):
        """GET a Helix endpoint, renewing the token once on 401."""
        for attempt in range(2):
            headers = {"Client-Id": self.client_id, "Authorization": f"Bearer {self.token.get()}"}
            try:
                resp, data = request_json(
                    self.session, "GET", url, timeout=self.timeout, params=params, headers=headers
                )
            except RateLimitError as exc:
                exc.retry_after = self._seconds_until_reset(exc.headers, exc.retry_after)
                raise
            except ApiError as exc:
                if exc.status == 401 and attempt == 0:
                    self.token.invalidate()
                    continue
                if exc.status == 401:
                    raise ApiError("Twitch rejected the access token (HTTP 401)", 401) from None
                raise
            if resp.headers.get("Ratelimit-Remaining") == "0":
                wait = self._seconds_until_reset(resp.headers, 60)
                self.block(wait, "Twitch rate limit reached")
            return data
        raise ApiError("Twitch authentication failed")

    @staticmethod
    def _seconds_until_reset(headers, default):
        try:
            return max(1, int(float(headers.get("Ratelimit-Reset")) - time.time()) + 1)
        except (TypeError, ValueError):
            return default

    # ---- monitoring -------------------------------------------------------------

    def _resolve_users(self, logins):
        """Find which logins exist (cached; missing ones re-checked hourly)."""
        now = time.time()
        todo = [
            l for l in logins
            if l not in self._users and now - self._missing.get(l, 0) > RECHECK_MISSING_AFTER
        ]
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            data = self._get(USERS_URL, [("login", l) for l in batch])
            found = {}
            for user in data.get("data", []):
                login = user["login"].lower()
                display = user.get("display_name") or login
                # Show the login too when the display name is in another script.
                found[login] = display if display.lower() == login else f"{display} ({login})"
                self._user_ids[login] = str(user.get("id", ""))
            for login in batch:
                if login in found:
                    self._users[login] = found[login]
                    self._missing.pop(login, None)
                else:
                    self._missing[login] = now

    def _fetch(self, names):
        if not self.configured:
            raise ApiError("missing TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET in .env")

        logins = {name: name.lower().lstrip("@") for name in names}
        self._validate_token()
        self._resolve_users(list(dict.fromkeys(logins.values())))

        existing = [l for l in dict.fromkeys(logins.values()) if l in self._users]
        live = {}
        for i in range(0, len(existing), BATCH_SIZE):
            batch = existing[i:i + BATCH_SIZE]
            params = [("user_login", l) for l in batch] + [("first", BATCH_SIZE)]
            data = self._get(STREAMS_URL, params)
            for stream in data.get("data", []):
                if stream.get("type") == "live":
                    live[stream["user_login"].lower()] = stream

        checked_at = time.time()
        results = []
        for name, login in logins.items():
            url = f"https://twitch.tv/{login}"
            if login not in self._users:
                results.append(self.unknown(name, "user not found on Twitch (typo, renamed or suspended?)"))
            elif login in live:
                stream = live[login]
                results.append(StreamResult(
                    self.key, name, LIVE, display_name=self._users[login],
                    viewers=stream.get("viewer_count"), title=stream.get("title"),
                    url=url, checked_at=checked_at,
                    extra={"user_id": str(stream.get("user_id") or self._user_ids.get(login, "")),
                           "started_at": stream.get("started_at")},
                ))
            else:
                results.append(StreamResult(
                    self.key, name, OFFLINE, display_name=self._users[login],
                    url=url, checked_at=checked_at,
                    extra={"user_id": self._user_ids.get(login, "")},
                ))
        return results
