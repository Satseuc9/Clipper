"""
Kick live status via the official Kick Public API (docs.kick.com).

Auth:   App access token, OAuth 2.1 client-credentials flow
        POST https://id.kick.com/oauth/token
Data:   GET https://api.kick.com/public/v1/channels?slug=a&slug=b
            (up to 50 slugs per request, each at most 25 characters)

Each channel in the response carries:
    slug, stream_title, stream.is_live, stream.viewer_count, stream.start_time

A slug missing from the response is reported UNKNOWN ("channel not found"),
never OFFLINE.

Rate limit: Kick does not publish numeric limits. A 429 pauses Kick polling
for the Retry-After period (60s if the header is absent).

Credentials (.env): KICK_CLIENT_ID, KICK_CLIENT_SECRET
"""

import os
import time

from platforms.base import LIVE, OFFLINE, PlatformMonitor, StreamResult
from utils.http import ApiError, AppAccessToken, request_json

TOKEN_URL = "https://id.kick.com/oauth/token"
CHANNELS_URL = "https://api.kick.com/public/v1/channels"

BATCH_SIZE = 50
MAX_SLUG_LENGTH = 25


class KickMonitor(PlatformMonitor):
    key = "kick"
    title = "KICK"

    def __init__(self, timeout=10, min_interval=30):
        super().__init__(timeout, min_interval)
        client_id = os.getenv("KICK_CLIENT_ID", "").strip()
        client_secret = os.getenv("KICK_CLIENT_SECRET", "").strip()
        self.configured = bool(client_id and client_secret)
        self.token = AppAccessToken(self.session, TOKEN_URL, client_id, client_secret, timeout)

    def _get_channels(self, slugs):
        """GET /public/v1/channels for up to 50 slugs, renewing the token once on 401."""
        params = [("slug", s) for s in slugs]
        for attempt in range(2):
            headers = {"Authorization": f"Bearer {self.token.get()}", "Accept": "application/json"}
            try:
                _, data = request_json(
                    self.session, "GET", CHANNELS_URL, timeout=self.timeout,
                    params=params, headers=headers,
                )
                return data.get("data") or []
            except ApiError as exc:
                if exc.status == 401 and attempt == 0:
                    self.token.invalidate()
                    continue
                raise
        raise ApiError("Kick rejected the access token (HTTP 401)", 401)

    def _fetch(self, names):
        if not self.configured:
            raise ApiError("missing KICK_CLIENT_ID / KICK_CLIENT_SECRET in .env")

        slugs = {name: name.lower().lstrip("@") for name in names}
        valid = [s for s in dict.fromkeys(slugs.values()) if len(s) <= MAX_SLUG_LENGTH]

        channels = {}
        for i in range(0, len(valid), BATCH_SIZE):
            for channel in self._get_channels(valid[i:i + BATCH_SIZE]):
                if isinstance(channel, dict) and channel.get("slug"):
                    channels[channel["slug"].lower()] = channel

        checked_at = time.time()
        results = []
        for name, slug in slugs.items():
            url = f"https://kick.com/{slug}"
            channel = channels.get(slug)
            if len(slug) > MAX_SLUG_LENGTH:
                results.append(self.unknown(name, f"invalid slug (Kick allows max {MAX_SLUG_LENGTH} characters)"))
                continue
            if channel is None:
                results.append(self.unknown(name, "channel not found on Kick (use the slug from kick.com/<slug>)"))
                continue

            stream = channel.get("stream")
            is_live = stream.get("is_live") if isinstance(stream, dict) else None
            extra = {"broadcaster_user_id": channel.get("broadcaster_user_id")}
            if is_live is True:
                extra["started_at"] = stream.get("start_time")
                results.append(StreamResult(
                    self.key, name, LIVE, viewers=stream.get("viewer_count"),
                    title=channel.get("stream_title"), url=url, checked_at=checked_at, extra=extra,
                ))
            elif is_live is False or ("stream" in channel and stream is None):
                # is_live false, or an explicit null stream: the channel is offline.
                results.append(StreamResult(self.key, name, OFFLINE, url=url,
                                            checked_at=checked_at, extra=extra))
            else:
                results.append(self.unknown(name, "unexpected response format from Kick"))
        return results
