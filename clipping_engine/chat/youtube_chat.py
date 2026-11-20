"""
YouTube live chat through the official YouTube Data API v3.

    GET https://www.googleapis.com/youtube/v3/liveChat/messages
        ?liveChatId=<liveStreamingDetails.activeLiveChatId>&part=snippet,authorDetails
        &maxResults=2000&pageToken=<nextPageToken>

The liveChatId comes from the stream-monitor's videos.list call (no extra
quota). Every list call costs quota (YOUTUBE_CHAT_UNITS_PER_CALL), so the
poller waits at least YOUTUBE_CHAT_POLL_SECONDS (or the API's
pollingIntervalMillis, whichever is longer) and stops for the day when
YOUTUBE_CHAT_DAILY_UNITS is spent. Polling slowly loses no timing detail:
every message carries its own publishedAt timestamp.

Google documents OAuth scopes for this method. Public live chats are
readable with an API key in practice; if Google rejects the key, chat is
marked unavailable for that stream (never invented).
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

from clipping_engine import config
from clipping_engine.chat.chat_log import parse_timestamp

log = logging.getLogger(__name__)

URL = "https://www.googleapis.com/youtube/v3/liveChat/messages"
STOP_REASONS = {"liveChatEnded", "liveChatNotFound", "liveChatDisabled", "forbidden", "keyInvalid"}

try:
    from zoneinfo import ZoneInfo
    PACIFIC = ZoneInfo("America/Los_Angeles")
except Exception:
    PACIFIC = timezone(timedelta(hours=-8))


class QuotaBudget:
    """Daily unit budget shared by all YouTube chat pollers (resets midnight Pacific)."""

    def __init__(self, daily_units):
        self.daily_units = daily_units
        self.used = 0
        self.day = None
        self._lock = threading.Lock()

    def spend(self, units):
        with self._lock:
            today = datetime.now(PACIFIC).date()
            if today != self.day:
                self.day, self.used = today, 0
            if self.used + units > self.daily_units:
                return False
            self.used += units
            return True


def _reason(resp):
    try:
        return resp.json()["error"]["errors"][0]["reason"]
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


class YouTubeChatPoller(threading.Thread):
    def __init__(self, live_chat_id, chat_log, budget, api_key=None, session=None, url=URL):
        super().__init__(name=f"youtube-chat-{live_chat_id[:12]}", daemon=True)
        self.live_chat_id = live_chat_id
        self.chat_log = chat_log
        self.budget = budget
        self.api_key = api_key if api_key is not None else os.getenv("YOUTUBE_API_KEY", "")
        self.http = session or requests.Session()
        self.url = url
        self._stop_event = threading.Event()
        self.status = "starting"
        self._seen = set()

    def stop(self):
        self._stop_event.set()

    def run(self):
        page_token = None
        failures = 0
        while not self._stop_event.is_set():
            if not self.api_key:
                self._unavailable("missing YOUTUBE_API_KEY")
                return
            if not self.budget.spend(config.YOUTUBE_CHAT_UNITS_PER_CALL):
                self._unavailable("YouTube chat quota budget used for today")
                self._stop_event.wait(3600)
                continue
            params = {"liveChatId": self.live_chat_id, "part": "snippet,authorDetails", "maxResults": 2000}
            if page_token:
                params["pageToken"] = page_token
            try:
                resp = self.http.get(self.url, params=params, headers={"X-Goog-Api-Key": self.api_key},
                                     timeout=config.REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as exc:
                failures += 1
                self._unavailable(f"network error ({type(exc).__name__})")
                self._stop_event.wait(min(300, 15 * failures))
                continue

            if not resp.ok:
                reason = _reason(resp)
                if resp.status_code in (401, 403, 404) and reason not in ("quotaExceeded", "rateLimitExceeded"):
                    self._unavailable(f"YouTube chat unavailable (HTTP {resp.status_code} {reason})".strip())
                    if reason in STOP_REASONS or resp.status_code in (401, 404):
                        return
                failures += 1
                self._unavailable(f"YouTube chat error (HTTP {resp.status_code} {reason})")
                self._stop_event.wait(min(900, 60 * failures))
                continue

            failures = 0
            data = resp.json()
            self.chat_log.coverage(True)
            self.status = "polling"
            for item in data.get("items", []):
                self._store(item)
            page_token = data.get("nextPageToken") or page_token
            if data.get("offlineAt"):
                self._unavailable("live chat ended")
                return
            wait = max(config.YOUTUBE_CHAT_POLL_SECONDS, (data.get("pollingIntervalMillis") or 0) / 1000)
            self._stop_event.wait(wait)
        self.chat_log.coverage(False, "recording stopped")

    def _store(self, item):
        if item.get("id") in self._seen:
            return
        self._seen.add(item.get("id"))
        snip = item.get("snippet", {})
        if snip.get("type") not in ("textMessageEvent", "superChatEvent", "superStickerEvent"):
            return
        text = (snip.get("textMessageDetails") or {}).get("messageText") or snip.get("displayMessage") or ""
        ts = parse_timestamp(snip.get("publishedAt"), time.time())
        author = (item.get("authorDetails") or {}).get("displayName") or snip.get("authorChannelId")
        self.chat_log.message(ts, author, text, (), item.get("id"))

    def _unavailable(self, reason):
        self.status = reason
        self.chat_log.coverage(False, reason)
        log.info("youtube chat %s: %s", self.live_chat_id[:12], reason)
