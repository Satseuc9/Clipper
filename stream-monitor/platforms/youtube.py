"""
YouTube live status via the official YouTube Data API v3 (API key).

search.list?eventType=live costs 100 quota units per call, which would use up
the default 10,000 units/day within minutes. This module uses 1-unit calls:

  once:       channels.list   (id=UC..., up to 50 per call, or forHandle=@name)
              -> channel title + "uploads" playlist ID
  every poll: playlistItems.list (uploads playlist, newest N videos)  1 unit/channel
              videos.list (snippet,liveStreamingDetails; up to 50 IDs)  1 unit/50 videos

A channel is LIVE when one of its recent uploads has
snippet.liveBroadcastContent == "live" and liveStreamingDetails.actualStartTime
set without actualEndTime. Viewers come from liveStreamingDetails.concurrentViewers
(absent when the owner hides the count).

Videos that can never become live again (regular uploads, ended broadcasts)
are remembered and not looked up twice, so videos.list is usually one call or
none per poll.

Quota: usage is counted per Pacific-time day. Polling is slowed so the
estimated daily use stays within YOUTUBE_DAILY_QUOTA * YOUTUBE_QUOTA_SHARE,
and stops until midnight Pacific if that budget or the real quota runs out.

Limitation: only public live streams that show up in the channel's uploads
playlist are detected (members-only or unlisted streams are not).

Credentials (.env): YOUTUBE_API_KEY  (sent in the X-Goog-Api-Key header,
never in the URL)
"""

import math
import os
import time
from datetime import datetime, timedelta, timezone

from platforms.base import LIVE, OFFLINE, PlatformMonitor, StreamResult
from utils.http import ApiError, RateLimitError, request_json

try:
    from zoneinfo import ZoneInfo

    PACIFIC = ZoneInfo("America/Los_Angeles")
except Exception:  # no tz database (e.g. Windows without the tzdata package)
    PACIFIC = timezone(timedelta(hours=-8))

API = "https://www.googleapis.com/youtube/v3"
BATCH_SIZE = 50
RECHECK_MISSING_AFTER = 3600
QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}
SHORT_LIMIT_REASONS = {"rateLimitExceeded", "userRateLimitExceeded"}


def seconds_until_pacific_midnight():
    now = datetime.now(PACIFIC)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
    return max(60, int((midnight - now).total_seconds()))


def google_error_reason(data):
    try:
        return data["error"]["errors"][0]["reason"]
    except (KeyError, IndexError, TypeError):
        return ""


class YouTubeMonitor(PlatformMonitor):
    key = "youtube"
    title = "YOUTUBE"

    def __init__(self, timeout=10, min_interval=60, daily_quota=10_000,
                 quota_share=0.9, videos_to_scan=10):
        super().__init__(timeout, min_interval)
        self.api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
        self.configured = bool(self.api_key)
        if self.configured:
            self.session.headers["X-Goog-Api-Key"] = self.api_key
        self.budget = max(1, int(daily_quota * quota_share))
        self.videos_to_scan = min(50, max(1, int(videos_to_scan)))
        self._channels = {}         # config entry -> {"id", "title", "uploads"}
        self._missing = {}          # config entry -> time it was last not found
        self._done_videos = set()   # video IDs that can no longer be live
        self._quota_day = None
        self._units_used = 0

    # ---- quota ------------------------------------------------------------------

    def required_interval(self, names):
        """Slowest safe polling rate: ~1 unit per channel + 1 videos.list call."""
        units_per_poll = len(names) + 1
        quota_interval = math.ceil(units_per_poll * 86400 / self.budget)
        return max(self.min_interval, quota_interval)

    def _spend(self, units=1):
        today = datetime.now(PACIFIC).date()
        if today != self._quota_day:
            self._quota_day, self._units_used = today, 0
        if self._units_used + units > self.budget:
            raise RateLimitError(
                f"daily YouTube quota budget used ({self._units_used}/{self.budget} units)",
                retry_after=seconds_until_pacific_midnight(),
            )
        self._units_used += units  # Google charges failed requests too

    def _call(self, endpoint, params):
        self._spend(1)
        try:
            _, data = request_json(
                self.session, "GET", f"{API}/{endpoint}", timeout=self.timeout, params=params
            )
            return data
        except RateLimitError:
            raise
        except ApiError as exc:
            reason = google_error_reason(exc.data)
            if reason in QUOTA_REASONS:
                raise RateLimitError(
                    "YouTube daily quota exhausted", retry_after=seconds_until_pacific_midnight()
                ) from None
            if reason in SHORT_LIMIT_REASONS:
                raise RateLimitError("YouTube rate limit hit", retry_after=120) from None
            raise

    # ---- monitoring -------------------------------------------------------------

    def _store_channel(self, name, item):
        self._channels[name] = {
            "id": item["id"],
            "title": item.get("snippet", {}).get("title") or name,
            "uploads": item["contentDetails"]["relatedPlaylists"]["uploads"],
        }
        self._missing.pop(name, None)

    def _resolve_channels(self, names):
        """Look up title + uploads playlist once per channel (cached)."""
        now = time.time()
        todo = [
            n for n in names
            if n not in self._channels and now - self._missing.get(n, 0) > RECHECK_MISSING_AFTER
        ]
        fields = "items(id,snippet/title,contentDetails/relatedPlaylists/uploads)"
        part = "snippet,contentDetails"

        ids = [n for n in todo if not n.startswith("@")]
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i:i + BATCH_SIZE]
            data = self._call("channels", {"part": part, "id": ",".join(batch),
                                           "maxResults": BATCH_SIZE, "fields": fields})
            found = {item["id"]: item for item in data.get("items", [])}
            for name in batch:
                if name in found:
                    self._store_channel(name, found[name])
                else:
                    self._missing[name] = now

        for handle in (n for n in todo if n.startswith("@")):
            data = self._call("channels", {"part": part, "forHandle": handle, "fields": fields})
            items = data.get("items") or []
            if items:
                self._store_channel(handle, items[0])
            else:
                self._missing[handle] = now

    def _recent_uploads(self, channel):
        """Newest video IDs of a channel, or an error string."""
        try:
            data = self._call("playlistItems", {
                "part": "contentDetails",
                "playlistId": channel["uploads"],
                "maxResults": self.videos_to_scan,
                "fields": "items/contentDetails/videoId",
            })
        except RateLimitError:
            raise
        except ApiError as exc:
            if exc.status == 404:
                return "uploads playlist not found (channel has no public videos?)"
            raise
        return [item["contentDetails"]["videoId"] for item in data.get("items", [])]

    def _check_videos(self, video_ids):
        """Return {video_id: {"title", "viewers"}} for videos that are live right now."""
        todo = [v for v in dict.fromkeys(video_ids) if v not in self._done_videos]
        live = {}
        fields = ("items(id,snippet(title,liveBroadcastContent),liveStreamingDetails("
                  "actualStartTime,actualEndTime,concurrentViewers,activeLiveChatId))")
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            data = self._call("videos", {"part": "snippet,liveStreamingDetails",
                                         "id": ",".join(batch), "fields": fields})
            returned = set()
            for item in data.get("items", []):
                vid = item["id"]
                returned.add(vid)
                state = item.get("snippet", {}).get("liveBroadcastContent")
                details = item.get("liveStreamingDetails")
                if (state == "live" and details and details.get("actualStartTime")
                        and not details.get("actualEndTime")):
                    viewers = details.get("concurrentViewers")  # a string in the API
                    live[vid] = {
                        "title": item["snippet"].get("title"),
                        "viewers": int(viewers) if str(viewers).isdigit() else None,
                        "started_at": details.get("actualStartTime"),
                        "live_chat_id": details.get("activeLiveChatId"),
                    }
                elif state not in ("live", "upcoming") and (not details or details.get("actualEndTime")):
                    self._done_videos.add(vid)   # normal upload or finished broadcast
                # "upcoming" videos are checked again next poll
            # IDs missing from the response are private/deleted: not publicly live.
            self._done_videos.update(set(batch) - returned)
        return live

    def _fetch(self, names):
        if not self.configured:
            raise ApiError("missing YOUTUBE_API_KEY in .env")

        self._resolve_channels(names)
        uploads = {n: self._recent_uploads(self._channels[n]) for n in names if n in self._channels}
        all_ids = [v for ids in uploads.values() if isinstance(ids, list) for v in ids]
        live = self._check_videos(all_ids)

        checked_at = time.time()
        results = []
        for name in names:
            channel = self._channels.get(name)
            if channel is None:
                results.append(self.unknown(name, "channel not found (check the channel ID or @handle)"))
                continue
            ids = uploads[name]
            if isinstance(ids, str):
                results.append(self.unknown(name, ids))
                continue
            live_here = [(v, live[v]) for v in ids if v in live]
            if live_here:
                vid, info = max(live_here, key=lambda x: x[1]["viewers"] or 0)
                results.append(StreamResult(
                    self.key, name, LIVE, display_name=channel["title"],
                    viewers=info["viewers"], title=info["title"],
                    url=f"https://www.youtube.com/watch?v={vid}", checked_at=checked_at,
                    extra={"channel_id": channel["id"], "video_id": vid,
                           "live_chat_id": info["live_chat_id"], "started_at": info["started_at"]},
                ))
            else:
                results.append(StreamResult(
                    self.key, name, OFFLINE, display_name=channel["title"],
                    url=f"https://www.youtube.com/channel/{channel['id']}", checked_at=checked_at,
                    extra={"channel_id": channel["id"]},
                ))
        return results
