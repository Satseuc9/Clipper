"""
Shared types for the platform modules.

Every platform module exposes a Monitor class with:
    check(names, requested_interval) -> list[StreamResult]

check() never raises: any failure becomes an UNKNOWN result with an error
message, so one broken platform cannot stop the others. An error is never
reported as OFFLINE.
"""

import time
from dataclasses import dataclass, field

from utils.http import ApiError, RateLimitError, new_session

LIVE = "LIVE"
OFFLINE = "OFFLINE"
UNKNOWN = "UNKNOWN"


@dataclass
class StreamResult:
    platform: str                  # "twitch", "kick", ...
    name: str                      # identifier exactly as written in config.py
    status: str                    # LIVE, OFFLINE or UNKNOWN
    display_name: str = ""         # nicer name when the API provides one
    viewers: int | None = None
    title: str | None = None
    url: str | None = None
    error: str | None = None       # why the status is UNKNOWN
    checked_at: float = field(default_factory=time.time)  # when the API confirmed it
    # Platform IDs other tools can reuse (e.g. the clipping engine):
    # twitch: user_id, started_at | kick: broadcaster_user_id, started_at
    # youtube: channel_id, video_id, live_chat_id, started_at
    extra: dict = field(default_factory=dict)

    @property
    def label(self):
        return self.display_name or self.name


def format_duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 120:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {seconds % 3600 // 60}m"


def dedupe(names):
    """Strip blanks and duplicates while keeping config order."""
    seen, out = set(), []
    for name in names or []:
        name = str(name).strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


class PlatformMonitor:
    key = "base"          # key used in config.STREAMERS
    title = "BASE"        # heading in the report
    available = True      # False when no official API exists

    def __init__(self, timeout=10, min_interval=0):
        self.timeout = timeout
        self.min_interval = min_interval
        self.session = new_session()
        self._cache = None             # (names, results) of the last real poll
        self._last_poll = None         # time.monotonic() of the last real poll
        self._blocked_until = 0.0      # monotonic time; set after rate limits
        self._block_reason = ""

    # ---- to be implemented by each platform ---------------------------------

    def _fetch(self, names):
        """Query the API for `names` and return a list of StreamResult."""
        raise NotImplementedError

    def required_interval(self, names):
        """Platform-specific lower bound on the polling interval, in seconds."""
        return self.min_interval

    # ---- shared logic ---------------------------------------------------------

    def poll_interval(self, names, requested_interval):
        return max(int(requested_interval), int(self.required_interval(dedupe(names))))

    def block(self, seconds, reason):
        """Stop calling the API for `seconds` (used for rate limits / quota)."""
        self._blocked_until = time.monotonic() + max(1, seconds)
        self._block_reason = reason

    def unknown(self, name, error):
        return StreamResult(self.key, name, UNKNOWN, error=error)

    def check(self, names, requested_interval):
        names = dedupe(names)
        if not names:
            return []

        now = time.monotonic()
        if now < self._blocked_until:
            wait = format_duration(self._blocked_until - now + 1)
            return [self.unknown(n, f"{self._block_reason}; retrying in {wait}") for n in names]

        # Reuse the last poll if this platform's interval has not elapsed yet.
        interval = self.poll_interval(names, requested_interval)
        if (
            self._cache is not None
            and self._cache[0] == names
            and now - self._last_poll < interval - 2  # 2s slack for timer jitter
        ):
            return self._cache[1]

        try:
            results = self._fetch(names)
        except RateLimitError as exc:
            self.block(exc.retry_after, str(exc))
            wait = format_duration(exc.retry_after)
            results = [self.unknown(n, f"{exc}; retrying in {wait}") for n in names]
        except ApiError as exc:
            results = [self.unknown(n, str(exc)) for n in names]
        except Exception as exc:  # a bug or an unexpected response shape
            results = [self.unknown(n, f"unexpected error ({type(exc).__name__})") for n in names]

        self._last_poll = time.monotonic()
        self._cache = (names, results)
        return results
