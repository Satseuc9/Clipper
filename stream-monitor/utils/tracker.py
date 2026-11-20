"""
In-memory status history: keeps the last confirmed status of every streamer,
detects OFFLINE -> LIVE and LIVE -> OFFLINE transitions, and decides what to
show when a check fails.
"""

import time
from dataclasses import dataclass, replace

from platforms.base import LIVE, OFFLINE, UNKNOWN, StreamResult


@dataclass
class StatusChange:
    time: float
    old: str
    new: StreamResult


class StatusTracker:
    def __init__(self, keep_last_status_seconds=600):
        self.keep_seconds = keep_last_status_seconds
        self.confirmed = {}   # (platform, name) -> last StreamResult that was LIVE/OFFLINE
        self.history = []     # recent StatusChange objects, newest last

    def update(self, platform, results, poll_interval):
        """
        Record one platform's results. Returns (rows_to_display, changes).

        - LIVE/OFFLINE results are stored; a change from the previous
          confirmed status is reported as a StatusChange.
        - An UNKNOWN result (failed check) shows the previous confirmed status
          if it is recent enough, marked stale with the error; otherwise it
          stays UNKNOWN. A failure never changes the stored status.
        """
        now = time.time()
        keep = max(self.keep_seconds, 2 * poll_interval)
        rows, changes = [], []
        for result in results:
            key = (platform, result.name.lower())
            previous = self.confirmed.get(key)

            if result.status in (LIVE, OFFLINE):
                if previous and previous.status != result.status:
                    changes.append(StatusChange(now, previous.status, result))
                self.confirmed[key] = result
                rows.append(result)
            elif previous and now - previous.checked_at <= keep:
                rows.append(replace(previous, error=result.error))  # stale but reliable
            else:
                rows.append(result)

        self.history.extend(changes)
        del self.history[:-50]
        return rows, changes


def is_stale(row):
    """A LIVE/OFFLINE row that is being shown from memory after a failed check."""
    return row.status != UNKNOWN and row.error is not None
