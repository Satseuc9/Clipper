"""
chat.jsonl: one JSON object per line, written while recording.

    {"kind": "msg", "ts": 1795140580.12, "user": "viewer32", "text": "KEKW",
     "emotes": ["KEKW"], "source": "twitch"}
    {"kind": "coverage", "ts": ..., "available": true,  "source": "twitch"}
    {"kind": "coverage", "ts": ..., "available": false, "reason": "..."}

`ts` is the platform's own message timestamp (epoch seconds). Coverage
markers record when chat collection was actually working, so analysis can
tell "chat was quiet" apart from "we had no chat data" - missing chat is
never treated as zero activity.
"""

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_FRACTION = re.compile(r"(\.\d{6})\d+")


def parse_timestamp(value, default=None):
    """RFC 3339 (incl. nanoseconds and 'Z') -> epoch seconds."""
    if not value:
        return default
    try:
        text = _FRACTION.sub(r"\1", str(value).strip()).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return default


class ChatLog:
    """Thread-safe appender for one session's chat.jsonl."""

    def __init__(self, session_dir, source):
        self.path = Path(session_dir) / "chat.jsonl"
        self.source = source
        self._lock = threading.Lock()
        self.messages = 0
        self.available = False

    def _write(self, record):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)

    def message(self, ts, user, text, emotes=(), msg_id=None):
        self.messages += 1
        self._write({"kind": "msg", "ts": float(ts), "user": user, "text": text,
                     "emotes": list(emotes), "id": msg_id, "source": self.source})

    def coverage(self, available, reason=None):
        if available == self.available and available:
            return
        self.available = available
        rec = {"kind": "coverage", "ts": time.time(), "available": bool(available), "source": self.source}
        if reason:
            rec["reason"] = reason
        self._write(rec)


def load_chat(session_dir):
    """Return (messages sorted by ts, coverage intervals [(start, end)])."""
    path = Path(session_dir) / "chat.jsonl"
    messages, intervals = [], []
    open_since = None
    if not path.exists():
        return messages, intervals
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue          # a half-written last line after a crash
            if rec.get("kind") == "msg":
                messages.append(rec)
            elif rec.get("kind") == "coverage":
                if rec.get("available") and open_since is None:
                    open_since = rec["ts"]
                elif not rec.get("available") and open_since is not None:
                    intervals.append((open_since, rec["ts"]))
                    open_since = None
    if open_since is not None:
        intervals.append((open_since, float("inf")))
    messages.sort(key=lambda m: m["ts"])
    return messages, intervals


def coverage_fraction(intervals, start, end):
    """Share of [start, end] during which chat collection was working."""
    if end <= start:
        return 0.0
    covered = sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)
    return min(1.0, covered / (end - start))
