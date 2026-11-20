"""
Segment bookkeeping: completion, timing, recovery, timeline lookups.

Two clocks matter:
  media time  seconds of video in one recording run (segments are contiguous)
  wall time   real clock time, which chat timestamps use

They drift apart: Streamlink starts with a burst of backlog, network stalls
pause the data, and Twitch ad breaks are cut out of the recording entirely.
So the recorder logs "media time T arrived at clock time W" once per second
(from FFmpeg's -progress output) into timing.jsonl, and every finished
segment adds one more sample (its end arrived at its file's mtime).

TimeMap turns those samples into a mapping. Data can never arrive before
it is broadcast, so for media time t:

    wall(t) = t + min(W_i - T_i  over samples with T_i >= t)

The minimum removes late arrivals (backlog, stalls); taking it only over
LATER samples means a gap (ad break) shifts everything after it, but not
what came before. Accuracy is about one second.
"""

import json
import logging
import re
import time
from pathlib import Path

import numpy as np

from clipping_engine.media import MediaError, media_timing
from clipping_engine.storage.recording_storage import (
    list_segment_files, list_sessions, read_json, segment_file, segment_index,
    session_segments, sidecar_for, update_json, write_json,
)

log = logging.getLogger(__name__)

MIN_SEGMENT_SECONDS = 1.0
TIMING_FILE = "timing.jsonl"
_OUT_TIME = re.compile(rb"out_time_us=(\d+)")


# ---------------------------------------------------------------- timing

class ProgressLogger:
    """Tails FFmpeg's -progress file and appends (wall, media) samples to timing.jsonl."""

    def __init__(self, session_dir, run, progress_path):
        self.out = Path(session_dir) / TIMING_FILE
        self.run = run
        self.path = Path(progress_path)
        self.pos = 0
        self.last_media = None

    def poll(self):
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.pos)
                chunk = fh.read()
        except OSError:
            return
        # Only consume complete lines.
        cut = chunk.rfind(b"\n") + 1
        self.pos += cut
        values = _OUT_TIME.findall(chunk[:cut])
        if not values:
            return
        media = int(values[-1]) / 1e6
        if self.last_media is not None and media <= self.last_media:
            return
        self.last_media = media
        with open(self.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run": self.run, "wall": round(time.time(), 3), "media": round(media, 3)}) + "\n")


class TimeMap:
    """Media time of one run <-> wall clock (see module docstring)."""

    def __init__(self, media, wall):
        media, wall = np.asarray(media, float), np.asarray(wall, float)
        order = np.argsort(media, kind="stable")
        self.m = media[order]
        offsets = wall[order] - self.m
        self.w = self.m + np.minimum.accumulate(offsets[::-1])[::-1]

    def wall(self, t):
        t = np.asarray(t, float)
        out = np.interp(t, self.m, self.w)
        out = np.where(t < self.m[0], self.w[0] - (self.m[0] - t), out)
        out = np.where(t > self.m[-1], self.w[-1] + (t - self.m[-1]), out)
        return float(out) if out.ndim == 0 else out

    def media(self, w):
        w = np.asarray(w, float)
        out = np.interp(w, self.w, self.m)
        out = np.where(w < self.w[0], self.m[0] - (self.w[0] - w), out)
        out = np.where(w > self.w[-1], self.m[-1] + (w - self.w[-1]), out)
        return float(out) if out.ndim == 0 else out


def _run_sidecars(session_dir, run):
    """All sidecars of a run (including deleted ones), with cum_start filled in."""
    sides = [s for s in session_segments(session_dir) if s.get("run") == run]
    cum = 0.0
    for s in sides:
        s.setdefault("cum_start", cum)
        cum = s["cum_start"] + s.get("duration", 0.0)
    return sides


def time_map(session_dir, run):
    media, wall = [], []
    path = Path(session_dir) / TIMING_FILE
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("run") == run:
                    media.append(rec["media"])
                    wall.append(rec["wall"])
    for s in _run_sidecars(session_dir, run):
        if s.get("duration") and s.get("mtime_end"):
            media.append(s["cum_start"] + s["duration"])
            wall.append(s["mtime_end"])
    return TimeMap(media, wall) if media else None


# ---------------------------------------------------------------- segments

def complete_segment(session_dir, seg_path, run, final_in_run, wall_end=None):
    """Write the sidecar for a finished segment and queue it for processing."""
    seg_path = Path(seg_path)
    side = {
        "index": segment_index(seg_path),
        "file": seg_path.name,
        "run": run,
        "final_in_run": bool(final_in_run),
        "attempts": 0,
        "error": None,
        "completed": time.time(),
    }
    try:
        media_start, duration = media_timing(seg_path)
    except MediaError as exc:
        media_start, duration = 0.0, 0.0
        side["error"] = str(exc)
    previous = [x for x in _run_sidecars(session_dir, run) if x["index"] < side["index"]]
    cum_start = (previous[-1]["cum_start"] + previous[-1].get("duration", 0.0)) if previous else 0.0
    side.update(
        media_start=media_start,
        duration=duration,
        cum_start=cum_start,                         # media time of the segment start within the run
        mtime_end=wall_end or seg_path.stat().st_mtime,
        size=seg_path.stat().st_size,
        status="queued" if duration >= MIN_SEGMENT_SECONDS else "skipped",
    )
    write_json(sidecar_for(seg_path), side)
    tm = time_map(session_dir, run)
    side["wall_start"], side["wall_end"] = tm.wall(cum_start), tm.wall(cum_start + duration)
    write_json(sidecar_for(seg_path), side)          # provisional; run_segments() recomputes
    log.info("segment ready: %s (%.1fs, run %s%s)", seg_path, duration, run, ", final" if final_in_run else "")
    return side


def recover_interrupted(active_sessions=()):
    """
    After a crash: finish sessions left in "recording" state and requeue
    segments that were being processed. Returns the number of segments queued.
    """
    active = {Path(p).resolve() for p in active_sessions}
    queued = 0
    for session in list_sessions():
        meta_path = session / "session.json"
        meta = read_json(meta_path, {}) or {}
        if meta.get("status") == "recording" and session.resolve() not in active:
            runs = meta.get("runs") or [{"run": 1}]
            run = runs[-1]["run"]
            files = list_segment_files(session)
            pending = [f for f in files if not sidecar_for(f).exists()]
            for f in pending:
                complete_segment(session, f, run, final_in_run=(f == files[-1]))
                queued += 1
            update_json(meta_path, status="ended", ended=time.time(), interrupted=True)
            log.info("recovered interrupted session %s (%d segments)", session, len(pending))
        for side in session_segments(session):
            if side.get("status") == "processing":
                update_json(segment_file(session, side["index"]).with_suffix(".json"), status="queued")
                queued += 1
    return queued


def run_segments(session_dir, run):
    """
    Segments of one recording run, in order (deleted ones excluded), with
    wall_start/wall_end computed from the run's current TimeMap.
    """
    sides = _run_sidecars(session_dir, run)
    tm = time_map(session_dir, run)
    out = []
    for s in sides:
        if s.get("deleted") or not s.get("duration"):
            continue
        if tm is not None:
            s["wall_start"] = tm.wall(s["cum_start"])
            s["wall_end"] = tm.wall(s["cum_start"] + s["duration"])
        out.append(s)
    return out


def wall_times(session_dir, side):
    """(wall_start, wall_end) of a segment."""
    for s in run_segments(session_dir, side["run"]):
        if s["index"] == side["index"]:
            return s["wall_start"], s["wall_end"]
    return side.get("wall_start"), side.get("wall_end")


def previous_segment(session_dir, side):
    for s in run_segments(session_dir, side["run"]):
        if s["index"] == side["index"] - 1:
            return s
    return None


def locate(session_dir, run, media_start, media_end):
    """
    Segments of `run` covering [media_start, media_end] (run media time).
    Returns (segment sidecars, offset of media_start inside the first segment).
    """
    segs = [s for s in run_segments(session_dir, run)
            if s["cum_start"] + s["duration"] > media_start and s["cum_start"] < media_end]
    if not segs:
        return [], 0.0
    indices = [s["index"] for s in segs]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise MediaError("needed segments are not contiguous (one was deleted?)")
    return segs, max(0.0, media_start - segs[0]["cum_start"])
