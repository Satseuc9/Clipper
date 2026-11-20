"""
Where recordings live and how they are cleaned up.

    recordings/<platform>/<streamer>/session_001/
        session.json          streamer, URLs, platform IDs, runs, status
        segment_0001.ts       10-minute MPEG-TS segments (FFmpeg segment muxer)
        segment_0001.json     segment sidecar: timing + processing status
        chat.jsonl            chat messages and chat-coverage markers
        clips.json            clips already made from this session (dedupe)
        logs/                 streamlink / ffmpeg logs

Segment status: "queued" -> "processing" -> "done" | "failed".
Cleanup only deletes segments that are processed AND no longer needed as
context by the next segment, so unprocessed footage is never lost.
"""

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

from clipping_engine import config

log = logging.getLogger(__name__)

SEGMENT_RE = re.compile(r"^segment_(\d{4,})\.ts$")
FINISHED = {"done", "failed", "skipped"}


# ---- small helpers -------------------------------------------------------------------

def safe_name(name):
    """Folder-safe version of a streamer name / channel ID / @handle."""
    cleaned = re.sub(r"[^A-Za-z0-9_.@-]+", "_", str(name).strip())
    return cleaned.strip("._") or "unnamed"


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    """Atomic JSON write (temp file + rename) so readers never see half a file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def update_json(path, **changes):
    data = read_json(path, {}) or {}
    data.update(changes)
    write_json(path, data)
    return data


# ---- layout --------------------------------------------------------------------------

def recordings_root():
    return Path(config.RECORDINGS_DIR)


def streamer_dir(platform, name):
    return recordings_root() / platform / safe_name(name)


def new_session_dir(platform, name, metadata):
    """Create the next session_NNN folder and write session.json."""
    base = streamer_dir(platform, name)
    base.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name.split("_")[1]) for p in base.glob("session_*") if p.name.split("_")[1].isdigit()]
    number = max(numbers, default=0) + 1
    while True:
        path = base / f"session_{number:03d}"
        try:
            path.mkdir()
            break
        except FileExistsError:
            number += 1
    (path / "logs").mkdir()
    meta = dict(metadata)
    meta.update({"session": path.name, "status": "recording", "created": time.time(), "runs": []})
    write_json(path / "session.json", meta)
    return path


def segment_file(session_dir, index):
    return Path(session_dir) / f"segment_{index:04d}.ts"


def sidecar_for(segment_path):
    return Path(segment_path).with_suffix(".json")


def segment_index(path):
    match = SEGMENT_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def list_segment_files(session_dir):
    files = [(segment_index(p), p) for p in Path(session_dir).glob("segment_*.ts")]
    return [p for i, p in sorted(f for f in files if f[0] is not None)]


def list_sessions():
    root = recordings_root()
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*/*/session_*") if (p / "session.json").exists())


def session_segments(session_dir):
    """Sidecars of a session, ordered by segment index."""
    out = []
    for seg in list_segment_files(session_dir):
        side = read_json(sidecar_for(seg))
        if side:
            out.append(side)
    return out


# ---- cleanup -------------------------------------------------------------------------

def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _deletable_segments(session_dir):
    """Segments that are processed and no longer needed by a later segment."""
    segs = session_segments(session_dir)
    by_index = {s["index"]: s for s in segs}
    out = []
    for side in segs:
        if side.get("status") not in FINISHED or side.get("deleted"):
            continue
        nxt = by_index.get(side["index"] + 1)
        # The next segment re-reads the tail of this one for context and
        # cross-boundary clips, so keep it until that segment is processed too.
        if side.get("final_in_run") or (nxt is not None and nxt.get("status") in FINISHED):
            out.append(side)
    return out


def session_fully_processed(session_dir):
    meta = read_json(Path(session_dir) / "session.json", {}) or {}
    if meta.get("status") == "recording":
        return False
    files = list_segment_files(session_dir)
    sides = [read_json(sidecar_for(f)) for f in files]
    return all(s and s.get("status") in FINISHED for s in sides)


def cleanup_recordings(now=None):
    """Apply RECORDING_RETENTION_HOURS and MAX_RECORDINGS_GB. Returns bytes freed."""
    now = now or time.time()
    freed = 0

    # 1. Whole sessions that are finished, processed and old.
    for session in list_sessions():
        meta = read_json(session / "session.json", {}) or {}
        ended = meta.get("ended") or meta.get("created", now)
        if session_fully_processed(session) and now - ended > config.RECORDING_RETENTION_HOURS * 3600:
            size = _dir_size(session)
            shutil.rmtree(session, ignore_errors=True)
            freed += size
            log.info("deleted old session %s (%.1f MB)", session, size / 1e6)

    # 2. Size limit: delete processed segments, oldest first.
    limit = config.MAX_RECORDINGS_GB * 1024 ** 3
    total = _dir_size(recordings_root()) if recordings_root().exists() else 0
    if total <= limit:
        return freed
    candidates = []
    for session in list_sessions():
        for side in _deletable_segments(session):
            seg = segment_file(session, side["index"])
            if seg.exists():
                candidates.append((seg.stat().st_mtime, seg, session))
    for _, seg, session in sorted(candidates):
        if total <= limit:
            break
        size = seg.stat().st_size
        seg.unlink(missing_ok=True)
        side_path = sidecar_for(seg)
        update_json(side_path, deleted=True)
        total -= size
        freed += size
        log.info("storage limit: deleted processed segment %s", seg)
    if total > limit:
        log.warning("recordings use %.1f GB (> %s GB) but the rest is not processed yet",
                    total / 1024 ** 3, config.MAX_RECORDINGS_GB)
    return freed
