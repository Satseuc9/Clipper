"""
Finished clips and the per-session clip registry.

    clips/<platform>/<streamer>/<date>_<session>_seg0004_94.mp4   vertical clip
    clips/<platform>/<streamer>/<same name>.json                   moment metadata

The registry (recordings/.../session_NNN/clips.json) lists every moment
already clipped in a session, in absolute time, so the same moment is never
clipped twice - even when it is found again from the next segment.
"""

import logging
import time
from pathlib import Path
from urllib.parse import quote

from clipping_engine import config
from clipping_engine.storage.recording_storage import _dir_size, read_json, safe_name, write_json

log = logging.getLogger(__name__)


def clips_root():
    return Path(config.CLIPS_DIR)


def clip_path(platform, streamer, session_name, segment_index, score, wall_time):
    folder = clips_root() / platform / safe_name(streamer)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(wall_time))
    return folder / f"{stamp}_{session_name}_seg{segment_index:04d}_{int(score)}.{config.OUTPUT_FORMAT}"


def public_url(path):
    """Link to a clip if CLIP_PUBLIC_BASE_URL is configured, else None."""
    base = config.CLIP_PUBLIC_BASE_URL
    if not base:
        return None
    rel = Path(path).resolve().relative_to(clips_root().resolve())
    return base.rstrip("/") + "/" + "/".join(quote(part) for part in rel.parts)


# ---- registry ----------------------------------------------------------------------------

def registry_path(session_dir):
    return Path(session_dir) / "clips.json"


def load_registry(session_dir):
    return read_json(registry_path(session_dir), []) or []


def overlaps_existing(session_dir, abs_start, abs_end, abs_peak, gap=None):
    """True if this moment was already clipped (overlapping window or nearby peak)."""
    gap = config.DEDUPE_GAP_SECONDS if gap is None else gap
    for entry in load_registry(session_dir):
        if abs(entry["abs_peak"] - abs_peak) < gap:
            return True
        overlap = min(abs_end, entry["abs_end"]) - max(abs_start, entry["abs_start"])
        if overlap > 0.5 * min(abs_end - abs_start, entry["abs_end"] - entry["abs_start"]):
            return True
    return False


def register_clip(session_dir, entry):
    reg = load_registry(session_dir)
    reg.append(entry)
    write_json(registry_path(session_dir), reg)


def update_clip(session_dir, clip_id, **changes):
    reg = load_registry(session_dir)
    for entry in reg:
        if entry.get("id") == clip_id:
            entry.update(changes)
    write_json(registry_path(session_dir), reg)


# ---- cleanup -----------------------------------------------------------------------------

def cleanup_clips():
    """Delete oldest clips above MAX_CLIPS_GB. Returns bytes freed."""
    root = clips_root()
    if not root.exists():
        return 0
    limit = config.MAX_CLIPS_GB * 1024 ** 3
    total = _dir_size(root)
    freed = 0
    if total <= limit:
        return 0
    for clip in sorted(root.rglob(f"*.{config.OUTPUT_FORMAT}"), key=lambda p: p.stat().st_mtime):
        if total <= limit:
            break
        size = clip.stat().st_size
        clip.unlink(missing_ok=True)
        for extra in (clip.with_suffix(".json"), clip.with_name(clip.stem + "_preview.mp4")):
            if extra.exists():
                size += extra.stat().st_size
                extra.unlink()
        total -= size
        freed += size
        log.info("clip storage limit: deleted %s", clip)
    return freed
