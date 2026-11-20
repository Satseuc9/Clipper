"""
Accurate clip extraction, including moments that cross segment boundaries.

The segments of one recording run are joined with FFmpeg's concat demuxer
(timestamps continue across files) and the clip is cut with input seeking
plus re-encoding, which is frame-accurate. Audio and video are re-encoded
together, so they stay in sync.
"""

import logging
from pathlib import Path

from clipping_engine.capture.segment_manager import locate
from clipping_engine.media import MediaError, concat_list, media_timing, run_ffmpeg
from clipping_engine.storage.recording_storage import segment_file

log = logging.getLogger(__name__)


def extract(session_dir, run, media_start, media_end, out_path):
    """
    Cut [media_start, media_end] (seconds of video in recording run `run`)
    from a session's segments into `out_path` (H.264/AAC MP4).
    """
    session_dir = Path(session_dir)
    segs, offset = locate(session_dir, run, media_start, media_end)
    if not segs:
        raise MediaError("no recording covers this time range")
    paths = [segment_file(session_dir, s["index"]) for s in segs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise MediaError(f"segment file missing: {missing[0].name}")
    duration = media_end - max(media_start, segs[0]["cum_start"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.with_suffix(".ffconcat")
    list_file.write_text(concat_list([p.resolve() for p in paths]))
    try:
        run_ffmpeg([
            "-ss", f"{offset:.3f}", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", str(out_path),
        ], timeout=900)
    finally:
        list_file.unlink(missing_ok=True)
    start, got = media_timing(out_path)
    if got < duration * 0.8:
        raise MediaError(f"extracted clip is too short ({got:.1f}s of {duration:.1f}s)")
    return out_path
