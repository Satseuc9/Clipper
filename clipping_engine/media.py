"""FFmpeg / FFprobe helpers shared by capture, analysis and editing."""

import contextlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from clipping_engine import config

AUDIO_RATE = 16_000


class MediaError(RuntimeError):
    pass


def run_ffmpeg(args, timeout=None, input_bytes=None, cwd=None):
    """Run ffmpeg with `args` (no leading 'ffmpeg'); raise MediaError with its stderr tail."""
    cmd = [config.FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y", *args]
    if input_bytes is None:
        cmd.insert(1, "-nostdin")
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, input=input_bytes, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise MediaError(f"ffmpeg timed out after {timeout}s") from None
    except FileNotFoundError:
        raise MediaError(f"ffmpeg not found ({config.FFMPEG_PATH}); install FFmpeg") from None
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-5:]
        raise MediaError("ffmpeg failed: " + " | ".join(tail))
    return proc.stdout


def probe(path):
    """Return ffprobe format + streams info as a dict."""
    cmd = [config.FFPROBE_PATH, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
    except FileNotFoundError:
        raise MediaError(f"ffprobe not found ({config.FFPROBE_PATH}); install FFmpeg") from None
    if proc.returncode != 0:
        raise MediaError("ffprobe failed: " + proc.stderr.decode(errors="replace").strip()[-300:])
    return json.loads(proc.stdout or b"{}")


def media_timing(path):
    """(start_time, duration) in seconds for a media file."""
    info = probe(path)
    fmt = info.get("format", {})
    start = float(fmt.get("start_time") or 0.0)
    duration = fmt.get("duration")
    if duration is None:
        durations = [float(s["duration"]) for s in info.get("streams", []) if s.get("duration")]
        duration = max(durations) if durations else 0.0
    return start, float(duration)


def video_size(path):
    for stream in probe(path).get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    return None


def has_audio(path):
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def decode_audio(inputs, start=0.0, duration=None):
    """
    Decode mono 16 kHz float32 audio.

    `inputs` is one media path, or a list of consecutive segment paths that
    are joined with the concat demuxer (used for context across segments).
    """
    paths = [inputs] if isinstance(inputs, (str, bytes)) or hasattr(inputs, "__fspath__") else list(inputs)
    with input_args(paths, start) as args:
        if duration:
            args += ["-t", f"{duration:.3f}"]
        args += ["-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "f32le", "pipe:1"]
        raw = run_ffmpeg(args, timeout=900)
    return np.frombuffer(raw, dtype=np.float32).copy()


@contextlib.contextmanager
def input_args(paths, start=None):
    """
    FFmpeg input arguments for one file, or for consecutive segments joined
    with the concat demuxer (list written to a temporary file).
    """
    paths = [paths] if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__") else list(paths)
    args = ["-ss", f"{start:.3f}"] if start else []
    if len(paths) == 1:
        yield args + ["-i", str(paths[0])]
        return
    fd, name = tempfile.mkstemp(suffix=".ffconcat")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(concat_list([Path(p).resolve() for p in paths]))
        yield args + ["-f", "concat", "-safe", "0", "-i", name]
    finally:
        os.unlink(name)


def _escape(path):
    return str(path).replace("'", "'\\''")


def concat_list(paths):
    """Text for an ffconcat file joining `paths` in order."""
    return "ffconcat version 1.0\n" + "".join(f"file '{_escape(p)}'\n" for p in paths)
