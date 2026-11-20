"""
Streamlink integration: version/plugin checks and the capture commands.

Capture is   streamlink --stdout <url> <quality>  |  ffmpeg -f segment ...
Streamlink fetches the stream (handling HLS, Twitch ad filtering, etc.) and
FFmpeg copies the audio/video packets - no re-encoding, so A/V sync is kept -
into consecutive, individually playable MPEG-TS segments.
"""

import functools
import logging
import re
import subprocess

from clipping_engine import config

log = logging.getLogger(__name__)

# Example URL per platform, used to ask Streamlink whether it has a plugin.
SAMPLE_URLS = {
    "twitch": "https://www.twitch.tv/twitch",
    "kick": "https://kick.com/kick",
    "youtube": "https://www.youtube.com/watch?v=jfKfPfyJRdk",
}


class StreamlinkError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def streamlink_version():
    """Installed Streamlink version string, or None if Streamlink is missing."""
    try:
        out = subprocess.run([config.STREAMLINK_PATH, "--version"], capture_output=True,
                             text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", out.stdout + out.stderr)
    return match.group(1) if match else None


@functools.lru_cache(maxsize=256)
def can_handle_url(url):
    """True if the installed Streamlink has a plugin for this URL (`--can-handle-url`)."""
    try:
        proc = subprocess.run([config.STREAMLINK_PATH, "--can-handle-url", url],
                              capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def platform_support():
    """{platform: (supported, message)} for every platform in CLIP_PLATFORMS."""
    version = streamlink_version()
    out = {}
    for platform in config.CLIP_PLATFORMS:
        if version is None:
            out[platform] = (False, f"Streamlink not found ({config.STREAMLINK_PATH})")
        elif platform not in SAMPLE_URLS:
            out[platform] = (False, "no capture support for this platform")
        elif can_handle_url(SAMPLE_URLS[platform]):
            out[platform] = (True, f"Streamlink {version} has a {platform} plugin")
        else:
            out[platform] = (False, f"Streamlink {version} has no {platform} plugin; upgrade Streamlink")
    return out


def streamlink_command(url):
    return [
        config.STREAMLINK_PATH,
        "--stdout",
        "--loglevel", "info",
        "--stream-timeout", str(config.STREAM_TIMEOUT_SECONDS),
        "--retry-open", "3",
        *config.STREAMLINK_EXTRA_ARGS,
        url,
        config.STREAM_QUALITY,
    ]


def ffmpeg_segment_command(session_dir, start_number, progress_path=None):
    """
    FFmpeg reads the stream on stdin and writes segment_NNNN.ts files.
    With progress_path, FFmpeg reports its output time every second; the
    recorder logs those reports against the clock (see segment_manager.TimeMap).
    """
    seconds = max(1, int(round(config.RECORDING_SEGMENT_MINUTES * 60)))
    progress = ["-progress", str(progress_path), "-stats_period", "1"] if progress_path else []
    return [
        config.FFMPEG_PATH, "-hide_banner", "-nostdin", "-loglevel", "warning",
        "-fflags", "+genpts+discardcorrupt",
        "-i", "pipe:0",
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(seconds),
        "-segment_format", "mpegts",
        "-reset_timestamps", "1",
        "-segment_start_number", str(start_number),
        *progress,
        str(session_dir / "segment_%04d.ts"),
    ]
