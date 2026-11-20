"""
Clipping engine settings.

Streamers are NOT listed here: the engine records every streamer from
stream-monitor/config.py (STREAMERS) on the platforms in CLIP_PLATFORMS.
Credentials live in .env files (see .env.example), never here.
"""

from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent

# ---- What to record ---------------------------------------------------------

# Platforms to record. TikTok is not supported (no official API / capture path).
CLIP_PLATFORMS = ["twitch", "kick", "youtube"]

# Optional allow-list of streamer names from stream-monitor/config.py.
# None = record every monitored streamer on CLIP_PLATFORMS.
# Example: {"twitch": ["streamer_one"], "kick": ["streamer_four"]}
CLIP_ONLY = None

# ---- Recording ----------------------------------------------------------------

RECORDINGS_DIR = ENGINE_DIR / "recordings"
CLIPS_DIR = ENGINE_DIR / "clips"
MODEL_DIR = ENGINE_DIR / "model_cache"

RECORDING_SEGMENT_MINUTES = 10
MAX_CONCURRENT_RECORDINGS = 3       # each live streamer adds one segment to analyse per RECORDING_SEGMENT_MINUTES

# Streamlink quality list, first available wins. 720p keeps analysis and disk use
# manageable; "best" is the fallback for channels without a 720p rendition.
# For sharper clips use "1080p60,1080p,936p60,900p60,720p60,720p,best".
STREAM_QUALITY = "720p60,720p,best"
STREAMLINK_PATH = "streamlink"
FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"
STREAMLINK_EXTRA_ARGS = []        # e.g. ["--twitch-low-latency"]
STREAM_TIMEOUT_SECONDS = 60       # streamlink gives up after this long without data
STALL_RESTART_SECONDS = 90        # restart if the current segment stops growing (ad breaks excluded)
MAX_AD_BREAK_SECONDS = 600        # a Twitch ad pause longer than this is treated as a stall
RECONNECT_DELAYS_SECONDS = [5, 15, 30, 60, 120]   # back-off between reconnects
RECORDING_GIVE_UP_MINUTES = 15    # stop reconnecting after this long without receiving any data...
RECORDING_RETRY_COOLDOWN_MINUTES = 10   # ...and wait this long before trying that streamer again

# ---- Storage limits -------------------------------------------------------------

MAX_RECORDINGS_GB = 100           # processed recordings are deleted oldest-first above this
RECORDING_RETENTION_HOURS = 48    # fully processed sessions older than this are deleted
MAX_CLIPS_GB = 50                 # finished clips are deleted oldest-first above this
KEEP_UNPROCESSED_RECORDINGS = True  # never delete segments that are not processed yet
MIN_FREE_DISK_GB = 20             # no new recordings below this; running ones stop below half of it

# HTTP timeout for chat, Discord and other API requests made by the engine.
REQUEST_TIMEOUT_SECONDS = 15

# ---- Chat -------------------------------------------------------------------------

ENABLE_CHAT_COLLECTION = True
# Seconds viewers typically need to react in chat after something happens.
CHAT_REACTION_DELAY_SECONDS = 4
# Extra offset if your chat timestamps are consistently ahead/behind the video.
CHAT_SYNC_OFFSET_SECONDS = 0.0
# YouTube chat polling: liveChatMessages.list costs quota on every call.
YOUTUBE_CHAT_POLL_SECONDS = 20
YOUTUBE_CHAT_UNITS_PER_CALL = 5
YOUTUBE_CHAT_DAILY_UNITS = 3000   # chat budget; stream-monitor's YouTube polling uses its own share
# Kick chat uses official webhooks and needs a public URL (see README).
KICK_WEBHOOK_ENABLED = False
KICK_WEBHOOK_HOST = "127.0.0.1"
KICK_WEBHOOK_PORT = 8787

# ---- Analysis -----------------------------------------------------------------------

ENABLE_AUDIO_ANALYSIS = True
ENABLE_CHAT_ANALYSIS = True
ENABLE_VIDEO_ANALYSIS = True
ENABLE_TRANSCRIPTION = True
ENABLE_AI_EVALUATION = True       # Claude; skipped automatically without ANTHROPIC_API_KEY

# Each segment is analysed together with the last ANALYSIS_CONTEXT_SECONDS of the
# previous one, so moments near a boundary get context and are not missed.
ANALYSIS_CONTEXT_SECONDS = 60
# Moments in the last SEGMENT_TAIL_GUARD_SECONDS of a segment are left to the
# next segment's analysis (which has the footage after them).
SEGMENT_TAIL_GUARD_SECONDS = 25

# Stage 1 keeps at most this many candidate moments per segment: the segment's
# strongest peaks relative to its own baseline, so every segment gets scored.
MAX_CANDIDATES_PER_SEGMENT = 8
STAGE1_MIN_INTEREST = 0.05        # ignore peaks below this (near-silence)
# Stage 2 (transcription, frames) runs on candidates whose pre-score reaches this.
STAGE2_MIN_PRESCORE = 25
# Stage 3 (Claude) evaluates at most this many candidates per segment...
AI_MAX_CANDIDATES_PER_SEGMENT = 3
# ...and only those whose score before AI reaches this.
AI_MIN_PRESCORE = 35

# Speech-to-text (faster-whisper). Models: tiny, base, small, medium, large-v3, ...
WHISPER_MODEL = "small"
WHISPER_DEVICE = "auto"           # "cpu", "cuda" or "auto"
WHISPER_COMPUTE_TYPE = "default"  # e.g. "int8" on CPU, "float16" on GPU
WHISPER_LANGUAGE = None           # None = auto-detect, or "en", "es", ...

# Claude model for the AI entertainment evaluation.
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "medium"          # low | medium | high | xhigh | max
AI_FRAMES_PER_CANDIDATE = 4       # still frames sent to Claude per moment

# Audio model thresholds (YAMNet scores are 0-1 per class).
LAUGHTER_THRESHOLD = 0.15
SCREAM_THRESHOLD = 0.20
CHEER_THRESHOLD = 0.25

# ---- Viral scoring -----------------------------------------------------------------------

SCORE_WEIGHTS = {
    "audio": 0.25,
    "chat": 0.25,
    "ai": 0.30,
    "speech": 0.20,
}
# If only one available signal is strong, the score is multiplied by this.
SINGLE_SIGNAL_FACTOR = 0.8
MINIMUM_VIRAL_SCORE = 75          # clips at or above this are rendered and sent
MAX_CLIPS_PER_SEGMENT = 2         # strongest moments per 10-minute segment

# ---- Clips -----------------------------------------------------------------------------------

DEFAULT_CLIP_DURATION = 60
MIN_CLIP_DURATION = 30
MAX_CLIP_DURATION = 90
PRE_CLIP_BUFFER_SECONDS = 15      # context before the moment (10-20 recommended)
POST_CLIP_BUFFER_SECONDS = 10     # time after the moment (5-15 recommended)
DEDUPE_GAP_SECONDS = 20           # moments closer than this are the same moment

OUTPUT_RESOLUTION = "1080x1920"
OUTPUT_FORMAT = "mp4"
OUTPUT_CRF = 20                   # x264 quality (lower = better/larger)
OUTPUT_PRESET = "veryfast"
TARGET_LOUDNESS_LUFS = -14        # typical for TikTok / Shorts / Reels
LAYOUT = "auto"                   # auto | face_crop | stacked | fit_blur
GAMEPLAY_FILL = False             # stacked layout: True crops gameplay sides to fill more height

CAPTIONS = True
ANIMATED_CAPTIONS = False         # word-by-word highlight captions
CAPTION_FONT = "DejaVu Sans"
CAPTION_FONT_SIZE = 78
CAPTION_MAX_WORDS = 5

# ---- Discord ----------------------------------------------------------------------------------

DISCORD_NOTIFICATIONS = True
DISCORD_MAX_UPLOAD_MB = 10        # default server upload limit; raise for boosted servers
DISCORD_SEND_PREVIEW_IF_TOO_LARGE = True   # attach a smaller preview when the clip is too big
# Base URL where CLIPS_DIR is reachable (your own file server / bucket), e.g.
# "https://files.example.com/clips". Used for links to clips that are too large.
CLIP_PUBLIC_BASE_URL = None

# ---- Processing -------------------------------------------------------------------------------

# Worker processes for analysis; recording is never blocked by analysis.
# "auto" = half of the Mac's performance cores (all cores / 4 elsewhere), between 1 and 3:
# 2 on M1/M2/M3/M4, 3 on Pro/Max chips. Or set a number.
ANALYSIS_WORKERS = "auto"
# Analysis runs at lower CPU priority (Unix "nice" 0-19) so the computer stays responsive;
# recording keeps normal priority. 0 = same priority as everything else.
ANALYSIS_NICENESS = 10
PROCESSING_MAX_ATTEMPTS = 3
PROCESSING_RETRY_DELAY_SECONDS = 60
