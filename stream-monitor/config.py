"""
Streamer list and monitor settings.

Edit STREAMERS to add or remove streamers. No other file needs to change.
Restart the script after editing.

Identifier format per platform:
  twitch  - login name, as in twitch.tv/<login>
  kick    - channel slug, as in kick.com/<slug>
  youtube - channel ID ("UC..." 24 characters) or handle ("@name")
  tiktok  - username, as in tiktok.com/@<username>  (no official LIVE API; see README)
"""

STREAMERS = {
    "twitch": [
        "streamer_one",
        "streamer_two",
        "streamer_three",
    ],
    "kick": [
        "streamer_four",
        "streamer_five",
    ],
    "youtube": [
        "CHANNEL_ID_ONE",
        "@channel_handle_two",
    ],
    "tiktok": [
        "username_one",
        "username_two",
    ],
}

# Seconds between monitoring cycles in continuous mode.
# Overridden by: python main.py --interval 30
CHECK_INTERVAL_SECONDS = 60

# Lowest polling interval allowed per platform, whatever interval you request.
# A platform is only re-queried when its own interval has elapsed; in between,
# its last results are reused. YouTube may be slowed further by its daily quota.
PLATFORM_MIN_INTERVAL_SECONDS = {
    "twitch": 10,   # Helix allows 800 points/min for app tokens; we use ~1 per 100 streamers
    "kick": 30,     # Kick does not publish a rate limit, so stay conservative
    "youtube": 60,  # also bounded by the quota budget below
}

# HTTP timeout for every API request, in seconds.
REQUEST_TIMEOUT_SECONDS = 10

# When a check fails, the last confirmed LIVE/OFFLINE status is kept for this
# many seconds (or two polling intervals of that platform, whichever is longer).
# After that the streamer is shown as UNKNOWN.
KEEP_LAST_STATUS_SECONDS = 600

# YouTube Data API quota. A Google Cloud project gets 10,000 units per day by
# default (resets at midnight Pacific Time). The script only spends
# YOUTUBE_QUOTA_SHARE of it, and slows its YouTube polling to stay inside it.
YOUTUBE_DAILY_QUOTA = 10_000
YOUTUBE_QUOTA_SHARE = 0.9

# How many of each channel's most recent uploads to inspect for an active live
# stream (1-50). The cost is the same 1 unit per channel for any value.
YOUTUBE_RECENT_VIDEOS_TO_SCAN = 10

# Number of recent status changes listed under the report (0 to hide the list).
RECENT_CHANGES_TO_SHOW = 5
