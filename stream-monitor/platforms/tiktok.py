"""
TikTok LIVE status: NOT AVAILABLE through an official API.

TikTok for Developers offers no public endpoint that tells a third-party app
whether an arbitrary account is live:

  - Login Kit / Display API: profile and video data only, and only for users
    who have logged in to your app and authorized it. No LIVE status field.
  - Research API: restricted to approved academic/non-profit researchers;
    its user and video endpoints have no LIVE status either.
  - Content Posting, Share Kit, Embed, Commercial Content and Data Portability
    APIs: unrelated to LIVE status.

Tools that report TikTok LIVE status (e.g. the "TikTokLive" library or paid
"live checker" services) read TikTok's undocumented internal web endpoints.
That is unofficial scraping, can break without notice and may violate
TikTok's terms, so it is deliberately not used here.

This module therefore makes no network requests and reports every configured
TikTok account as UNKNOWN ("API access unavailable"). It is never shown as
OFFLINE. If TikTok publishes an official LIVE API, implement it in _fetch()
following the pattern of the other platform modules.
"""

from platforms.base import PlatformMonitor

REASON = "API access unavailable (TikTok has no official LIVE status API)"


class TikTokMonitor(PlatformMonitor):
    key = "tiktok"
    title = "TIKTOK"
    available = False

    def _fetch(self, names):
        return [self.unknown(name, REASON) for name in names]
