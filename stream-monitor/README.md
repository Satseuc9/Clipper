# Stream Monitor

A small Python script that checks whether a list of streamers is **LIVE**, **OFFLINE** or **UNKNOWN** on Twitch, Kick, YouTube and TikTok, and prints a color-coded report in the terminal.

It is not a website, dashboard or service. It has no database. It calls each platform's official API and prints the results.

```
============================================================
STREAMER LIVE MONITOR
============================================================

TWITCH
[ LIVE ]    streamer_one | 12,450 viewers | Ranked grind
[ OFFLINE ] streamer_two

KICK
[ LIVE ]    streamer_three | 5,200 viewers | IRL

YOUTUBE
[ OFFLINE ] Some Channel

TIKTOK
[ UNKNOWN ] streamer_five | API access unavailable (TikTok has no official LIVE status API)

------------------------------------------------------------
TOTAL: 5
LIVE: 2
OFFLINE: 2
UNKNOWN: 1

Last checked: 2026-11-20 01:30:15
```

Colors: **green** = LIVE, **gray** = OFFLINE, **yellow** = UNKNOWN or API error.

---

## 1. Install

You need Python 3.10 or newer.

```bash
cd stream-monitor

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The dependencies are `requests` (HTTP), `python-dotenv` (reads `.env`) and `colorama` (colors on Windows terminals).

## 2. Add credentials

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and fill in the keys for the platforms you use. `.env.example` explains where to get each one:

| Platform | Variables | Where to get them |
|---|---|---|
| Twitch | `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET` | https://dev.twitch.tv/console/apps → Register Your Application |
| Kick | `KICK_CLIENT_ID`, `KICK_CLIENT_SECRET` | kick.com → Settings → [Developer](https://kick.com/settings/developer) (2FA must be on) |
| YouTube | `YOUTUBE_API_KEY` | Google Cloud Console → enable *YouTube Data API v3* → Credentials → API key |
| TikTok | none | No official API exists. See [TikTok](#tiktok-live-not-available) |

If a platform has no credentials, its streamers are shown as UNKNOWN. The other platforms still work.

`.env` is git-ignored. The script never prints keys or tokens. Error messages leave out URLs and headers, and the YouTube key is sent in a request header instead of the URL.

## 3. List your streamers

Edit `STREAMERS` in `config.py`. You don't need to change any other file.

```python
STREAMERS = {
    "twitch":  ["streamer_one", "streamer_two"],      # login from twitch.tv/<login>
    "kick":    ["streamer_four"],                     # slug from kick.com/<slug>
    "youtube": ["UCxxxxxxxxxxxxxxxxxxxxxx", "@somehandle"],  # channel ID or @handle
    "tiktok":  ["username_one"],                      # tiktok.com/@<username>
}
```

Restart the script after editing.

## 4. Run

```bash
python main.py                 # continuous monitoring, every 60 s (default). CTRL+C to stop.
python main.py --once          # one check, print the report, exit
python main.py --interval 30   # continuous monitoring every 30 s (minimum 10)
```

---

## Continuous mode

- Every cycle clears the screen and redraws the full status list. It ends with
  `LIVE: x | OFFLINE: y | UNKNOWN: z` and `Next check in N seconds...`.
- Statuses are kept in memory. A notification is printed only when a confirmed status changes:

  ```
  [01:30:15] 🔴 STREAMER IS NOW LIVE
  Platform: Twitch
  Username: example_streamer
  Viewers: 12,500
  Title: Ranked grind
  URL: https://twitch.tv/example_streamer

  [02:15:30] ⚫ STREAMER WENT OFFLINE
  Platform: Kick
  Username: another_streamer
  ```

- There is no notification for the first status seen at startup, because nothing has changed yet.
- The last few changes stay listed under **RECENT CHANGES**, so you can still see a notification after the screen redraws. Set `RECENT_CHANGES_TO_SHOW = 0` to hide this list.
- CTRL+C stops the script cleanly.

### What happens when a check fails

- **An error is never shown as OFFLINE.** Failed requests, rate limits, unknown accounts and unexpected responses are all UNKNOWN.
- If a check fails but the streamer had a confirmed LIVE or OFFLINE status recently, that status stays on screen with a yellow note, for example `(last confirmed 2m ago; check failed: HTTP 503)`. "Recently" means within `KEEP_LAST_STATUS_SECONDS` (default 10 min) or two polling intervals of that platform, whichever is longer.
- If there is no recent confirmed status, the streamer is shown as UNKNOWN.
- A failed check never triggers a LIVE or OFFLINE notification. The last confirmed status is used to detect the next real change.
- Each platform is checked on its own, so one failing API does not affect the others.

### Polling intervals and API limits

The requested interval is a maximum rate, not a guarantee. A platform is only queried again when its own interval has passed. In between, its last results are reused and the report shows a note such as `(refreshes every 16m to stay within API limits)`.

| Platform | Lowest interval | Why |
|---|---|---|
| Twitch | 10 s | The limit is 800 points/min per app token. One check uses 1 point per 100 streamers. |
| Kick | 30 s | Kick publishes no numeric limit, so the script stays conservative. |
| YouTube | 60 s, or slower to fit the quota | The default quota is 10,000 units/day. See below. |

You can change these in `PLATFORM_MIN_INTERVAL_SECONDS` in `config.py`.

---

## How each platform is checked

### Twitch: Helix API

- **Auth:** app access token from the client-credentials flow (`POST https://id.twitch.tv/oauth2/token`). The token is renewed before it expires and after any `401`. It is validated hourly with `GET https://id.twitch.tv/oauth2/validate`, as Twitch asks.
- **Account check:** `GET /helix/users?login=…` (up to 100 per call). Results are cached, and missing logins are checked again hourly. Get Streams only returns live channels, so a typo would otherwise look OFFLINE. With this check, a typo shows as UNKNOWN ("user not found").
- **Live status:** `GET /helix/streams?user_login=…&first=100` (up to 100 logins per call). The script reads `viewer_count` and `title`.
- **Rate limit:** the script reads `Ratelimit-Remaining` and `Ratelimit-Reset`. If the bucket is empty or the API returns a `429`, Twitch polling pauses until the reset time.

### Kick: Kick Public API

- **Auth:** app access token from the OAuth 2.1 client-credentials flow (`POST https://id.kick.com/oauth/token`). It is renewed before expiry and after a `401`.
- **Live status:** `GET https://api.kick.com/public/v1/channels?slug=a&slug=b` (up to 50 slugs per call, 25 characters each). The script reads `stream.is_live`, `stream.viewer_count` and `stream_title`.
- A slug that Kick does not return is shown as UNKNOWN ("channel not found"). Use the slug from the channel URL, which is usually the username in lowercase.
- **Rate limit:** on a `429`, Kick polling pauses for `Retry-After` seconds (60 s if the header is missing).

### YouTube: YouTube Data API v3

`search.list?eventType=live` costs **100 units** per call, so a 60-second loop would use up the daily quota within a few hours. This script uses only 1-unit calls:

1. Once per channel: `channels.list` (by `id`, up to 50 per call, or `forHandle=@name`) to get the channel title and its "uploads" playlist.
2. Every poll: `playlistItems.list` on the uploads playlist, reading the newest `YOUTUBE_RECENT_VIDEOS_TO_SCAN` videos. This costs 1 unit per channel.
3. Every poll: `videos.list` with `snippet,liveStreamingDetails`, up to 50 videos for 1 unit. A channel is LIVE when one of its recent videos has `liveBroadcastContent == "live"`, a start time and no end time. Viewer count comes from `concurrentViewers`, which is missing if the owner hides it. Videos that can't go live again (normal uploads and ended streams) are remembered and not looked up again, so this is usually one call or none.

**Quota math:** each poll costs about (number of channels + 1) units. The script allows itself `YOUTUBE_DAILY_QUOTA × YOUTUBE_QUOTA_SHARE` units (9,000 by default) and slows YouTube polling to match:

| YouTube channels | Effective YouTube refresh |
|---|---|
| up to ~5 | every 60 s |
| 10 | ~2 min |
| 50 | ~8 min |
| 100 | ~16 min |

If the quota runs out anyway, for example because another app uses the same Google Cloud project, YouTube polling stops until midnight Pacific Time, when the quota resets. Meanwhile YouTube streamers show their last confirmed status, then UNKNOWN. If you need faster checks, you can request more quota from Google.

**Limitation:** this method detects public live streams that appear in a channel's uploads playlist. Members-only and unlisted streams are not detected. A stream that was scheduled long ago and has more than `YOUTUBE_RECENT_VIDEOS_TO_SCAN` newer uploads may be missed, so raise the value if that happens. It can go up to 50 at no extra cost per channel. Premieres show as live while they play.

### TikTok LIVE: not available

TikTok has no official, publicly available API that tells a third-party app whether an account is live:

- **Login Kit / Display API** returns profile and video data, and only for users who logged in to *your* app and authorized it. It has no LIVE field.
- The **Research API** is limited to approved academic and non-profit researchers, and it has no LIVE status either.
- The **Content Posting, Share Kit, Embed, Commercial Content and Data Portability** APIs do not cover LIVE status.

Tools that do report TikTok LIVE status, such as the `TikTokLive` library and paid "live checker" services, read TikTok's undocumented internal web endpoints. That is unofficial scraping. It can break without notice and may break TikTok's terms, so this project does not use it. `platforms/tiktok.py` makes no network requests and shows every TikTok account as **UNKNOWN – API access unavailable**, never OFFLINE. If TikTok releases an official LIVE API, it can be added in that module's `_fetch()`.

---

## Configuration reference (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `STREAMERS` | example lists | Streamers to check, per platform |
| `CHECK_INTERVAL_SECONDS` | `60` | Seconds between cycles in continuous mode (`--interval` overrides it) |
| `PLATFORM_MIN_INTERVAL_SECONDS` | twitch 10, kick 30, youtube 60 | Lowest allowed polling interval per platform |
| `REQUEST_TIMEOUT_SECONDS` | `10` | Timeout for every HTTP request |
| `KEEP_LAST_STATUS_SECONDS` | `600` | How long a confirmed status is kept on screen after failed checks |
| `YOUTUBE_DAILY_QUOTA` | `10000` | Your Google Cloud project's daily quota |
| `YOUTUBE_QUOTA_SHARE` | `0.9` | Share of that quota this script may use |
| `YOUTUBE_RECENT_VIDEOS_TO_SCAN` | `10` | Recent uploads inspected per channel (1–50) |
| `RECENT_CHANGES_TO_SHOW` | `5` | Status changes listed under the report (0 hides the list) |

## Reliability details

- Every request has a timeout.
- Network errors and `5xx` responses are retried once.
- Tokens are fetched automatically and renewed on expiry or `401`.
- Twitch and Kick `429` responses and YouTube quota errors pause only that platform, until the time the API gives.
- Requests are batched: 100 logins per call for Twitch, 50 slugs per call for Kick, 50 channels or videos per call for YouTube.
- Account lookups are cached, and finished YouTube videos are not looked up again.
- API text such as stream titles is cleaned of control characters before printing.
- Colors turn off when output is not a terminal, or when `NO_COLOR` is set.

## Troubleshooting

| Message | Fix |
|---|---|
| `missing ... in .env` | Copy `.env.example` to `.env` and fill in that platform's keys. |
| `authentication failed ... check the client ID/secret` | The ID or secret is wrong, or the secret was regenerated. |
| `user not found on Twitch` | Use the login from the URL. The account may also have been renamed or suspended. |
| `channel not found on Kick` | Use the slug from `kick.com/<slug>`. |
| `channel not found (check the channel ID or @handle)` | YouTube IDs start with `UC` and are 24 characters long. Handles start with `@`. |
| `HTTP 403: ... YouTube Data API v3 has not been used / is disabled` | Enable the API in your Google Cloud project. |
| `YouTube daily quota exhausted` | Wait until midnight Pacific Time, or reduce the number of channels. |

## Project structure

```
stream-monitor/
├── main.py              # CLI, monitoring loop, report assembly
├── config.py            # your streamer list and settings
├── requirements.txt
├── .env.example         # credential template (copy to .env)
├── platforms/
│   ├── base.py          # shared result type, caching, error handling
│   ├── twitch.py
│   ├── kick.py
│   ├── youtube.py
│   └── tiktok.py
└── utils/
    ├── http.py          # timeouts, retries, rate-limit errors, app tokens
    ├── tracker.py       # status memory and change detection
    └── terminal.py      # colors and output
```

## Official documentation used

- Twitch: [Get Streams](https://dev.twitch.tv/docs/api/reference/#get-streams), [Get Users](https://dev.twitch.tv/docs/api/reference/#get-users), [Client credentials flow](https://dev.twitch.tv/docs/authentication/getting-tokens-oauth/#client-credentials-grant-flow), [Rate limits](https://dev.twitch.tv/docs/api/guide/#twitch-rate-limits), [Validating tokens](https://dev.twitch.tv/docs/authentication/validate-tokens/)
- Kick: [Kick Dev Docs](https://docs.kick.com/) ([source on GitHub](https://github.com/KickEngineering/KickDevDocs)): OAuth 2.1 app access token, Channels API
- YouTube: [videos.list](https://developers.google.com/youtube/v3/docs/videos/list), [playlistItems.list](https://developers.google.com/youtube/v3/docs/playlistItems/list), [channels.list](https://developers.google.com/youtube/v3/docs/channels/list), [Quota costs](https://developers.google.com/youtube/v3/determine_quota_cost)
- TikTok: [TikTok for Developers products](https://developers.tiktok.com/products/)
