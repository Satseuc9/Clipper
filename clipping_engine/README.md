# AI Clipping Engine

The clipping engine records your streamers while they are live and finds their most entertaining moments. It turns those moments into captioned 9:16 clips and posts them to your Discord channel.

It builds on [`stream-monitor`](../stream-monitor). You keep adding streamers to `stream-monitor/config.py`, and the monitor still decides who is live. The engine plugs into the monitor's loop, so there is no second monitoring app. It runs in a terminal, with no website, dashboard or database.

```
stream-monitor says LIVE ──► Streamlink ─► FFmpeg ─► 10-minute segments ─┐
                              chat (official APIs) ─► chat.jsonl ────────┤
                                                                         ▼
           Stage 1  whole segment, cheap: audio events, loudness, chat spikes, scene changes
           Stage 2  candidates only: speech-to-text, spoken cues, motion
           Stage 3  best few: Claude judges transcript + chat + audio + frames
           Stage 4  winners: cut (across segments if needed) ─► captions ─► 1080x1920 ─► Discord
```

---

## What was tested, and how

Every component is covered by `clipping_engine/tests` (64 tests). Platforms that can't be reached from a test machine are replaced by local servers that speak the documented formats.

| Component | Tested with |
|---|---|
| Streamlink plugin check (Twitch, Kick, YouTube) | the real Streamlink 8.6.3 `--can-handle-url` |
| Recording, 10-minute segments, start/stop, no duplicates, multiple streamers, concurrency limit, reconnect after a dropped connection, crash recovery, storage cleanup | real Streamlink + FFmpeg recording a live HLS stream that FFmpeg generates and a local web server serves |
| Twitch ad breaks don't trigger reconnects; real stalls still do; giving up on dead streams; low-disk guard; removed streamers stop recording | a fake `streamlink` that pauses output and logs exactly like Streamlink's ad filter |
| Timestamp sync (wall clock ↔ video), including the start-up burst and ad-break gaps | real captures + time-map unit tests |
| Twitch chat (EventSub WebSocket, subscriptions, `session_reconnect`, duplicate messages, device-code login, token refresh) | local mock WebSocket + Helix endpoints |
| YouTube chat (`liveChatMessages.list` paging, chat ended, chat disabled, quota budget) | mocked API responses in the documented format |
| Kick chat (webhook signature verification, emote parsing, subscriptions) | real RSA signatures checked as Kick documents |
| Chat spike / laugh / emoji / "clip it" detection against a baseline | synthetic chat |
| Laughter and sound-event detection | the real YAMNet model on real recordings (ESC-50 laughter, clapping, rain, dogs; LibriSpeech speech) |
| Viral scoring, missing signals, single-signal penalty, dedupe, clip bounds | unit tests |
| Claude evaluation (request shape, structured output, refusals, API errors) | mocked Anthropic client |
| Cross-segment clip extraction accuracy | a tone whose pitch rises with time: the cut lands within ±0.5 s |
| Vertical layouts, face-aware cropping, burned captions, loudness (−14 LUFS) | real FFmpeg + YuNet on a real face photo |
| Discord (multipart upload, 429 retry, size limit → preview/link, no @mentions) | local mock webhook server |
| Processing queue (ordering, retries, permanent failure, crashed worker process) | real worker processes |
| Everything together | mock mode on a prerecorded video with chat → scored moment → clip → Discord |

**First live run (Twitch, 7 streams at once):** recording and segmenting worked; it exposed ad-break reconnect loops, a candidate threshold that was too strict for real stream audio, an analysis backlog and recordings that never stopped. All four are fixed and covered by tests.

**Not run against the real services yet:** Twitch/Kick/YouTube chat, Kick and YouTube capture, a real Discord channel, and real Claude calls. The machine this was built on had no network access to those hosts and no credentials. Speech-to-text was not run with real Whisper weights either (HuggingFace was unreachable there). The faster-whisper API it uses was checked against the installed package. Run `python -m clipping_engine check`, then a short live test, before relying on it.

---

## 1. Requirements

- **Python 3.10+**
- **FFmpeg** (with libx264 and libass, which standard builds include):
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt install ffmpeg`
- **Disk:** about 2.5–4 GB per hour per streamer at 1080p60 while segments wait to be processed. Limits are configurable.
- **CPU/GPU:** recording itself is cheap, because packets are copied without re-encoding. Analysis is the heavy part:
  - Whisper `small` needs roughly a minute of CPU time for the candidate windows of a 10-minute segment (a rough estimate; it depends on how much is said). An NVIDIA GPU is much faster.
  - Rendering a 60 s clip takes 20–60 s of CPU time.
- **Models**, downloaded automatically on first use into `clipping_engine/model_cache/`:

  | Model | Used for | Source | Size |
  |---|---|---|---|
  | YAMNet (Apache-2.0) | laughter, screams, cheering, gasps, unusual sounds | storage.googleapis.com (MediaPipe models) | 4 MB |
  | YuNet (MIT) | face detection for cropping | OpenCV Zoo on GitHub | 0.2 MB |
  | Whisper (`WHISPER_MODEL`) | speech-to-text with word timestamps | HuggingFace, via faster-whisper | 75 MB (tiny) – 3 GB (large-v3) |

  The YAMNet runtime (`ai-edge-litert`) has wheels for Windows, Linux and Apple-silicon Macs. On Intel Macs it is unavailable, and audio analysis falls back to loudness and pause detection.

## 2. Install

From the repository root (the folder that contains `clipping_engine/` and `stream-monitor/`):

```bash
python -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install -r clipping_engine/requirements.txt
```

This also installs the monitor's dependencies. Optional: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` for GPU Whisper. See the faster-whisper docs.

## 3. Credentials

| Where | Variable | Needed for |
|---|---|---|
| `stream-monitor/.env` | `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET` | live detection (as before), Twitch chat |
| `stream-monitor/.env` | `KICK_CLIENT_ID`, `KICK_CLIENT_SECRET` | live detection, Kick chat subscriptions |
| `stream-monitor/.env` | `YOUTUBE_API_KEY` | live detection, YouTube chat |
| `clipping_engine/.env` | `DISCORD_WEBHOOK_URL` | posting clips |
| `clipping_engine/.env` | `ANTHROPIC_API_KEY` | Claude evaluation (optional) |
| `clipping_engine/.env` | `TWITCH_CHAT_CLIENT_ID` / `_SECRET` | only if Twitch rejects the device login for your monitor app (see below) |

```bash
cp clipping_engine/.env.example clipping_engine/.env      # then fill it in
python -m clipping_engine auth twitch                     # one-time login for Twitch chat
python -m clipping_engine check                           # shows what is ready
```

`auth twitch` prints a URL and a code. Open the URL, log in with the account that should read chat (it can be your own), and approve. The token is saved to `clipping_engine/.tokens/twitch_user.json` (git-ignored, owner-only permissions) and refreshed automatically. Device-code login needs the Twitch app's settings to allow it. If Twitch rejects it, register a second app at dev.twitch.tv with Client Type **Public** and put its ID in `TWITCH_CHAT_CLIENT_ID`.

Secrets never appear in output or logs. The webhook URL, keys and tokens are only sent to their own services.

## 4. Run

```bash
python -m clipping_engine                    # monitor + record + clip; CTRL+C to stop
python -m clipping_engine --interval 30      # monitor every 30 s (platform limits still apply)
```

The terminal shows the monitor's normal report, followed by a **CLIPPING ENGINE** section: recordings in progress, chat status per stream, the analysis queue and recent results. Details go to `clipping_engine/logs/engine.log` and `worker.log`.

CTRL+C finishes the current segments cleanly. Press it again to stop immediately. Anything not yet analysed is processed on the next start.

### Mock mode: try it without a live stream

```bash
python -m clipping_engine mock path/to/vod.mp4 --chat chat.jsonl --streamer someone --no-discord
python -m clipping_engine mock vod.mp4 --segment-minutes 2        # shorter segments for a quick test
```

The video is split exactly like a live recording and runs through the full pipeline. The command prints each segment's top moments as JSON and lists the clips it made. Leave out `--no-discord` to send them. The chat file is optional, with one JSON object per line:

```json
{"t": 272.4, "user": "viewer1", "text": "KEKW", "emotes": ["KEKW"]}
```

`t` is seconds from the start of the video. Use `"ts"` for epoch seconds instead.

---

## How it works

### Recording

- When the monitor reports a streamer as **LIVE** (a fresh result, not one kept from memory after an API error), a `Recorder` starts:

  ```
  streamlink --stdout <url> 1080p60,1080p,...,best  |  ffmpeg -c copy -f segment -segment_time 600 ...
  ```

  FFmpeg copies the audio and video packets without re-encoding, so A/V sync is untouched. Each segment is a standalone MPEG-TS file with its own timestamps.
- The recorder stops on **OFFLINE**, or when the streamer is removed from `stream-monitor/config.py`. **UNKNOWN** never stops a recording.
- There is one recorder per streamer, up to `MAX_CONCURRENT_RECORDINGS`, each in its own threads and child processes.
- Before the first capture, the engine runs `streamlink --can-handle-url` to check that the installed Streamlink has a plugin for each platform. `check` shows the result.
- **Reconnects:** if the stream drops, stalls (no data for `STALL_RESTART_SECONDS`) or Streamlink exits while the monitor still says LIVE, the recorder reconnects with back-off. Segment numbers continue in the same session folder. After `RECORDING_GIVE_UP_MINUTES` without receiving any data it gives up, and waits `RECORDING_RETRY_COOLDOWN_MINUTES` before trying that streamer again.
- **Twitch ad breaks:** Streamlink removes ads and deliberately writes nothing until the ad ends. The recorder reads Streamlink's log ("Filtering out segments and pausing stream output" / "Resuming stream output") and doesn't treat that pause as a stall, up to `MAX_AD_BREAK_SECONDS`. The status line counts the ad breaks skipped.
- **Disk guard:** no new recordings start below `MIN_FREE_DISK_GB` free; running recordings stop cleanly below half of it.
- **Folders:**

  ```
  recordings/<platform>/<streamer>/session_001/
      segment_0001.ts + segment_0001.json   (timing + processing status)
      chat.jsonl  timing.jsonl  clips.json  session.json  logs/
  ```

- **Storage:** `MAX_RECORDINGS_GB` and `RECORDING_RETENTION_HOURS`. A segment is deleted only once it has been processed and the next segment no longer needs it as context, so unprocessed footage is never deleted. `MAX_CLIPS_GB` limits finished clips.

### Time synchronisation (chat ↔ video)

The video clock and the real clock drift apart:
- Streamlink starts with a few seconds of backlog.
- Network stalls pause the data.
- Twitch ad breaks are filtered out of the recording, so the video skips over them.

While recording, FFmpeg reports its output time every second, and the engine logs it next to the clock (`timing.jsonl`). A time-map built from these samples converts video time to clock time and back, gaps included. Chat messages keep the platform's own timestamps, so they line up with the video to about a second. Viewers react a few seconds after the moment, so chat is shifted back by `CHAT_REACTION_DELAY_SECONDS`. Use `CHAT_SYNC_OFFSET_SECONDS` if your chat is consistently early or late.

### Chat (official access only)

| Platform | How | Notes |
|---|---|---|
| Twitch | EventSub WebSocket, `channel.chat.message` v1 | Needs a **user** token with `user:read:chat`, from `auth twitch`. One shared socket; subscriptions are added and removed with recordings; reconnects and `session_reconnect` are handled. |
| YouTube | Data API `liveChatMessages.list` | The live chat ID comes from the monitor at no extra quota. Every call costs quota (`YOUTUBE_CHAT_UNITS_PER_CALL`, default 5), so the poller waits `YOUTUBE_CHAT_POLL_SECONDS` (20 s) and stops at `YOUTUBE_CHAT_DAILY_UNITS`. No timing is lost, because each message carries its own timestamp. Google lists OAuth scopes for this method, so if your API key is refused, chat is marked unavailable for that stream. |
| Kick | Official webhooks, `chat.message.sent` | Kick's only official real-time chat access. It needs a **public URL**: enable webhooks in kick.com/settings/developer, point it at a tunnel (`cloudflared tunnel --url http://127.0.0.1:8787` or ngrok) and set `KICK_WEBHOOK_ENABLED = True`. Every request is RSA-signature-verified against Kick's public key. **Off by default.** |

The engine never makes up chat. Collection writes coverage markers, so analysis can tell "chat was quiet" apart from "we had no chat data". Missing chat removes the chat signal and is never counted as zero activity. Unofficial chat sockets (Kick's Pusher channel, YouTube page scraping) are deliberately not used.

### Analysis

Each finished segment is analysed together with the last `ANALYSIS_CONTEXT_SECONDS` (60 s) of the one before it. The status line shows, for every segment, the moments found, the best score, which signals were available and any problem (for example `[audio+video; no chat, no ai] ⚠ audio classifier unavailable`). Moments in the final `SEGMENT_TAIL_GUARD_SECONDS` (25 s) are left to the next segment, which has the footage after them. Nothing at a boundary is missed or cut short.

- **Stage 1, everything, cheap:**
  - YAMNet sound events every 0.48 s, plus loudness and silence.
  - Chat activity against its own baseline (median of the previous 10 minutes).
  - Scene changes, decoding keyframes only.

  These are combined into an interest curve. Its strongest peaks relative to the segment's own baseline (up to `MAX_CANDIDATES_PER_SEGMENT`, ignoring anything below `STAGE1_MIN_INTEREST`) become candidates, so every eventful segment gets scored. Deciding what is clip-worthy is left to the scoring.
- **Stage 2, candidates with a pre-score of at least `STAGE2_MIN_PRESCORE`:**
  - faster-whisper transcription with word timestamps, around the moment only.
  - Spoken cues: "oh my god", "no way", laughter in speech, a jump in speaking rate, short quotable lines.
  - Motion intensity.
- **Stage 3, at most `AI_MAX_CANDIDATES_PER_SEGMENT` with a score of at least `AI_MIN_PRESCORE`:** Claude receives the timestamped transcript, the chat reaction summary, the audio and video signals, and `AI_FRAMES_PER_CANDIDATE` still frames. It returns structured JSON: entertainment score, category (funny, fail, clutch play, argument, story, quote…), whether the clip makes sense on its own, suggested start and end, a title and the reason.
- **Signals detected:**
  - **Audio:** laughter, sustained laughter with crowd or babble (YAMNet can't count people), screams and shouts, cheering, gasps, dramatic pauses followed by a reaction, emotional shifts, unusual sounds, loudness spikes.
  - **Chat:** spike vs baseline, laugh, surprise, hype and negative reactions, emoji/emote bursts, repeated messages, unique-chatter surges, "clip it" requests.
- **Loudness is not entertainment.** A loud moment with no human reaction is capped at 40/100 for audio. Speaker identification (diarization) is not included: it needs a separate model with its own licence.

### Viral score (0–100)

```
score = Σ weight_i × score_i / Σ weight_i     over the AVAILABLE components only
```

| Component | Default weight |
|---|---|
| audio reactions | 0.25 |
| chat activity | 0.25 |
| AI entertainment (Claude) | 0.30 |
| speech / context | 0.20 |

- Weights are in `SCORE_WEIGHTS`.
- A missing signal (no chat API, no Anthropic key, nothing said) drops out of both sums instead of counting as zero.
- If fewer than two components are strong (50 or more), the score is multiplied by `SINGLE_SIGNAL_FACTOR` (0.8), so no clip rests on one signal.
- Clips are made for moments scoring at least `MINIMUM_VIRAL_SCORE` (75), up to `MAX_CLIPS_PER_SEGMENT` per segment.
- Duplicates are removed within a segment and, through `clips.json`, across segments.

Each moment is stored in the segment's sidecar like this:

```json
{
  "streamer": "example_streamer", "platform": "twitch", "segment": "segment_0004",
  "start_timestamp": "04:32", "end_timestamp": "05:18", "viral_score": 94,
  "reason": "Unexpected funny interaction followed by intense laughter and a big increase in chat activity.",
  "signals": {"laughter_detected": true, "chat_spike": true, "emoji_spike": true, "exciting_audio": true, "...": "..."},
  "components": {"audio": 96, "chat": 91, "ai": 95, "speech": 88}
}
```

### Clips

- **Boundaries:**
  - Without Claude: `PRE_CLIP_BUFFER_SECONDS` before and `POST_CLIP_BUFFER_SECONDS` after the activity, kept within `MIN_CLIP_DURATION`–`MAX_CLIP_DURATION` (30–90 s). If Claude's suggested start and end make sense, they are used instead (it aims for `DEFAULT_CLIP_DURATION`).
  - Cuts are then snapped to sentence starts and ends from the word timestamps, or to the quietest point when there is no transcript, so nobody is cut off mid-word.
- **Extraction** joins consecutive segments with FFmpeg's concat demuxer and cuts frame-accurately, so moments that cross a segment boundary work.
- **Vertical output**: 1080×1920, H.264 + AAC, loudness-normalised to `TARGET_LOUDNESS_LUFS` (−14). YuNet faces on sampled frames choose the layout:
  - `face_crop`: a large face (IRL / just chatting) gets a 9:16 crop centred on it.
  - `stacked`: a small fixed face (webcam on gameplay) puts the webcam enlarged on top and the **full** gameplay frame below, so no HUD is cut. `GAMEPLAY_FILL = True` crops the sides to make the game bigger instead.
  - `fit_blur`: with no reliable face, the full frame sits over a blurred copy of itself. Nothing is cropped.
- **Captions** are burned in from the word timestamps. `ANIMATED_CAPTIONS = True` switches to word-by-word highlighting.

### Discord

Clips at or above `MINIMUM_VIRAL_SCORE` are posted with: streamer, platform, viral score, duration, timestamp in stream time (or recording time), the reason, the detected signals and the livestream link. YouTube links jump to the moment.

The video is attached when it fits under `DISCORD_MAX_UPLOAD_MB` (10 MB by default; raise it for boosted servers). Larger clips are never uploaded. Instead, a compressed 720p preview is attached and the message links to the full clip, at `CLIP_PUBLIC_BASE_URL` if you host the clips folder, otherwise the local path. Rate limits (429) are retried. AI-written text can't @-mention anyone.

### Reliability

- Capture runs in its own threads and child processes. Analysis runs in separate worker processes, so analysis can never block or crash recording.
- Segments are a persistent queue (sidecar status), processed in order per session and in parallel across streamers (`ANALYSIS_WORKERS`, default 2). The status line warns when the queue grows faster than the workers can process it. Each live streamer adds one segment per `RECORDING_SEGMENT_MINUTES`, so with many streamers raise `ANALYSIS_WORKERS` (CPU cores ÷ 3 is a good start), record fewer streamers, or record at 720p.
- Failures retry after `PROCESSING_RETRY_DELAY_SECONDS`, up to `PROCESSING_MAX_ATTEMPTS`, then the segment is marked `failed` with the error.
- A crashed worker process is replaced.
- After a crash or restart, interrupted sessions are closed and half-processed segments are requeued.
- Results are saved before rendering, so a retry never re-analyses or re-sends a clip.

---

## Costs and limits

- **Claude:** at most `AI_MAX_CANDIDATES_PER_SEGMENT` (3) calls per 10-minute segment, each with a short transcript and 4 small frames. With `claude-opus-5-5` at medium effort, that is roughly $0.05 per call, so at most about $1 per streamer per live hour, usually much less. Set `CLAUDE_MODEL`, `CLAUDE_EFFORT`, `AI_MIN_PRESCORE` or `ENABLE_AI_EVALUATION` to change it. The request enables Anthropic's server-side refusal fallback, so a declined request is retried on a fallback model instead of failing.
- **YouTube quota** (10,000 units/day by default) is shared by the monitor and YouTube chat. Keep `YOUTUBE_CHAT_DAILY_UNITS` plus the monitor's usage under it.
- **Twitch:** EventSub limits how many WebSocket subscriptions one connection can hold. A personal streamer list is far below that limit.

## Respect the platforms and creators

Record and clip only streams you have the right to use: your own, or creators who allow it. Follow Twitch's, Kick's and YouTube's terms of service. The engine uses official APIs and supported Streamlink plugins only, never bypasses access controls, and keeps recordings local. Check copyright before publishing clips (music in streams especially).

## Configuration reference (`config.py`)

| Group | Settings |
|---|---|
| What to record | `CLIP_PLATFORMS`, `CLIP_ONLY` |
| Recording | `RECORDING_SEGMENT_MINUTES`, `MAX_CONCURRENT_RECORDINGS`, `STREAM_QUALITY`, `STREAMLINK_EXTRA_ARGS`, `STREAM_TIMEOUT_SECONDS`, `STALL_RESTART_SECONDS`, `MAX_AD_BREAK_SECONDS`, `RECONNECT_DELAYS_SECONDS`, `RECORDING_GIVE_UP_MINUTES`, `RECORDING_RETRY_COOLDOWN_MINUTES` |
| Storage | `RECORDINGS_DIR`, `CLIPS_DIR`, `MAX_RECORDINGS_GB`, `RECORDING_RETENTION_HOURS`, `MAX_CLIPS_GB`, `MIN_FREE_DISK_GB` |
| Chat | `ENABLE_CHAT_COLLECTION`, `CHAT_REACTION_DELAY_SECONDS`, `CHAT_SYNC_OFFSET_SECONDS`, `YOUTUBE_CHAT_*`, `KICK_WEBHOOK_*` |
| Analysis | `ENABLE_AUDIO_ANALYSIS`, `ENABLE_CHAT_ANALYSIS`, `ENABLE_VIDEO_ANALYSIS`, `ENABLE_TRANSCRIPTION`, `ENABLE_AI_EVALUATION`, `ANALYSIS_CONTEXT_SECONDS`, `SEGMENT_TAIL_GUARD_SECONDS`, `MAX_CANDIDATES_PER_SEGMENT`, `STAGE1_MIN_INTEREST`, `STAGE2_MIN_PRESCORE`, `AI_*`, `WHISPER_*`, `CLAUDE_*`, `LAUGHTER_THRESHOLD`, `SCREAM_THRESHOLD`, `CHEER_THRESHOLD` |
| Scoring | `SCORE_WEIGHTS`, `SINGLE_SIGNAL_FACTOR`, `MINIMUM_VIRAL_SCORE`, `MAX_CLIPS_PER_SEGMENT`, `DEDUPE_GAP_SECONDS` |
| Clips | `DEFAULT_CLIP_DURATION`, `MIN_CLIP_DURATION`, `MAX_CLIP_DURATION`, `PRE_CLIP_BUFFER_SECONDS`, `POST_CLIP_BUFFER_SECONDS`, `OUTPUT_RESOLUTION`, `OUTPUT_FORMAT`, `OUTPUT_CRF`, `OUTPUT_PRESET`, `TARGET_LOUDNESS_LUFS`, `LAYOUT`, `GAMEPLAY_FILL`, `CAPTIONS`, `ANIMATED_CAPTIONS`, `CAPTION_FONT`, `CAPTION_FONT_SIZE`, `CAPTION_MAX_WORDS` |
| Discord | `DISCORD_NOTIFICATIONS`, `DISCORD_MAX_UPLOAD_MB`, `DISCORD_SEND_PREVIEW_IF_TOO_LARGE`, `CLIP_PUBLIC_BASE_URL` |
| Processing | `ANALYSIS_WORKERS`, `PROCESSING_MAX_ATTEMPTS`, `PROCESSING_RETRY_DELAY_SECONDS` |

## Tests

```bash
python -m pytest clipping_engine/tests            # ~3.5 minutes; needs FFmpeg and Streamlink
```

The first run downloads small test files into `clipping_engine/tests/.cache/` (git-ignored): the YAMNet and YuNet models, a few ESC-50 sound clips (CC BY-NC, used only locally for testing), a LibriSpeech utterance and OpenCV's sample face photo. The tests never call Twitch, Kick, YouTube, Discord or Claude, and they don't download Whisper weights.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `not recording – Streamlink ... has no kick plugin` | `pip install -U streamlink` |
| Recording keeps reconnecting | See `recordings/.../logs/runNN_streamlink.log`. Kick sometimes blocks automated clients; Twitch subscriber-only streams need a logged-in Streamlink (`--twitch-api-header`, in `STREAMLINK_EXTRA_ARGS`). |
| `no chat – Twitch chat not authorized` | `python -m clipping_engine auth twitch` |
| YouTube `no chat – ... HTTP 403` | Google refused key-only access for that chat; the stream is analysed without chat |
| `transcription unavailable` in the notes | The Whisper download failed or `ENABLE_TRANSCRIPTION` is off; set `WHISPER_MODEL` to a local CTranslate2 model folder to work offline |
| No clips / "0 moments" | Read the segment's status line or `analysis_notes` in `segment_NNNN.json`. `no chat` → run `auth twitch`. `audio classifier unavailable` → YAMNet's runtime isn't installed (no Intel-Mac build). Low `stage 1 peak interest` on every segment means nothing stood out. Lower `MINIMUM_VIRAL_SCORE` while tuning. |
| "stream stalled" / reconnect loops | Update to this version (Twitch ad breaks were mistaken for stalls). Persistent stalls on many streams at once usually mean the connection can't carry them all at 1080p60. |
| "analysis is falling behind" | Raise `ANALYSIS_WORKERS`, record fewer streamers, or set `STREAM_QUALITY = "720p60,720p,best"`. |
| Discord says too large | Raise `DISCORD_MAX_UPLOAD_MB` for boosted servers, or set `CLIP_PUBLIC_BASE_URL` |

## Project structure

```
clipping_engine/
├── main.py                 CLI: run / check / auth / mock
├── config.py               all settings
├── monitor_bridge.py       hooks into stream-monitor's loop (LIVE/OFFLINE -> recordings)
├── pipeline.py             processing queue, worker processes, retries, recovery
├── processing.py           per-segment stages 1-4
├── media.py                FFmpeg/FFprobe helpers
├── capture/                streamlink_manager.py, recording_manager.py, segment_manager.py (timing)
├── chat/                   twitch_chat.py, youtube_chat.py, kick_chat.py, chat_manager.py, chat_log.py, chat_analyzer.py
├── analysis/               audio_analyzer.py, laughter_detector.py (YAMNet), speech_transcriber.py,
│                           video_analyzer.py, sentiment_analyzer.py, ai_evaluator.py (Claude), viral_scorer.py
├── editing/                clip_extractor.py, caption_generator.py, vertical_formatter.py
├── notifications/          discord_notifier.py
├── storage/                recording_storage.py, clip_storage.py
└── tests/                  pytest suite
```
