# Clipper
Clipping for streamers

## Tools

- [`stream-monitor/`](stream-monitor/): terminal script that checks whether streamers are live on Twitch, Kick, YouTube and TikTok. See its [README](stream-monitor/README.md).
- [`clipping_engine/`](clipping_engine/): records streamers while the monitor reports them live, finds the most clip-worthy moments (audio, chat, speech and Claude), renders captioned 9:16 clips and posts them to Discord. Run it with `python -m clipping_engine`. See its [README](clipping_engine/README.md).
