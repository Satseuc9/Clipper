"""
AI clipping engine.

    python -m clipping_engine                    monitor + record + clip (default; CTRL+C to stop)
    python -m clipping_engine --interval 30      same, monitor every 30 s
    python -m clipping_engine check              check FFmpeg, Streamlink, models, credentials
    python -m clipping_engine auth twitch        one-time login for Twitch chat (EventSub)
    python -m clipping_engine discord            test the Discord webhook (--send-pending posts unsent clips)
    python -m clipping_engine mock VIDEO [--chat CHAT.jsonl] [--streamer NAME] [--no-discord]
                                                 analyse a prerecorded video, no live connection

Run these from the repository root (the folder that contains clipping_engine/).
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):                       # `python clipping_engine/main.py`
    sys.path[0] = str(Path(__file__).resolve().parent.parent)
    import runpy
    runpy.run_module("clipping_engine", run_name="__main__")
    sys.exit()

from dotenv import load_dotenv

from clipping_engine import config

log = logging.getLogger("clipping_engine")


def setup_logging(verbose=False):
    logs = config.ENGINE_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(logs / "engine.log", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(threadName)s %(name)s %(levelname)s %(message)s")


def load_env():
    # The monitor's .env (Twitch/Kick/YouTube keys) first, then the engine's own.
    load_dotenv(config.ENGINE_DIR.parent / "stream-monitor" / ".env")
    load_dotenv(config.ENGINE_DIR / ".env")


# ---------------------------------------------------------------- live engine

def run_engine(interval=None):
    from clipping_engine.capture.recording_manager import RecordingManager
    from clipping_engine.chat.chat_manager import ChatManager
    from clipping_engine.monitor_bridge import load_monitor, make_hook
    from clipping_engine.pipeline import Pipeline

    monitor_main, monitor_config, is_stale = load_monitor()
    pipeline = Pipeline()
    chat = ChatManager()
    recorder = RecordingManager(on_segment=pipeline.notify, chat_manager=chat)
    for platform, (ok, message) in recorder.support.items():
        log.info("capture %s: %s", platform, message)
    pipeline.start()
    hook = make_hook(recorder, chat, pipeline, is_stale)
    try:
        return monitor_main.run_monitor(interval or monitor_config.CHECK_INTERVAL_SECONDS, on_cycle=hook)
    finally:
        print("Stopping recordings (finishing the last segments; CTRL+C again to force)...", flush=True)
        try:
            recorder.stop_all(timeout=90)
        except KeyboardInterrupt:
            recorder.kill_all()            # never leave capture processes running in the background
        finally:
            recorder.kill_all()            # anything still alive after the timeout
            chat.shutdown()
            pipeline.stop()
        print("Stopped. Unprocessed segments will be processed on the next start.")


# ---------------------------------------------------------------- mock mode

def run_mock(video, chat_file=None, streamer="mock_streamer", platform="mock", discord=True):
    """Segment a prerecorded video like a live capture and run the full pipeline on it."""
    from clipping_engine.capture import segment_manager
    from clipping_engine.capture.streamlink_manager import ffmpeg_segment_command
    from clipping_engine.media import media_timing
    from clipping_engine.pipeline import Pipeline
    from clipping_engine.storage import recording_storage as storage

    video = Path(video).resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    if not discord:
        config.DISCORD_NOTIFICATIONS = False
    meta = {"platform": platform, "streamer": streamer, "display_name": streamer,
            "url": video.as_uri(), "title": video.stem, "extra": {}, "mock": True}
    session = storage.new_session_dir(platform, streamer, meta)
    print(f"Mock session: {session}")

    cmd = ffmpeg_segment_command(session, 1)
    cmd[cmd.index("pipe:0")] = str(video)
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not storage.list_segment_files(session):
        print("Stream copy into MPEG-TS failed; re-encoding to H.264/AAC first...")
        for f in storage.list_segment_files(session):
            f.unlink()
        reenc = cmd[:cmd.index("-c")] + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                         "-c:a", "aac", "-b:a", "160k"] + cmd[cmd.index("copy") + 1:]
        subprocess.run(reenc, check=True, capture_output=True)

    files = storage.list_segment_files(session)
    total = sum(media_timing(f)[1] for f in files)
    base = time.time() - total - 5            # pretend the video was received live, ending now
    end = base
    for i, f in enumerate(files):
        end += media_timing(f)[1]
        os.utime(f, (end, end))
        segment_manager.complete_segment(session, f, run=1, final_in_run=(i == len(files) - 1))
    storage.update_json(session / "session.json", status="ended", ended=time.time(), created=base,
                        runs=[{"run": 1, "started": base, "first_segment": 1}])
    if chat_file:
        n = import_chat(chat_file, session, base, total)
        print(f"Imported {n} chat messages.")
    print(f"{len(files)} segment(s), {total / 60:.1f} min. Analysing...")

    pipeline = Pipeline()
    pipeline.start()
    try:
        pipeline.drain()
    finally:
        pipeline.stop(wait=True)
    for side in storage.session_segments(session):
        print(f"\n{side['file']}: {side['status']}" + (f" ({side.get('error')})" if side.get("error") else ""))
        for note in side.get("analysis_notes") or []:
            print(f"  note: {note}")
        for m in (side.get("moments") or [])[:3]:
            print(json.dumps({k: m[k] for k in ("streamer", "platform", "segment", "start_timestamp",
                                               "end_timestamp", "viral_score", "reason", "signals")}, indent=2))
    clips = [e["path"] for e in json.loads((session / "clips.json").read_text())] \
        if (session / "clips.json").exists() else []
    print(f"\nClips ({len(clips)}): " + ("\n  " + "\n  ".join(clips) if clips else "none above "
                                       f"MINIMUM_VIRAL_SCORE={config.MINIMUM_VIRAL_SCORE}"))
    return 0


def import_chat(chat_file, session, base, total):
    """
    Mock chat: JSON lines with "t" (seconds from the start of the video) or
    "ts" (epoch seconds), plus "user" and "text" (optional "emotes").
    """
    from clipping_engine.chat.chat_log import ChatLog
    log_ = ChatLog(session, "mock")
    with open(session / "chat.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "coverage", "ts": base - 1, "available": True, "source": "mock"}) + "\n")
    n = 0
    with open(chat_file, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            ts = rec["ts"] if "ts" in rec else base + float(rec["t"])
            log_.message(ts, rec.get("user", "viewer"), rec.get("text", ""), rec.get("emotes") or ())
            n += 1
    with open(session / "chat.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "coverage", "ts": base + total + 1, "available": False,
                             "source": "mock", "reason": "end of mock video"}) + "\n")
    return n


# ---------------------------------------------------------------- check / auth

def run_check():
    from clipping_engine.analysis import ai_evaluator
    from clipping_engine.capture.streamlink_manager import platform_support, streamlink_version
    from clipping_engine.notifications import discord_notifier

    def row(ok, name, detail):
        print(f"[{'OK' if ok else '--'}] {name}: {detail}")

    row(bool(shutil.which(config.FFMPEG_PATH)), "FFmpeg", shutil.which(config.FFMPEG_PATH) or "not found")
    version = streamlink_version()
    row(bool(version), "Streamlink", version or "not found (pip install streamlink)")
    for platform, (ok, message) in platform_support().items():
        row(ok, f"  capture {platform}", message)
    for name in ("yamnet.tflite", "face_detection_yunet_2023mar.onnx"):
        from clipping_engine.analysis.model_files import ModelUnavailable, model_path
        try:
            row(True, f"model {name}", model_path(name))
        except ModelUnavailable as exc:
            row(False, f"model {name}", exc)
    try:
        import faster_whisper  # noqa: F401  (availability check)
        row(True, "faster-whisper", f"model '{config.WHISPER_MODEL}' downloads on first use")
    except ImportError:
        row(False, "faster-whisper", "not installed")
    row(ai_evaluator.available(), "Claude (AI evaluation)",
        config.CLAUDE_MODEL if ai_evaluator.available() else "ANTHROPIC_API_KEY not set (AI signal skipped)")
    row(discord_notifier.enabled(), "Discord", "webhook configured (test it: python -m clipping_engine discord)"
        if discord_notifier.enabled() else discord_notifier.status())
    from clipping_engine.chat.twitch_chat import TwitchUserToken, chat_client_credentials
    cid, secret = chat_client_credentials()
    row(TwitchUserToken(cid, secret).available, "Twitch chat",
        "authorized" if TwitchUserToken(cid, secret).available else "run: python -m clipping_engine auth twitch")
    row(bool(os.getenv("YOUTUBE_API_KEY")), "YouTube chat", "uses YOUTUBE_API_KEY" if os.getenv("YOUTUBE_API_KEY")
        else "YOUTUBE_API_KEY not set")
    row(config.KICK_WEBHOOK_ENABLED, "Kick chat", f"webhooks on port {config.KICK_WEBHOOK_PORT}"
        if config.KICK_WEBHOOK_ENABLED else "off (needs a public webhook URL, see README)")
    return 0


def run_auth(service):
    if service != "twitch":
        sys.exit("only 'twitch' needs an interactive login")
    from clipping_engine.chat.twitch_chat import TwitchAuthError, TwitchUserToken, chat_client_credentials
    cid, secret = chat_client_credentials()
    if not cid:
        sys.exit("TWITCH_CLIENT_ID is not set (stream-monitor/.env or clipping_engine/.env)")
    try:
        TwitchUserToken(cid, secret).device_login()
    except TwitchAuthError as exc:
        sys.exit(f"Twitch login failed: {exc}")
    return 0


def unsent_clips():
    """(session_dir, registry entry) for clips that were made but never posted to Discord."""
    from clipping_engine.storage import clip_storage
    from clipping_engine.storage.recording_storage import list_sessions
    out = []
    for session in list_sessions():
        for entry in clip_storage.load_registry(session):
            if not entry.get("notified") and Path(entry.get("path", "")).exists():
                out.append((session, entry))
    return out


def run_discord(send_pending=False):
    """Check the webhook, post a test message, and optionally post clips that were never sent."""
    from clipping_engine.notifications import discord_notifier
    from clipping_engine.storage import clip_storage
    from clipping_engine.storage.recording_storage import read_json
    ok, message = discord_notifier.send_test()
    print(f"[{'OK' if ok else '--'}] Discord: {message}")
    pending = unsent_clips()
    print(f"Clips made but not yet posted: {len(pending)}")
    if not pending:
        print(f"(Only moments scoring at least MINIMUM_VIRAL_SCORE={config.MINIMUM_VIRAL_SCORE} become clips.)")
    if not ok:
        return 1
    if pending and not send_pending:
        print("Post them with: python -m clipping_engine discord --send-pending")
    if send_pending:
        sent = 0
        for session, entry in pending:
            clip = read_json(Path(entry["path"]).with_suffix(".json"))
            result = discord_notifier.notify(clip)
            if result["sent"]:
                clip_storage.update_clip(session, entry["id"], notified=True, attached=result["attached"])
                sent += 1
            else:
                print(f"  failed: {Path(entry['path']).name}: {result['error']}")
        print(f"Posted {sent} of {len(pending)} clip(s).")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m clipping_engine",
                                     description="Record live streams and turn their best moments into vertical clips.")
    parser.add_argument("--interval", type=int, help="monitor interval in seconds (default: stream-monitor config)")
    parser.add_argument("--verbose", action="store_true", help="also print the log to the terminal")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="check dependencies and credentials")
    auth = sub.add_parser("auth", help="one-time login for chat access")
    auth.add_argument("service", choices=["twitch"])
    disc = sub.add_parser("discord", help="test the Discord webhook and post clips that were never sent")
    disc.add_argument("--send-pending", action="store_true", help="post clips that were made but never sent")
    mock = sub.add_parser("mock", help="analyse a prerecorded video without live connections")
    mock.add_argument("video")
    mock.add_argument("--chat", help="JSON lines chat file (t or ts, user, text)")
    mock.add_argument("--streamer", default="mock_streamer")
    mock.add_argument("--platform", default="mock")
    mock.add_argument("--no-discord", action="store_true")
    mock.add_argument("--segment-minutes", type=float, help="override RECORDING_SEGMENT_MINUTES for this run")
    args = parser.parse_args(argv)

    load_env()
    setup_logging(args.verbose)
    if args.command == "check":
        return run_check()
    if args.command == "discord":
        return run_discord(args.send_pending)
    if args.command == "auth":
        return run_auth(args.service)
    if args.command == "mock":
        if args.segment_minutes:
            config.RECORDING_SEGMENT_MINUTES = args.segment_minutes
        return run_mock(args.video, args.chat, args.streamer, args.platform, discord=not args.no_discord)
    return run_engine(args.interval)
