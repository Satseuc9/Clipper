"""
Integration: monitor -> recording, processing queue (retries, ordering,
crashed workers), and a full mock-mode run that produces a clip and
sends it to (mock) Discord.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from clipping_engine import config
from clipping_engine.capture.recording_manager import RecordingManager
from clipping_engine.monitor_bridge import load_monitor, make_hook, targets_from_rows
from clipping_engine.pipeline import Pipeline
from clipping_engine.storage import recording_storage as storage
from clipping_engine.tests.conftest import needs_ffmpeg, needs_streamlink, wait_for

monitor_main, monitor_config, is_stale = load_monitor()
from platforms.base import LIVE, OFFLINE, UNKNOWN, StreamResult  # noqa: E402  (stream-monitor)


def row(platform, name, status, url="", error=None, **extra):
    return StreamResult(platform, name, status, url=url, error=error, extra=extra)


# ---------------------------------------------------------------- monitor -> recording

def test_rows_to_recording_decisions(monkeypatch):
    rows = [row("twitch", "live_one", LIVE, "https://twitch.tv/live_one", user_id="1"),
            row("kick", "gone", OFFLINE, "https://kick.com/gone"),
            row("youtube", "flaky", UNKNOWN, error="HTTP 503"),
            row("twitch", "stale_live", LIVE, "https://twitch.tv/stale_live", error="HTTP 500"),  # kept from memory
            row("tiktok", "tt", LIVE, "https://tiktok.com/@tt")]
    start, stop = targets_from_rows([("ALL", "", rows)], is_stale)
    assert [t.name for t in start] == ["live_one"] and start[0].extra == {"user_id": "1"}
    assert stop == [("kick", "gone")]
    monkeypatch.setattr(config, "CLIP_ONLY", {"twitch": ["someone_else"]})
    assert targets_from_rows([("ALL", "", rows)], is_stale)[0] == []


@needs_streamlink
@needs_ffmpeg
def test_monitor_live_starts_and_offline_stops_recording(engine, live_hls):
    rec = RecordingManager()
    pipeline = Pipeline(process_fn=lambda *a: {"candidates": 0, "clips": [], "sent": 0}, use_processes=False)
    hook = make_hook(rec, None, pipeline, is_stale)

    lines = hook([("TWITCH", "", [row("twitch", "streamer_one", LIVE, live_hls.url, user_id="9")])], [])
    assert any("streamer_one: recording" in l or "streamer_one: starting" in l for l in lines)
    recorder = rec.recorders[("twitch", "streamer_one")]
    assert wait_for(lambda: recorder.segments >= 1, 30)
    hook([("TWITCH", "", [row("twitch", "streamer_one", LIVE, live_hls.url)])], [])    # still live: no duplicate
    assert len(list((engine.RECORDINGS_DIR / "twitch" / "streamer_one").glob("session_*"))) == 1

    hook([("TWITCH", "", [row("twitch", "streamer_one", UNKNOWN, error="timeout")])], [])
    assert recorder.is_alive() and not recorder.stopping, "UNKNOWN never stops a recording"

    hook([("TWITCH", "", [row("twitch", "streamer_one", OFFLINE)])], [])
    recorder.join(40)
    assert not recorder.is_alive()
    meta = storage.read_json(recorder.session_dir / "session.json")
    assert meta["status"] == "ended" and meta["extra"] == {"user_id": "9"}


class FakePlatform:
    key, title, available = "twitch", "TWITCH", True

    def __init__(self, results):
        self.results = results

    def poll_interval(self, names, interval):
        return interval

    def check(self, names, interval):
        return self.results

    def unknown(self, name, error):
        return row("twitch", name, UNKNOWN, error=error)


def test_hook_runs_inside_the_real_monitor_loop(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(monitor_config, "STREAMERS", {"twitch": ["a"]})
    monkeypatch.setattr(monitor_main, "build_monitors",
                        lambda: [FakePlatform([row("twitch", "a", LIVE, "https://twitch.tv/a", user_id="5")])])
    monitor_main.run_monitor(10, once=True, on_cycle=lambda sections, changes: calls.append(sections) or ["HOOK OK"])
    out = capsys.readouterr().out
    assert calls and calls[0][0][2][0].extra == {"user_id": "5"}
    assert "HOOK OK" in out and "[ LIVE ]" in out


def test_monitor_platforms_expose_ids_for_chat(monkeypatch):
    """The monitor's results carry the IDs chat needs (Twitch user_id, YouTube live_chat_id, Kick id)."""
    from platforms.kick import KickMonitor
    from platforms.youtube import YouTubeMonitor
    import requests
    from requests.structures import CaseInsensitiveDict

    def resp(body):
        r = requests.Response()
        r.status_code, r._content, r.headers = 200, json.dumps(body).encode(), CaseInsensitiveDict()
        return r

    class S:
        headers = {}

        def __init__(self, handler):
            self.handler = handler

        def request(self, method, url, timeout=None, params=None, data=None, headers=None):
            return resp(self.handler(url, params))

    def yt(url, params):
        if url.endswith("/channels"):
            return {"items": [{"id": "UC1", "snippet": {"title": "One"},
                               "contentDetails": {"relatedPlaylists": {"uploads": "UU1"}}}]}
        if url.endswith("/playlistItems"):
            return {"items": [{"contentDetails": {"videoId": "v1"}}]}
        return {"items": [{"id": "v1", "snippet": {"title": "live", "liveBroadcastContent": "live"},
                           "liveStreamingDetails": {"actualStartTime": "2026-11-20T01:00:00Z",
                                                    "concurrentViewers": "10", "activeLiveChatId": "CHAT1"}}]}
    monkeypatch.setenv("YOUTUBE_API_KEY", "k")
    y = YouTubeMonitor()
    y.configured = True
    y.session = S(yt)
    r = y.check(["UC1"], 60)[0]
    assert r.status == LIVE and r.extra["live_chat_id"] == "CHAT1" and r.extra["video_id"] == "v1"

    def kick(url, params):
        if url.endswith("/oauth/token"):
            return {"access_token": "t", "expires_in": 3600}
        return {"data": [{"slug": "abc", "broadcaster_user_id": 77, "stream_title": "x",
                          "stream": {"is_live": True, "viewer_count": 5, "start_time": "2026-11-20T01:00:00Z"}}]}
    k = KickMonitor()
    k.configured = True
    k.session = S(kick)
    k.token.session = k.session
    r = k.check(["abc"], 60)[0]
    assert r.status == LIVE and r.extra == {"broadcaster_user_id": 77, "started_at": "2026-11-20T01:00:00Z"}


# ---------------------------------------------------------------- processing queue

def fake_session(n=3, platform="twitch", name="queue"):
    session = storage.new_session_dir(platform, name, {"platform": platform, "streamer": name})
    for i in range(1, n + 1):
        seg = storage.segment_file(session, i)
        seg.write_bytes(b"x")
        storage.write_json(storage.sidecar_for(seg), {"index": i, "file": seg.name, "run": 1, "status": "queued",
                                                      "attempts": 0, "duration": 600.0, "final_in_run": i == n})
    storage.update_json(session / "session.json", status="ended")
    return session


def statuses(session):
    return [s["status"] for s in storage.session_segments(session)]


def test_queue_processes_in_order_and_retries(engine, monkeypatch):
    monkeypatch.setattr(config, "PROCESSING_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(config, "PROCESSING_MAX_ATTEMPTS", 3)
    session = fake_session()
    order, failures = [], {"n": 0}

    def process(sess, index, snapshot):
        order.append(index)
        if index == 2 and failures["n"] < 2:          # API outage: fails twice, then works
            failures["n"] += 1
            raise ConnectionError("Anthropic API unreachable")
        return {"candidates": 1, "best_score": 50, "clips": [], "sent": 0}

    p = Pipeline(process_fn=process, use_processes=False)
    p.start()
    assert p.drain(30)
    p.stop()
    assert statuses(session) == ["done", "done", "done"]
    assert order == [1, 2, 2, 2, 3], "segment 3 waits for segment 2"
    side2 = storage.session_segments(session)[1]
    assert side2["attempts"] == 3


def test_permanent_failure_is_marked_failed(engine, monkeypatch):
    monkeypatch.setattr(config, "PROCESSING_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(config, "PROCESSING_MAX_ATTEMPTS", 2)
    session = fake_session(n=2)

    def process(sess, index, snapshot):
        if index == 1:
            raise RuntimeError("corrupt segment")
        return {"candidates": 0, "clips": [], "sent": 0}

    p = Pipeline(process_fn=process, use_processes=False)
    p.start()
    assert p.drain(30)
    p.stop()
    assert statuses(session) == ["failed", "done"], "a broken segment does not block the session forever"
    assert "corrupt segment" in storage.session_segments(session)[0]["error"]


def crash_on_first_attempt(session, index, snapshot):
    """Runs in a real worker process; kills it the first time (like an OOM kill)."""
    from clipping_engine.processing import apply_snapshot
    apply_snapshot(snapshot)
    side = storage.read_json(storage.sidecar_for(storage.segment_file(session, index)))
    if side["attempts"] == 1:
        os._exit(1)
    return {"candidates": 0, "clips": [], "sent": 0, "pid": os.getpid()}


def test_crashed_worker_process_is_replaced(engine, monkeypatch):
    monkeypatch.setattr(config, "PROCESSING_RETRY_DELAY_SECONDS", 0)
    session = fake_session(n=1)
    p = Pipeline(process_fn=crash_on_first_attempt, use_processes=True)
    p.start()
    assert p.drain(120)
    p.stop()
    side = storage.session_segments(session)[0]
    assert side["status"] == "done" and side["attempts"] == 2
    assert side["result"]["pid"] != os.getpid(), "work ran in a separate process"


# ---------------------------------------------------------------- full mock run

@needs_ffmpeg
def test_mock_mode_end_to_end(engine, discord, tmp_path, monkeypatch):
    """Prerecorded video + chat -> segments -> analysis -> vertical clip -> Discord."""
    from clipping_engine import main
    from clipping_engine.media import decode_audio
    from clipping_engine.tests.test_analysis import ESC, SPEECH, fetch

    monkeypatch.setattr(config, "MODEL_DIR", Path(__file__).parent / ".cache" / "models")
    monkeypatch.setattr(config, "RECORDING_SEGMENT_MINUTES", 0.5)
    monkeypatch.setattr(config, "ENABLE_TRANSCRIPTION", False)       # no Whisper download in tests
    monkeypatch.setattr(config, "ENABLE_AI_EVALUATION", False)       # no Claude calls in tests
    monkeypatch.setattr(config, "PROCESSING_RETRY_DELAY_SECONDS", 1)

    rate, seconds = 16000, 75
    speech = decode_audio(fetch("speech.ogg", SPEECH))
    laugh = decode_audio(fetch("esc_1-36164-B-26.wav", ESC + "1-36164-B-26.wav"))
    track = np.tile(speech, int(np.ceil(seconds * rate / len(speech))))[: seconds * rate] * 0.5
    track[40 * rate:40 * rate + len(laugh)] += laugh * 1.6
    (tmp_path / "a.f32").write_bytes(np.clip(track, -1, 1).astype(np.float32).tobytes())
    video = tmp_path / "vod.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={seconds}",
                    "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", str(tmp_path / "a.f32"), "-c:v", "libx264",
                    "-preset", "ultrafast", "-g", "60", "-c:a", "aac", "-shortest", str(video)], check=True)
    rng = np.random.default_rng(7)
    chat = [{"t": float(t), "user": f"u{rng.integers(80)}", "text": "hello chat"} for t in rng.uniform(0, seconds, 70)]
    chat += [{"t": float(t), "user": f"v{rng.integers(200)}", "text": rng.choice(["KEKW", "LMAOOO 😂", "💀💀", "CLIP IT"])}
             for t in rng.uniform(43, 52, 45)]
    chat_file = tmp_path / "chat.jsonl"
    chat_file.write_text("\n".join(json.dumps(c) for c in sorted(chat, key=lambda c: c["t"])))

    assert main.run_mock(video, chat_file, streamer="mock_streamer", platform="twitch", discord=True) == 0

    session = next((engine.RECORDINGS_DIR / "twitch" / "mock_streamer").glob("session_*"))
    sides = storage.session_segments(session)
    assert [s["status"] for s in sides] == ["done"] * len(sides) and len(sides) == 3
    moments = [m for s in sides for m in (s.get("moments") or [])]
    best = max(moments, key=lambda m: m["viral_score"])
    for key in ("streamer", "platform", "segment", "start_timestamp", "end_timestamp", "viral_score", "reason", "signals"):
        assert key in best
    assert best["viral_score"] >= config.MINIMUM_VIRAL_SCORE
    assert best["signals"]["laughter_detected"] and best["signals"]["chat_spike"] and best["signals"]["emoji_spike"]
    assert best["abs_start"] - sides[0]["wall_start"] < 40 < best["abs_end"] - sides[0]["wall_start"]

    registry = json.loads((session / "clips.json").read_text())
    assert len(registry) == 1 and registry[0]["notified"], "one clip, sent once (no duplicates)"
    clip = Path(registry[0]["path"])
    info = json.loads(clip.with_suffix(".json").read_text())
    assert clip.exists() and info["layout"] == "fit_blur" and 30 <= info["duration"] <= 90
    assert len(discord.requests) == 1 and discord.requests[0]["files"]["files[0]"][0] == clip.name


@needs_streamlink
@needs_ffmpeg
def test_streamer_removed_from_monitor_config_stops_recording(engine, live_hls):
    rec = RecordingManager()
    pipeline = Pipeline(process_fn=lambda *a: {}, use_processes=False)
    hook = make_hook(rec, None, pipeline, is_stale)
    hook([("TWITCH", "", [row("twitch", "hasanabi", LIVE, live_hls.url)])], [])
    recorder = rec.recorders[("twitch", "hasanabi")]
    assert wait_for(lambda: recorder.segments >= 1, 30)
    hook([("TWITCH", "", [row("twitch", "someone_else", OFFLINE)])], [])      # hasanabi no longer listed
    recorder.join(40)
    assert not recorder.is_alive()


def test_status_line_explains_empty_results():
    from clipping_engine.pipeline import describe
    text = describe({"candidates": 0, "best_score": None, "clips": [], "signals": {"audio": True, "video": True,
                     "chat": False}, "notes": ["audio classifier unavailable: ImportError: no module"]})
    assert "audio+video" in text and "no chat" in text and "no speech" in text and "no ai" in text
    assert "audio classifier unavailable" in text
