"""Recording engine: real Streamlink + FFmpeg against a local live HLS stream."""

import json
import sys
from pathlib import Path

import numpy as np

import pytest

from clipping_engine.capture import segment_manager
from clipping_engine.capture.recording_manager import RecordingManager, StreamTarget
from clipping_engine.capture.streamlink_manager import can_handle_url, platform_support, streamlink_version
from clipping_engine.storage import recording_storage as storage
from clipping_engine.tests.conftest import needs_ffmpeg, needs_streamlink, wait_for

pytestmark = [needs_streamlink, needs_ffmpeg]


def target(name, url, platform="twitch"):
    return StreamTarget(platform, name, url, extra={"user_id": "123"})


def sidecars(session_dir):
    return storage.session_segments(session_dir)


def test_streamlink_supports_each_platform():
    assert streamlink_version()
    support = platform_support()
    for platform in ("twitch", "kick", "youtube"):
        ok, message = support[platform]
        assert ok, message
    assert not can_handle_url("https://example.com/not-a-stream")


def test_live_starts_recording_and_offline_stops_it(engine, live_hls):
    done = []
    rm = RecordingManager(on_segment=lambda d, s: done.append(s))
    assert rm.ensure_recording(target("Streamer_One", live_hls.url)) == "started"
    rec = rm.recorders[("twitch", "streamer_one")]
    assert wait_for(lambda: len(done) >= 2, 30), "no segments recorded"

    assert rm.stop_recording("twitch", "Streamer_One")
    rec.join(40)
    assert not rec.is_alive()

    sides = sidecars(rec.session_dir)
    assert len(sides) >= 3
    assert [s["index"] for s in sides] == list(range(1, len(sides) + 1))
    assert sides[-1]["final_in_run"] and not any(s["final_in_run"] for s in sides[:-1])
    for s in sides[:-1]:
        assert 1.5 <= s["duration"] <= 3.5, s           # 2-second segments (cut on keyframes)
        assert s["status"] == "queued"
    # Segments of a run are contiguous in wall-clock time, and anchored close to
    # when the data was actually received.
    timed = segment_manager.run_segments(rec.session_dir, 1)
    for a, b in zip(timed, timed[1:]):
        assert abs(b["wall_start"] - a["wall_end"]) < 0.05
    assert abs(timed[-1]["wall_end"] - timed[-1]["mtime_end"]) < 2.5
    meta = json.loads((rec.session_dir / "session.json").read_text())
    assert meta["status"] == "ended" and meta["platform"] == "twitch"
    assert rec.session_dir.parent.name == "Streamer_One" and rec.session_dir.name == "session_001"


def test_no_duplicate_recording_for_same_streamer(engine, live_hls):
    rm = RecordingManager()
    t = target("dupe", live_hls.url)
    assert rm.ensure_recording(t) == "started"
    assert rm.ensure_recording(t) == "recording"
    assert rm.ensure_recording(target("DUPE", live_hls.url)) == "recording"
    assert len(rm.recorders) == 1
    rm.stop_all(40)
    assert len(list((engine.RECORDINGS_DIR / "twitch" / "dupe").glob("session_*"))) == 1


def test_multiple_simultaneous_recordings_and_limit(engine, live_hls, monkeypatch):
    monkeypatch.setattr(engine, "MAX_CONCURRENT_RECORDINGS", 2)
    rm = RecordingManager()
    assert rm.ensure_recording(target("a", live_hls.url)) == "started"
    assert rm.ensure_recording(target("b", live_hls.url, platform="kick")) == "started"
    assert "MAX_CONCURRENT_RECORDINGS" in rm.ensure_recording(target("c", live_hls.url))
    recs = list(rm.recorders.values())
    assert wait_for(lambda: all(r.segments >= 1 for r in recs), 30)
    rm.stop_all(40)
    assert (engine.RECORDINGS_DIR / "twitch" / "a" / "session_001" / "segment_0001.ts").exists()
    assert (engine.RECORDINGS_DIR / "kick" / "b" / "session_001" / "segment_0001.ts").exists()
    assert not (engine.RECORDINGS_DIR / "twitch" / "c").exists()


def test_reconnects_after_connection_drop(engine, live_hls):
    rm = RecordingManager()
    rm.ensure_recording(target("flaky", live_hls.url))
    rec = rm.recorders[("twitch", "flaky")]
    assert wait_for(lambda: rec.segments >= 1, 30)

    live_hls.stop()                                   # connection lost
    assert wait_for(lambda: rec.state == "reconnecting", 30)
    live_hls.start()                                  # stream back
    assert wait_for(lambda: rec.run_number >= 2 and rec.state == "recording", 40)
    before = rec.segments
    assert wait_for(lambda: rec.segments > before, 30)
    rm.stop_all(40)

    sides = sidecars(rec.session_dir)
    runs = {s["run"] for s in sides}
    assert len(runs) >= 2, "second run should continue the same session"
    indices = [s["index"] for s in sides]
    assert indices == sorted(set(indices)), "segment numbers continue without overwriting"
    first_run = [s for s in sides if s["run"] == min(runs)]
    assert first_run[-1]["final_in_run"]


def test_unsupported_url_is_refused(engine):
    rm = RecordingManager()
    message = rm.ensure_recording(target("nope", "https://example.com/not-a-stream"))
    assert "cannot open" in message
    assert not rm.recorders


def test_recovery_after_crash(engine, live_hls):
    rm = RecordingManager()
    rm.ensure_recording(target("crashy", live_hls.url))
    rec = rm.recorders[("twitch", "crashy")]
    assert wait_for(lambda: rec.segments >= 1, 30)
    rm.stop_all(40)
    session = rec.session_dir
    # Simulate a crash: session still "recording", last sidecar missing, one segment mid-processing.
    storage.update_json(session / "session.json", status="recording")
    last = storage.list_segment_files(session)[-1]
    storage.sidecar_for(last).unlink()
    first = storage.sidecar_for(storage.list_segment_files(session)[0])
    storage.update_json(first, status="processing")

    queued = segment_manager.recover_interrupted()
    assert queued >= 2
    assert storage.read_json(first)["status"] == "queued"
    recovered = storage.read_json(storage.sidecar_for(last))
    assert recovered["status"] == "queued" and recovered["final_in_run"]
    assert storage.read_json(session / "session.json")["status"] == "ended"


def test_cleanup_never_deletes_unprocessed(engine, live_hls, monkeypatch):
    rm = RecordingManager()
    rm.ensure_recording(target("cleanup", live_hls.url))
    rec = rm.recorders[("twitch", "cleanup")]
    assert wait_for(lambda: rec.segments >= 3, 40)
    rm.stop_all(40)
    files = storage.list_segment_files(rec.session_dir)
    # Mark only the first segment processed; the second (its neighbour) is not.
    storage.update_json(storage.sidecar_for(files[0]), status="done")
    monkeypatch.setattr(engine, "MAX_RECORDINGS_GB", 0)          # force the size limit
    storage.cleanup_recordings()
    assert all(f.exists() for f in files), "segment 1 is still needed as context for segment 2"
    for f in files:
        storage.update_json(storage.sidecar_for(f), status="done")
    storage.cleanup_recordings()
    assert not any(f.exists() for f in files)
    assert all(storage.read_json(storage.sidecar_for(f))["deleted"] for f in files)


def test_time_map_handles_startup_burst_and_ad_gap():
    """Burst: 6 s of backlog arrive at once. Ad break: 90 s cut out of the recording at media 60 s."""
    media = [0, 3, 6] + list(range(7, 61)) + list(range(61, 121))
    wall = [100, 100, 100] + [94 + m for m in range(7, 61)] + [94 + 90 + m for m in range(61, 121)]
    tm = segment_manager.TimeMap(media, wall)
    assert tm.wall(3) == pytest.approx(97)          # backlog content was live 6 s before it arrived
    assert tm.wall(30) == pytest.approx(124)
    assert tm.wall(59) == pytest.approx(153)        # before the ad: not shifted
    assert tm.wall(90) == pytest.approx(274)        # after the ad: shifted by 90 s
    assert tm.media(274) == pytest.approx(90)       # chat after the ad maps back correctly
    assert tm.media(124) == pytest.approx(30)
    assert 60 <= tm.media(200) <= 61                # chat during the ad -> where the video resumes
    assert list(tm.wall(np.array([30.0, 90.0]))) == pytest.approx([124, 274])


def test_capture_logs_timing_samples(engine, live_hls):
    rm = RecordingManager()
    rm.ensure_recording(target("timed", live_hls.url))
    rec = rm.recorders[("twitch", "timed")]
    assert wait_for(lambda: rec.segments >= 6, 60)          # past the start-up burst
    rm.stop_all(40)
    lines = [json.loads(l) for l in (rec.session_dir / "timing.jsonl").read_text().splitlines()]
    assert len(lines) >= 5 and all(l["run"] == 1 for l in lines)
    media = [l["media"] for l in lines]
    assert media == sorted(media)
    tm = segment_manager.time_map(rec.session_dir, 1)
    last = lines[-1]
    assert abs(tm.wall(last["media"]) - last["wall"]) < 3      # steady state: received ~live


# ---------------------------------------------------------------- lessons from the first live run

FAKE = Path(__file__).with_name("fake_streamlink.py")


@pytest.fixture
def fake_streamlink(engine, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("uses an executable script as the streamlink binary")
    monkeypatch.setattr(engine, "STREAMLINK_PATH", str(FAKE))
    monkeypatch.setattr(engine, "STALL_RESTART_SECONDS", 8)     # > FFmpeg's ~5 s initial probe
    from clipping_engine.capture import streamlink_manager
    streamlink_manager.can_handle_url.cache_clear()
    streamlink_manager.streamlink_version.cache_clear()
    yield monkeypatch
    streamlink_manager.can_handle_url.cache_clear()
    streamlink_manager.streamlink_version.cache_clear()


def test_twitch_ad_break_is_not_a_stall(fake_streamlink):
    """Streamlink pauses output during ad breaks; that must not trigger a reconnect."""
    fake_streamlink.setenv("FAKE_PAUSE_AT", "10")
    fake_streamlink.setenv("FAKE_PAUSE_SECONDS", "15")           # > STALL_RESTART_SECONDS
    rm = RecordingManager()
    rm.ensure_recording(target("adbreak", "https://twitch.tv/adbreak"))
    rec = rm.recorders[("twitch", "adbreak")]
    assert wait_for(lambda: rec.segments >= 1, 20), "recording before the ad"
    assert wait_for(lambda: rec.state == "recording (ad break)", 20)
    assert wait_for(lambda: rec.state == "recording", 20), "resumes after the ad"
    before = rec.segments
    assert wait_for(lambda: rec.segments > before, 15)
    rm.stop_all(30)
    assert rec.run_number == 1, "no reconnect during the ad break"
    assert rec.ad_breaks == 1
    assert "1 ad break(s) skipped" in "".join(rm.status_lines()) or not rec.is_alive()


def test_real_stall_still_restarts(fake_streamlink):
    fake_streamlink.setenv("FAKE_PAUSE_AT", "10")
    fake_streamlink.setenv("FAKE_PAUSE_SECONDS", "15")
    fake_streamlink.setenv("FAKE_PAUSE_SILENT", "1")              # no ad-filter log lines
    rm = RecordingManager()
    rm.ensure_recording(target("stally", "https://twitch.tv/stally"))
    rec = rm.recorders[("twitch", "stally")]
    assert wait_for(lambda: rec.segments >= 1, 20), "recording before the stall"
    assert rec.run_number == 1
    assert wait_for(lambda: rec.run_number >= 2, 30), "a genuine stall is restarted"
    rm.stop_all(30)
    assert "stalled" in (rec.last_error or "")


def test_gives_up_and_cools_down(engine, monkeypatch):
    from clipping_engine.tests.conftest import free_port
    monkeypatch.setattr(engine, "RECORDING_GIVE_UP_MINUTES", 0.05)       # 3 seconds
    monkeypatch.setattr(engine, "RECORDING_RETRY_COOLDOWN_MINUTES", 10)
    rm = RecordingManager()
    dead = target("nobody", f"hls://http://127.0.0.1:{free_port()}/live.m3u8")
    assert rm.ensure_recording(dead) == "started"
    rec = rm.recorders[("twitch", "nobody")]
    rec.join(90)
    assert rec.state == "gave up" and "without data" in rec.last_error
    rm.reap()
    assert "retrying in" in rm.ensure_recording(dead), "no endless restart loop"
    assert not rm.recorders


def test_low_disk_space_blocks_new_recordings(engine, monkeypatch):
    from clipping_engine.capture import recording_manager
    monkeypatch.setattr(recording_manager, "free_gb", lambda path: 3.0)
    rm = RecordingManager()
    assert "GB free" in rm.ensure_recording(target("full", "https://twitch.tv/full"))
    assert not rm.recorders


def test_folder_names_keep_underscores(engine):
    assert storage.safe_name("caseoh_") == "caseoh_" and storage.safe_name("caseoh") == "caseoh"
    assert storage.safe_name("..") == "unnamed" and storage.safe_name("a/b") == "a_b"
