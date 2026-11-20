"""
Clip extraction, captions and vertical formatting with real FFmpeg.

Extraction accuracy is checked with a "chirp" recording: the audio pitch
rises 20 Hz per second, so measuring the pitch at the start and end of an
extracted clip tells exactly which part of the recording it came from.
"""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import requests

from clipping_engine import config
from clipping_engine.capture import segment_manager
from clipping_engine.capture.streamlink_manager import ffmpeg_segment_command
from clipping_engine.editing import caption_generator, clip_extractor, vertical_formatter
from clipping_engine.media import decode_audio, media_timing, probe
from clipping_engine.storage import recording_storage as storage
from clipping_engine.tests.conftest import needs_ffmpeg

pytestmark = needs_ffmpeg
CACHE = Path(__file__).parent / ".cache"


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def dominant_freq(audio, rate=16000):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return np.fft.rfftfreq(len(audio), 1 / rate)[int(np.argmax(spectrum))]


@pytest.fixture
def chirp_session(engine, tmp_path, monkeypatch):
    """A 30 s recording (pitch = 200 + 20*t Hz) split into 10 s segments like a live capture."""
    monkeypatch.setattr(engine, "RECORDING_SEGMENT_MINUTES", 10 / 60)
    src = tmp_path / "source.ts"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
           "-f", "lavfi", "-i", "aevalsrc=0.5*sin(2*PI*(200*t+10*t*t)):s=48000",
           "-t", "30", "-c:v", "libx264", "-g", "30", "-preset", "ultrafast", "-c:a", "aac", "-f", "mpegts", str(src))
    session = storage.new_session_dir("twitch", "chirp", {"platform": "twitch", "streamer": "chirp"})
    with open(src, "rb") as fh:
        subprocess.run(ffmpeg_segment_command(session, 1), stdin=fh, check=True, capture_output=True)
    base = 1_800_000_000.0
    files = storage.list_segment_files(session)
    assert len(files) == 3
    end = base
    for i, f in enumerate(files):
        _, dur = media_timing(f)
        end += dur
        os.utime(f, (end, end))          # as if received live
        segment_manager.complete_segment(session, f, run=1, final_in_run=(i == len(files) - 1))
    return session, base


def test_cross_segment_extraction_is_accurate(chirp_session, tmp_path):
    session, base = chirp_session
    out = clip_extractor.extract(session, 1, 8.0, 14.0, tmp_path / "clip.mp4")
    start, duration = media_timing(out)
    assert duration == pytest.approx(6.0, abs=0.15)
    streams = {s["codec_type"]: s for s in probe(out)["streams"]}
    assert streams["video"]["codec_name"] == "h264" and streams["audio"]["codec_name"] == "aac"
    assert abs(float(streams["video"]["duration"]) - float(streams["audio"]["duration"])) < 0.1   # A/V in sync

    audio = decode_audio(out)
    head, tail = audio[:8000], audio[-8000:]                       # first / last 0.5 s
    t_head = (dominant_freq(head) - 200) / 20                       # recording time at clip start
    t_tail = (dominant_freq(tail) - 200) / 20
    assert t_head == pytest.approx(8.25, abs=0.5), t_head            # crosses the 10 s boundary...
    assert t_tail == pytest.approx(13.75, abs=0.5), t_tail           # ...into segment 2


def test_extraction_refuses_missing_segment(chirp_session, tmp_path):
    session, base = chirp_session
    storage.update_json(storage.sidecar_for(storage.segment_file(session, 2)), deleted=True)
    with pytest.raises(Exception):
        clip_extractor.extract(session, 1, 8.0, 22.0, tmp_path / "clip.mp4")


def words(text, start=0.5, step=0.4):
    out = []
    t = start
    for w in text.split():
        out.append({"start": t, "end": t + step * 0.8, "word": w})
        t += step
    return out


def test_caption_grouping_and_animation():
    ws = words("no way he actually did that. bro what is happening right now")
    static = caption_generator.build_ass(ws, 0, 10, animated=False)
    animated = caption_generator.build_ass(ws, 0, 10, animated=True)
    static_lines = [l for l in static.splitlines() if l.startswith("Dialogue")]
    anim_lines = [l for l in animated.splitlines() if l.startswith("Dialogue")]
    assert static_lines[0].endswith("no way he actually did")
    assert static_lines[1].endswith("that.")                      # sentence end closes a phrase
    assert len(anim_lines) == len([w for w in ws])                # one event per spoken word
    assert caption_generator.HIGHLIGHT in anim_lines[0]
    assert "PlayResY: 1920" in static
    assert caption_generator._clean("{\\b1}hi}") == "b1hi"         # no ASS injection from transcripts


def _sample(name, url):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            pytest.skip(f"could not download test image {name}")
        path.write_bytes(r.content)
    return path


@pytest.fixture
def face_image():
    # OpenCV's sample image, used only for testing face detection.
    return _sample("lena.jpg", "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg")


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", CACHE / "models")
    from clipping_engine.analysis import model_files
    try:
        model_files.model_path("face_detection_yunet_2023mar.onnx")
    except model_files.ModelUnavailable as exc:
        pytest.skip(str(exc))


def check_vertical(path, expect_audio=True):
    info = probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p"
    if expect_audio:
        a = next(s for s in info["streams"] if s["codec_type"] == "audio")
        assert a["codec_name"] == "aac"


def first_frame(path, t=1.0):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(1920, 1080, 3)


def test_fit_blur_layout_with_captions_and_loudness(tmp_path, models):
    src = tmp_path / "game.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-f", "lavfi",
           "-i", "sine=frequency=330:sample_rate=48000", "-af", "volume=-30dB", "-t", "4",
           "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(src))
    ws = words("WHAT JUST HAPPENED", start=0.2, step=1.0)
    with_caps = vertical_formatter.render(src, tmp_path / "v.mp4", words=ws, clip_start=0, clip_end=4)
    assert with_caps["layout"] == "fit_blur" and with_caps["captions"]
    check_vertical(with_caps["path"])

    no_caps = vertical_formatter.render(src, tmp_path / "plain.mp4", words=None)
    a, b = first_frame(with_caps["path"]).astype(int), first_frame(no_caps["path"]).astype(int)
    diff = np.abs(a - b).mean(axis=2)
    rows = np.flatnonzero(diff.mean(axis=1) > 8)
    assert len(rows) > 30, "captions should change a band of rows"
    assert rows.min() > 960, "captions sit in the lower half"

    loud = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(with_caps["path"]), "-af", "ebur128", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    integrated = float(re.findall(r"I:\s+(-?\d+\.\d) LUFS", loud)[-1])
    assert integrated == pytest.approx(config.TARGET_LOUDNESS_LUFS, abs=2.5), integrated


def test_face_crop_layout_keeps_face_centered(tmp_path, face_image, models):
    src = tmp_path / "irl.mp4"
    ffmpeg("-loop", "1", "-i", str(face_image), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "3",
           "-vf", "scale=720:720,pad=1280:720:380:0", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src))
    out = vertical_formatter.render(src, tmp_path / "v.mp4")
    assert out["layout"] == "face_crop", out["layout_info"]
    check_vertical(out["path"])
    from clipping_engine.analysis.video_analyzer import detect_faces
    faces = detect_faces([first_frame(out["path"])])[0]
    assert faces, "face must survive the crop"
    x, y, w, h, _ = max(faces, key=lambda f: f[2] * f[3])
    assert 0.3 < x + w / 2 < 0.7


def test_stacked_layout_for_webcam_overlay(tmp_path, face_image, models):
    src = tmp_path / "gaming.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-loop", "1", "-i", str(face_image),
           "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "3",
           "-filter_complex", "[1:v]scale=260:260[cam];[0:v][cam]overlay=1000:440[v]", "-map", "[v]", "-map", "2:a",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src))
    out = vertical_formatter.render(src, tmp_path / "v.mp4")
    assert out["layout"] == "stacked", out["layout_info"]
    check_vertical(out["path"])
    from clipping_engine.analysis.video_analyzer import detect_faces
    faces = detect_faces([first_frame(out["path"])])[0]
    assert faces
    top_face = max(faces, key=lambda f: f[2] * f[3])
    assert top_face[1] + top_face[3] / 2 < 0.42, "webcam panel (with the face) is on top"
    assert top_face[2] > 0.25, "webcam is enlarged"


def test_discord_preview_fits_limit(tmp_path):
    src = tmp_path / "big.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30", "-f", "lavfi", "-i", "sine=frequency=440",
           "-t", "6", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "5", "-c:a", "aac", str(src))
    limit = 2 * 1024 * 1024
    assert src.stat().st_size > limit
    preview = vertical_formatter.discord_preview(src, tmp_path / "p.mp4", limit)
    assert preview and preview.stat().st_size <= limit
