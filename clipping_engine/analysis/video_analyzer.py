"""
Video analysis.

Stage 1 (whole segment, cheap): only keyframes are decoded, at 64x36 gray,
to get a visual-change curve (scene cuts, sudden action). Decoding only
keyframes keeps this to a few seconds for a 10-minute 1080p segment.

Stage 2 (candidates only): motion intensity at 4 fps around the moment and
still frames sent to Claude, which judges what is actually happening
(gameplay, events, reactions) - that is the "video understanding" part.

Faces: YuNet (OpenCV) on sampled frames, used to decide the vertical layout.
"""

import logging
import subprocess

import numpy as np

from clipping_engine import config
from clipping_engine.media import MediaError, input_args, run_ffmpeg

log = logging.getLogger(__name__)

SMALL_W, SMALL_H = 64, 36


def keyframe_change_curve(inputs, seconds, start=0.0):
    """Per-second visual change (0-1) from consecutive keyframes. None on failure."""
    if not config.ENABLE_VIDEO_ANALYSIS:
        return None
    try:
        with input_args(inputs, start) as in_args:
            raw = run_ffmpeg(["-skip_frame", "nokey", *in_args, "-t", f"{seconds:.3f}", "-an",
                              "-vf", f"scale={SMALL_W}:{SMALL_H},format=gray",
                              "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"], timeout=600)
    except MediaError as exc:
        log.warning("keyframe analysis failed: %s", exc)
        return None
    frames = np.frombuffer(raw, np.uint8)
    n = len(frames) // (SMALL_W * SMALL_H)
    if n < 2:
        return None
    frames = frames[: n * SMALL_W * SMALL_H].reshape(n, SMALL_H, SMALL_W).astype(np.float32)
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2)) / 255.0
    # Keyframes are (close to) evenly spaced in live streams; spread them over the window.
    times = np.linspace(0, seconds, n)[1:]
    curve = np.zeros(int(np.ceil(seconds)))
    idx = np.clip(times.astype(int), 0, len(curve) - 1)
    np.maximum.at(curve, idx, diffs)
    ref = np.median(diffs) + 1e-3
    return np.clip((curve / ref - 1.5) / 6, 0, 1)


def _decode_small(inputs, start, duration, fps):
    with input_args(inputs, start) as in_args:
        raw = run_ffmpeg([*in_args, "-t", f"{duration:.3f}", "-an",
                          "-vf", f"fps={fps},scale={SMALL_W * 2}:{SMALL_H * 2},format=gray",
                          "-f", "rawvideo", "pipe:1"], timeout=300)
    size = SMALL_W * 2 * SMALL_H * 2
    n = len(raw) // size
    return np.frombuffer(raw[: n * size], np.uint8).reshape(n, SMALL_H * 2, SMALL_W * 2).astype(np.float32)


def motion_stats(inputs, start, duration, fps=4):
    """Motion intensity and scene cuts around a moment. None on failure."""
    try:
        frames = _decode_small(inputs, start, duration, fps)
    except MediaError as exc:
        log.warning("motion analysis failed: %s", exc)
        return None
    if len(frames) < 3:
        return None
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2)) / 255.0
    med = float(np.median(diffs)) + 1e-4
    cuts = int((diffs > max(0.12, 6 * med)).sum())
    return {"motion_mean": round(float(diffs.mean()), 4), "motion_peak": round(float(diffs.max()), 4),
            "scene_cuts": cuts, "sudden_change": bool(diffs.max() > 5 * med and diffs.max() > 0.05)}


def extract_frame_jpeg(path, t, width=640):
    """One frame at time t (seconds into `path`) as JPEG bytes."""
    return run_ffmpeg(["-ss", f"{max(0.0, t):.3f}", "-i", str(path), "-frames:v", "1",
                       "-vf", f"scale={width}:-2", "-q:v", "4", "-f", "image2", "-c:v", "mjpeg", "pipe:1"],
                      timeout=60)


def sample_frames(path, fps=1.0, width=640, max_frames=90):
    """Decoded BGR frames (numpy) sampled from a clip, for face detection."""
    probe = subprocess.run([config.FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=60)
    w, h = (int(x) for x in probe.stdout.strip().split(",")[:2])
    height = int(round(h * width / w / 2) * 2)
    raw = run_ffmpeg(["-i", str(path), "-an", "-vf", f"fps={fps},scale={width}:{height}",
                      "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], timeout=300)
    size = width * height * 3
    n = len(raw) // size
    return np.frombuffer(raw[: n * size], np.uint8).reshape(n, height, width, 3), (w, h)


_face_detector = None


def detect_faces(frames, min_score=0.7):
    """
    Faces per frame as normalised (x, y, w, h, score) tuples (0-1 of the frame).
    Returns None if the face model is unavailable.
    """
    global _face_detector
    try:
        import cv2
        from clipping_engine.analysis.model_files import ModelUnavailable, model_path
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass
        if _face_detector is None:
            _face_detector = cv2.FaceDetectorYN.create(str(model_path("face_detection_yunet_2023mar.onnx")),
                                                       "", (320, 320), min_score, 0.3, 5000)
    except (ImportError, ModelUnavailable, Exception) as exc:   # noqa: B014
        log.warning("face detection unavailable: %s", exc)
        return None
    out = []
    for frame in frames:
        h, w = frame.shape[:2]
        _face_detector.setInputSize((w, h))
        _, faces = _face_detector.detect(np.ascontiguousarray(frame))
        out.append([] if faces is None else
                   [(f[0] / w, f[1] / h, f[2] / w, f[3] / h, float(f[-1])) for f in faces])
    return out
