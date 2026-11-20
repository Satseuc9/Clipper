"""
9:16 vertical rendering: 1080x1920, H.264 + AAC, loudness-normalised,
captions burned in.

Layouts (LAYOUT = "auto" picks one from face detection on sampled frames):

  face_crop  A large face fills much of the frame (IRL / just chatting):
             crop a 9:16 window centred on the face.
  stacked    A small face sits in one place (webcam overlay on gameplay):
             webcam crop on top, the FULL gameplay frame below (nothing of
             the game is cut off; GAMEPLAY_FILL=True crops the sides to make
             the game bigger instead), blurred background behind.
  fit_blur   No reliable face: the full frame, scaled to the width, over a
             blurred, zoomed copy of itself. Nothing is cropped away.

If the face moves a lot (a static crop would lose it), face_crop falls back
to fit_blur.
"""

import logging
import os
from pathlib import Path

import numpy as np

from clipping_engine import config
from clipping_engine.analysis.video_analyzer import detect_faces, sample_frames
from clipping_engine.editing.caption_generator import build_ass
from clipping_engine.media import has_audio, run_ffmpeg, video_size

log = logging.getLogger(__name__)

OUT_W, OUT_H = (int(v) for v in config.OUTPUT_RESOLUTION.split("x"))


def _even(v):
    return max(2, int(round(v / 2)) * 2)


def choose_layout(src):
    """(layout name, info) from face detection on ~1 frame per second."""
    if config.LAYOUT != "auto":
        return config.LAYOUT, {"reason": "fixed in config"}
    try:
        frames, _ = sample_frames(src, fps=1.0, width=640, max_frames=60)
    except Exception as exc:
        return "fit_blur", {"reason": f"could not sample frames ({exc})"}
    faces = detect_faces(frames)
    if faces is None:
        return "fit_blur", {"reason": "face detector unavailable"}
    largest = [max(f, key=lambda b: b[2] * b[3]) for f in faces if f]
    share = len(largest) / max(1, len(faces))
    if share < 0.5:
        return "fit_blur", {"reason": f"face in only {share:.0%} of frames", "face_share": share}
    arr = np.array([b[:4] for b in largest])
    cx, cy = np.median(arr[:, 0] + arr[:, 2] / 2), np.median(arr[:, 1] + arr[:, 3] / 2)
    fw, fh = np.median(arr[:, 2]), np.median(arr[:, 3])
    spread = float(np.std(arr[:, 0] + arr[:, 2] / 2))
    info = {"face_share": round(share, 2), "face_center": (round(float(cx), 3), round(float(cy), 3)),
            "face_size": (round(float(fw), 3), round(float(fh), 3)), "face_spread": round(spread, 3)}
    if fw >= 0.12:
        if spread > 0.12:
            info["reason"] = "large face but it moves too much for a fixed crop"
            return "fit_blur", info
        return "face_crop", info
    info["reason"] = "small static face: webcam overlay"
    return "stacked", info


def _filter_graph(layout, info, w, h):
    """Video filter graph from [0:v] to [v] (before captions)."""
    blur_bg = (f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
               f"boxblur=24:4,eq=brightness=-0.08[bg]")
    if layout == "face_crop":
        cx = info["face_center"][0] * w
        cw = min(w, _even(h * OUT_W / OUT_H))
        x = int(min(max(0, cx - cw / 2), w - cw))
        return f"[0:v]crop={cw}:{h}:{x}:0,scale={OUT_W}:{OUT_H},setsar=1[v]", 420
    if layout == "stacked":
        fx, fy = info["face_center"]
        fw, fh = info["face_size"]
        cam_w = min(w, _even(fw * w * 3.2))
        cam_h = min(h, _even(cam_w * 3 / 4))
        cam_x = int(min(max(0, fx * w - cam_w / 2), w - cam_w))
        cam_y = int(min(max(0, fy * h - cam_h / 2), h - cam_h))
        top_h = _even(OUT_W * 3 / 4)                              # 1080x810 webcam panel
        if config.GAMEPLAY_FILL:
            game_h = OUT_H - top_h
            game = f"[0:v]scale=-2:{game_h},crop={OUT_W}:{game_h}[game]"
        else:
            game_h = _even(OUT_W * h / w)
            game = f"[0:v]scale={OUT_W}:{game_h}[game]"
        graph = (f"{blur_bg};[0:v]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},scale={OUT_W}:{top_h}[cam];{game};"
                 f"[bg][cam]overlay=0:0[t];[t][game]overlay=0:{top_h},setsar=1[v]")
        # Captions go in the free band under the gameplay (two lines ~ 200 px tall).
        free = OUT_H - top_h - game_h
        caption_margin = max(60, free - 230) if free >= 300 else 160
        return graph, caption_margin
    fg_h = _even(OUT_W * h / w)
    y = max(0, (OUT_H - fg_h) // 2 - 120)
    graph = f"{blur_bg};[0:v]scale={OUT_W}:{fg_h}[fg];[bg][fg]overlay=0:{y},setsar=1[v]"
    return graph, max(140, OUT_H - y - fg_h - 260)


def render(src, out, words=None, clip_start=0.0, clip_end=None):
    """
    Render `src` (horizontal clip) to `out` as a vertical short.
    `words` are transcript words on the same timeline as clip_start/clip_end.
    Returns {"path", "layout", "layout_info", "captions"}.
    """
    src, out = Path(src), Path(out)
    w, h = video_size(src)
    layout, info = choose_layout(src)
    graph, margin_v = _filter_graph(layout, info, w, h)
    filters = graph
    captions = bool(config.CAPTIONS and words)
    ass_name = out.stem + ".ass"
    if captions:
        ass_text = build_ass(words, clip_start, clip_end if clip_end is not None else clip_start + 1e9,
                             margin_v=margin_v)
        (out.parent / ass_name).write_text(ass_text, encoding="utf-8")
        filters = graph.replace("[v]", "[pre]") + f";[pre]ass={ass_name}[v]"
    audio = has_audio(src)
    args = ["-i", str(src.resolve()), "-filter_complex", filters, "-map", "[v]"]
    if audio:
        args += ["-map", "0:a:0", "-af",
                 f"loudnorm=I={config.TARGET_LOUDNESS_LUFS}:TP=-1.5:LRA=11,aresample=48000",
                 "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
    args += ["-c:v", "libx264", "-preset", config.OUTPUT_PRESET, "-crf", str(config.OUTPUT_CRF),
             "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart", str(out.resolve())]
    try:
        run_ffmpeg(args, timeout=1800, cwd=str(out.parent))   # cwd: ass= gets a plain file name
    finally:
        if captions and not os.getenv("KEEP_CAPTION_FILES"):
            (out.parent / ass_name).unlink(missing_ok=True)
    return {"path": out, "layout": layout, "layout_info": info, "captions": captions}


def discord_preview(src, out, max_bytes):
    """Smaller 720x1280 copy sized to fit under `max_bytes` (Discord upload limit)."""
    from clipping_engine.media import media_timing
    _, duration = media_timing(src)
    budget_bits = max_bytes * 8 * 0.9
    audio_bps = 96_000
    video_bps = int(budget_bits / max(1.0, duration) - audio_bps)
    if video_bps < 250_000:
        return None
    run_ffmpeg(["-i", str(src), "-vf", "scale=720:1280", "-c:v", "libx264", "-preset", "veryfast",
                "-b:v", str(video_bps), "-maxrate", str(video_bps), "-bufsize", str(video_bps * 2),
                "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)], timeout=900)
    return out if Path(out).stat().st_size <= max_bytes else None
