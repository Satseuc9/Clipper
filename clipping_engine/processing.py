"""
Per-segment processing (runs in a worker process).

    Stage 1  whole segment, cheap   audio events + loudness, chat activity,
                                    keyframe changes -> candidate moments
    Stage 2  candidates only        transcription, speech cues, motion
    Stage 3  best few candidates    Claude evaluates context + frames
    Stage 4  selected moments       extract, captions, 9:16 render, Discord

Each segment is analysed with the last ANALYSIS_CONTEXT_SECONDS of the
previous segment, and owns the moments from where the previous segment
stopped looking up to SEGMENT_TAIL_GUARD_SECONDS before its own end (the
rest belongs to the next segment, which has the footage that follows).
Results are written to the segment sidecar before rendering, so a retry
after a crash resumes instead of re-analysing or re-sending clips.
"""

import logging
import time
from pathlib import Path

from clipping_engine import config
from clipping_engine.analysis import audio_analyzer, viral_scorer
from clipping_engine.analysis import speech_transcriber as stt
from clipping_engine.analysis import video_analyzer
from clipping_engine.capture.segment_manager import previous_segment, run_segments, time_map
from clipping_engine.chat import chat_analyzer
from clipping_engine.chat.chat_log import parse_timestamp
from clipping_engine.media import decode_audio
from clipping_engine.storage import clip_storage
from clipping_engine.storage.recording_storage import read_json, segment_file, sidecar_for, update_json

log = logging.getLogger(__name__)
RATE = 16_000


class Window:
    """Analysis window: [optional tail of previous segment] + this segment."""

    def __init__(self, session_dir, side):
        self.session_dir = Path(session_dir)
        current = next((s for s in run_segments(session_dir, side["run"]) if s["index"] == side["index"]), side)
        self.side = side = dict(side, cum_start=current.get("cum_start", 0.0))
        self.seg_path = segment_file(session_dir, side["index"])
        self.tm = time_map(session_dir, side["run"])
        prev = previous_segment(session_dir, side)
        prev_path = segment_file(session_dir, prev["index"]) if prev else None
        if prev and prev_path.exists() and config.ANALYSIS_CONTEXT_SECONDS > 0:
            self.prev, self.prev_path = prev, prev_path
            self.context = min(config.ANALYSIS_CONTEXT_SECONDS, prev["duration"])
        else:
            self.prev, self.prev_path, self.context = None, None, 0.0
        self.seconds = self.context + side["duration"]
        self.media0 = side["cum_start"] - self.context        # run media time of window second 0
        self.start_wall = self.to_abs(0.0)
        self.wall_start = self.to_abs(self.context)
        self.wall_end = self.to_abs(self.seconds)

    @property
    def inputs(self):
        return [self.prev_path, self.seg_path] if self.prev else [self.seg_path]

    @property
    def input_start(self):
        return self.prev["duration"] - self.context if self.prev else 0.0

    def owned(self):
        guard = config.SEGMENT_TAIL_GUARD_SECONDS
        start = max(0.0, self.context - guard) if self.prev else 0.0
        end = self.seconds if self.side.get("final_in_run") else max(start, self.seconds - guard)
        return start, end

    def locate(self, t):
        """Window time -> (file, seconds into that file)."""
        if self.prev and t < self.context:
            return self.prev_path, self.input_start + t
        return self.seg_path, t - self.context

    def to_media(self, t):
        return self.media0 + t

    def to_abs(self, t):
        """Window seconds -> wall clock (through the run's TimeMap)."""
        if self.tm is None:
            return time.time() - self.seconds + t
        return self.tm.wall(self.media0 + t)

    def to_window(self, wall):
        """Wall clock (scalar or array) -> window seconds."""
        return self.tm.media(wall) - self.media0

    def seg_relative(self, t):
        return t - self.context

    def segment_label(self, start, end):
        """(segment name, start, end) relative to the segment the clip starts in."""
        if self.prev and start < self.context:
            return self.prev_path.stem, self.input_start + start, self.input_start + end
        return self.seg_path.stem, start - self.context, end - self.context


# ---------------------------------------------------------------- stages

def stage1(win, audio):
    feats, notes = None, []
    if config.ENABLE_AUDIO_ANALYSIS:
        try:
            feats = audio_analyzer.analyze(audio, use_classifier=True)
        except Exception as exc:                     # model download / runtime problem
            notes.append(f"audio classifier unavailable: {type(exc).__name__}: {exc}")
            feats = audio_analyzer.analyze(audio, use_classifier=False)
    chat = chat_analyzer.load_window(win.session_dir, win.start_wall, win.wall_end) \
        if config.ENABLE_CHAT_ANALYSIS else None
    curves = {
        "audio": audio_analyzer.interest_curve(feats) if feats is not None else None,
        "chat": chat_analyzer.activity_curve(chat, win.to_window, win.seconds, wall_start=win.start_wall)
                if chat and win.tm is not None else None,
        "video": video_analyzer.keyframe_change_curve(win.inputs, win.seconds, win.input_start),
    }
    if chat is not None and not chat.available:
        from clipping_engine.chat.chat_log import unavailable_reason
        notes.append(f"no chat data: {unavailable_reason(win.session_dir) or 'chat was not collected'}")
    curve = viral_scorer.stage1_curve(curves)
    own_start, own_end = win.owned()
    candidates = viral_scorer.find_candidates(curve, own_start, own_end)
    if curve is not None and own_end > own_start:
        notes.append(f"stage 1 peak interest {float(curve[int(own_start):int(own_end)].max()):.2f}")
    for c in candidates:
        c["audio"] = audio_analyzer.score_moment(feats, c["event_start"], c["event_end"]) if feats else None
        c["chat"] = chat_analyzer.score_moment(chat, win.to_abs(c["event_start"]), win.to_abs(c["event_end"])) \
            if chat is not None else None
        c["prescore"], _ = viral_scorer.combine({"audio": _s(c["audio"]), "chat": _s(c["chat"])})
    return feats, chat, candidates, notes, {k: v is not None for k, v in curves.items()}


def _s(component):
    return None if component is None else component["score"]


def stage2(win, audio, candidates, transcripts, notes):
    for c in candidates:
        if c["prescore"] < config.STAGE2_MIN_PRESCORE:
            continue
        a, b = max(0.0, c["peak"] - 35), min(win.seconds, c["peak"] + 20)
        transcript = transcribe_range(audio, a, b, transcripts, notes)
        c["speech"] = stt.score_speech(transcript, c["event_start"], c["event_end"]) if transcript else None
        c["video"] = video_analyzer.motion_stats(win.inputs, win.input_start + max(0, c["event_start"] - 2),
                                                 c["event_end"] - c["event_start"] + 4)
        c["score_before_ai"], _ = viral_scorer.combine(
            {"audio": _s(c["audio"]), "chat": _s(c["chat"]), "speech": _s(c.get("speech"))})


def transcribe_range(audio, a, b, transcripts, notes):
    """Transcribe [a, b] of the window, reusing earlier transcriptions. Returns merged transcript."""
    if transcripts.get("unavailable"):
        return None
    covered = transcripts.setdefault("ranges", [])
    todo = [(a, b)]
    for ca, cb in covered:
        todo = [piece for x, y in todo for piece in ((x, min(y, ca)), (max(x, cb), y)) if piece[1] - piece[0] > 1.0]
    for x, y in todo:
        try:
            t = stt.transcribe(audio[int(x * RATE):int(y * RATE)], offset=x)
        except stt.TranscriptionUnavailable as exc:
            transcripts["unavailable"] = True
            notes.append(f"transcription unavailable: {exc}")
            return None
        transcripts.setdefault("parts", []).append(t)
        covered.append((x, y))
    return stt.merge(transcripts.get("parts", []))


def stage3(win, candidates, transcripts, meta, notes):
    from clipping_engine.analysis import ai_evaluator
    if not ai_evaluator.available():
        notes.append("AI evaluation skipped (no ANTHROPIC_API_KEY or disabled)")
        return
    ranked = sorted((c for c in candidates if c.get("score_before_ai", 0) >= config.AI_MIN_PRESCORE),
                    key=lambda c: -c["score_before_ai"])[: config.AI_MAX_CANDIDATES_PER_SEGMENT]
    transcript = stt.merge(transcripts.get("parts", [])) if transcripts.get("parts") else None
    for c in ranked:
        a, b = max(0.0, c["peak"] - 35), min(win.seconds, c["peak"] + 20)
        lines = "\n".join(f"{s['start']:.1f}: {s['text']}" for s in (transcript or {}).get("segments", [])
                          if s["end"] > a and s["start"] < b)
        frames = []
        n = max(1, config.AI_FRAMES_PER_CANDIDATE)
        for i in range(n):
            t = c["event_start"] - 3 + (c["event_end"] - c["event_start"] + 6) * i / max(1, n - 1)
            t = min(max(0.0, t), win.seconds - 0.5)
            path, local = win.locate(t)
            try:
                frames.append((t, video_analyzer.extract_frame_jpeg(path, local)))
            except Exception:
                pass
        payload = {
            "streamer": meta.get("display_name") or meta.get("streamer"), "platform": meta.get("platform"),
            "stream_title": meta.get("title"), "window_start": a, "window_end": b, "peak": c["peak"],
            "event_start": c["event_start"], "event_end": c["event_end"], "transcript_lines": lines,
            "audio": {k: v for k, v in (c["audio"] or {}).items() if k != "score"} or None,
            "chat": {k: v for k, v in (c["chat"] or {}).items() if k != "score"} or None,
            "video": c.get("video"),
        }
        try:
            c["ai"] = ai_evaluator.evaluate(payload, frames)
        except ai_evaluator.AIUnavailable as exc:
            notes.append(f"AI evaluation failed: {exc}")
            break


# ---------------------------------------------------------------- moments & clips

def finalize(win, candidates, transcripts, feats, meta):
    transcript = stt.merge(transcripts.get("parts", [])) if transcripts.get("parts") else None
    moments = []
    for c in candidates:
        ai = c.get("ai")
        components = {"audio": _s(c["audio"]), "chat": _s(c["chat"]),
                      "ai": ai["entertainment_score"] if ai else None, "speech": _s(c.get("speech"))}
        score, details = viral_scorer.combine(components)
        start, end = clip_window(win, c, ai, transcript, feats)
        signals = {}
        for part in (c["audio"], c["chat"], c.get("speech")):
            signals.update((part or {}).get("signals", {}))
        if c.get("video"):
            signals["visual_change"] = bool(c["video"].get("sudden_change"))
        signals["ai_selected"] = bool(ai and ai["entertainment_score"] >= 70)
        seg_name, rel_start, rel_end = win.segment_label(start, end)
        moment = {
            "streamer": meta.get("streamer"), "platform": meta.get("platform"),
            "segment": seg_name,
            "start_timestamp": viral_scorer.fmt_ts(rel_start),
            "end_timestamp": viral_scorer.fmt_ts(rel_end),
            "crosses_segments": bool(win.prev and start < win.context < end),
            "viral_score": round(score),
            "reason": (ai or {}).get("reason") or viral_scorer.explain({"signals": signals, "chat": c["chat"]}),
            "signals": {k: bool(v) for k, v in signals.items()},
            "components": components, "scoring": details,
            "category": (ai or {}).get("category"), "title": (ai or {}).get("title"),
            "peak": c["peak"], "clip_start": start, "clip_end": end,
            "run": win.side["run"], "media_start": win.to_media(start), "media_end": win.to_media(end),
            "abs_start": win.to_abs(start), "abs_end": win.to_abs(end), "abs_peak": win.to_abs(c["peak"]),
            "audio": c["audio"], "chat": c["chat"], "speech": c.get("speech"), "video": c.get("video"),
        }
        moments.append(moment)
    moments.sort(key=lambda m: -m["viral_score"])
    return moments


def clip_window(win, c, ai, transcript, feats):
    """Clip start/end (window seconds): AI suggestion if sane, else buffers + natural boundaries."""
    lo, hi = config.MIN_CLIP_DURATION, config.MAX_CLIP_DURATION
    if ai:
        a, b = float(ai.get("clip_start", -1)), float(ai.get("clip_end", -1))
        if 0 <= a < c["peak"] < b <= win.seconds and lo * 0.8 <= b - a <= hi:
            return _natural(a, b, transcript, feats, win)
    a, b = viral_scorer.clip_bounds(c["event_start"], c["event_end"], c["peak"], win.seconds)
    return _natural(a, b, transcript, feats, win)


def _natural(a, b, transcript, feats, win):
    if transcript and transcript.get("words"):
        a2, b2 = stt.natural_start(transcript, a), stt.natural_end(transcript, b)
    elif feats is not None:
        a2, b2 = audio_analyzer.quietest_point(feats, a), audio_analyzer.quietest_point(feats, b)
    else:
        a2, b2 = a, b
    a2, b2 = max(0.0, a2), min(win.seconds, b2)
    if b2 - a2 > config.MAX_CLIP_DURATION + 4:      # natural snapping must not blow up the length
        a2, b2 = a, b
    return round(a2, 2), round(b2, 2)


def stream_timestamp(meta, abs_start, abs_end):
    started = parse_timestamp((meta.get("extra") or {}).get("started_at")) or meta.get("created")
    label = "stream" if (meta.get("extra") or {}).get("started_at") else "recording"
    if not started:
        return "", None
    return (f"{viral_scorer.fmt_ts(abs_start - started)} – {viral_scorer.fmt_ts(abs_end - started)} "
            f"({label} time)"), abs_start - started


def source_url(meta, offset):
    url = meta.get("url", "")
    extra = meta.get("extra") or {}
    if meta.get("platform") == "youtube" and extra.get("video_id") and offset is not None:
        return f"https://www.youtube.com/watch?v={extra['video_id']}&t={max(0, int(offset))}s"
    return url


def render_moment(win, moment, audio, transcripts, meta, notes):
    """Extract + caption + 9:16 render one moment. Returns the clip dict."""
    from clipping_engine.editing import clip_extractor, vertical_formatter
    transcript = transcribe_range(audio, moment["clip_start"], moment["clip_end"], transcripts, notes) \
        if not transcripts.get("unavailable") else None
    words = [dict(w, start=w["start"] - moment["clip_start"], end=w["end"] - moment["clip_start"])
             for w in stt.words_between(transcript, moment["clip_start"], moment["clip_end"])]
    final = clip_storage.clip_path(moment["platform"], moment["streamer"], win.session_dir.name,
                                   win.side["index"], moment["viral_score"], moment["abs_start"])
    raw = final.with_name(final.stem + "_source.mp4")
    clip_extractor.extract(win.session_dir, moment["run"], moment["media_start"], moment["media_end"], raw)
    try:
        result = vertical_formatter.render(raw, final, words=words, clip_start=0.0,
                                           clip_end=moment["clip_end"] - moment["clip_start"])
    finally:
        raw.unlink(missing_ok=True)
    stamp, offset = stream_timestamp(meta, moment["abs_start"], moment["abs_end"])
    clip = dict(moment, path=str(final), layout=result["layout"], captions=result["captions"],
                duration=round(moment["clip_end"] - moment["clip_start"], 1),
                streamer_display=meta.get("display_name") or meta.get("streamer"),
                stream_timestamp=stamp or f"{moment['segment']} {moment['start_timestamp']}–{moment['end_timestamp']}",
                source_url=source_url(meta, offset), created=time.time())
    from clipping_engine.storage.recording_storage import write_json
    write_json(final.with_suffix(".json"), {k: v for k, v in clip.items()})
    return clip


def process_segment(session_dir, index, snapshot=None):
    """Worker entry point. Returns a short summary dict."""
    if snapshot:
        apply_snapshot(snapshot)
    session_dir = Path(session_dir)
    side_path = sidecar_for(segment_file(session_dir, index))
    side = read_json(side_path)
    meta = read_json(session_dir / "session.json", {}) or {}
    win = Window(session_dir, side)
    notes = []
    transcripts = {}
    audio = decode_audio(win.inputs, start=win.input_start, duration=win.seconds)

    if side.get("moments") is None:
        feats, chat, candidates, n1, curves = stage1(win, audio)
        notes += n1
        stage2(win, audio, candidates, transcripts, notes)
        stage3(win, candidates, transcripts, meta, notes)
        moments = finalize(win, candidates, transcripts, feats, meta)
        moments = viral_scorer.dedupe(moments)
        update_json(side_path, moments=moments, analysis_notes=notes, signals_available=curves,
                    chat_coverage=round(chat.coverage, 2) if chat else None, analyzed=time.time())
    else:
        moments = side["moments"]                              # resume after a crash
        curves = side.get("signals_available") or {}
        notes = list(side.get("analysis_notes") or [])

    selected = [m for m in moments if m["viral_score"] >= config.MINIMUM_VIRAL_SCORE][: config.MAX_CLIPS_PER_SEGMENT]
    clips, sent = [], 0
    for m in selected:
        clip_id = f"{session_dir.name}-{index:04d}-{int(m['abs_peak'])}"
        existing = next((e for e in clip_storage.load_registry(session_dir) if e.get("id") == clip_id), None)
        if existing is None:
            if clip_storage.overlaps_existing(session_dir, m["abs_start"], m["abs_end"], m["abs_peak"]):
                continue                                           # already clipped from another segment
            clip = render_moment(win, m, audio, transcripts, meta, notes)
            existing = {"id": clip_id, "abs_start": m["abs_start"], "abs_end": m["abs_end"],
                        "abs_peak": m["abs_peak"], "score": m["viral_score"], "path": clip["path"], "notified": False}
            clip_storage.register_clip(session_dir, existing)
        clips.append(existing["path"])
        if not existing.get("notified"):
            from clipping_engine.notifications import discord_notifier
            if discord_notifier.enabled():
                clip = read_json(Path(existing["path"]).with_suffix(".json"))
                result = discord_notifier.notify(clip)
                if result["sent"]:
                    clip_storage.update_clip(session_dir, clip_id, notified=True, attached=result["attached"])
                    sent += 1
                else:
                    notes.append(f"Discord: {result['error']}")
                    raise RuntimeError(f"Discord notification failed: {result['error']}")
    if notes:
        update_json(side_path, analysis_notes=notes)
    best = moments[0]["viral_score"] if moments else None
    return {"candidates": len(moments), "best_score": best, "clips": clips, "sent": sent, "notes": notes,
            "signals": curves, "ai": any(m["components"].get("ai") is not None for m in moments),
            "speech": any(m["components"].get("speech") is not None for m in moments)}


def apply_snapshot(snapshot):
    for key, value in snapshot.items():
        setattr(config, key, value)


def config_snapshot():
    return {k: getattr(config, k) for k in dir(config) if k.isupper()}
