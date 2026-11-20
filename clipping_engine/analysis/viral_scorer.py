"""
Viral moment scoring (0-100) and candidate selection.

Components (each 0-100, or None when the signal is unavailable):
    audio  - reactions in the audio (laughter, screams, cheering, gasps, pauses)
    chat   - chat reaction relative to the stream's own baseline
    ai     - Claude's entertainment judgement (transcript + frames + signals)
    speech - spoken cues: exclamations, laughter in speech, rate surges, quotes

final = sum(w_i * s_i) / sum(w_i) over AVAILABLE components only, so a
missing signal (no chat API, no API key) never disqualifies a moment.
Corroboration: when fewer than two components are strong (>= 50), the
score is multiplied by SINGLE_SIGNAL_FACTOR - one signal alone is not enough.
"""


import numpy as np

from clipping_engine import config

STRONG = 50.0


def combine(components, weights=None):
    """(score, details) from {component: score or None}."""
    weights = weights or config.SCORE_WEIGHTS
    avail = {k: float(v) for k, v in components.items() if v is not None and weights.get(k, 0) > 0}
    if not avail:
        return 0.0, {"used": [], "missing": list(components), "corroborated": False}
    total = sum(weights[k] for k in avail)
    score = sum(weights[k] * v for k, v in avail.items()) / total
    strong = [k for k, v in avail.items() if v >= STRONG]
    corroborated = len(strong) >= 2
    if not corroborated:
        score *= config.SINGLE_SIGNAL_FACTOR
    return round(min(100.0, score), 1), {
        "used": sorted(avail), "missing": sorted(k for k in components if k not in avail),
        "strong": sorted(strong), "corroborated": corroborated,
        "effective_weights": {k: round(weights[k] / total, 3) for k in avail},
    }


def stage1_curve(curves):
    """Combine per-second curves {name: array or None} into one interest curve."""
    weights = {"audio": 0.5, "chat": 0.35, "video": 0.15}
    present = {k: v for k, v in curves.items() if v is not None and len(v)}
    if not present:
        return None
    n = max(len(v) for v in present.values())
    total = sum(weights[k] for k in present)
    curve = np.zeros(n)
    for k, v in present.items():
        curve[: len(v)] += weights[k] / total * np.asarray(v, float)
    # A moment where two curves agree beats one curve alone.
    if len(present) > 1:
        agree = np.minimum.reduce([np.pad(np.asarray(v, float), (0, n - len(v))) for v in present.values()])
        curve += 0.3 * agree
    return np.convolve(curve, np.ones(3) / 3, mode="same")


def find_candidates(curve, own_start, own_end, max_candidates=None, gap=None):
    """
    Peaks of the interest curve inside [own_start, own_end) (seconds of the window).
    Returns [{"peak", "event_start", "event_end", "strength"}], strongest first.
    """
    if curve is None or not len(curve):
        return []
    max_candidates = max_candidates or config.MAX_CANDIDATES_PER_SEGMENT
    gap = gap or config.DEDUPE_GAP_SECONDS
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med))) * 1.4826
    # Relative to this segment's own baseline: real streams (game audio, music,
    # speech mixed together) rarely reach high absolute values, and Stage 1 only
    # has to pick which moments get scored, not decide what is clip-worthy.
    threshold = max(config.STAGE1_MIN_INTEREST, med + 2 * mad)
    order = np.argsort(-curve)
    picked = []
    for t in order:
        value = float(curve[t])
        if value < threshold or len(picked) >= max_candidates:
            break
        if not (own_start <= t < own_end):
            continue
        if any(abs(t - p["peak"]) < gap for p in picked):
            continue
        half = value * 0.5
        a = t
        while a > 0 and curve[a - 1] >= half and t - a < 15:
            a -= 1
        b = t
        while b < len(curve) - 1 and curve[b + 1] >= half and b - t < 15:
            b += 1
        picked.append({"peak": float(t), "event_start": float(a), "event_end": float(b + 1),
                       "strength": round(value, 3)})
    return picked


def dedupe(moments, gap=None):
    """Keep the best-scoring moment of every cluster of overlapping/nearby moments."""
    gap = gap or config.DEDUPE_GAP_SECONDS
    kept = []
    for m in sorted(moments, key=lambda m: -m["viral_score"]):
        clash = False
        for k in kept:
            overlap = min(m["clip_end"], k["clip_end"]) - max(m["clip_start"], k["clip_start"])
            shorter = min(m["clip_end"] - m["clip_start"], k["clip_end"] - k["clip_start"])
            if abs(m["peak"] - k["peak"]) < gap or overlap > 0.5 * shorter:
                clash = True
                break
        if not clash:
            kept.append(m)
    return kept


def clip_bounds(event_start, event_end, peak, total, pre=None, post=None, lo=None, hi=None):
    """Buffer-based clip window, clamped to [MIN_CLIP_DURATION, MAX_CLIP_DURATION]."""
    pre = config.PRE_CLIP_BUFFER_SECONDS if pre is None else pre
    post = config.POST_CLIP_BUFFER_SECONDS if post is None else post
    lo = config.MIN_CLIP_DURATION if lo is None else lo
    hi = config.MAX_CLIP_DURATION if hi is None else hi
    start, end = event_start - pre, event_end + post
    if end - start < lo:                         # too short: add context, 2/3 before
        missing = lo - (end - start)
        start -= missing * 2 / 3
        end += missing / 3
    if end - start > hi:                         # too long: keep the peak and its reaction
        start = max(start, peak - hi * 0.6)
        end = start + hi
    shift = max(0.0, -start) - max(0.0, end - total)
    start, end = start + shift, end + shift
    return max(0.0, start), min(total, end)


def fmt_ts(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def explain(moment):
    """Readable reason built from the signals (used when Claude is not available)."""
    s = moment["signals"]
    parts = []
    if s.get("laughter_detected"):
        parts.append("sustained laughter" if s.get("multiple_laughter") else "laughter in the audio")
    if s.get("scream_detected"):
        parts.append("a scream/shout")
    if s.get("dramatic_pause"):
        parts.append("a dramatic pause followed by a reaction")
    if s.get("unusual_sound"):
        parts.append("an unusual sound")
    if s.get("chat_spike"):
        inc = moment.get("chat", {}).get("increase_pct")
        parts.append(f"chat activity up {inc}%" if inc is not None else "a chat spike")
    if s.get("emoji_spike"):
        parts.append("a burst of emotes/emojis")
    if s.get("clip_requests"):
        parts.append("viewers asking to clip it")
    if s.get("memorable_quote"):
        parts.append("a quotable line")
    if not parts:
        return "Several moderate signals peaked together."
    text = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return text[0].upper() + text[1:] + "."
