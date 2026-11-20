"""
Audio analysis: reactions, not volume.

Per-second features for an analysis window:
  - YAMNet group scores (laughter, screams/shouts, cheering, gasps, crowd)
  - rare "unusual" sound events (relative to this window)
  - loudness (dB) and silence, for intensity jumps and dramatic pauses

Loudness on its own is weak evidence (games are loud, music is loud): a
loudness jump only adds a small bonus on top of an actual reaction class,
and a loud moment with no reaction caps at a low score.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from clipping_engine import config

RATE = 16_000
FRAME = 1_600            # 0.1 s loudness frames


@dataclass
class AudioFeatures:
    seconds: int
    loudness: np.ndarray                     # dB per second
    rms10: np.ndarray                        # dB per 0.1 s (for boundary snapping)
    silence: np.ndarray                      # share of each second below the silence level
    groups: dict = field(default_factory=dict)   # group -> per-second max score (YAMNet)
    unusual: np.ndarray = None
    unusual_label: list = None
    classifier: bool = False                 # YAMNet ran
    z: np.ndarray = None                     # loudness z-scores (cached)


def _per_second_max(times, values, seconds):
    out = np.zeros(seconds, dtype=np.float32)
    idx = np.clip(times.astype(int), 0, seconds - 1)
    np.maximum.at(out, idx, values)
    return out


def analyze(audio, use_classifier=True):
    """Compute AudioFeatures for 16 kHz mono float32 audio."""
    audio = np.asarray(audio, dtype=np.float32)
    seconds = max(1, int(math.ceil(len(audio) / RATE)))
    n10 = len(audio) // FRAME
    frames = audio[: n10 * FRAME].reshape(n10, FRAME) if n10 else np.zeros((1, FRAME), np.float32)
    power10 = (frames.astype(np.float64) ** 2).mean(axis=1)
    rms10 = 10 * np.log10(power10 + 1e-10)
    pad = seconds * 10 - len(power10)
    p = np.pad(power10, (0, max(0, pad)), constant_values=1e-10)[: seconds * 10].reshape(seconds, 10)
    loudness = 10 * np.log10(p.mean(axis=1) + 1e-10)
    level = np.median(rms10) - 18 if len(rms10) else -60
    silent10 = np.pad(rms10 < max(level, -55), (0, max(0, pad)), constant_values=True)[: seconds * 10]
    silence = silent10.reshape(seconds, 10).mean(axis=1)

    feats = AudioFeatures(seconds, loudness, rms10, silence)
    if use_classifier and config.ENABLE_AUDIO_ANALYSIS:
        from clipping_engine.analysis.laughter_detector import get_model
        model = get_model()
        times, scores = model.scores(audio)
        for group, values in model.grouped(scores).items():
            feats.groups[group] = _per_second_max(times, values, seconds)
        unusual, rare_names = model.unusual(scores)
        feats.unusual = _per_second_max(times, unusual, seconds)
        labels = [""] * seconds
        if rare_names:
            rare_idx = np.array([model.labels.index(n) for n in rare_names])
            top = rare_idx[scores[:, rare_idx].argmax(axis=1)]
            for t, lab, u in zip(times, top, unusual):
                if u >= 0.4:
                    labels[min(seconds - 1, int(t))] = model.labels[lab]
        feats.unusual_label = labels
        feats.classifier = True
    feats.z = _loudness_z(feats.loudness)
    return feats


def _g(f, name):
    return f.groups.get(name, np.zeros(f.seconds, dtype=np.float32))


def _loudness_z(loudness, window=60):
    """Robust z-score of each second's loudness against the surrounding minute."""
    z = np.zeros(len(loudness))
    for t in range(len(loudness)):
        ref = loudness[max(0, t - window):t + window]
        med = np.median(ref)
        mad = np.median(np.abs(ref - med)) * 1.4826 + 1.0
        z[t] = (loudness[t] - med) / mad
    return z


def loudness_z(f):
    if f.z is None:
        f.z = _loudness_z(f.loudness)
    return f.z


def dramatic_pauses(f, min_silence=0.6, gap=3, react_within=3):
    """Seconds where >= gap seconds of near-silence are followed by a reaction."""
    reaction = (_g(f, "laughter") >= 0.6 * config.LAUGHTER_THRESHOLD) \
        | (_g(f, "scream") >= config.SCREAM_THRESHOLD) | (_g(f, "gasp") >= 0.2) | (loudness_z(f) >= 2.5)
    out = np.zeros(f.seconds, dtype=bool)
    for t in range(gap, f.seconds):
        if f.silence[t - gap:t].mean() >= min_silence and f.silence[t] < 0.5 and reaction[t:t + react_within].any():
            out[t] = True
    return out


def interest_curve(f):
    """Per-second audio interest (0-1) for Stage 1 candidate search."""
    laugh = np.minimum(1, _g(f, "laughter") / 0.4)
    scream = np.minimum(1, _g(f, "scream") / 0.5)
    cheer = np.minimum(1, _g(f, "cheer") / 0.5)
    gasp = np.minimum(1, _g(f, "gasp") / 0.4)
    z = loudness_z(f)
    voiced = np.maximum(_g(f, "speech"), _g(f, "scream")) >= 0.3 if f.classifier else np.ones(f.seconds, bool)
    loud = np.clip((z - 1.5) / 3, 0, 1) * 0.35 * voiced
    pause = dramatic_pauses(f) * 0.6
    unusual = (np.minimum(1, f.unusual / 0.7) * 0.45) if f.unusual is not None else 0
    curve = np.maximum.reduce([laugh, 0.9 * scream, 0.7 * cheer, 0.6 * gasp,
                               np.asarray(loud, float), pause.astype(float), np.asarray(unusual, float) * np.ones(f.seconds)])
    return np.convolve(curve, np.ones(3) / 3, mode="same")


def score_moment(f, t0, t1):
    """Audio evidence (0-100) for a moment spanning seconds [t0, t1] of the window."""
    a, b = max(0, int(t0) - 2), min(f.seconds, int(math.ceil(t1)) + 3)
    if b <= a:
        return None
    sl = slice(a, b)
    laugh = _g(f, "laughter")[sl]
    laugh_peak = float(laugh.max()) if f.classifier else 0.0
    laugh_secs = int((laugh >= config.LAUGHTER_THRESHOLD).sum()) if f.classifier else 0
    scream_peak = float(_g(f, "scream")[sl].max()) if f.classifier else 0.0
    cheer_peak = float(_g(f, "cheer")[sl].max()) if f.classifier else 0.0
    gasp_peak = float(_g(f, "gasp")[sl].max()) if f.classifier else 0.0
    crowd = float(_g(f, "crowd")[sl].max()) if f.classifier else 0.0
    unusual_peak = float(f.unusual[sl].max()) if f.unusual is not None else 0.0
    unusual_label = next((l for l in (f.unusual_label or [])[a:b] if l), None) if unusual_peak >= 0.5 else None
    z = loudness_z(f)
    z_peak = float(z[sl].max())
    pause = bool(dramatic_pauses(f)[sl].any())
    before = f.loudness[max(0, a - 15):a]
    shift = float(f.loudness[sl].mean() - before.mean()) if len(before) else 0.0

    laugh_c = min(1.0, laugh_peak / 0.45) * (0.6 + 0.4 * min(1.0, laugh_secs / 4))
    scream_c = min(1.0, scream_peak / 0.5)
    cheer_c = min(1.0, cheer_peak / 0.5)
    gasp_c = min(1.0, gasp_peak / 0.4)
    unusual_c = min(1.0, unusual_peak / 0.7)
    intensity_c = min(1.0, max(0.0, z_peak - 1.5) / 3)
    reactions = [laugh_c >= 0.35, scream_c >= 0.4, cheer_c >= 0.5, gasp_c >= 0.5]
    human = max(laugh_c, 0.85 * scream_c, 0.7 * cheer_c, 0.6 * gasp_c)
    base = max(human, 0.35 * unusual_c)
    bonus = 0.12 * intensity_c + (0.15 if pause else 0) + 0.08 * max(0, sum(reactions) - 1)
    total = base + bonus
    if human < 0.3:
        total = min(total, 0.4)             # loud or odd sounds but no human reaction: capped
    score = 100 * min(1.0, total)

    signals = {
        "laughter_detected": laugh_peak >= config.LAUGHTER_THRESHOLD and laugh_secs >= 1,
        "multiple_laughter": laugh_secs >= 3 and crowd >= 0.2,
        "scream_detected": scream_peak >= config.SCREAM_THRESHOLD,
        "cheering": cheer_peak >= config.CHEER_THRESHOLD,
        "surprised_reaction": gasp_peak >= 0.2 or (pause and scream_peak >= config.SCREAM_THRESHOLD),
        "exciting_audio": scream_peak >= config.SCREAM_THRESHOLD or cheer_peak >= config.CHEER_THRESHOLD
                          or (z_peak >= 2.5 and (laugh_peak >= 0.1 or scream_peak >= 0.1)),
        "loudness_spike": z_peak >= 2.5,
        "dramatic_pause": pause,
        "emotional_shift": abs(shift) >= 6,
        "unusual_sound": unusual_label is not None,
    }
    return {
        "score": round(score, 1),
        "laughter_peak": round(laugh_peak, 3),
        "laughter_seconds": laugh_secs,
        "scream_peak": round(scream_peak, 3),
        "cheer_peak": round(cheer_peak, 3),
        "gasp_peak": round(gasp_peak, 3),
        "loudness_z": round(z_peak, 2),
        "unusual_sound": unusual_label,
        "signals": signals,
    }


def quietest_point(f, t, radius=2.0):
    """Time (s) of the lowest 0.1 s loudness within +-radius of t (natural cut point)."""
    a, b = max(0, int((t - radius) * 10)), min(len(f.rms10), int((t + radius) * 10) + 1)
    if b <= a:
        return t
    return (a + int(np.argmin(f.rms10[a:b]))) / 10.0
