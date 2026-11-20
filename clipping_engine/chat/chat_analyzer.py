"""
Chat activity analysis, relative to the stream's own baseline.

A busy channel always has fast chat, so absolute message counts mean
nothing. Every metric is compared with the same session's recent normal
activity (median of 10-second bins over the previous BASELINE_SECONDS).

Viewers react a few seconds after something happens (stream latency +
typing), so chat at time t is attributed to video around t - delay
(CHAT_REACTION_DELAY_SECONDS + CHAT_SYNC_OFFSET_SECONDS).
"""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from clipping_engine import config
from clipping_engine.analysis import sentiment_analyzer as lexicon
from clipping_engine.chat.chat_log import coverage_fraction, load_chat

BASELINE_SECONDS = 600
MIN_BASELINE_RATE = 0.05     # messages/second floor, so tiny chats don't explode the ratio
REACTION_WINDOW = 12         # seconds of chat examined after a moment


@dataclass
class ChatData:
    available: bool
    messages: list
    intervals: list
    delay: float
    coverage: float


def load_window(session_dir, wall_start, wall_end):
    """Chat needed to analyse [wall_start, wall_end]: baseline before it, reactions after it."""
    messages, intervals = load_chat(session_dir)
    delay = config.CHAT_REACTION_DELAY_SECONDS + config.CHAT_SYNC_OFFSET_SECONDS
    lo, hi = wall_start - BASELINE_SECONDS, wall_end + delay + REACTION_WINDOW
    msgs = [m for m in messages if lo <= m["ts"] <= hi]
    for m in msgs:
        m.setdefault("_flags", lexicon.classify(m.get("text", ""), m.get("emotes") or ()))
    cov = coverage_fraction(intervals, wall_start, wall_end)
    return ChatData(available=cov >= 0.5, messages=msgs, intervals=intervals, delay=delay, coverage=cov)


def _in(msgs, a, b):
    return [m for m in msgs if a <= m["ts"] < b]


def _baseline(data, before):
    """(messages/sec, reaction share, emoji/sec, unique chatters/sec) before `before`."""
    lo = before - BASELINE_SECONDS
    window = _in(data.messages, lo, before)
    covered = sum(max(0.0, min(before, b) - max(lo, a)) for a, b in data.intervals)
    if covered < 30 or not window:
        return MIN_BASELINE_RATE, 0.1, MIN_BASELINE_RATE, 0.5
    edges = np.arange(lo, before + 10, 10.0)
    counts, _ = np.histogram([m["ts"] for m in window], bins=edges)
    counts = counts[counts > 0] if (counts > 0).sum() >= 3 else counts
    rate = max(MIN_BASELINE_RATE, float(np.median(counts)) / 10.0) if len(counts) else MIN_BASELINE_RATE
    reaction = sum(1 for m in window if _is_reaction(m)) / len(window)
    emoji_rate = max(MIN_BASELINE_RATE, sum(m["_flags"]["emoji_count"] for m in window) / covered)
    # Unique chatters per 10-second bin (unique counts don't scale with window length).
    bins = {}
    for m in window:
        bins.setdefault(int((m["ts"] - lo) // 10), set()).add(m.get("user"))
    users_per_bin = max(0.5, float(np.median([len(u) for u in bins.values()])))
    return rate, reaction, emoji_rate, users_per_bin


def _is_reaction(m):
    f = m["_flags"]
    return f["laugh"] or f["hype"] or f["surprise"]


def activity_curve(data, start, seconds, wall_start=None):
    """
    Per-second chat excitement (0-1) for each second of the analysis window,
    shifted back by the reaction delay. None when chat was not available.

    `start` is the wall-clock time of second 0, or a function mapping wall
    times to window seconds (pass wall_start then) - the recorder's TimeMap,
    which accounts for gaps such as skipped ad breaks.
    """
    if not data.available:
        return None
    n = int(math.ceil(seconds))
    if callable(start):
        to_window = start
    else:
        wall_start = start
        to_window = lambda w: np.asarray(w, float) - wall_start     # noqa: E731
    rate, *_ = _baseline(data, wall_start)
    if not data.messages:
        return np.zeros(n)
    times = np.asarray(to_window(np.array([m["ts"] for m in data.messages]) - data.delay), float)
    weights = np.array([1.5 if _is_reaction(m) else 1.0 for m in data.messages])
    counts = np.zeros(n)
    idx = np.floor(times).astype(int)
    ok = (idx >= 0) & (idx < n)
    np.add.at(counts, idx[ok], weights[ok])
    smooth = np.convolve(counts, np.ones(5) / 5.0, mode="same")    # 5-second window
    ratio = smooth / rate
    return np.clip(np.log2(np.maximum(ratio, 1e-9)) / math.log2(6), 0, 1)


def score_moment(data, abs_start, abs_end):
    """
    Chat evidence for a moment spanning [abs_start, abs_end] (wall clock).
    Returns None when chat data was not available for it.
    """
    a, b = abs_start + data.delay, abs_end + data.delay + REACTION_WINDOW
    if not data.available or coverage_fraction(data.intervals, a, b) < 0.5:
        return None
    rate, base_reaction, base_emoji, base_users = _baseline(data, abs_start)
    window = _in(data.messages, a, b)
    seconds = max(1.0, b - a)
    n = len(window)
    win_rate = n / seconds
    ratio = win_rate / rate

    laugh = sum(1 for m in window if m["_flags"]["laugh"])
    hype = sum(1 for m in window if m["_flags"]["hype"])
    surprise = sum(1 for m in window if m["_flags"]["surprise"])
    negative = sum(1 for m in window if m["_flags"]["negative"])
    clip_requests = sum(1 for m in window if m["_flags"]["clip_request"])
    emoji_rate = sum(m["_flags"]["emoji_count"] for m in window) / seconds
    users = {m.get("user") for m in window}
    reaction_share = (sum(1 for m in window if _is_reaction(m)) / n) if n else 0.0

    normalized = Counter(lexicon.normalize(m.get("text", "")) for m in window if m.get("text"))
    repeated = sum(c for t, c in normalized.items() if c >= 3)
    repeat_share = repeated / n if n else 0.0

    s_spike = min(1.0, max(0.0, math.log2(max(ratio, 1e-9)) / math.log2(6)))
    s_reaction = min(1.0, max(0.0, (reaction_share - base_reaction) / 0.4))
    s_repeat = min(1.0, repeat_share / 0.4)
    users_per_bin = len(users) / (seconds / 10.0)
    s_users = min(1.0, max(0.0, math.log2(max(users_per_bin / base_users, 1e-9)) / math.log2(4)))
    s_emoji = min(1.0, max(0.0, math.log2(max(emoji_rate / base_emoji, 1e-9)) / math.log2(6)))
    score = 100 * (0.35 * s_spike + 0.25 * s_reaction + 0.15 * s_repeat + 0.15 * s_users + 0.10 * s_emoji)
    score += min(15, 5 * clip_requests)
    if n < 3:
        score *= n / 3.0          # too few messages to mean much

    signals = {
        "chat_spike": ratio >= 2.0 and n >= 5,
        "emoji_spike": emoji_rate / base_emoji >= 2.5 and emoji_rate * seconds >= 5,
        "laugh_reaction": laugh >= 3 and laugh / max(n, 1) >= 0.25,
        "surprise_reaction": surprise >= 3 and surprise / max(n, 1) >= 0.25,
        "hype_reaction": hype >= 3 and hype / max(n, 1) >= 0.25,
        "negative_reaction": negative >= 3 and negative / max(n, 1) >= 0.25,
        "repeated_messages": repeat_share >= 0.2 and repeated >= 3,
        "clip_requests": clip_requests > 0,
    }
    return {
        "score": round(min(100.0, score), 1),
        "messages": n,
        "rate_per_sec": round(win_rate, 2),
        "baseline_per_sec": round(rate, 2),
        "increase_pct": round((ratio - 1) * 100),
        "unique_chatters": len(users),
        "laugh_messages": laugh,
        "hype_messages": hype,
        "surprise_messages": surprise,
        "negative_messages": negative,
        "clip_requests": clip_requests,
        "top_messages": [t for t, _ in normalized.most_common(6)],
        "signals": signals,
    }
