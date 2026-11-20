"""
Burned-in captions from word-level timestamps (ASS subtitles via libass).

Static mode: short phrases (<= CAPTION_MAX_WORDS words), broken at
sentence ends and pauses, shown while they are spoken.
Animated mode (ANIMATED_CAPTIONS): the same phrases, with the word being
spoken highlighted and slightly enlarged - the word-by-word style common on
TikTok/Shorts.
"""

import re

from clipping_engine import config

HIGHLIGHT = "&H0000F0FF&"     # ASS BGR: yellow
WHITE = "&H00FFFFFF&"


def _ts(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _clean(word):
    """Remove ASS control characters from a word."""
    return re.sub(r"[{}\\]", "", word).strip()


def group_words(words, max_words=None, max_gap=0.6):
    max_words = max_words or config.CAPTION_MAX_WORDS
    groups, current = [], []
    for w in words:
        if not _clean(w["word"]):
            continue
        if current and (len(current) >= max_words or w["start"] - current[-1]["end"] > max_gap
                        or re.search(r"[.!?…]$", current[-1]["word"])):
            groups.append(current)
            current = []
        current.append(w)
    if current:
        groups.append(current)
    return groups


def build_ass(words, clip_start, clip_end, margin_v=420, animated=None, width=1080, height=1920):
    """ASS document for the words between clip_start and clip_end (times shifted to the clip)."""
    animated = config.ANIMATED_CAPTIONS if animated is None else animated
    rel = [dict(w, start=w["start"] - clip_start, end=w["end"] - clip_start)
           for w in words if w["end"] > clip_start and w["start"] < clip_end]
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{config.CAPTION_FONT},{config.CAPTION_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,6,2,2,70,70,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    groups = group_words(rel)
    for gi, group in enumerate(groups):
        g_start = max(0.0, group[0]["start"])
        nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else clip_end - clip_start
        g_end = min(max(group[-1]["end"] + 0.3, g_start + 0.4), nxt)
        tokens = [_clean(w["word"]) for w in group]
        if not animated:
            events.append(f"Dialogue: 0,{_ts(g_start)},{_ts(g_end)},Caption,,0,0,0,,{' '.join(tokens)}")
            continue
        for i, w in enumerate(group):
            a = max(g_start, w["start"])
            b = group[i + 1]["start"] if i + 1 < len(group) else g_end
            if b <= a:
                continue
            parts = []
            for j, tok in enumerate(tokens):
                if j == i:
                    parts.append(r"{\c" + HIGHLIGHT + r"\fscx112\fscy112}" + tok + r"{\c" + WHITE + r"\fscx100\fscy100}")
                else:
                    parts.append(tok)
            events.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Caption,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(events) + "\n"
