"""
Reaction lexicon for chat messages and transcripts.

This is not generic positive/negative sentiment: it recognises the specific
reactions that make a moment clip-worthy - laughter, hype, surprise,
negativity/controversy and explicit "clip it" requests - in streaming slang,
emotes and emojis. It is one input among several; no moment is selected on
text alone.
"""

import re
import unicodedata

LAUGH_TOKENS = {
    "lol", "lmao", "lmfao", "rofl", "kekw", "kek", "kekl", "omegalul", "lul", "lulw", "lule",
    "icant", "pepelaugh", "xd", "xdd", "haha", "hahaha", "ahaha", "jaja", "jajaja", "kkk",
    "dead", "dying", "crying", "pmsl", "lolol", "lmaooo",
}
LAUGH_EMOJI = {"😂", "🤣", "💀", "😭", "😹", "☠", "☠️"}

HYPE_TOKENS = {
    "pog", "poggers", "pogchamp", "pogu", "poggies", "w", "ww", "www", "letsgo", "lets go",
    "goated", "insane", "clutch", "huge", "gg", "ez", "hype", "fire", "cracked", "sheesh",
    "peepoclap", "clap", "ez clap",
}
HYPE_EMOJI = {"🔥", "🐐", "👏", "💯", "🙌", "🎉", "⚡"}

SURPRISE_TOKENS = {
    "what", "wtf", "omg", "noway", "no way", "holy", "wait", "huh", "monkas", "omegalul?",
    "bruh", "bro", "nah", "wow", "yo", "omfg", "wut", "wha", "pause",
}
SURPRISE_EMOJI = {"😮", "😱", "😳", "🤯", "😲", "👀", "⁉", "‼"}

NEGATIVE_TOKENS = {
    "l", "ll", "lll", "ratio", "cringe", "yikes", "boo", "booo", "trash", "toxic", "cancelled",
    "monkaw", "weirdchamp", "residentsleeper", "ff", "throw", "throwing", "sus", "drama", "beef",
}
NEGATIVE_EMOJI = {"😬", "🤢", "👎", "😡", "🤮", "😤"}

CLIP_PATTERNS = re.compile(r"\b(clip\s*(it|that|this)|clipped|clipper|someone\s+clip)\b", re.I)

# Spoken reactions in transcripts.
SPEECH_EXCLAMATIONS = re.compile(
    r"\b(oh my god|oh my gosh|no way|what the|are you (kidding|serious)|let'?s go|holy (shit|crap|moly)|"
    r"bro what|dude|i can'?t believe|what just happened|did you see that|oh no|let me go|"
    r"wait wait|stop it|shut up|that'?s crazy|that'?s insane|i'?m dead)\b", re.I)
SPEECH_LAUGH = re.compile(r"\b(ha(ha)+|he(he)+|lol|lmao)\b", re.I)

_WORD = re.compile(r"[\w']+", re.UNICODE)


def _is_emoji(ch):
    return unicodedata.category(ch) == "So" or ch in LAUGH_EMOJI | HYPE_EMOJI | SURPRISE_EMOJI | NEGATIVE_EMOJI


def normalize(text):
    """Lowercase, collapse elongations (LMAOOOO -> lmaoo), squash spaces."""
    text = re.sub(r"(.)\1{2,}", r"\1\1", text.lower())
    return " ".join(text.split())


def emojis(text):
    return [ch for ch in text if _is_emoji(ch)]


def classify(text, emotes=()):
    """Reaction flags for one chat message."""
    norm = normalize(text)
    words = set(_WORD.findall(norm)) | {e.lower() for e in emotes}
    joined = norm.replace(" ", "")
    ems = set(emojis(text))
    # Collapse repeated letters fully for token matching (lmaoo -> lmao, kekww -> kekw).
    squashed = {re.sub(r"(.)\1+", r"\1", w) for w in words}
    tokens = words | squashed
    return {
        "laugh": bool(tokens & LAUGH_TOKENS or ems & LAUGH_EMOJI or "haha" in joined or "lmao" in joined),
        "hype": bool(tokens & HYPE_TOKENS or ems & HYPE_EMOJI or "letsgo" in joined),
        "surprise": bool(tokens & SURPRISE_TOKENS or ems & SURPRISE_EMOJI or "noway" in joined
                         or norm.count("?") >= 2 or norm.count("!") >= 2),
        "negative": bool(tokens & NEGATIVE_TOKENS or ems & NEGATIVE_EMOJI),
        "clip_request": bool(CLIP_PATTERNS.search(norm)),
        "emoji_count": len(emojis(text)) + len(emotes),
        "caps": len(text) >= 4 and text.upper() == text and any(c.isalpha() for c in text),
    }


def speech_features(words_text):
    """Reaction cues in a transcript snippet."""
    exclaim = len(SPEECH_EXCLAMATIONS.findall(words_text))
    laughs = len(SPEECH_LAUGH.findall(words_text))
    return {"exclamations": exclaim, "laughs": laughs, "questions": words_text.count("?"),
            "emphasis": words_text.count("!")}
