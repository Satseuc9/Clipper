"""
Speech-to-text with word timestamps (faster-whisper) and speech-based cues.

Only candidate windows are transcribed (Stage 2), not whole segments, so a
10-minute segment typically costs a few minutes of audio, not ten.

Model weights are downloaded by faster-whisper from HuggingFace into
MODEL_DIR/whisper on first use; set WHISPER_MODEL to a local folder to use
pre-downloaded CTranslate2 weights instead.

Speaker identification: not done (it needs a separate diarization model,
e.g. pyannote, with its own licence and a HuggingFace token). Captions and
context keep Whisper's sentence segmentation, which follows turn changes
reasonably well.
"""

import logging
import re

import numpy as np

from clipping_engine import config
from clipping_engine.analysis import sentiment_analyzer as lexicon

log = logging.getLogger(__name__)
_model = None


class TranscriptionUnavailable(RuntimeError):
    pass


def get_model():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise TranscriptionUnavailable("faster-whisper is not installed") from None
        try:
            _model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                                  compute_type=config.WHISPER_COMPUTE_TYPE,
                                  download_root=str(config.MODEL_DIR / "whisper"))
        except Exception as exc:
            raise TranscriptionUnavailable(f"could not load Whisper model '{config.WHISPER_MODEL}': "
                                           f"{type(exc).__name__}: {str(exc)[:200]}") from None
    return _model


def transcribe(audio, offset=0.0):
    """
    Transcribe 16 kHz mono audio. Times are returned relative to the
    analysis window (offset added). Raises TranscriptionUnavailable.
    """
    if not config.ENABLE_TRANSCRIPTION:
        raise TranscriptionUnavailable("transcription disabled in config")
    model = get_model()
    segments, info = model.transcribe(
        np.asarray(audio, dtype=np.float32), language=config.WHISPER_LANGUAGE, word_timestamps=True,
        vad_filter=True, condition_on_previous_text=False, beam_size=5,
    )
    words, sentences = [], []
    for seg in segments:
        sentences.append({"start": seg.start + offset, "end": seg.end + offset, "text": seg.text.strip()})
        for w in seg.words or []:
            words.append({"start": w.start + offset, "end": w.end + offset,
                          "word": w.word.strip(), "p": round(float(w.probability), 3)})
    return {"language": getattr(info, "language", None), "words": words, "segments": sentences}


def merge(transcripts):
    """Merge transcripts of several windows (sorted, overlaps de-duplicated)."""
    words, sentences = [], []
    for t in transcripts:
        words += t["words"]
        sentences += t["segments"]
    words.sort(key=lambda w: w["start"])
    sentences.sort(key=lambda s: s["start"])
    dedup = []
    for w in words:
        if dedup and abs(w["start"] - dedup[-1]["start"]) < 0.05 and w["word"] == dedup[-1]["word"]:
            continue
        dedup.append(w)
    return {"words": dedup, "segments": sentences}


def words_between(transcript, t0, t1):
    return [w for w in (transcript or {}).get("words", []) if w["end"] > t0 and w["start"] < t1]


def text_between(transcript, t0, t1):
    return " ".join(w["word"] for w in words_between(transcript, t0, t1)).strip()


def score_speech(transcript, t0, t1):
    """
    Speech/context evidence (0-100) for [t0, t1]: spoken exclamations,
    laughter in speech, a jump in speaking rate and short quotable lines.
    Returns None when nothing was said (the signal is missing, not zero).
    """
    words = words_between(transcript, t0 - 3, t1 + 3)
    if len(words) < 3:
        return None
    text = " ".join(w["word"] for w in words)
    cues = lexicon.speech_features(text)
    all_words = (transcript or {}).get("words", [])
    span = max(1.0, all_words[-1]["end"] - all_words[0]["start"]) if all_words else 1.0
    base_rate = len(all_words) / span
    rate = len(words) / max(1.0, (t1 + 3) - (t0 - 3))
    surge = rate / max(base_rate, 0.5)

    # Short, punchy sentences that contain a cue make good quotes.
    quote = None
    for s in (transcript or {}).get("segments", []):
        if s["end"] > t0 - 3 and s["start"] < t1 + 3:
            n = len(s["text"].split())
            if 3 <= n <= 14 and (lexicon.SPEECH_EXCLAMATIONS.search(s["text"]) or s["text"].endswith("!")):
                quote = s["text"]
                break

    c_excl = min(1.0, cues["exclamations"] / 2)
    c_laugh = min(1.0, cues["laughs"] / 2)
    c_rate = min(1.0, max(0.0, surge - 1.0) / 1.0)
    c_quote = 1.0 if quote else 0.0
    c_emph = min(1.0, (cues["emphasis"] + cues["questions"]) / 4)
    score = 100 * min(1.0, 0.35 * c_excl + 0.2 * c_laugh + 0.2 * c_rate + 0.15 * c_quote + 0.1 * c_emph)
    return {
        "score": round(score, 1),
        "text": text[:600],
        "quote": quote,
        "exclamations": cues["exclamations"],
        "speech_laughs": cues["laughs"],
        "speech_rate_ratio": round(surge, 2),
        "signals": {"speech_highlight": score >= 50, "memorable_quote": quote is not None,
                    "excited_speech": c_rate >= 0.5 or c_excl >= 0.5},
    }


_SENTENCE_END = re.compile(r"[.!?…]$")


def natural_start(transcript, t, limit=4.0):
    """Move a cut start back to the beginning of the sentence/phrase around t."""
    words = (transcript or {}).get("words", [])
    best = t
    for i, w in enumerate(words):
        if w["start"] > t:
            break
        if w["end"] < t - limit:
            continue
        prev = words[i - 1] if i else None
        boundary = prev is None or _SENTENCE_END.search(prev["word"]) or w["start"] - prev["end"] >= 0.6
        if boundary and t - limit <= w["start"] <= t:
            best = w["start"] - 0.2
    # Never cut in the middle of a word that spans t.
    for w in words:
        if w["start"] < best < w["end"]:
            best = w["start"] - 0.2
    return max(0.0, best)


def natural_end(transcript, t, limit=4.0):
    """Move a cut end forward to the end of the sentence/phrase around t."""
    words = (transcript or {}).get("words", [])
    for i, w in enumerate(words):
        if w["end"] < t:
            continue
        if w["start"] > t + limit:
            break
        nxt = words[i + 1] if i + 1 < len(words) else None
        if _SENTENCE_END.search(w["word"]) or nxt is None or nxt["start"] - w["end"] >= 0.6:
            return w["end"] + 0.3
    for w in words:
        if w["start"] < t < w["end"]:
            return w["end"] + 0.3
    return t
