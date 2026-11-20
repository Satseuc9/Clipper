"""
Analysis engine: audio (real YAMNet on real recordings), speech cues,
reaction lexicon, viral scoring, and the Claude evaluator (mocked client).

Audio test data is downloaded on first run into tests/.cache:
  - ESC-50 (Piczak, CC BY-NC 3.0): laughing / clapping / rain / dog clips
  - a LibriSpeech utterance (CC BY 4.0) from the librosa sample data
These files are used locally for testing only and are not committed.
"""

import json
import types
from pathlib import Path

import numpy as np
import pytest
import requests

from clipping_engine import config
from clipping_engine.analysis import ai_evaluator, sentiment_analyzer, speech_transcriber, viral_scorer
from clipping_engine.tests.conftest import needs_ffmpeg

CACHE = Path(__file__).parent / ".cache"
ESC = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
SPEECH = "https://raw.githubusercontent.com/librosa/data/main/audio/5703-47212-0000.ogg"


def fetch(name, url):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except requests.RequestException:
            pytest.skip(f"could not download {name}")
        path.write_bytes(r.content)
    return path


@pytest.fixture(scope="module")
def yamnet():
    config.MODEL_DIR = CACHE / "models"
    from clipping_engine.analysis import laughter_detector, model_files
    try:
        return laughter_detector.get_model()
    except model_files.ModelUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def clips():
    from clipping_engine.media import decode_audio
    names = {"laugh1": "1-36164-B-26.wav", "laugh2": "1-30039-A-26.wav", "clap": "1-104089-A-22.wav",
             "rain": "1-17367-A-10.wav", "dog": "1-100032-A-0.wav"}
    out = {k: decode_audio(fetch(f"esc_{v}", ESC + v)) for k, v in names.items()}
    out["speech"] = decode_audio(fetch("speech.ogg", SPEECH))
    return out


# ---------------------------------------------------------------- audio

@needs_ffmpeg
def test_yamnet_separates_laughter_from_other_sounds(yamnet, clips):
    from clipping_engine.analysis.laughter_detector import GROUPS
    peaks = {}
    for name, audio in clips.items():
        _, scores = yamnet.scores(audio)
        peaks[name] = float(yamnet.grouped(scores)["laughter"].max())
    assert peaks["laugh1"] >= 0.3 and peaks["laugh2"] >= config.LAUGHTER_THRESHOLD, peaks
    for other in ("clap", "rain", "dog", "speech"):
        assert peaks[other] < 0.05, peaks
    assert "Laughter" in GROUPS["laughter"]


def stream_audio(clips, seconds=120):
    rate = 16000
    track = np.tile(clips["speech"], int(np.ceil(seconds * rate / len(clips["speech"]))))[: seconds * rate] * 0.5
    track[40 * rate:40 * rate + len(clips["laugh1"])] += clips["laugh1"] * 1.5          # laughter at 40 s
    rng = np.random.default_rng(0)
    track[80 * rate:83 * rate] += rng.normal(0, 0.5, 3 * rate).astype(np.float32)       # loud noise at 80 s
    return np.clip(track, -1, 1).astype(np.float32)


@needs_ffmpeg
def test_audio_scores_reactions_not_loudness(yamnet, clips):
    from clipping_engine.analysis import audio_analyzer
    f = audio_analyzer.analyze(stream_audio(clips))
    laugh = audio_analyzer.score_moment(f, 40, 45)
    speech = audio_analyzer.score_moment(f, 15, 20)
    noise = audio_analyzer.score_moment(f, 80, 83)
    assert laugh["signals"]["laughter_detected"] and laugh["score"] >= 70, laugh
    assert speech["score"] < 30 and not speech["signals"]["laughter_detected"], speech
    assert noise["signals"]["loudness_spike"] and noise["score"] <= 40, "loud noise alone must stay low"
    curve = audio_analyzer.interest_curve(f)
    assert 38 <= int(np.argmax(curve)) <= 46


def test_dramatic_pause_detection():
    from clipping_engine.analysis import audio_analyzer
    rate = 16000
    rng = np.random.default_rng(1)
    track = (rng.normal(0, 0.05, 60 * rate)).astype(np.float32)
    track[30 * rate:34 * rate] *= 0.001                     # 4 s of silence
    track[34 * rate:36 * rate] = rng.normal(0, 0.6, 2 * rate)  # sudden loud reaction
    f = audio_analyzer.analyze(track, use_classifier=False)
    assert audio_analyzer.dramatic_pauses(f)[33:36].any()
    assert audio_analyzer.score_moment(f, 33, 36)["signals"]["dramatic_pause"]
    assert 29.0 <= audio_analyzer.quietest_point(f, 31.5, 2.0) <= 33.5


# ---------------------------------------------------------------- lexicon & speech

def test_reaction_lexicon():
    c = sentiment_analyzer.classify
    assert c("LMAOOOOO")["laugh"] and c("💀💀")["laugh"] and c("KEKW", ["KEKW"])["laugh"]
    assert c("NO WAY")["surprise"] and c("what?!?!")["surprise"]
    assert c("POGGERS")["hype"] and c("W")["hype"]
    assert c("L ratio")["negative"]
    assert c("someone clip that")["clip_request"]
    assert not any(c("what time does the stream end")[k] for k in ("laugh", "hype", "negative", "clip_request"))


def make_transcript(spec):
    words, segments = [], []
    for start, text in spec:
        t = start
        seg_words = text.split()
        for w in seg_words:
            words.append({"start": t, "end": t + 0.3, "word": w, "p": 0.9})
            t += 0.35
        segments.append({"start": start, "end": t, "text": text})
    return {"words": words, "segments": segments}


def test_speech_scoring_and_natural_boundaries():
    tr = make_transcript([(0.0, "okay so we just go left here."), (5.0, "and then we check the corner."),
                          (10.0, "oh my god no way!"), (12.0, "did you see that? that's insane!"),
                          (20.0, "anyway back to the game.")])
    hot = speech_transcriber.score_speech(tr, 10, 14)
    calm = speech_transcriber.score_speech(tr, 0, 3)
    assert hot["score"] > calm["score"] + 30 and hot["signals"]["memorable_quote"]
    assert speech_transcriber.score_speech({"words": [], "segments": []}, 0, 5) is None   # missing, not zero
    # Natural cuts: a start inside a sentence moves back to its first word, an end to the sentence end.
    assert speech_transcriber.natural_start(tr, 6.2) == pytest.approx(4.8, abs=0.01)
    # "did you see that?" ends at the question mark: 4th word ends at 12.0 + 3*0.35 + 0.3 = 13.35 (+0.3 s tail).
    assert speech_transcriber.natural_end(tr, 12.5) == pytest.approx(13.65, abs=0.01)
    assert speech_transcriber.natural_end(tr, 13.5) == pytest.approx(12.0 + 5 * 0.35 + 0.3 + 0.3, abs=0.01)


def test_transcriber_uses_word_timestamps(monkeypatch):
    """faster-whisper API shape: transcribe() -> (segments generator, info); segment.words with start/end/word."""
    Word = types.SimpleNamespace
    seg = types.SimpleNamespace(start=1.0, end=2.0, text=" no way ",
                                words=[Word(start=1.0, end=1.3, word=" no", probability=0.9),
                                       Word(start=1.4, end=1.9, word=" way", probability=0.8)])
    calls = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            calls.update(kwargs)
            return iter([seg]), types.SimpleNamespace(language="en")

    monkeypatch.setattr(speech_transcriber, "_model", FakeModel())
    monkeypatch.setattr(config, "ENABLE_TRANSCRIPTION", True)
    out = speech_transcriber.transcribe(np.zeros(16000, np.float32), offset=100.0)
    assert calls["word_timestamps"] and calls["vad_filter"]
    assert out["words"][0] == {"start": 101.0, "end": 101.3, "word": "no", "p": 0.9}
    assert out["segments"][0]["text"] == "no way" and out["language"] == "en"


def test_real_faster_whisper_signature():
    import inspect
    fw = pytest.importorskip("faster_whisper")
    params = inspect.signature(fw.WhisperModel.transcribe).parameters
    for name in ("word_timestamps", "vad_filter", "condition_on_previous_text", "language", "beam_size"):
        assert name in params
    assert "download_root" in inspect.signature(fw.WhisperModel.__init__).parameters


# ---------------------------------------------------------------- viral scoring

def test_score_example_from_spec():
    """Laughter + chat +300% + laughing emojis + funny statement -> high score."""
    score, d = viral_scorer.combine({"audio": 92, "chat": 95, "ai": 96, "speech": 85})
    assert score >= 90 and d["corroborated"]


def test_missing_signals_do_not_disqualify():
    full, _ = viral_scorer.combine({"audio": 90, "chat": 88, "ai": None, "speech": None})
    assert full >= 85, "weights are renormalised over available signals"


def test_single_signal_is_not_enough(monkeypatch):
    monkeypatch.setattr(config, "SINGLE_SIGNAL_FACTOR", 0.8)
    solo, d = viral_scorer.combine({"audio": 100, "chat": 10, "ai": None, "speech": None})
    assert not d["corroborated"] and solo < config.MINIMUM_VIRAL_SCORE
    only_audio, _ = viral_scorer.combine({"audio": 90, "chat": None, "ai": None, "speech": None})
    assert only_audio == pytest.approx(72.0)


def test_weights_are_configurable(monkeypatch):
    a, _ = viral_scorer.combine({"audio": 100, "chat": 60, "ai": 60, "speech": 60})
    monkeypatch.setattr(config, "SCORE_WEIGHTS", {"audio": 0.7, "chat": 0.1, "ai": 0.1, "speech": 0.1})
    b, _ = viral_scorer.combine({"audio": 100, "chat": 60, "ai": 60, "speech": 60})
    assert b > a


def test_candidates_dedupe_and_bounds():
    curve = np.zeros(600)
    curve[100:106] = [0.3, 0.6, 0.9, 0.7, 0.4, 0.2]
    curve[110] = 0.8                                  # same moment (within DEDUPE_GAP)
    curve[400:403] = [0.5, 0.95, 0.5]
    cands = viral_scorer.find_candidates(curve, 0, 600)
    peaks = sorted(int(c["peak"]) for c in cands)
    assert peaks == [102, 401]
    assert viral_scorer.find_candidates(curve, 0, 300) and all(c["peak"] < 300 for c in viral_scorer.find_candidates(curve, 0, 300))

    m = lambda s, p, a, b: {"viral_score": s, "peak": p, "clip_start": a, "clip_end": b}
    kept = viral_scorer.dedupe([m(80, 100, 85, 130), m(90, 105, 88, 135), m(70, 300, 280, 320)])
    assert [k["viral_score"] for k in kept] == [90, 70]

    start, end = viral_scorer.clip_bounds(100, 104, 102, 600, pre=15, post=10, lo=30, hi=90)
    assert start == pytest.approx(85 - (30 - 29) * 2 / 3, abs=0.01) and end - start == pytest.approx(30, abs=0.01)
    start, end = viral_scorer.clip_bounds(5, 200, 100, 600, pre=15, post=10, lo=30, hi=90)
    assert end - start == pytest.approx(90) and start <= 100 <= end
    start, end = viral_scorer.clip_bounds(2, 4, 3, 600)
    assert start == 0.0


def test_output_matches_spec_shape():
    moment = {"signals": {"laughter_detected": True, "chat_spike": True}, "chat": {"increase_pct": 300}}
    assert "chat activity up 300%" in viral_scorer.explain(moment)
    assert viral_scorer.fmt_ts(272) == "04:32" and viral_scorer.fmt_ts(3872) == "1:04:32"


# ---------------------------------------------------------------- Claude evaluator (mocked)

class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def fake_client(messages):
    return types.SimpleNamespace(beta=types.SimpleNamespace(messages=messages))


def response(payload, stop="end_turn"):
    return types.SimpleNamespace(stop_reason=stop, model="claude-opus-5-5",
                                 content=[types.SimpleNamespace(type="text", text=json.dumps(payload))])


CANDIDATE = {"streamer": "s", "platform": "twitch", "window_start": 0, "window_end": 55, "peak": 30,
             "event_start": 28, "event_end": 34, "transcript_lines": "30.0: no way he did that",
             "audio": {"laughter_peak": 0.6}, "chat": {"increase_pct": 300}, "video": None}


def test_ai_evaluator_request_and_parsing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(config, "ENABLE_AI_EVALUATION", True)
    msgs = FakeMessages(response({"entertainment_score": 140, "category": "funny", "standalone": True,
                                  "clip_start": 12.0, "clip_end": 50.0, "title": "He really did that",
                                  "reason": "Unexpected funny interaction followed by laughter."}))
    monkeypatch.setattr(ai_evaluator, "_client", fake_client(msgs))
    out = ai_evaluator.evaluate(CANDIDATE, [(30.0, b"\xff\xd8jpeg")])
    assert out["entertainment_score"] == 100 and out["category"] == "funny"       # clamped
    call = msgs.calls[0]
    assert call["model"] == config.CLAUDE_MODEL
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == config.CLAUDE_EFFORT
    assert call["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in call["betas"]
    assert "thinking" not in call and "temperature" not in call
    images = [b for b in call["messages"][0]["content"] if b["type"] == "image"]
    assert images and images[0]["source"]["media_type"] == "image/jpeg"


def test_ai_evaluator_failures_are_missing_signals(monkeypatch):
    import anthropic
    monkeypatch.setattr(config, "ENABLE_AI_EVALUATION", True)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ai_evaluator.AIUnavailable):
        ai_evaluator.evaluate(CANDIDATE, [])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(ai_evaluator, "_client", fake_client(FakeMessages(response({}, stop="refusal"))))
    with pytest.raises(ai_evaluator.AIUnavailable, match="declined"):
        ai_evaluator.evaluate(CANDIDATE, [])
    import httpx2
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    monkeypatch.setattr(ai_evaluator, "_client", fake_client(FakeMessages(error=err)))
    with pytest.raises(ai_evaluator.AIUnavailable, match="reach"):
        ai_evaluator.evaluate(CANDIDATE, [])
