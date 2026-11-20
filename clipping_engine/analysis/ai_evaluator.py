"""
Stage 3: Claude judges whether a candidate moment would work as a short.

Only the few strongest candidates per segment (after cheap audio/chat and
transcription stages) reach this step. Claude receives the timestamped
transcript, the chat reaction summary, the detected audio/video signals and
a handful of still frames, and returns structured JSON:
entertainment score, category, whether it makes sense on its own, a
suggested start/end and a short explanation.

No ANTHROPIC_API_KEY (or ENABLE_AI_EVALUATION = False) -> the AI component
is simply missing and the score is computed from the other signals.
"""

import base64
import json
import logging
import os

from clipping_engine import config

log = logging.getLogger(__name__)

CATEGORIES = ["funny", "fail", "clutch_play", "impressive_skill", "unexpected_event", "argument_drama",
              "emotional", "story", "memorable_quote", "scare_reaction", "wholesome", "other", "not_clipworthy"]

SCHEMA = {
    "type": "object",
    "properties": {
        "entertainment_score": {"type": "integer", "description": "0-100 likelihood this works as a viral short"},
        "category": {"type": "string", "enum": CATEGORIES},
        "standalone": {"type": "boolean", "description": "understandable without watching the rest of the stream"},
        "clip_start": {"type": "number", "description": "suggested clip start, seconds on the given timeline"},
        "clip_end": {"type": "number", "description": "suggested clip end, seconds on the given timeline"},
        "title": {"type": "string", "description": "short hook title, max 60 characters"},
        "reason": {"type": "string", "description": "one or two sentences explaining the score"},
    },
    "required": ["entertainment_score", "category", "standalone", "clip_start", "clip_end", "title", "reason"],
    "additionalProperties": False,
}

SYSTEM = """You review moments from livestreams and decide whether each would make an entertaining, shareable \
short-form clip (TikTok, YouTube Shorts, Reels).

You get: a timeline in seconds, a timestamped transcript of the streamer, what the audio models heard \
(laughter, screams, cheering, gasps, unusual sounds), how chat reacted compared with its normal activity, \
and still frames from the moment.

Judge it like an experienced clip editor:
- High scores need a real payoff: something genuinely funny, surprising, impressive, dramatic, emotional, \
or quotable - and a viewer who never saw the stream must understand it within a few seconds.
- Loud audio, busy gameplay or a fast chat are not entertaining by themselves. Routine gameplay, \
reading donations, waiting screens, ads, or talking with no payoff score low.
- Chat spam and laughter are strong evidence only when they react to something visible or said.
- Missing data (no chat, no transcript) is not a reason to score low; judge what you have.
- Pick clip_start/clip_end on the given timeline so the clip starts with the setup (no mid-sentence start) \
and ends right after the payoff/reaction. Aim for about {target} seconds, between {min_d} and {max_d}.
- Score 0-100: 85+ = would very likely go viral, 70-84 = solid clip, 40-69 = mildly interesting, below 40 = skip.
Write the title and reason in English, without hashtags or emojis."""


class AIUnavailable(RuntimeError):
    pass


_client = None


def available():
    return config.ENABLE_AI_EVALUATION and bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(timeout=120.0, max_retries=3)
    return _client


def build_content(candidate, frames):
    """User message content: context text + JPEG frames."""
    lines = [
        f"Streamer: {candidate['streamer']} on {candidate['platform']}",
        f"Stream title: {candidate.get('stream_title') or 'unknown'}",
        f"Timeline: {candidate['window_start']:.1f}s to {candidate['window_end']:.1f}s; "
        f"the detected moment is around {candidate['peak']:.1f}s "
        f"(activity {candidate['event_start']:.1f}s-{candidate['event_end']:.1f}s).",
        "",
        "Transcript (seconds: words):",
        candidate.get("transcript_lines") or "(no speech detected or transcription unavailable)",
        "",
        "Audio signals: " + json.dumps(candidate.get("audio") or "unavailable"),
        "Chat reaction: " + json.dumps(candidate.get("chat") or "chat data unavailable for this stream"),
        "Video: " + json.dumps(candidate.get("video") or "unavailable"),
    ]
    content = [{"type": "text", "text": "\n".join(lines)}]
    for t, jpeg in frames:
        content.append({"type": "text", "text": f"Frame at {t:.1f}s:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(jpeg).decode()}})
    content.append({"type": "text", "text": "Evaluate this moment."})
    return content


def evaluate(candidate, frames):
    """Return Claude's evaluation dict, or raise AIUnavailable."""
    if not available():
        raise AIUnavailable("no ANTHROPIC_API_KEY (or AI evaluation disabled)")
    import anthropic
    system = SYSTEM.format(target=config.DEFAULT_CLIP_DURATION, min_d=config.MIN_CLIP_DURATION,
                           max_d=config.MAX_CLIP_DURATION)
    try:
        response = get_client().beta.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=16000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": build_content(candidate, frames)}],
            output_config={"effort": config.CLAUDE_EFFORT,
                           "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        raise AIUnavailable("Anthropic API key rejected") from None
    except anthropic.RateLimitError:
        raise AIUnavailable("Anthropic rate limit (after retries)") from None
    except anthropic.APIStatusError as exc:
        raise AIUnavailable(f"Anthropic API error {exc.status_code}") from None
    except anthropic.APIConnectionError:
        raise AIUnavailable("could not reach the Anthropic API") from None

    if response.stop_reason == "refusal":
        raise AIUnavailable("Claude declined to evaluate this moment")
    if response.stop_reason == "max_tokens":
        raise AIUnavailable("Claude response was cut off")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise AIUnavailable("empty response")
    try:
        result = json.loads(text)
    except ValueError:
        raise AIUnavailable("response was not valid JSON") from None
    result["entertainment_score"] = max(0, min(100, int(result.get("entertainment_score", 0))))
    if result.get("category") not in CATEGORIES:
        result["category"] = "other"
    result["title"] = str(result.get("title", ""))[:80]
    result["reason"] = str(result.get("reason", ""))[:600]
    result["model"] = getattr(response, "model", config.CLAUDE_MODEL)
    return result
