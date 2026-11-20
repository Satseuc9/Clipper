"""Discord notifications against a local webhook server (multipart uploads, 429, size limits)."""

import subprocess

import pytest


from clipping_engine import config
from clipping_engine.notifications import discord_notifier
from clipping_engine.tests.conftest import needs_ffmpeg


def make_clip(path, seconds=3, crf="30"):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=size=1080x1920:rate=30:duration={seconds}", "-f", "lavfi", "-i", "sine=f=440",
                    "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-crf", crf, "-c:a", "aac",
                    str(path)], check=True)
    return path


def clip_dict(path):
    return {"path": str(path), "platform": "twitch", "streamer": "examplestreamer",
            "streamer_display": "ExampleStreamer", "viral_score": 94, "duration": 46,
            "stream_timestamp": "04:32 – 05:18 (stream time)", "segment": "segment_0004",
            "start_timestamp": "04:32", "end_timestamp": "05:18", "title": "He did NOT just do that",
            "reason": "Unexpected funny interaction followed by intense laughter and a big chat spike. @everyone",
            "source_url": "https://twitch.tv/examplestreamer",
            "signals": {"laughter_detected": True, "chat_spike": True, "emoji_spike": True, "exciting_audio": False}}


@needs_ffmpeg
def test_clip_is_uploaded_with_details(discord, tmp_path):
    clip = make_clip(tmp_path / "clip.mp4")
    result = discord_notifier.notify(clip_dict(clip))
    assert result == {"sent": True, "attached": "clip", "error": None}
    req = discord.requests[0]
    assert req["content_type"].startswith("multipart/form-data")
    assert req["files"]["files[0]"] == ("clip.mp4", clip.stat().st_size)
    embed = req["payload"]["embeds"][0]
    assert embed["title"] == "🔥 POTENTIAL VIRAL CLIP DETECTED"
    for text in ("ExampleStreamer", "Twitch", "94/100", "46 seconds", "04:32 – 05:18", "Why it was selected",
                 "https://twitch.tv/examplestreamer", "Unexpected funny interaction"):
        assert text in embed["description"]
    assert req["payload"]["allowed_mentions"] == {"parse": []}, "AI text must not ping anyone"
    assert "laughter detected" in embed["fields"][0]["value"]
    assert "wait=true" in req["path"]


@needs_ffmpeg
def test_rate_limit_is_retried(discord, tmp_path):
    discord.script = [429, 200]
    result = discord_notifier.notify(clip_dict(make_clip(tmp_path / "c.mp4")))
    assert result["sent"] and len(discord.requests) == 2


@needs_ffmpeg
def test_too_large_sends_preview_and_link(discord, tmp_path, monkeypatch):
    clip = make_clip(tmp_path / "big.mp4", seconds=6, crf="4")
    monkeypatch.setattr(config, "DISCORD_MAX_UPLOAD_MB", 1)
    monkeypatch.setattr(config, "CLIPS_DIR", tmp_path)
    monkeypatch.setattr(config, "CLIP_PUBLIC_BASE_URL", "https://files.example.com/clips")
    assert clip.stat().st_size > 1024 * 1024
    result = discord_notifier.notify(clip_dict(clip))
    assert result == {"sent": True, "attached": "preview", "error": None}
    req = discord.requests[0]
    name, size = req["files"]["files[0]"]
    assert name == "big_preview.mp4" and size <= 1024 * 1024
    assert "https://files.example.com/clips/big.mp4" in req["payload"]["embeds"][0]["description"]


@needs_ffmpeg
def test_too_large_without_preview_sends_link_only(discord, tmp_path, monkeypatch):
    clip = make_clip(tmp_path / "big.mp4", seconds=4, crf="4")
    monkeypatch.setattr(config, "DISCORD_MAX_UPLOAD_MB", 0.2)
    monkeypatch.setattr(config, "DISCORD_SEND_PREVIEW_IF_TOO_LARGE", False)
    monkeypatch.setattr(config, "CLIP_PUBLIC_BASE_URL", None)
    result = discord_notifier.notify(clip_dict(clip))
    assert result == {"sent": True, "attached": None, "error": None}
    req = discord.requests[0]
    assert not req["files"], "never attempt an oversized upload"
    assert str(clip) in req["payload"]["embeds"][0]["description"]


@needs_ffmpeg
def test_failures_are_reported_without_leaking_the_webhook(discord, tmp_path, monkeypatch):
    monkeypatch.setattr(discord_notifier.time, "sleep", lambda s: None)
    discord.script = [500, 500, 500, 500]
    result = discord_notifier.notify(clip_dict(make_clip(tmp_path / "c.mp4")))
    assert not result["sent"] and "500" in result["error"]
    assert "SECRET-TOKEN" not in result["error"]


def test_not_configured(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert not discord_notifier.enabled()
    assert discord_notifier.notify({"path": "x"})["sent"] is False


@pytest.mark.parametrize("url,problem", [
    ("", "not set"),
    ("https://discord.com/channels/1234567890/9876543210", "channel link"),
    ("https://discord.com/api/webhooks/123", "doesn't look like"),
    ("https://discord.com/api/webhooks/1234567890/AbC-def_123", None),
    ("https://discordapp.com/api/webhooks/1234567890/AbC-def_123", None),
    ('"https://discord.com/api/webhooks/1234567890/AbC"', None),        # quoted in .env
])
def test_webhook_url_problems(monkeypatch, url, problem):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", url)
    found = discord_notifier.url_problem()
    if problem is None:
        assert found is None
    else:
        assert problem in found and "1234567890" not in found, "never echo the URL"


def test_send_test_message(discord):
    ok, message = discord_notifier.send_test()
    assert ok and discord.requests[0]["payload"]["content"].startswith("✅ Clipper test message")
    discord.script = [404]
    ok, message = discord_notifier.send_test()
    assert not ok and "deleted" in message and "SECRET-TOKEN" not in message


@needs_ffmpeg
def test_send_pending_posts_clips_made_before_discord_was_set_up(discord, engine, tmp_path, capsys):
    from clipping_engine import main
    from clipping_engine.storage import clip_storage, recording_storage as storage
    session = storage.new_session_dir("twitch", "xqc", {"platform": "twitch", "streamer": "xqc"})
    clip = make_clip(tmp_path / "clip.mp4")
    storage.write_json(clip.with_suffix(".json"), clip_dict(clip))
    clip_storage.register_clip(session, {"id": "c1", "abs_start": 0, "abs_end": 40, "abs_peak": 20,
                                         "score": 90, "path": str(clip), "notified": False})
    assert main.run_discord(send_pending=False) == 0
    assert "Clips made but not yet posted: 1" in capsys.readouterr().out
    assert main.run_discord(send_pending=True) == 0
    assert "Posted 1 of 1" in capsys.readouterr().out
    assert clip_storage.load_registry(session)[0]["notified"] is True
    assert discord.requests[-1]["files"]["files[0]"][0] == "clip.mp4"
    assert main.unsent_clips() == []
