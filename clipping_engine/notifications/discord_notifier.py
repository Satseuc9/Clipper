"""
Discord webhook notifications for finished clips.

POST <DISCORD_WEBHOOK_URL>?wait=true as multipart/form-data:
    payload_json = {"embeds": [...], "allowed_mentions": {"parse": []}}
    files[0]     = the clip (only if it fits DISCORD_MAX_UPLOAD_MB)

Clips over the limit are never uploaded: a smaller preview is attached
instead (DISCORD_SEND_PREVIEW_IF_TOO_LARGE) and the message links to the
full clip (CLIP_PUBLIC_BASE_URL) or names its local path.
429 responses are retried after Discord's retry_after. The webhook URL
(which contains a secret token) is never logged or printed.
"""

import json
import logging
import os
import time
from pathlib import Path

import requests

from clipping_engine import config

log = logging.getLogger(__name__)

PLATFORM_NAMES = {"twitch": "Twitch", "kick": "Kick", "youtube": "YouTube"}
COLOR = 0xFF4500


class DiscordError(RuntimeError):
    pass


def webhook_url():
    return os.getenv("DISCORD_WEBHOOK_URL", "").strip()


def enabled():
    return config.DISCORD_NOTIFICATIONS and bool(webhook_url())


def build_payload(clip, link=None, attachment_note=None):
    """Embed for a clip dict (see processing.render_moment)."""
    platform = PLATFORM_NAMES.get(clip["platform"], clip["platform"])
    lines = [
        f"**Streamer:** {clip['streamer_display']}",
        f"**Platform:** {platform}",
        f"**Viral Score:** {round(clip['viral_score'])}/100",
        "",
        f"**Duration:** {round(clip['duration'])} seconds",
        f"**Timestamp:** {clip['stream_timestamp']}",
        "",
        "**Why it was selected:**",
        clip["reason"],
    ]
    if clip.get("title"):
        lines.insert(0, f"*{clip['title']}*\n")
    if clip.get("source_url"):
        lines.append(f"\n**Livestream:** {clip['source_url']}")
    if link:
        lines.append(f"**Full clip:** {link}")
    if attachment_note:
        lines.append(f"\n{attachment_note}")
    fields = [{"name": "Signals", "value": ", ".join(k.replace("_", " ") for k, v in clip["signals"].items() if v)[:1000]
               or "-", "inline": False}]
    return {
        "allowed_mentions": {"parse": []},
        "embeds": [{
            "title": "🔥 POTENTIAL VIRAL CLIP DETECTED",
            "description": "\n".join(lines)[:4000],
            "color": COLOR,
            "fields": fields,
            "footer": {"text": f"{clip['segment']} {clip['start_timestamp']}–{clip['end_timestamp']}"},
        }],
    }


def _post(url, payload, file_path=None, session=None, attempts=4):
    http = session or requests
    for attempt in range(attempts):
        try:
            if file_path:
                with open(file_path, "rb") as fh:
                    resp = http.post(url, params={"wait": "true"}, timeout=config.REQUEST_TIMEOUT_SECONDS * 8,
                                     data={"payload_json": json.dumps(payload)},
                                     files={"files[0]": (Path(file_path).name, fh, "video/mp4")})
            else:
                resp = http.post(url, params={"wait": "true"}, json=payload, timeout=config.REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            if attempt == attempts - 1:
                raise DiscordError(f"network error ({type(exc).__name__})") from None
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429:
            try:
                wait = float(resp.json().get("retry_after", 1))
            except ValueError:
                wait = float(resp.headers.get("Retry-After", 1))
            time.sleep(min(60.0, wait) + 0.1)
            continue
        if resp.status_code == 413:
            return "too_large"
        if resp.status_code >= 500 and attempt < attempts - 1:
            time.sleep(2 ** attempt)
            continue
        if not resp.ok:
            raise DiscordError(f"Discord returned HTTP {resp.status_code}")
        return "ok"
    raise DiscordError("Discord rate limit: gave up after retries")


def notify(clip, session=None):
    """
    Send one clip. Returns {"sent": bool, "attached": "clip"|"preview"|None, "error": str|None}.
    """
    url = webhook_url()
    if not (config.DISCORD_NOTIFICATIONS and url):
        return {"sent": False, "attached": None, "error": "Discord notifications not configured"}
    from clipping_engine.storage.clip_storage import public_url
    from clipping_engine.editing.vertical_formatter import discord_preview

    path = Path(clip["path"])
    limit = int(config.DISCORD_MAX_UPLOAD_MB * 1024 * 1024) - 64 * 1024   # room for the form fields
    link = public_url(path) or None
    try:
        if path.stat().st_size <= limit:
            result = _post(url, build_payload(clip, link), path, session)
            if result == "ok":
                return {"sent": True, "attached": "clip", "error": None}
        note_target = link or f"`{path}` (on the recording machine)"
        preview = None
        if config.DISCORD_SEND_PREVIEW_IF_TOO_LARGE:
            preview = discord_preview(path, path.with_name(path.stem + "_preview.mp4"), limit)
        if preview:
            payload = build_payload(clip, link, f"Attached: compressed preview. Full quality: {note_target}")
            if _post(url, payload, preview, session) == "ok":
                return {"sent": True, "attached": "preview", "error": None}
        payload = build_payload(clip, link, f"Clip is larger than Discord's upload limit. Full clip: {note_target}")
        _post(url, payload, None, session)
        return {"sent": True, "attached": None, "error": None}
    except DiscordError as exc:
        log.warning("discord notification failed: %s", exc)
        return {"sent": False, "attached": None, "error": str(exc)}
