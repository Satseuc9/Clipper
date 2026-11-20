"""
Twitch chat through the official EventSub WebSocket transport.

Subscription: channel.chat.message (version 1)
    condition: {"broadcaster_user_id": <streamer>, "user_id": <your reading account>}
    transport: {"method": "websocket", "session_id": <from session_welcome>}

WebSocket subscriptions need a USER access token with the user:read:chat
scope (app tokens are only accepted for webhook transports). The token is
obtained once with the OAuth device code flow:

    python -m clipping_engine auth twitch

and refreshed automatically afterwards (stored in .tokens/twitch_user.json).

One WebSocket is shared by every live Twitch streamer; subscriptions are
added when a recording starts and deleted when it stops. On disconnect the
client reconnects and re-subscribes; `session_reconnect` messages are
followed to the new URL without losing subscriptions. Chat coverage gaps
are written to chat.jsonl so analysis knows when chat was missing.
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import requests
import websockets

from clipping_engine import config
from clipping_engine.chat.chat_log import parse_timestamp

log = logging.getLogger(__name__)

DEVICE_URL = "https://id.twitch.tv/oauth2/device"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
SUBSCRIPTIONS_URL = "https://api.twitch.tv/helix/eventsub/subscriptions"
WS_URL = "wss://eventsub.wss.twitch.tv/ws"
SCOPES = "user:read:chat"
TOKEN_FILE = config.ENGINE_DIR / ".tokens" / "twitch_user.json"
TIMEOUT = 15
SUBSCRIBE_RETRY_SECONDS = 60


class TwitchAuthError(RuntimeError):
    pass


def chat_client_credentials():
    """Client ID/secret for chat: TWITCH_CHAT_* if set, else the monitor's app."""
    chat_id = (os.getenv("TWITCH_CHAT_CLIENT_ID") or "").strip()
    if chat_id:                                   # a separate (usually public) app for chat
        return chat_id, (os.getenv("TWITCH_CHAT_CLIENT_SECRET") or "").strip()
    return (os.getenv("TWITCH_CLIENT_ID") or "").strip(), (os.getenv("TWITCH_CLIENT_SECRET") or "").strip()


class TwitchUserToken:
    """User access + refresh token for the account that reads chat."""

    def __init__(self, client_id, client_secret="", token_file=TOKEN_FILE, session=None,
                 token_url=TOKEN_URL, validate_url=VALIDATE_URL, device_url=DEVICE_URL):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_file = Path(token_file)
        self.http = session or requests.Session()
        self.token_url, self.validate_url, self.device_url = token_url, validate_url, device_url
        self.data = {}
        self.user_id = None
        self.login = None
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            self.data = json.loads(self.token_file.read_text())
        except (OSError, ValueError):
            self.data = {}

    def _save(self):
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.token_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.token_file)

    @property
    def available(self):
        return bool(self.client_id and self.data.get("refresh_token"))

    def _store(self, payload):
        self.data = {
            "access_token": payload["access_token"],
            "refresh_token": payload.get("refresh_token", self.data.get("refresh_token")),
            "expires_at": time.time() + float(payload.get("expires_in") or 3600),
            "scope": payload.get("scope"),
        }
        self._save()

    # ---- one-time login -----------------------------------------------------------------

    def device_login(self, show=print, poll_limit=1800):
        """OAuth device code flow: the user opens a URL and confirms a code."""
        resp = self.http.post(self.device_url, data={"client_id": self.client_id, "scopes": SCOPES},
                              timeout=TIMEOUT)
        if not resp.ok:
            raise TwitchAuthError(f"device code request failed: HTTP {resp.status_code} {resp.text[:200]}")
        dev = resp.json()
        show(f"Open {dev['verification_uri']} and enter the code {dev['user_code']} "
             f"(log in with the account that should read chat).")
        interval = int(dev.get("interval", 5))
        deadline = time.time() + min(poll_limit, int(dev.get("expires_in", 1800)))
        while time.time() < deadline:
            time.sleep(interval)
            r = self.http.post(self.token_url, data={
                "client_id": self.client_id, "scopes": SCOPES, "device_code": dev["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }, timeout=TIMEOUT)
            if r.ok:
                self._store(r.json())
                info = self.validate()
                show(f"Twitch chat authorized for {info.get('login')}.")
                return info
            message = (r.json() if r.content else {}).get("message", "")
            if message == "authorization_pending":
                continue
            if message == "slow_down":
                interval += 5
                continue
            raise TwitchAuthError(f"device authorization failed: {message or r.status_code}")
        raise TwitchAuthError("device authorization timed out")

    # ---- runtime ---------------------------------------------------------------------------

    def refresh(self):
        if not self.data.get("refresh_token"):
            raise TwitchAuthError("no Twitch chat token; run: python -m clipping_engine auth twitch")
        body = {"client_id": self.client_id, "grant_type": "refresh_token",
                "refresh_token": self.data["refresh_token"]}
        if self.client_secret:
            body["client_secret"] = self.client_secret
        r = self.http.post(self.token_url, data=body, timeout=TIMEOUT)
        if not r.ok:
            raise TwitchAuthError(f"token refresh failed (HTTP {r.status_code}); "
                                  "run: python -m clipping_engine auth twitch")
        self._store(r.json())

    def access_token(self):
        with self._lock:
            if not self.data.get("access_token") or time.time() > self.data.get("expires_at", 0) - 300:
                self.refresh()
            return self.data["access_token"]

    def validate(self):
        """Check the token and learn which account it belongs to."""
        for attempt in range(2):
            r = self.http.get(self.validate_url, headers={"Authorization": f"OAuth {self.access_token()}"},
                              timeout=TIMEOUT)
            if r.status_code == 401 and attempt == 0:
                self.refresh()
                continue
            if not r.ok:
                raise TwitchAuthError(f"token validation failed (HTTP {r.status_code})")
            info = r.json()
            if SCOPES not in (info.get("scopes") or []):
                raise TwitchAuthError("token lacks user:read:chat; run: python -m clipping_engine auth twitch")
            self.user_id, self.login = info.get("user_id"), info.get("login")
            return info
        raise TwitchAuthError("token validation failed")


class TwitchChat:
    """Shared EventSub WebSocket client for all live Twitch streamers."""

    def __init__(self, token, client_id, ws_url=WS_URL, subscriptions_url=SUBSCRIPTIONS_URL, session=None):
        self.token = token
        self.client_id = client_id
        self.ws_url = ws_url
        self.subscriptions_url = subscriptions_url
        self.http = session or requests.Session()
        self.desired = {}          # broadcaster_user_id -> ChatLog
        self.subscribed = {}       # broadcaster_user_id -> subscription id
        self._failed = {}          # broadcaster_user_id -> monotonic time of last failed attempt
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self._stop = threading.Event()
        self._seen = OrderedDict()  # recent message ids (dedupe across reconnects)
        self.status = "idle"
        self._thread = None

    # ---- called from other threads ----------------------------------------------------------

    def add(self, broadcaster_id, chat_log):
        with self._lock:
            self.desired[str(broadcaster_id)] = chat_log
        self._changed.set()

    def remove(self, broadcaster_id):
        with self._lock:
            chat_log = self.desired.pop(str(broadcaster_id), None)
        if chat_log:
            chat_log.coverage(False, "recording stopped")
        self._changed.set()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=lambda: asyncio.run(self._main()),
                                            name="twitch-chat", daemon=True)
            self._thread.start()

    def stop(self, timeout=10):
        self._stop.set()
        self._changed.set()
        if self._thread:
            self._thread.join(timeout)

    # ---- REST (blocking; run in an executor) -----------------------------------------------------

    def _headers(self):
        return {"Client-Id": self.client_id, "Authorization": f"Bearer {self.token.access_token()}",
                "Content-Type": "application/json"}

    def _subscribe(self, broadcaster_id, session_id):
        body = {"type": "channel.chat.message", "version": "1",
                "condition": {"broadcaster_user_id": broadcaster_id, "user_id": self.token.user_id},
                "transport": {"method": "websocket", "session_id": session_id}}
        for attempt in range(2):
            r = self.http.post(self.subscriptions_url, headers=self._headers(), data=json.dumps(body),
                               timeout=TIMEOUT)
            if r.status_code == 401 and attempt == 0:
                self.token.refresh()
                continue
            if r.status_code == 202:
                return r.json()["data"][0]["id"], None
            detail = (r.json() if r.content else {}).get("message", "")
            return None, f"subscription failed (HTTP {r.status_code}{': ' + detail if detail else ''})"
        return None, "subscription failed (unauthorized)"

    def _unsubscribe(self, sub_id):
        try:
            self.http.delete(self.subscriptions_url, headers=self._headers(), params={"id": sub_id},
                             timeout=TIMEOUT)
        except requests.RequestException:
            pass   # the subscription dies with the socket anyway

    # ---- asyncio side ---------------------------------------------------------------------------

    async def _main(self):
        loop = asyncio.get_running_loop()
        backoff = 1
        while not self._stop.is_set():
            with self._lock:
                wanted = bool(self.desired)
            if not wanted:
                self.status = "idle"
                await loop.run_in_executor(None, self._changed.wait, 5)
                self._changed.clear()
                continue
            try:
                await loop.run_in_executor(None, self.token.validate)
                await self._session(self.ws_url)
                backoff = 1
            except Exception as exc:  # network, auth, protocol: mark gap, retry
                self.status = f"error: {type(exc).__name__}: {str(exc)[:120]}"
                log.warning("twitch chat: %s", self.status)
                self._mark_all(False, "chat connection lost")
                self.subscribed.clear()
                await asyncio.sleep(min(backoff, 60))
                backoff *= 2

    def _mark_all(self, available, reason=None):
        with self._lock:
            logs = list(self.desired.values())
        for chat_log in logs:
            chat_log.coverage(available, reason)

    async def _session(self, url):
        loop = asyncio.get_running_loop()
        ws = await websockets.connect(url, open_timeout=TIMEOUT, ping_interval=None, max_size=2 ** 20)
        try:
            session_id, keepalive = await self._welcome(ws)
            self.subscribed.clear()
            self._failed.clear()
            self.status = "connected"
            last_msg = time.monotonic()
            while not self._stop.is_set():
                await self._sync_subscriptions(loop, session_id)
                with self._lock:
                    if not self.desired:
                        return      # nothing to watch: close the socket
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    if time.monotonic() - last_msg > keepalive + 5:
                        raise ConnectionError("keepalive timeout")
                    continue
                last_msg = time.monotonic()
                msg = json.loads(raw)
                mtype = msg.get("metadata", {}).get("message_type")
                if mtype == "notification":
                    self._on_notification(msg)
                elif mtype == "session_reconnect":
                    new_url = msg["payload"]["session"]["reconnect_url"]
                    new_ws = await websockets.connect(new_url, open_timeout=TIMEOUT, ping_interval=None)
                    session_id, keepalive = await self._welcome(new_ws)   # subscriptions carry over
                    await ws.close()
                    ws = new_ws
                elif mtype == "revocation":
                    sub = msg["payload"]["subscription"]
                    bid = sub.get("condition", {}).get("broadcaster_user_id")
                    self.subscribed.pop(bid, None)
                    with self._lock:
                        chat_log = self.desired.get(bid)
                    if chat_log:
                        chat_log.coverage(False, f"subscription revoked ({sub.get('status')})")
        finally:
            self._mark_all(False, "chat connection closed")
            await ws.close()

    async def _welcome(self, ws):
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=TIMEOUT))
        if msg.get("metadata", {}).get("message_type") != "session_welcome":
            raise ConnectionError("expected session_welcome")
        session = msg["payload"]["session"]
        return session["id"], float(session.get("keepalive_timeout_seconds") or 10)

    async def _sync_subscriptions(self, loop, session_id):
        now = time.monotonic()
        with self._lock:
            desired = dict(self.desired)
        missing = [b for b in desired if b not in self.subscribed
                   and now - self._failed.get(b, -1e9) >= SUBSCRIBE_RETRY_SECONDS]
        extra = [b for b in self.subscribed if b not in desired]
        if not missing and not extra:
            return
        for bid in extra:
            await loop.run_in_executor(None, self._unsubscribe, self.subscribed.pop(bid))
        for bid in missing:
            sub_id, error = await loop.run_in_executor(None, self._subscribe, bid, session_id)
            if sub_id:
                self.subscribed[bid] = sub_id
                self._failed.pop(bid, None)
                desired[bid].coverage(True)
            else:
                self._failed[bid] = time.monotonic()
                desired[bid].coverage(False, error)
                log.warning("twitch chat for %s: %s (retrying in %ss)", bid, error, SUBSCRIBE_RETRY_SECONDS)

    def _on_notification(self, msg):
        if msg["metadata"].get("subscription_type") != "channel.chat.message":
            return
        ev = msg["payload"]["event"]
        mid = ev.get("message_id")
        if mid in self._seen:
            return
        self._seen[mid] = None
        while len(self._seen) > 5000:
            self._seen.popitem(last=False)
        with self._lock:
            chat_log = self.desired.get(str(ev.get("broadcaster_user_id")))
        if not chat_log:
            return
        message = ev.get("message") or {}
        emotes = [f.get("text") for f in message.get("fragments") or [] if f.get("type") == "emote"]
        ts = parse_timestamp(msg["metadata"].get("message_timestamp"), time.time())
        chat_log.message(ts, ev.get("chatter_user_login"), message.get("text", ""), emotes, mid)
