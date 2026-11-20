"""
Chat collection and analysis.

Twitch: a local WebSocket server speaking the EventSub WebSocket protocol
(session_welcome / notification / session_keepalive / session_reconnect)
plus a local Helix subscriptions endpoint. YouTube: liveChatMessages.list
responses in the documented format. Kick: webhooks signed with a real RSA
key, verified exactly as Kick documents.
"""

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from websockets.asyncio.server import serve

from clipping_engine import config
from clipping_engine.chat import chat_analyzer
from clipping_engine.chat.chat_log import ChatLog, coverage_fraction, load_chat, parse_timestamp
from clipping_engine.chat.kick_chat import KickChat, parse_content
from clipping_engine.chat.twitch_chat import TwitchChat, TwitchUserToken
from clipping_engine.chat.youtube_chat import QuotaBudget, YouTubeChatPoller
from clipping_engine.tests.conftest import free_port, wait_for


# ---------------------------------------------------------------- helpers

class FakeResponse:
    def __init__(self, status, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body or {})
        self.content = self.text.encode()
        self.ok = 200 <= status < 300

    def json(self):
        return self._body if self._body is not None else json.loads(self.text)

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_timestamp_parsing_handles_nanoseconds():
    assert parse_timestamp("2023-07-19T14:56:51.634234626Z") == pytest.approx(1689778611.634234, abs=1e-5)
    assert parse_timestamp("2025-01-14T16:08:06Z") == 1736870886.0
    assert parse_timestamp("garbage", 5) == 5


def test_chat_log_coverage(tmp_path):
    log = ChatLog(tmp_path, "twitch")
    log.coverage(True)
    log.message(100.0, "a", "hi")
    log.coverage(False, "disconnected")
    log.coverage(True)
    msgs, intervals = load_chat(tmp_path)
    assert len(msgs) == 1 and len(intervals) == 2
    assert intervals[1][1] == float("inf")


# ---------------------------------------------------------------- Twitch EventSub

class FakeTwitchToken:
    user_id = "999"
    login = "reader"
    available = True

    def access_token(self):
        return "user-token"

    def validate(self):
        return {"user_id": "999", "scopes": ["user:read:chat"]}

    def refresh(self):
        pass


def notification(broadcaster, msg_id, text, fragments=None, ts="2026-11-20T01:30:15.123456789Z"):
    return {
        "metadata": {"message_id": "m-" + msg_id, "message_type": "notification", "message_timestamp": ts,
                     "subscription_type": "channel.chat.message", "subscription_version": "1"},
        "payload": {
            "subscription": {"id": "sub-1", "type": "channel.chat.message", "version": "1", "status": "enabled",
                             "condition": {"broadcaster_user_id": broadcaster, "user_id": "999"}},
            "event": {
                "broadcaster_user_id": broadcaster, "broadcaster_user_login": "streamer",
                "chatter_user_id": "4145994", "chatter_user_login": "viewer32", "chatter_user_name": "viewer32",
                "message_id": msg_id,
                "message": {"text": text, "fragments": fragments or [{"type": "text", "text": text}]},
                "message_type": "text",
            },
        },
    }


class MockEventSub:
    """EventSub WebSocket + Helix subscription endpoint on localhost."""

    def __init__(self):
        self.ws_port, self.http_port = free_port(), free_port()
        self.posts, self.deletes = [], []
        self.subscribed = threading.Event()
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        threading.Thread(target=self._run_ws, daemon=True).start()
        self._ready.wait(10)
        outer = self

        class Helix(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.posts.append((dict(self.headers), body))
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "sub-1", "status": "enabled"}]}).encode())
                outer.subscribed.set()

            def do_DELETE(self):
                outer.deletes.append(self.path)
                self.send_response(204)
                self.end_headers()

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.http_port), Helix)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def _run_ws(self):
        asyncio.set_event_loop(self.loop)

        async def handler(ws):
            path = ws.request.path
            sid = "S2" if path.startswith("/reconnect") else "S1"
            await ws.send(json.dumps({
                "metadata": {"message_id": "w", "message_type": "session_welcome",
                             "message_timestamp": "2026-11-20T01:30:00Z"},
                "payload": {"session": {"id": sid, "status": "connected", "keepalive_timeout_seconds": 10,
                                        "reconnect_url": None}}}))
            if sid == "S1":
                while not self.subscribed.is_set():
                    await asyncio.sleep(0.05)
                await ws.send(json.dumps(notification("111", "a1", "KEKW that was insane",
                                                      [{"type": "emote", "text": "KEKW"},
                                                       {"type": "text", "text": " that was insane"}])))
                await ws.send(json.dumps(notification("111", "a1", "duplicate delivery")))
                await ws.send(json.dumps({"metadata": {"message_type": "session_keepalive"}, "payload": {}}))
                await ws.send(json.dumps({
                    "metadata": {"message_type": "session_reconnect"},
                    "payload": {"session": {"id": "S1", "status": "reconnecting",
                                            "reconnect_url": f"ws://127.0.0.1:{self.ws_port}/reconnect"}}}))
                await asyncio.sleep(5)
            else:
                await ws.send(json.dumps(notification("111", "b2", "after reconnect 😂")))
                await asyncio.sleep(30)

        async def main():
            self._done = asyncio.Event()
            async with serve(handler, "127.0.0.1", self.ws_port) as server:
                self._ready.set()
                await self._done.wait()
                server.close(close_connections=True)

        self.loop.run_until_complete(main())
        self.loop.close()

    def close(self):
        self.httpd.shutdown()
        self.loop.call_soon_threadsafe(self._done.set)


def test_twitch_eventsub_chat_collection(tmp_path):
    mock = MockEventSub()
    try:
        chat = TwitchChat(FakeTwitchToken(), "client-id", ws_url=f"ws://127.0.0.1:{mock.ws_port}/ws",
                          subscriptions_url=f"http://127.0.0.1:{mock.http_port}/helix/eventsub/subscriptions")
        log = ChatLog(tmp_path, "twitch")
        chat.start()
        chat.add("111", log)
        assert wait_for(lambda: log.messages >= 2, 20), chat.status
        headers, body = mock.posts[0]
        assert body["type"] == "channel.chat.message" and body["version"] == "1"
        assert body["condition"] == {"broadcaster_user_id": "111", "user_id": "999"}
        assert body["transport"] == {"method": "websocket", "session_id": "S1"}
        assert headers["Client-Id"] == "client-id" and headers["Authorization"] == "Bearer user-token"
        assert len(mock.posts) == 1, "subscriptions carry over on session_reconnect"

        chat.remove("111")
        assert wait_for(lambda: mock.deletes, 10)
        chat.stop()
        msgs, intervals = load_chat(tmp_path)
        assert [m["text"] for m in msgs] == ["KEKW that was insane", "after reconnect 😂"]   # duplicate dropped
        assert msgs[0]["emotes"] == ["KEKW"] and msgs[0]["user"] == "viewer32"
        assert msgs[0]["ts"] == pytest.approx(parse_timestamp("2026-11-20T01:30:15.123456Z"))
        assert intervals and intervals[0][1] != float("inf")       # coverage closed on remove
    finally:
        mock.close()


def test_twitch_device_code_login_and_refresh(tmp_path):
    calls = []

    class Session:
        def post(self, url, data=None, timeout=None):
            calls.append((url, data))
            if url.endswith("/device"):
                return FakeResponse(200, {"device_code": "dc", "user_code": "ABCD1234", "interval": 0,
                                          "expires_in": 60, "verification_uri": "https://twitch.tv/activate"})
            if data.get("grant_type", "").endswith("device_code"):
                pending = sum(1 for _, d in calls if d.get("device_code")) == 1
                if pending:
                    return FakeResponse(400, {"status": 400, "message": "authorization_pending"})
                return FakeResponse(200, {"access_token": "at1", "refresh_token": "rt1", "expires_in": 14000,
                                          "scope": ["user:read:chat"], "token_type": "bearer"})
            if data.get("grant_type") == "refresh_token":
                return FakeResponse(200, {"access_token": "at2", "refresh_token": "rt2", "expires_in": 14000})
            raise AssertionError(url)

        def get(self, url, headers=None, timeout=None):
            return FakeResponse(200, {"client_id": "cid", "login": "reader", "user_id": "999",
                                      "scopes": ["user:read:chat"], "expires_in": 14000})

    shown = []
    token = TwitchUserToken("cid", "secret", token_file=tmp_path / "t.json", session=Session())
    info = token.device_login(show=shown.append, poll_limit=10)
    assert info["user_id"] == "999" and "ABCD1234" in shown[0]
    assert token.available and token.access_token() == "at1"
    assert (tmp_path / "t.json").stat().st_mode & 0o077 == 0, "token file is private"
    token.data["expires_at"] = 0
    assert token.access_token() == "at2"
    refresh_call = [d for _, d in calls if d.get("grant_type") == "refresh_token"][0]
    assert refresh_call["client_secret"] == "secret" and refresh_call["refresh_token"] == "rt1"


# ---------------------------------------------------------------- YouTube

def yt_item(i, text, ts, kind="textMessageEvent"):
    return {"kind": "youtube#liveChatMessage", "id": f"id{i}",
            "snippet": {"type": kind, "liveChatId": "chat1", "publishedAt": ts, "hasDisplayContent": True,
                        "displayMessage": text, "textMessageDetails": {"messageText": text}},
            "authorDetails": {"displayName": f"user{i}", "channelId": f"UC{i}"}}


def test_youtube_chat_polling(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "YOUTUBE_CHAT_POLL_SECONDS", 0.05)
    pages = [
        {"items": [yt_item(1, "LMAO", "2026-11-20T01:00:00.5Z"), yt_item(2, "no way 😱", "2026-11-20T01:00:01Z"),
                   yt_item(3, "", "2026-11-20T01:00:01Z", kind="newSponsorEvent")],
         "nextPageToken": "p2", "pollingIntervalMillis": 10},
        {"items": [yt_item(2, "no way 😱", "2026-11-20T01:00:01Z"), yt_item(4, "clip it", "2026-11-20T01:00:03Z")],
         "nextPageToken": "p3", "pollingIntervalMillis": 10, "offlineAt": "2026-11-20T01:00:04Z"},
    ]
    seen = []

    class Session:
        def get(self, url, params=None, headers=None, timeout=None):
            seen.append((params, headers))
            return FakeResponse(200, pages[len(seen) - 1])

    log = ChatLog(tmp_path, "youtube")
    poller = YouTubeChatPoller("chat1", log, QuotaBudget(100), api_key="KEY", session=Session())
    poller.start()
    poller.join(10)
    msgs, intervals = load_chat(tmp_path)
    assert [m["text"] for m in msgs] == ["LMAO", "no way 😱", "clip it"]
    assert seen[0][0]["part"] == "snippet,authorDetails" and seen[0][0]["liveChatId"] == "chat1"
    assert seen[1][0]["pageToken"] == "p2"
    assert seen[0][1]["X-Goog-Api-Key"] == "KEY" and "key" not in seen[0][0]
    assert intervals and intervals[-1][1] != float("inf")         # ended -> coverage closed


def test_youtube_chat_unavailable_and_budget(tmp_path):
    class Denied:
        def get(self, *a, **k):
            return FakeResponse(403, {"error": {"code": 403, "errors": [{"reason": "liveChatDisabled"}]}})

    log = ChatLog(tmp_path, "youtube")
    p = YouTubeChatPoller("chat1", log, QuotaBudget(100), api_key="KEY", session=Denied())
    p.start()
    p.join(5)
    assert not p.is_alive() and "liveChatDisabled" in p.status
    assert load_chat(tmp_path) == ([], [])          # no data, and never marked as covered

    budget = QuotaBudget(10)
    assert budget.spend(5) and budget.spend(5) and not budget.spend(5)


# ---------------------------------------------------------------- Kick

def test_kick_webhook_signature_and_parsing(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    subs = []

    class Session:
        def get(self, url, timeout=None, **k):
            return FakeResponse(200, {"data": {"public_key": pem}, "message": "OK"})

        def post(self, url, json=None, data=None, headers=None, timeout=None):
            if url.endswith("/oauth/token"):
                return FakeResponse(200, {"access_token": "app", "expires_in": "3600", "token_type": "Bearer"})
            subs.append((json, headers))
            return FakeResponse(200, {"data": [{"name": "chat.message.sent", "version": 1,
                                                "subscription_id": "01SUB", "error": ""}], "message": "OK"})

        def delete(self, url, params=None, headers=None, timeout=None):
            subs.append(("DELETE", params))
            return FakeResponse(200, {"data": {}, "message": "OK"})

    kick = KickChat(host="127.0.0.1", port=0, client_id="k", client_secret="s", session=Session())
    kick.start()
    log = ChatLog(tmp_path, "kick")
    kick.add(123456789, log)
    assert subs[0][0] == {"broadcaster_user_id": 123456789, "method": "webhook",
                          "events": [{"name": "chat.message.sent", "version": 1}]}
    assert subs[0][1]["Authorization"] == "Bearer app"

    body = json.dumps({
        "message_id": "unique_message_id_123",
        "broadcaster": {"user_id": 123456789, "username": "broadcaster_name", "channel_slug": "broadcaster_channel"},
        "sender": {"user_id": 987654321, "username": "sender_name", "channel_slug": "sender_channel"},
        "content": "Hello [emote:4148074:HYPERCLAP] [emote:37226:KEKW]",
        "emotes": [{"emote_id": "4148074", "positions": [{"s": 6, "e": 30}]}],
        "created_at": "2025-01-14T16:08:06Z",
    }).encode()
    mid, ts = "01JMSGID", "2025-01-14T16:08:06Z"
    sig = base64.b64encode(key.sign(f"{mid}.{ts}.".encode() + body, padding.PKCS1v15(), hashes.SHA256())).decode()
    url = f"http://127.0.0.1:{kick.port}/kick/webhook"
    hdrs = {"Kick-Event-Message-Id": mid, "Kick-Event-Message-Timestamp": ts, "Kick-Event-Signature": sig,
            "Kick-Event-Type": "chat.message.sent", "Kick-Event-Version": "1"}
    assert log.available is False
    assert requests.post(url, data=body, headers=hdrs, timeout=5).status_code == 200
    assert requests.post(url, data=body, headers=hdrs, timeout=5).status_code == 200    # redelivery: deduped
    forged = dict(hdrs, **{"Kick-Event-Message-Id": "01FORGED"})
    assert requests.post(url, data=body, headers=forged, timeout=5).status_code == 401
    kick.remove(123456789)
    kick.shutdown()

    msgs, intervals = load_chat(tmp_path)
    assert len(msgs) == 1
    assert msgs[0]["text"] == "Hello HYPERCLAP KEKW" and msgs[0]["emotes"] == ["HYPERCLAP", "KEKW"]
    assert msgs[0]["user"] == "sender_name" and msgs[0]["ts"] == 1736870886.0
    assert intervals, "coverage starts only once a webhook really arrived"
    assert subs[-1] == ("DELETE", {"id": "01SUB"})
    assert kick.rejected == 1
    assert parse_content("[emote:1:a][emote:2:b]") == ("ab", ["a", "b"])


# ---------------------------------------------------------------- analysis

def synthetic_chat(tmp_path, start, seconds=900, spike_at=None, seed=1):
    """1 msg/s of ordinary chat, plus a 10-second laughing spike at `spike_at`."""
    rng = np.random.default_rng(seed)
    log = ChatLog(tmp_path, "twitch")
    with open(tmp_path / "chat.jsonl", "a") as fh:
        fh.write(json.dumps({"kind": "coverage", "ts": start, "available": True}) + "\n")
    normal = ["what game is this", "hello chat", "first time here", "nice", "how long is stream", "gg"]
    for t in np.sort(rng.uniform(start, start + seconds, seconds)):
        log.message(t, f"u{rng.integers(200)}", normal[rng.integers(len(normal))])
    if spike_at is not None:
        laughs = ["KEKW", "LMAOOOO 😂😂", "💀💀💀", "KEKW KEKW", "im dead", "CLIP IT"]
        for t in np.sort(rng.uniform(spike_at, spike_at + 10, 70)):
            log.message(t, f"u{rng.integers(400)}", laughs[rng.integers(len(laughs))])


def test_chat_spike_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHAT_REACTION_DELAY_SECONDS", 4)
    monkeypatch.setattr(config, "CHAT_SYNC_OFFSET_SECONDS", 0)
    start = 1_800_000_000.0
    seg_start = start + 600
    spike = seg_start + 120                      # chat explodes 2 min into the segment
    synthetic_chat(tmp_path, start, spike_at=spike)

    data = chat_analyzer.load_window(tmp_path, seg_start, seg_start + 300)
    assert data.available
    moment = chat_analyzer.score_moment(data, spike - 4, spike + 4)       # event ~4 s before chat reacts
    quiet = chat_analyzer.score_moment(data, seg_start + 200, seg_start + 208)
    assert moment["signals"]["chat_spike"] and moment["signals"]["laugh_reaction"]
    assert moment["signals"]["emoji_spike"] and moment["signals"]["clip_requests"]
    assert moment["increase_pct"] >= 200
    assert moment["score"] >= 70, moment
    assert quiet["score"] < 30, quiet
    assert not quiet["signals"]["chat_spike"]

    # Timestamp synchronisation: the activity curve peaks at the video second the
    # moment happened (chat time minus the reaction delay).
    curve = chat_analyzer.activity_curve(data, seg_start, 300)
    peak = int(np.argmax(curve))
    assert 112 <= peak <= 126, peak


def test_missing_chat_is_not_zero(tmp_path):
    log = ChatLog(tmp_path, "kick")
    log.coverage(False, "Kick chat needs webhooks")
    data = chat_analyzer.load_window(tmp_path, 1000, 1600)
    assert not data.available
    assert chat_analyzer.score_moment(data, 1100, 1110) is None
    assert chat_analyzer.activity_curve(data, 1000, 600) is None
    assert coverage_fraction([(0, 50)], 0, 100) == 0.5


def test_chat_client_credentials_fallback(monkeypatch):
    from clipping_engine.chat.twitch_chat import chat_client_credentials
    monkeypatch.setenv("TWITCH_CLIENT_ID", "monitor-app")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "monitor-secret")
    monkeypatch.setenv("TWITCH_CHAT_CLIENT_ID", "")            # empty lines from .env.example
    monkeypatch.setenv("TWITCH_CHAT_CLIENT_SECRET", "")
    assert chat_client_credentials() == ("monitor-app", "monitor-secret")
    monkeypatch.setenv("TWITCH_CHAT_CLIENT_ID", "public-app")
    assert chat_client_credentials() == ("public-app", "")
