"""
Kick chat through the official Kick Public API webhooks.

Kick's only official real-time chat access is the `chat.message.sent`
webhook event (docs.kick.com -> Events). It requires:

  1. Webhooks enabled on your Kick app (kick.com/settings/developer) with a
     PUBLIC https URL pointing at this machine, e.g. through Cloudflare
     Tunnel or ngrok -> http://KICK_WEBHOOK_HOST:KICK_WEBHOOK_PORT/
  2. KICK_WEBHOOK_ENABLED = True in config.py.

Subscriptions are created with the app access token:
    POST https://api.kick.com/public/v1/events/subscriptions
    {"broadcaster_user_id": 123, "events": [{"name": "chat.message.sent", "version": 1}],
     "method": "webhook"}
and removed with DELETE ...?id=<subscription_id> when recording stops.

Every request is verified with Kick's public key (RSA PKCS#1 v1.5, SHA-256,
over "<Kick-Event-Message-Id>.<Kick-Event-Message-Timestamp>.<raw body>").
Unsigned or wrongly signed requests are rejected.

Chat coverage starts only when the first webhook for a channel actually
arrives, so a misconfigured tunnel shows up as "no chat data", not as a
quiet chat.
"""

import base64
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from clipping_engine import config
from clipping_engine.chat.chat_log import parse_timestamp

log = logging.getLogger(__name__)

TOKEN_URL = "https://id.kick.com/oauth/token"
SUBSCRIPTIONS_URL = "https://api.kick.com/public/v1/events/subscriptions"
PUBLIC_KEY_URL = "https://api.kick.com/public/v1/public-key"
EMOTE_RE = re.compile(r"\[emote:(\d+):([^\]]+)\]")
TIMEOUT = 15


def parse_content(content):
    """'Hi [emote:37226:KEKW]' -> ('Hi KEKW', ['KEKW'])."""
    names = [m.group(2) for m in EMOTE_RE.finditer(content or "")]
    return EMOTE_RE.sub(lambda m: m.group(2), content or ""), names


def load_public_key(text):
    """Accept the endpoint's JSON ({"data": {"public_key": PEM}}) or a bare PEM."""
    try:
        text = json.loads(text)["data"]["public_key"]
    except (ValueError, KeyError, TypeError):
        pass
    return serialization.load_pem_public_key(text.encode() if isinstance(text, str) else text)


def verify_signature(public_key, message_id, timestamp, body, signature_b64):
    try:
        public_key.verify(base64.b64decode(signature_b64), f"{message_id}.{timestamp}.".encode() + body,
                          padding.PKCS1v15(), hashes.SHA256())
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


class KickChat:
    def __init__(self, host=None, port=None, client_id=None, client_secret=None, session=None,
                 token_url=TOKEN_URL, subscriptions_url=SUBSCRIPTIONS_URL, public_key_url=PUBLIC_KEY_URL):
        self.host = host or config.KICK_WEBHOOK_HOST
        self.port = config.KICK_WEBHOOK_PORT if port is None else port
        self.client_id = client_id if client_id is not None else os.getenv("KICK_CLIENT_ID", "")
        self.client_secret = client_secret if client_secret is not None else os.getenv("KICK_CLIENT_SECRET", "")
        self.http = session or requests.Session()
        self.token_url, self.subscriptions_url, self.public_key_url = token_url, subscriptions_url, public_key_url
        self.logs = {}             # broadcaster_user_id (str) -> ChatLog
        self.subscriptions = {}    # broadcaster_user_id -> subscription id
        self._token = None
        self._token_expires = 0
        self._public_key = None
        self._seen = OrderedDict()
        self._lock = threading.Lock()
        self.server = None
        self.status = "idle"
        self.rejected = 0

    # ---- auth / keys -------------------------------------------------------------------------

    def _app_token(self):
        if self._token and time.time() < self._token_expires - 300:
            return self._token
        r = self.http.post(self.token_url, data={"client_id": self.client_id, "client_secret": self.client_secret,
                                                 "grant_type": "client_credentials"}, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        self._token_expires = time.time() + float(data.get("expires_in") or 3600)
        return self._token

    def public_key(self, refresh=False):
        if self._public_key is None or refresh:
            r = self.http.get(self.public_key_url, timeout=TIMEOUT)
            r.raise_for_status()
            self._public_key = load_public_key(r.text)
        return self._public_key

    # ---- server -------------------------------------------------------------------------------

    def start(self):
        if self.server:
            return
        self.public_key()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                code = outer.handle_webhook(self.headers, body)
                self.send_response(code)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, name="kick-webhooks", daemon=True).start()
        self.status = f"listening on {self.host}:{self.port}"

    def shutdown(self):
        for bid in list(self.subscriptions):
            self.remove(bid)
        if self.server:
            self.server.shutdown()
            self.server = None

    def handle_webhook(self, headers, body):
        mid = headers.get("Kick-Event-Message-Id", "")
        ts = headers.get("Kick-Event-Message-Timestamp", "")
        sig = headers.get("Kick-Event-Signature", "")
        ok = verify_signature(self.public_key(), mid, ts, body, sig)
        if not ok:
            # The key may have been rotated: fetch it again once before rejecting.
            try:
                ok = verify_signature(self.public_key(refresh=True), mid, ts, body, sig)
            except requests.RequestException:
                ok = False
        if not ok:
            self.rejected += 1
            return 401
        if headers.get("Kick-Event-Type") != "chat.message.sent":
            return 200
        with self._lock:
            if mid in self._seen:
                return 200
            self._seen[mid] = None
            while len(self._seen) > 5000:
                self._seen.popitem(last=False)
        try:
            event = json.loads(body)
        except ValueError:
            return 400
        bid = str((event.get("broadcaster") or {}).get("user_id"))
        chat_log = self.logs.get(bid)
        if chat_log:
            chat_log.coverage(True)        # delivery proven to work
            text, emotes = parse_content(event.get("content", ""))
            when = parse_timestamp(event.get("created_at")) or parse_timestamp(ts) or time.time()
            sender = event.get("sender") or {}
            chat_log.message(when, sender.get("username"), text, emotes, event.get("message_id"))
        return 200

    # ---- subscriptions ------------------------------------------------------------------------------

    def add(self, broadcaster_user_id, chat_log):
        bid = str(broadcaster_user_id)
        self.logs[bid] = chat_log
        body = {"broadcaster_user_id": int(bid), "method": "webhook",
                "events": [{"name": "chat.message.sent", "version": 1}]}
        try:
            r = self.http.post(self.subscriptions_url, json=body, timeout=TIMEOUT,
                               headers={"Authorization": f"Bearer {self._app_token()}"})
            r.raise_for_status()
            entry = (r.json().get("data") or [{}])[0]
            if entry.get("error"):
                raise RuntimeError(entry["error"])
            self.subscriptions[bid] = entry.get("subscription_id")
            chat_log.coverage(False, "waiting for the first Kick webhook")
        except Exception as exc:
            chat_log.coverage(False, f"Kick subscription failed ({type(exc).__name__}: {str(exc)[:100]})")
            log.warning("kick chat subscribe %s failed: %s", bid, exc)

    def remove(self, broadcaster_user_id):
        bid = str(broadcaster_user_id)
        chat_log = self.logs.pop(bid, None)
        if chat_log:
            chat_log.coverage(False, "recording stopped")
        sub_id = self.subscriptions.pop(bid, None)
        if sub_id:
            try:
                self.http.delete(self.subscriptions_url, params={"id": sub_id}, timeout=TIMEOUT,
                                 headers={"Authorization": f"Bearer {self._app_token()}"})
            except Exception as exc:
                log.warning("kick chat unsubscribe %s failed: %s", bid, exc)
