"""
Starts and stops chat collection alongside recordings.

Chat runs independently of video capture: a chat failure never affects a
recording, and a recording without chat is simply analysed without it.
Every reason chat is unavailable is written to chat.jsonl as a coverage
marker and shown in the status lines.
"""

import logging

from clipping_engine import config
from clipping_engine.chat.chat_log import ChatLog

log = logging.getLogger(__name__)


class ChatManager:
    def __init__(self):
        self.sessions = {}         # target.key -> (platform_id, ChatLog, collector or poller)
        self.notes = {}            # target.key -> why chat is unavailable
        self._twitch = None
        self._kick = None
        self._yt_budget = None

    # ---- lazily created collectors ------------------------------------------------------------

    def _twitch_client(self):
        if self._twitch is None:
            from clipping_engine.chat.twitch_chat import TwitchChat, TwitchUserToken, chat_client_credentials
            cid, secret = chat_client_credentials()
            token = TwitchUserToken(cid, secret)
            if not token.available:
                return None
            self._twitch = TwitchChat(token, cid)
            self._twitch.start()
        return self._twitch

    def _kick_client(self):
        if self._kick is None:
            from clipping_engine.chat.kick_chat import KickChat
            kick = KickChat()
            kick.start()
            self._kick = kick
        return self._kick

    # ---- per recording -------------------------------------------------------------------------

    def start(self, target, session_dir):
        chat_log = ChatLog(session_dir, target.platform)
        if not config.ENABLE_CHAT_COLLECTION:
            return self._unavailable(target, chat_log, "chat collection disabled in config")
        try:
            if target.platform == "twitch":
                bid = target.extra.get("user_id")
                client = self._twitch_client()
                if not bid:
                    return self._unavailable(target, chat_log, "Twitch user ID unknown")
                if client is None:
                    return self._unavailable(target, chat_log,
                                             "Twitch chat not authorized (python -m clipping_engine auth twitch)")
                client.add(bid, chat_log)
                self.sessions[target.key] = (bid, chat_log, client)
            elif target.platform == "youtube":
                chat_id = target.extra.get("live_chat_id")
                if not chat_id:
                    return self._unavailable(target, chat_log, "this broadcast has no active live chat")
                from clipping_engine.chat.youtube_chat import QuotaBudget, YouTubeChatPoller
                if self._yt_budget is None:
                    self._yt_budget = QuotaBudget(config.YOUTUBE_CHAT_DAILY_UNITS)
                poller = YouTubeChatPoller(chat_id, chat_log, self._yt_budget)
                poller.start()
                self.sessions[target.key] = (chat_id, chat_log, poller)
            elif target.platform == "kick":
                if not config.KICK_WEBHOOK_ENABLED:
                    return self._unavailable(target, chat_log,
                                             "Kick chat needs webhooks (KICK_WEBHOOK_ENABLED, see README)")
                bid = target.extra.get("broadcaster_user_id")
                if not bid:
                    return self._unavailable(target, chat_log, "Kick broadcaster ID unknown")
                client = self._kick_client()
                client.add(bid, chat_log)
                self.sessions[target.key] = (bid, chat_log, client)
            else:
                return self._unavailable(target, chat_log, "no official chat access for this platform")
            self.notes.pop(target.key, None)
        except Exception as exc:
            log.exception("chat start failed for %s", target.key)
            self._unavailable(target, chat_log, f"chat start failed ({type(exc).__name__})")

    def _unavailable(self, target, chat_log, reason):
        chat_log.coverage(False, reason)
        self.notes[target.key] = reason

    def stop(self, target):
        self.notes.pop(target.key, None)
        entry = self.sessions.pop(target.key, None)
        if not entry:
            return
        pid, chat_log, collector = entry
        try:
            if hasattr(collector, "remove"):
                collector.remove(pid)
            else:
                collector.stop()
        except Exception:
            log.exception("chat stop failed for %s", target.key)
        chat_log.coverage(False, "recording stopped")

    def shutdown(self):
        for key in list(self.sessions):
            _, chat_log, collector = self.sessions.pop(key)
            if not hasattr(collector, "remove"):
                collector.stop()
            chat_log.coverage(False, "engine stopped")
        if self._twitch:
            self._twitch.stop()
        if self._kick:
            self._kick.shutdown()

    def status_lines(self):
        lines = []
        for (platform, name), (_, chat_log, collector) in self.sessions.items():
            state = getattr(collector, "status", "")
            lines.append(f"  💬 {platform}/{name}: {chat_log.messages} chat messages ({state})")
        for (platform, name), reason in self.notes.items():
            lines.append(f"  💬 {platform}/{name}: no chat - {reason}")
        return lines
