"""Conversation state shared by chat and voice.

One ``Session`` holds the transcript, the verified customer and the traces. Chat
turns and voice turns append to the same history, so switching modes mid-
conversation keeps context: the voice agent is primed with the chat so far, and
voice transcripts flow back into the history the chat agent reads.

The store is in-process with TTL + LRU eviction. That is the right size for a
single-instance demo; the ``SessionStore`` boundary is where Redis would go for
horizontal scale.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .tools.mock_data import Customer, MockBackend


@dataclass
class HistoryItem:
    message: dict[str, Any]  # OpenAI chat-format message
    mode: str  # chat | voice
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    backend: MockBackend = field(default_factory=MockBackend)
    history: list[HistoryItem] = field(default_factory=list)
    customer: Customer | None = None
    escalations: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    voice_sessions_started: int = 0
    voice_pending: dict = field(default_factory=lambda: {"spans": [], "retrievals": []})

    def add(self, message: dict[str, Any], mode: str) -> None:
        self.history.append(HistoryItem(message=message, mode=mode))

    def chat_messages(self, limit: int = 40) -> list[dict[str, Any]]:
        """History as API messages, trimmed without orphaning tool results."""
        items = [h.message for h in self.history[-limit:]]
        while items and items[0].get("role") == "tool":
            items.pop(0)
        return items

    def transcript(self, limit: int = 16) -> list[dict[str, str]]:
        """User/assistant text only, with the mode each turn happened in."""
        out = []
        for h in self.history:
            role, content = h.message.get("role"), h.message.get("content")
            if role in ("user", "assistant") and content:
                out.append({"role": role, "text": content, "mode": h.mode})
        return out[-limit:]

    def customer_context(self) -> str:
        if not self.customer:
            return "Customer identity: NOT verified yet."
        c = self.customer.profile()
        return (
            "Customer identity: VERIFIED. "
            f"Name {c['name']}, id {c['customer_id']}, plan {c['tier']} ({c['billing_cycle']}), "
            f"account status {c['contract_status']}, renewal {c['renewal_date']}."
        )


class SessionStore:
    def __init__(self, ttl_s: int = 7200, max_sessions: int = 2000) -> None:
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def _evict(self) -> None:
        now = time.time()
        for sid in [sid for sid, s in self._sessions.items() if now - s.last_seen > self.ttl_s]:
            del self._sessions[sid]
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def create(self) -> Session:
        self._evict()
        session = Session(id=uuid.uuid4().hex)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session:
            session.last_seen = time.time()
            self._sessions.move_to_end(session_id)
        return session

    def get_or_create(self, session_id: str | None) -> Session:
        return (self.get(session_id) if session_id else None) or self.create()
