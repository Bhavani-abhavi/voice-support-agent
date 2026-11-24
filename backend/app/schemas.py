"""Request bodies for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Simulation = Literal["none", "tool_flaky", "tool_down", "llm_rate_limit", "llm_timeout"]


class ChatIn(BaseModel):
    session_id: str
    message: str = Field(min_length=1)
    simulate: Simulation = "none"


class VoiceSessionIn(BaseModel):
    session_id: str


class VoiceToolIn(BaseModel):
    session_id: str
    call_id: str
    name: str
    arguments: str = "{}"


class TranscriptIn(BaseModel):
    session_id: str
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)
    interrupted: bool = False


class UsageIn(BaseModel):
    session_id: str
    usage: dict[str, Any] = Field(default_factory=dict)
    latency: dict[str, Any] = Field(default_factory=dict)
    user_text: str = ""
    status: str = "completed"
