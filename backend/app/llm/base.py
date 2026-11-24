"""Provider-neutral LLM types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol


class LLMError(Exception):
    """Base for upstream model failures the orchestrator knows how to handle."""

    retryable = True
    user_message = "The AI service had a problem."


class LLMRateLimited(LLMError):
    user_message = "The AI service is rate limiting requests."


class LLMTimeout(LLMError):
    user_message = "The AI service took too long to respond."


class LLMUnavailable(LLMError):
    user_message = "The AI service is unavailable."


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class LLMResult:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    model: str
    ttft_ms: float | None
    duration_ms: float
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


OnToken = Callable[[str], Awaitable[None]]


class LLMProvider(Protocol):
    name: str
    chat_model: str
    verifier_model: str

    async def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], on_token: OnToken | None
    ) -> LLMResult: ...

    async def complete_json(
        self, messages: list[dict[str, Any]], schema: dict[str, Any], name: str
    ) -> tuple[dict[str, Any], LLMResult]: ...
