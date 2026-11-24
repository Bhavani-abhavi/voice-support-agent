"""OpenAI Chat Completions provider with streaming + tool-call accumulation."""

from __future__ import annotations

import json
import time
from typing import Any

import openai
from openai import AsyncOpenAI

from .base import LLMResult, LLMRateLimited, LLMTimeout, LLMUnavailable, OnToken, ToolCall, Usage


def _map_error(exc: Exception) -> Exception:
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimited(str(exc))
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeout(str(exc))
    if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError)):
        return LLMUnavailable(str(exc))
    return exc


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: AsyncOpenAI, chat_model: str, verifier_model: str, timeout_s: float) -> None:
        self.client = client
        self.chat_model = chat_model
        self.verifier_model = verifier_model
        self.timeout_s = timeout_s

    async def stream_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], on_token: OnToken | None) -> LLMResult:
        started = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        usage = Usage()
        finish = None
        try:
            stream = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                tools=tools or openai.NOT_GIVEN,
                temperature=0.2,
                stream=True,
                stream_options={"include_usage": True},
                timeout=self.timeout_s,
            )
            async for chunk in stream:
                if chunk.usage:
                    details = getattr(chunk.usage, "prompt_tokens_details", None)
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        cached_tokens=getattr(details, "cached_tokens", 0) or 0,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish = choice.finish_reason or finish
                delta = choice.delta
                if delta.content:
                    if ttft is None:
                        ttft = (time.perf_counter() - started) * 1000
                    text_parts.append(delta.content)
                    if on_token:
                        await on_token(delta.content)
                for tc in delta.tool_calls or []:
                    if ttft is None:
                        ttft = (time.perf_counter() - started) * 1000
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors
            raise _map_error(exc) from exc
        return LLMResult(
            text="".join(text_parts),
            tool_calls=[ToolCall(c["id"], c["name"], c["arguments"] or "{}") for _, c in sorted(calls.items())],
            usage=usage,
            model=self.chat_model,
            ttft_ms=ttft,
            duration_ms=(time.perf_counter() - started) * 1000,
            finish_reason=finish,
        )

    async def complete_json(self, messages: list[dict[str, Any]], schema: dict[str, Any], name: str) -> tuple[dict[str, Any], LLMResult]:
        started = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(
                model=self.verifier_model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
                timeout=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            raise _map_error(exc) from exc
        content = resp.choices[0].message.content or "{}"
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None) if u else None
        result = LLMResult(
            text=content,
            tool_calls=[],
            usage=Usage(u.prompt_tokens if u else 0, u.completion_tokens if u else 0, getattr(details, "cached_tokens", 0) or 0),
            model=self.verifier_model,
            ttft_ms=None,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return json.loads(content), result
