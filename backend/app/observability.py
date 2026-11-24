"""Per-turn tracing: spans, LLM calls (with the exact prompt sent), tokens, cost.

Each chat turn produces one trace; each voice response produces one trace. Traces
stream to the UI's observability drawer at the end of the turn and are kept on the
session (capped) so a reload can show history.
"""

from __future__ import annotations

import copy
import time
import uuid
from typing import Any

from .llm.base import LLMResult
from .llm.pricing import text_cost

_MAX_CONTENT = 4000


def _clip_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m.get("content"), str) and len(m["content"]) > _MAX_CONTENT:
            m["content"] = m["content"][:_MAX_CONTENT] + " …[truncated]"
    return out


class TurnTrace:
    def __init__(self, mode: str, user_text: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.mode = mode
        self.user_text = user_text
        self.wall_start = time.time()
        self._t0 = time.perf_counter()
        self.spans: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.retrievals: list[dict[str, Any]] = []
        self.guardrail: dict[str, Any] = {}
        self.errors: list[str] = []
        self.ttft_ms: float | None = None

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def span(self, name: str, kind: str, start_ms: float, duration_ms: float, status: str = "ok", **attrs: Any) -> None:
        self.spans.append(
            {"name": name, "kind": kind, "start_ms": round(start_ms, 1), "duration_ms": round(duration_ms, 1), "status": status, **attrs}
        )

    def add_llm(self, purpose: str, result: LLMResult, messages: list[dict[str, Any]], start_ms: float) -> None:
        cost, priced = text_cost(result.model, result.usage.input_tokens, result.usage.output_tokens, result.usage.cached_tokens)
        self.llm_calls.append(
            {
                "purpose": purpose,
                "model": result.model,
                "ttft_ms": round(result.ttft_ms, 1) if result.ttft_ms is not None else None,
                "duration_ms": round(result.duration_ms, 1),
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cached_tokens": result.usage.cached_tokens,
                "cost_usd": round(cost, 6),
                "priced": priced,
                "finish_reason": result.finish_reason,
                "prompt": _clip_messages(messages),
                "output": result.text or [{"tool": c.name, "arguments": c.arguments} for c in result.tool_calls],
            }
        )
        self.span(purpose, "llm", start_ms, result.duration_ms, model=result.model,
                  tokens=result.usage.input_tokens + result.usage.output_tokens)

    def finish(self, outcome: str) -> dict[str, Any]:
        total_in = sum(c["input_tokens"] for c in self.llm_calls)
        total_out = sum(c["output_tokens"] for c in self.llm_calls)
        return {
            "id": self.id,
            "mode": self.mode,
            "user_text": self.user_text,
            "started_at": self.wall_start,
            "outcome": outcome,
            "latency_ms": round(self.now_ms(), 1),
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": round(sum(c["cost_usd"] for c in self.llm_calls), 6),
            "llm_calls": self.llm_calls,
            "spans": self.spans,
            "retrievals": self.retrievals,
            "guardrail": self.guardrail,
            "errors": self.errors,
        }
