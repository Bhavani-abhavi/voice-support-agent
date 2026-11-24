"""The chat agent: an explicit state machine around a tool-calling loop.

    THINKING --(tool calls)--> RUNNING_TOOLS --> THINKING ... (max N steps)
        |
        +--(text)--> VERIFYING --grounded--> DONE
                         |
                         +--ungrounded--> ESCALATING --> DONE (handoff message)

Every transition is streamed to the UI as an event, so the customer sees
"Verifying account…", "Searching knowledge base…" live, and the observability
drawer receives the full trace at the end of the turn.

Failure handling:
* LLM rate limits / timeouts: bounded exponential backoff, surfaced as a visible
  "retrying" state; after the last retry the turn ends with a typed ``error`` event
  and the session history is rolled back so the user can simply retry.
* Tool failures: retried once inside the executor, then returned to the model as a
  structured error it must not paper over (prompt rule 5 + verifier).
* Runaway loops: hard step budget, then escalation.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from ..config import Settings
from ..llm.base import LLMError, LLMProvider, LLMRateLimited, LLMResult, LLMTimeout
from ..observability import TurnTrace
from ..rag.retriever import Retriever
from ..session import Session
from ..tools.registry import TOOLS, ToolContext, execute, openai_chat_tools
from .prompts import chat_system_prompt
from .verifier import Verdict, verify

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]

_SENSITIVE_ARGS = {"zip_code"}


def _safe_args(raw: str) -> dict[str, Any]:
    try:
        args = json.loads(raw) if raw else {}
    except ValueError:
        return {"_raw": raw[:200]}
    return {k: ("•••" if k in _SENSITIVE_ARGS else v) for k, v in args.items()}


class AgentOrchestrator:
    def __init__(self, llm: LLMProvider, retriever: Retriever, settings: Settings) -> None:
        self.llm = llm
        self.retriever = retriever
        self.settings = settings
        self.tools = openai_chat_tools()

    async def run_turn(self, session: Session, user_text: str, emit: Emit, simulate: str = "none") -> dict[str, Any]:
        trace = TurnTrace("chat", user_text)
        ctx = ToolContext(session=session, retriever=self.retriever, simulate=simulate, mode="chat")
        history_mark = len(session.history)
        session.add({"role": "user", "content": user_text}, "chat")
        chunks: dict[str, dict[str, Any]] = {}
        tool_evidence: list[dict[str, Any]] = []
        escalated: dict[str, Any] | None = None
        final_text = ""

        try:
            for step in range(self.settings.max_agent_steps):
                hold = any(r["confidence"] == "borderline" for r in ctx.retrievals)
                await emit("status", {"state": "thinking", "label": "Thinking" if step == 0 else "Reviewing results"})
                if hold:
                    await emit("status", {"state": "buffering", "label": "Borderline match: checking the answer before showing it"})
                messages = [{"role": "system", "content": chat_system_prompt(session)}, *session.chat_messages()]
                result = await self._call_llm(messages, trace, emit, simulate, f"agent step {step + 1}", stream=not hold)
                if result.tool_calls:
                    session.add(
                        {
                            "role": "assistant",
                            "content": result.text or None,
                            "tool_calls": [
                                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                                for tc in result.tool_calls
                            ],
                        },
                        "chat",
                    )
                    for tc in result.tool_calls:
                        outcome = await self._run_tool(tc.id, tc.name, tc.arguments, ctx, trace, emit, chunks, tool_evidence)
                        if tc.name == "create_escalation_ticket" and outcome.ok:
                            escalated = outcome.output
                    continue
                final_text = result.text.strip()
                break
            else:
                trace.errors.append("step_budget_exhausted")

            if not final_text:  # step budget exhausted or empty answer: hand off, don't guess
                escalated = escalated or await self._escalate(
                    ctx, trace, emit, chunks, tool_evidence, f"Agent could not complete the request: {user_text[:200]}", "other"
                )
                final_text = self._handoff_text(escalated, failed=False)
                await emit("final", {"text": final_text, "replaced": True})
                verdict = Verdict("grounded", "handoff", "Handoff message; no factual claims.")
            else:
                verdict = await self._verify(user_text, final_text, chunks, tool_evidence, ctx, trace, emit)
                if not verdict.grounded and not (verdict.status == "unverified" and not self._was_borderline(ctx)):
                    escalated = escalated or await self._escalate(
                        ctx, trace, emit, chunks, tool_evidence,
                        f"Answer failed grounding check ({verdict.reason[:120]}). Customer asked: {user_text[:200]}", "other",
                    )
                    final_text = self._handoff_text(escalated, failed=False, retracted=True)
                    await emit("final", {"text": final_text, "replaced": True})
                else:
                    await emit("final", {"text": final_text, "replaced": False})

            session.add({"role": "assistant", "content": final_text}, "chat")
            confidence = self._turn_confidence(ctx)
            trace.guardrail = {
                "retrieval_confidence": confidence,
                "verification": verdict.status,
                "method": verdict.method,
                "reason": verdict.reason,
                "unsupported_claims": verdict.unsupported,
                "cited": verdict.cited,
                "escalated": bool(escalated),
            }
            await emit("verification", {
                "status": verdict.status,
                "method": verdict.method,
                "reason": verdict.reason,
                "unsupported": verdict.unsupported,
                "confidence": confidence,
                "citations": [{"id": c, **chunks[c]} for c in verdict.cited if c in chunks],
            })
            outcome_label = "escalated" if escalated else "answered"
        except LLMError as err:
            del session.history[history_mark:]  # roll back so a retry starts clean (tools are idempotent)
            trace.errors.append(type(err).__name__)
            await emit("error", {"code": type(err).__name__, "message": f"{err.user_message} Your message wasn't lost; try again.", "retryable": True})
            outcome_label = "error"

        record = trace.finish(outcome_label)
        session.traces.append(record)
        del session.traces[:-50]
        await emit("trace", record)
        return record

    # -- pieces ---------------------------------------------------------------------

    async def _call_llm(self, messages, trace: TurnTrace, emit: Emit, simulate: str, purpose: str, stream: bool) -> LLMResult:
        attempt = 0
        while True:
            start = trace.now_ms()
            emitted = False

            async def on_token(text: str) -> None:
                nonlocal emitted
                if trace.ttft_ms is None:
                    trace.ttft_ms = trace.now_ms()
                emitted = True
                await emit("token", {"text": text})

            try:
                if simulate == "llm_rate_limit" and attempt == 0:
                    raise LLMRateLimited("simulated 429 from the model provider")
                if simulate == "llm_timeout":
                    await asyncio.sleep(0.5)
                    raise LLMTimeout("simulated upstream timeout")
                result = await asyncio.wait_for(
                    self.llm.stream_chat(messages, self.tools, on_token if stream else None),
                    timeout=self.settings.llm_timeout_s + 5,
                )
                trace.add_llm(purpose, result, messages, start)
                return result
            except (LLMError, asyncio.TimeoutError) as exc:
                err = exc if isinstance(exc, LLMError) else LLMTimeout("orchestrator deadline")
                trace.span(f"{purpose} failed", "llm", start, trace.now_ms() - start, status="error", error=type(err).__name__)
                if emitted or attempt >= self.settings.llm_max_retries:
                    raise err from exc
                attempt += 1
                delay = 0.8 * (2 ** (attempt - 1))
                await emit("status", {
                    "state": "retrying",
                    "label": f"{err.user_message} Retrying in {delay:.1f}s (attempt {attempt + 1} of {self.settings.llm_max_retries + 1})",
                })
                await asyncio.sleep(delay)

    async def _run_tool(self, call_id, name, raw_args, ctx: ToolContext, trace: TurnTrace, emit: Emit, chunks, tool_evidence):
        spec = TOOLS.get(name)
        await emit("tool_start", {"call_id": call_id, "name": name, "label": spec.label if spec else name, "args": _safe_args(raw_args)})
        start = trace.now_ms()
        n_retrievals = len(ctx.retrievals)

        async def on_retry(tool: str, attempt: int, message: str) -> None:
            await emit("tool_retry", {"call_id": call_id, "name": tool, "attempt": attempt + 1, "message": f"{message}, retrying"})

        outcome = await execute(
            name, raw_args, ctx,
            timeout_s=self.settings.tool_timeout_s, max_retries=self.settings.tool_max_retries, on_retry=on_retry,
        )
        trace.span(name, "tool", start, outcome.duration_ms, status="ok" if outcome.ok else "error",
                   attempts=outcome.attempts, summary=outcome.summary())
        if len(ctx.retrievals) > n_retrievals:
            trace.retrievals.append(ctx.retrievals[-1])
        for r in outcome.output.get("results", []) if name == "search_policies" else []:
            chunks[r["id"]] = {"source": r["source"], "text": r["text"], "score": r["score"]}
        tool_evidence.append({"tool": name, "output": outcome.output})
        ctx.session.add({"role": "tool", "tool_call_id": call_id, "content": json.dumps(outcome.output)}, ctx.mode)
        await emit("tool_end", {
            "call_id": call_id, "name": name, "ok": outcome.ok, "duration_ms": round(outcome.duration_ms, 1),
            "attempts": outcome.attempts, "summary": outcome.summary(), "error": outcome.error,
            "retrieval": ctx.retrievals[-1] if len(ctx.retrievals) > n_retrievals else None,
        })
        if name == "create_escalation_ticket" and outcome.ok:
            await emit("escalation", outcome.output)
        return outcome

    async def _escalate(self, ctx, trace, emit, chunks, tool_evidence, summary: str, category: str) -> dict[str, Any]:
        call_id = f"call_guardrail_{trace.id}"
        args = json.dumps({"summary": summary, "category": category, "priority": "normal"})
        ctx.session.add({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "create_escalation_ticket", "arguments": args}}]}, ctx.mode)
        outcome = await self._run_tool(call_id, "create_escalation_ticket", args, ctx, trace, emit, chunks, tool_evidence)
        return outcome.output

    async def _verify(self, user_text, answer, chunks, tool_evidence, ctx, trace: TurnTrace, emit: Emit) -> Verdict:
        await emit("status", {"state": "verifying", "label": "Checking answer against sources"})
        start = trace.now_ms()
        verdict = await verify(self.llm, user_text, answer, chunks, tool_evidence)
        if verdict.llm and verdict.prompt:
            trace.add_llm("grounding verifier", verdict.llm, verdict.prompt, start)
        else:
            trace.span("grounding verifier", "verify", start, trace.now_ms() - start, status=verdict.status, method=verdict.method)
        if self._was_borderline(ctx) and verdict.grounded:
            # Buffered answer: release it now that it has passed. (Counts as first token.)
            if trace.ttft_ms is None:
                trace.ttft_ms = trace.now_ms()
            await emit("token", {"text": answer})
        return verdict

    @staticmethod
    def _was_borderline(ctx: ToolContext) -> bool:
        return any(r["confidence"] == "borderline" for r in ctx.retrievals)

    @staticmethod
    def _turn_confidence(ctx: ToolContext) -> str | None:
        if not ctx.retrievals:
            return None
        order = {"low": 0, "borderline": 1, "high": 2}
        return min((r["confidence"] for r in ctx.retrievals), key=order.__getitem__)

    @staticmethod
    def _handoff_text(ticket: dict[str, Any] | None, failed: bool, retracted: bool = False) -> str:
        lead = "I want to be sure you get an accurate answer, so rather than guess, " if retracted else ""
        if not ticket or "ticket_id" not in ticket:
            return lead + "I wasn't able to complete that. Please try again in a moment, or ask for a human agent."
        return (f"{lead}I've passed this to a human specialist: ticket {ticket['ticket_id']} ({ticket['priority']} priority). "
                f"You'll hear back within {ticket['first_response_sla']}, and they'll have the full context.")
