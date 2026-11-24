"""Live voice over the OpenAI Realtime API (WebRTC).

Architecture: the browser holds the audio (WebRTC straight to OpenAI, so there is
no audio relay hop through our server), while every *decision* still goes through
this backend:

* ``mint``      creates a short-lived client secret. The real API key never reaches
                the browser. The session is primed with the shared conversation so far.
* ``run_tool``  executes the model's function calls with the same registry, identity
                checks, refund limits and retrieval confidence gate as chat.
* ``record_*``  writes voice transcripts into the shared session history (so chat can
                continue where voice left off) and turns Realtime usage into traces.

Retrieval gating is the voice guardrail: a low-confidence search returns no passages
and an instruction to escalate, because spoken audio can't be retracted after the fact.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .config import Settings
from .llm.pricing import realtime_cost
from .rag.retriever import Retriever
from .session import Session
from .tools.registry import TOOLS, ToolContext, execute, realtime_tools
from .agent.prompts import voice_instructions


class VoiceError(Exception):
    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.message, self.status = message, status


class VoiceService:
    def __init__(self, settings: Settings, retriever: Retriever) -> None:
        self.settings = settings
        self.retriever = retriever

    def session_config(self, session: Session) -> dict[str, Any]:
        s = self.settings
        return {
            "type": "realtime",
            "model": s.realtime_model,
            "instructions": voice_instructions(session),
            "audio": {
                "input": {
                    "transcription": {"model": s.transcribe_model},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.55,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 550,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {"voice": s.realtime_voice},
            },
            "tools": realtime_tools(),
            "tool_choice": "auto",
        }

    async def mint(self, session: Session) -> dict[str, Any]:
        if not self.settings.voice_enabled:
            raise VoiceError("Voice mode needs OPENAI_API_KEY on the server.", 503)
        body = {"session": self.session_config(session), "expires_after": {"anchor": "created_at", "seconds": 60}}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/realtime/client_secrets",
                    headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                    json=body,
                )
        except httpx.HTTPError as exc:
            raise VoiceError(f"Couldn't reach the voice service ({type(exc).__name__}).") from exc
        if resp.status_code == 429:
            raise VoiceError("The voice service is rate limiting requests. Try again shortly.", 429)
        if resp.status_code >= 400:
            detail = resp.json().get("error", {}).get("message", resp.text[:200]) if resp.headers.get("content-type", "").startswith("application/json") else resp.text[:200]
            raise VoiceError(f"Voice session was refused: {detail}", 502)
        data = resp.json()
        session.voice_sessions_started += 1
        session.voice_pending = {"spans": [], "retrievals": [], "started": time.time()}
        return {
            "client_secret": data.get("value"),
            "expires_at": data.get("expires_at"),
            "model": self.settings.realtime_model,
            "max_session_s": self.settings.voice_max_session_s,
        }

    async def run_tool(self, session: Session, call_id: str, name: str, arguments: str) -> dict[str, Any]:
        ctx = ToolContext(session=session, retriever=self.retriever, mode="voice")
        session.add({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]}, "voice")
        outcome = await execute(name, arguments, ctx, timeout_s=self.settings.tool_timeout_s, max_retries=self.settings.tool_max_retries)
        session.add({"role": "tool", "tool_call_id": call_id, "content": json.dumps(outcome.output)}, "voice")
        pending = getattr(session, "voice_pending", None) or {"spans": [], "retrievals": []}
        pending["spans"].append({"name": name, "kind": "tool", "duration_ms": round(outcome.duration_ms, 1),
                                 "status": "ok" if outcome.ok else "error", "summary": outcome.summary()})
        pending["retrievals"].extend(ctx.retrievals)
        session.voice_pending = pending
        if name == "create_escalation_ticket" and outcome.ok:
            pass  # already recorded on session.escalations by the tool
        spec = TOOLS.get(name)
        return {
            "output": json.dumps(outcome.output),
            "ok": outcome.ok,
            "label": spec.label if spec else name,
            "summary": outcome.summary(),
            "duration_ms": round(outcome.duration_ms, 1),
            "retrieval": ctx.retrievals[-1] if ctx.retrievals else None,
            "escalation": outcome.output if name == "create_escalation_ticket" and outcome.ok else None,
        }

    @staticmethod
    def record_transcript(session: Session, role: str, text: str, interrupted: bool = False) -> None:
        text = text.strip()
        if text and role in ("user", "assistant"):
            session.add({"role": role, "content": text + (" [interrupted by customer]" if interrupted else "")}, "voice")

    def record_usage(self, session: Session, usage: dict[str, Any], latency: dict[str, Any], user_text: str, status: str) -> dict[str, Any]:
        cost, priced = realtime_cost(self.settings.realtime_model, usage)
        pending = getattr(session, "voice_pending", None) or {"spans": [], "retrievals": []}
        ind, outd = usage.get("input_token_details") or {}, usage.get("output_token_details") or {}
        record = {
            "id": f"v{int(time.time() * 1000) % 10**9}",
            "mode": "voice",
            "user_text": user_text,
            "started_at": time.time(),
            "outcome": "interrupted" if status == "cancelled" else ("escalated" if any(s["name"] == "create_escalation_ticket" for s in pending["spans"]) else "answered"),
            "latency_ms": latency.get("response_ms"),
            "ttft_ms": latency.get("first_audio_ms"),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cost_usd": round(cost, 6),
            "token_detail": {
                "audio_in": ind.get("audio_tokens", 0), "text_in": ind.get("text_tokens", 0),
                "cached_in": ind.get("cached_tokens", 0),
                "audio_out": outd.get("audio_tokens", 0), "text_out": outd.get("text_tokens", 0),
            },
            "llm_calls": [{
                "purpose": "realtime response", "model": self.settings.realtime_model,
                "ttft_ms": latency.get("first_audio_ms"), "duration_ms": latency.get("response_ms"),
                "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
                "cached_tokens": ind.get("cached_tokens", 0), "cost_usd": round(cost, 6), "priced": priced,
                "prompt": [{"role": "system", "content": voice_instructions(session)}],
                "output": status,
            }],
            "spans": pending["spans"],
            "retrievals": pending["retrievals"],
            "guardrail": {
                "retrieval_confidence": min((r["confidence"] for r in pending["retrievals"]),
                                            key={"low": 0, "borderline": 1, "high": 2}.__getitem__, default=None),
                "verification": "gated-at-retrieval",
                "method": "voice: low-confidence searches return no passages",
            },
            "errors": [],
        }
        session.voice_pending = {"spans": [], "retrievals": []}
        session.traces.append(record)
        del session.traces[:-50]
        return record
