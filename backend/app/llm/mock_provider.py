"""Deterministic scripted agent used when no API key is configured.

It speaks the same protocol as the real provider (streams tokens, emits tool calls,
reads tool results back), so the orchestrator, tools, guardrails, SSE stream and UI
are exercised end to end in tests and in the offline demo. It is a stand-in for the
model, not a second implementation of the business rules: refunds, limits and
retrieval confidence still come from the real tool layer.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from .base import LLMResult, OnToken, ToolCall, Usage

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_ZIP = re.compile(r"\b\d{5}\b")
_INVOICE = re.compile(r"\bINV-\d+\b", re.I)


def _intent(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(human|person|agent|representative|manager)\b", t) and re.search(r"\b(talk|speak|want|need|get)\b", t):
        return "human"
    if re.search(r"refund|charged twice|double charged|duplicate|money back|billed in error|overcharged", t):
        return "refund"
    if re.search(r"auto.?renew", t) and re.search(r"off|stop|disable|cancel|turn", t):
        return "auto_renew_off"
    if re.fullmatch(r"\s*(hi|hello|hey|good (morning|afternoon|evening))[!. ]*", t):
        return "greeting"
    if _EMAIL.search(t) and len(t.split()) <= 8:
        return "identity"
    return "question"


def _first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


class MockProvider:
    name = "mock"
    chat_model = "mock-scripted-agent"
    verifier_model = "mock-scripted-agent"

    def __init__(self, token_delay_s: float = 0.012) -> None:
        self.token_delay_s = token_delay_s
        self._n = 0

    # -- protocol ---------------------------------------------------------------

    async def stream_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], on_token: OnToken | None) -> LLMResult:
        started = time.perf_counter()
        await asyncio.sleep(0.15)  # a little realism for the latency panel
        decision = self._decide(messages)
        ttft = (time.perf_counter() - started) * 1000
        if isinstance(decision, ToolCall):
            return self._result("", [decision], messages, started, ttft)
        if on_token:
            for word in re.findall(r"\S+\s*", decision):
                await on_token(word)
                await asyncio.sleep(self.token_delay_s)
        return self._result(decision, [], messages, started, ttft)

    async def complete_json(self, messages, schema, name):  # pragma: no cover - verifier uses heuristic
        raise NotImplementedError

    # -- the script ---------------------------------------------------------------

    def _call(self, name: str, args: dict) -> ToolCall:
        self._n += 1
        return ToolCall(id=f"call_mock_{self._n}", name=name, arguments=json.dumps(args))

    def _decide(self, messages: list[dict[str, Any]]) -> ToolCall | str:
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        user_text = messages[last_user]["content"]
        verified = "VERIFIED." in messages[0]["content"]

        # Tool results produced so far in this turn, by tool name (latest wins).
        names: dict[str, str] = {}
        done: dict[str, dict] = {}
        for m in messages[last_user + 1 :]:
            for tc in m.get("tool_calls") or []:
                names[tc["id"]] = tc["function"]["name"]
            if m["role"] == "tool":
                done[names.get(m["tool_call_id"], "?")] = json.loads(m["content"])

        failed = next((r for r in done.values() if r.get("error") and r["error"] != "identity_not_verified"), None)
        if failed and "create_escalation_ticket" not in done:
            return self._call("create_escalation_ticket", {
                "summary": f"Tool failure during request: {user_text[:160]}", "category": "technical", "priority": "high"})
        if "create_escalation_ticket" in done:
            t = done["create_escalation_ticket"]
            lead = "Sorry, one of our systems isn't responding right now, so I couldn't finish that. " if failed else ""
            return (f"{lead}I've passed this to a human specialist: ticket {t['ticket_id']} ({t['priority']} priority). "
                    f"You'll hear back within {t['first_response_sla']}, and you won't need to repeat yourself.")

        users = [m["content"] for m in messages if m["role"] == "user"]
        intent = _intent(user_text)
        if intent == "identity":  # follow-up with credentials: resume the earlier request
            intent = next((i for i in map(_intent, reversed(users[:-1])) if i not in ("identity", "greeting")), "question")
        all_user = " ".join(users)
        email, zips = _EMAIL.search(all_user), _ZIP.findall(all_user)

        if intent == "greeting":
            return ("Hi! I'm the Orbit Notes support assistant. I can answer questions about plans, billing, refunds "
                    "and sync, and once I've verified your account I can check invoices or fix billing problems.")
        if intent == "human":
            return self._call("create_escalation_ticket", {
                "summary": f"Customer asked for a person: {user_text[:160]}", "category": "other", "priority": "normal"})

        if intent in ("refund", "auto_renew_off"):
            if not verified:
                if "verify_customer" in done:
                    return "Those details don't match an account I can find. Could you double-check the email and billing ZIP code?"
                if email and zips:
                    return self._call("verify_customer", {"email": email.group(0), "zip_code": zips[-1]})
                return "I can help with that. First I need to verify your account: what's the email on the account and your billing ZIP code?"
            if intent == "auto_renew_off":
                if "update_account_field" not in done:
                    return self._call("update_account_field", {"field": "auto_renew", "value": "false"})
                return "Done: auto-renew is now off, so your plan won't renew at the end of this billing period."
            if "search_policies" not in done:
                return self._call("search_policies", {"query": "refund policy for duplicate charges and refund windows"})
            if "list_invoices" not in done:
                return self._call("list_invoices", {})
            policy = done["search_policies"]
            cite = policy["results"][0]["id"] if policy.get("results") else None
            if "issue_refund" not in done:
                invoices = done["list_invoices"]["invoices"]
                named = _INVOICE.search(user_text)
                target = (
                    next((i for i in invoices if named and i["invoice_id"].upper() == named.group(0).upper()), None)
                    or next((i for i in invoices if i["flag"] and i["status"] == "paid"), None)
                    or next((i for i in invoices if "error" in i["description"].lower() and i["status"] == "paid"), None)
                    or next((i for i in invoices if i["status"] == "paid"), None)
                )
                if not target:
                    return "I don't see any refundable invoices on your account."
                reason = "duplicate charge" if target["flag"] else "customer request"
                return self._call("issue_refund", {"invoice_id": target["invoice_id"], "amount_usd": target["amount_usd"], "reason": reason})
            r = done["issue_refund"]
            ref = f" [{cite}]" if cite else ""
            if r["status"] == "refunded":
                return (f"Done. I've refunded ${r['amount_usd']:.2f} for {r['invoice_id']}. It goes back to your original "
                        f"payment method and should arrive in 5 to 10 business days{ref}. Refund reference: {r['refund_id']}.")
            if r["status"] == "requires_human":
                return self._call("create_escalation_ticket", {
                    "summary": f"Refund above agent limit: {r['reason']}", "category": "refund", "priority": "high"})
            return f"I'm sorry, I can't refund that one. {r['reason']}{ref} I can help you turn off auto-renew so it doesn't happen again."

        # General policy question.
        if "search_policies" not in done:
            return self._call("search_policies", {"query": user_text})
        result = done["search_policies"]
        if result["confidence"] == "low" or not result.get("results"):
            return self._call("create_escalation_ticket", {
                "summary": f"Question not covered by the knowledge base: {user_text[:160]}", "category": "other", "priority": "normal"})
        top = result["results"][0]
        return f"{_first_sentences(top['text'])} [{top['id']}]"

    def _result(self, text: str, calls: list[ToolCall], messages: list[dict], started: float, ttft: float) -> LLMResult:
        prompt_chars = sum(len(json.dumps(m)) for m in messages)
        out_chars = len(text) + sum(len(c.arguments) + len(c.name) for c in calls)
        return LLMResult(
            text=text,
            tool_calls=calls,
            usage=Usage(input_tokens=prompt_chars // 4, output_tokens=max(out_chars // 4, 1)),
            model=self.chat_model,
            ttft_ms=ttft,
            duration_ms=(time.perf_counter() - started) * 1000,
            finish_reason="tool_calls" if calls else "stop",
        )
