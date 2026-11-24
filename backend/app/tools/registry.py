"""Tool definitions, validation and a resilient executor.

Business rules live here, in code, not in the prompt: identity must be verified
before any account tool runs, refund limits are computed from the invoice and the
customer's plan, and escalation tickets carry the plan's SLA. The model decides
*which* tool to call; it cannot talk its way past these checks.

Both agents (chat orchestrator and voice) execute tools through ``execute()``, so
the guardrails, retries and traces are identical in either mode.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field, ValidationError

from ..rag.retriever import Retriever
from ..session import Session


class ToolFailure(Exception):
    def __init__(self, code: str, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


# --- argument schemas -------------------------------------------------------------


class VerifyArgs(BaseModel):
    email: str = Field(description="Email address on the account")
    zip_code: str = Field(description="Billing ZIP code on the account")


class SearchArgs(BaseModel):
    query: str = Field(description="What to look up in the support knowledge base, in plain words")


class NoArgs(BaseModel):
    pass


class RefundArgs(BaseModel):
    invoice_id: str = Field(description="Invoice to refund, e.g. INV-1043")
    amount_usd: float = Field(gt=0, description="Amount to refund in USD")
    reason: str = Field(description="Short reason, e.g. 'duplicate charge'")


class UpdateArgs(BaseModel):
    field: str = Field(description="One of: billing_email, notifications, auto_renew")
    value: str = Field(description="New value. notifications: all|important|none. auto_renew: true|false")


class TicketArgs(BaseModel):
    summary: str = Field(description="What the customer needs, so a human doesn't have to re-ask")
    category: str = Field(description="billing | refund | contract | security | technical | other")
    priority: str = Field(description="urgent | high | normal")


@dataclass
class ToolContext:
    session: Session
    retriever: Retriever
    simulate: str = "none"
    mode: str = "chat"
    attempt: int = 0
    retrievals: list[dict] = field(default_factory=list)


Handler = Callable[[BaseModel, ToolContext], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    label: str  # shown live in the UI while the tool runs
    description: str
    args: type[BaseModel]
    handler: Handler
    needs_verified_customer: bool = False
    billing_system: bool = False  # affected by the "billing system down" simulation

    def json_schema(self) -> dict[str, Any]:
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        return schema


# --- handlers ---------------------------------------------------------------------


async def _verify(args: VerifyArgs, ctx: ToolContext) -> dict:
    customer = ctx.session.backend.find(args.email)
    if not customer or customer.zip_code != args.zip_code.strip():
        # Same message either way: don't reveal whether the email has an account.
        return {"verified": False, "message": "Those details don't match an account. Ask the customer to check them."}
    ctx.session.customer = customer
    return {"verified": True, **customer.profile()}


async def _search(args: SearchArgs, ctx: ToolContext) -> dict:
    tier = ctx.session.customer.tier if ctx.session.customer else None
    result = await ctx.retriever.search(args.query, tier)
    ctx.retrievals.append(
        {
            "query": args.query,
            "tier_filter": result.tiers,
            "confidence": result.confidence,
            "top_score": round(result.top_score, 3),
            "thresholds": result.thresholds,
            "scores": result.extra.get("all_scores", []),
            "chunks": [h.chunk_id for h in result.hits],
            "latency_ms": round(result.latency_ms, 1),
        }
    )
    if result.confidence != "low":
        ctx.session.policy_checked = True
    return result.to_tool_output()


async def _list_invoices(_: NoArgs, ctx: ToolContext) -> dict:
    assert ctx.session.customer
    return {"invoices": [i.public() for i in ctx.session.customer.invoices]}


def _refund_limit(customer, invoice, today) -> tuple[float, str | None, str]:
    """Return (max auto-approvable amount, denial reason or None, policy reference)."""
    age = (today - invoice.date).days
    remaining = round(invoice.amount - invoice.refunded, 2)
    if invoice.duplicate_of:
        if age <= 90:
            return remaining, None, "refunds:1 (duplicate charges, full refund within 90 days)"
        return 0, "Duplicate charges can only be refunded within 90 days.", "refunds:1"
    if customer.tier == "free":
        return 0, "Free plan has no charges; treat as a billing error and escalate.", "refunds:5"
    if customer.tier == "pro":
        window = 14 if customer.billing_cycle == "monthly" else 30
        ref = "refunds:2 (Pro monthly, 14 days)" if window == 14 else "refunds:3 (Pro annual, 30 days)"
        if age > window:
            return 0, f"This charge is {age} days old; Pro {customer.billing_cycle} refunds are only allowed within {window} days.", ref
        return min(remaining, 150.0), None, ref + "; agent limit $150"
    return min(remaining, 500.0), None, "refunds:4 (Business, agent limit $500)"


async def _issue_refund(args: RefundArgs, ctx: ToolContext) -> dict:
    customer = ctx.session.customer
    assert customer
    if not ctx.session.policy_checked:
        # Mirrors refunds policy: "Agents never promise a refund before checking the invoice and this policy."
        return {
            "status": "policy_not_checked",
            "reason": "Look up the refund policy with search_policies before issuing a refund, then try again.",
        }
    invoice = next((i for i in customer.invoices if i.id.upper() == args.invoice_id.upper()), None)
    if not invoice:
        return {"status": "denied", "reason": f"Invoice {args.invoice_id} is not on this account."}
    if invoice.status == "refunded":
        return {"status": "denied", "reason": f"{invoice.id} has already been refunded."}
    limit, denial, policy = _refund_limit(customer, invoice, ctx.session.backend.today)
    if denial:
        return {"status": "denied", "reason": denial, "policy": policy}
    if args.amount_usd > invoice.amount - invoice.refunded + 0.001:
        return {"status": "denied", "reason": f"Amount exceeds the refundable balance of ${invoice.amount - invoice.refunded:.2f}."}
    if args.amount_usd > limit + 0.001:
        return {
            "status": "requires_human",
            "reason": f"${args.amount_usd:,.2f} is above the agent approval limit of ${limit:,.2f}. Create an escalation ticket.",
            "policy": policy,
        }
    invoice.refunded = round(invoice.refunded + args.amount_usd, 2)
    invoice.status = "refunded" if invoice.refunded >= invoice.amount else "partially_refunded"
    return {
        "status": "refunded",
        "refund_id": ctx.session.backend.next_refund_id(),
        "invoice_id": invoice.id,
        "amount_usd": args.amount_usd,
        "arrives": "5 to 10 business days, to the original payment method",
        "policy": policy,
    }


_ALLOWED_FIELDS = {"billing_email", "notifications", "auto_renew"}


async def _update_account(args: UpdateArgs, ctx: ToolContext) -> dict:
    customer = ctx.session.customer
    assert customer
    if args.field not in _ALLOWED_FIELDS:
        return {"status": "denied", "reason": f"Field '{args.field}' can't be changed by the agent. Allowed: {sorted(_ALLOWED_FIELDS)}"}
    if args.field == "billing_email":
        if "@" not in args.value:
            return {"status": "denied", "reason": "That isn't a valid email address."}
        old, customer.billing_email = customer.billing_email, args.value.strip()
    elif args.field == "notifications":
        if args.value not in ("all", "important", "none"):
            return {"status": "denied", "reason": "notifications must be all, important or none."}
        old, customer.notifications = customer.notifications, args.value
    else:
        new = args.value.strip().lower() in ("true", "yes", "on", "1")
        old, customer.auto_renew = customer.auto_renew, new
    return {"status": "updated", "field": args.field, "old_value": old, "new_value": getattr(customer, args.field)}


_SLA = {"business": "4 hours", "pro": "24 hours", "free": "3 business days"}


async def _create_ticket(args: TicketArgs, ctx: ToolContext) -> dict:
    tier = ctx.session.customer.tier if ctx.session.customer else "free"
    priority = args.priority if args.priority in ("urgent", "high", "normal") else "normal"
    ticket = {
        "ticket_id": ctx.session.backend.next_ticket_id(),
        "status": "open",
        "queue": "human-specialists",
        "priority": priority,
        "category": args.category,
        "summary": args.summary,
        "customer_id": ctx.session.customer.id if ctx.session.customer else None,
        "first_response_sla": _SLA[tier],
        "created_via": ctx.mode,
    }
    ctx.session.escalations.append(ticket)
    return ticket


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec("verify_customer", "Verifying account", "Verify the customer's identity with their account email and billing ZIP code. Returns plan tier, billing cycle and contract status. Required before any account-specific tool.", VerifyArgs, _verify),
        ToolSpec("search_policies", "Searching knowledge base", "Search the support policies (filtered to the verified customer's plan). Use for any question about policies, prices, limits or how-to steps. Returns a confidence level.", SearchArgs, _search),
        ToolSpec("list_invoices", "Checking invoices", "List the verified customer's recent invoices, with duplicate-charge flags.", NoArgs, _list_invoices, needs_verified_customer=True, billing_system=True),
        ToolSpec("issue_refund", "Issuing refund", "Issue a refund on an invoice. The billing system enforces the refund policy and the agent's approval limit; it may return denied or requires_human.", RefundArgs, _issue_refund, needs_verified_customer=True, billing_system=True),
        ToolSpec("update_account_field", "Updating account", "Change an allowed account setting: billing_email, notifications (all|important|none) or auto_renew (true|false).", UpdateArgs, _update_account, needs_verified_customer=True, billing_system=True),
        ToolSpec("create_escalation_ticket", "Escalating to a human", "Hand the conversation to a human specialist with a structured ticket. Use when confidence is low, a request is above your limits, a system is down, or the customer asks for a person.", TicketArgs, _create_ticket),
    ]
}


def openai_chat_tools() -> list[dict]:
    return [
        {"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.json_schema()}}
        for s in TOOLS.values()
    ]


def realtime_tools() -> list[dict]:
    return [
        {"type": "function", "name": s.name, "description": s.description, "parameters": s.json_schema()}
        for s in TOOLS.values()
    ]


# --- executor -----------------------------------------------------------------------


@dataclass
class ToolOutcome:
    name: str
    ok: bool
    output: dict[str, Any]
    duration_ms: float
    attempts: int
    error: str | None = None

    def summary(self) -> str:
        o = self.output
        if not self.ok:
            return o.get("message", "failed")
        if self.name == "verify_customer":
            return f"{o.get('name')} · {o.get('tier')} · {o.get('contract_status')}" if o.get("verified") else "details didn't match"
        if self.name == "search_policies":
            n = len(o.get("results", []))
            return f"{o.get('confidence')} confidence · {n} source{'s' if n != 1 else ''} · top {o.get('top_score')}"
        if self.name == "list_invoices":
            return f"{len(o.get('invoices', []))} invoices"
        if self.name == "issue_refund":
            return f"{o.get('status')} · ${o.get('amount_usd', 0):.2f}" if o.get("status") == "refunded" else o.get("status", "")
        if self.name == "create_escalation_ticket":
            return f"{o.get('ticket_id')} · {o.get('priority')} · SLA {o.get('first_response_sla')}"
        return o.get("status", "done")


OnRetry = Callable[[str, int, str], Awaitable[None]]


async def _run_once(spec: ToolSpec, args: BaseModel, ctx: ToolContext, timeout_s: float) -> dict:
    if ctx.simulate == "tool_flaky" and ctx.attempt == 0:
        raise ToolFailure("timeout", f"{spec.label} timed out", retryable=True)
    if ctx.simulate == "tool_down" and spec.billing_system:
        raise ToolFailure("unavailable", "The billing system is not responding", retryable=True)
    try:
        return await asyncio.wait_for(spec.handler(args, ctx), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise ToolFailure("timeout", f"{spec.label} timed out", retryable=True) from exc


async def execute(
    name: str,
    raw_args: str | dict,
    ctx: ToolContext,
    *,
    timeout_s: float = 4.0,
    max_retries: int = 1,
    on_retry: OnRetry | None = None,
) -> ToolOutcome:
    started = time.perf_counter()
    spec = TOOLS.get(name)
    elapsed = lambda: (time.perf_counter() - started) * 1000  # noqa: E731
    if not spec:
        return ToolOutcome(name, False, {"error": "unknown_tool", "message": f"No tool named {name}"}, elapsed(), 0, "unknown_tool")
    try:
        data = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        args = spec.args.model_validate(data or {})
    except (ValueError, ValidationError) as exc:
        msg = f"Invalid arguments for {name}: {exc}"[:400]
        return ToolOutcome(name, False, {"error": "invalid_arguments", "message": msg}, elapsed(), 0, "invalid_arguments")
    if spec.needs_verified_customer and not ctx.session.customer:
        return ToolOutcome(
            name, False,
            {"error": "identity_not_verified", "message": "Verify the customer first with verify_customer (email + ZIP code)."},
            elapsed(), 0, "identity_not_verified",
        )

    attempt = 0
    while True:
        ctx.attempt = attempt
        try:
            output = await _run_once(spec, args, ctx, timeout_s)
            return ToolOutcome(name, True, output, elapsed(), attempt + 1)
        except ToolFailure as failure:
            if failure.retryable and attempt < max_retries:
                attempt += 1
                if on_retry:
                    await on_retry(name, attempt, failure.message)
                await asyncio.sleep(0.4 * attempt)
                continue
            return ToolOutcome(
                name, False,
                {
                    "error": failure.code,
                    "message": f"{failure.message} (after {attempt + 1} attempt{'s' if attempt else ''})",
                    "instruction": "Do not guess the result. Apologise, and offer to escalate to a human.",
                    "retryable": failure.retryable,
                },
                elapsed(), attempt + 1, failure.code,
            )
