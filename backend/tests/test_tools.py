from datetime import date

import pytest

from app.config import Settings
from app.llm.embeddings import HashEmbedder
from app.rag.retriever import Retriever
from app.rag.store import VectorStore
from app.session import Session
from app.tools.mock_data import MockBackend
from app.tools.registry import ToolContext, execute


@pytest.fixture()
def ctx():
    s = Settings(llm_provider="mock")
    session = Session(id="t", backend=MockBackend(date(2026, 11, 24)))
    return ToolContext(session=session, retriever=Retriever(HashEmbedder(), VectorStore("t"), s.thresholds))


async def verify(ctx, email="alex.rivera@example.com", zip_code="64111"):
    out = await execute("verify_customer", {"email": email, "zip_code": zip_code}, ctx)
    ctx.session.policy_checked = True  # tests below exercise refund rules, not the lookup gate
    return out


async def test_refund_blocked_until_policy_checked(ctx):
    await execute("verify_customer", {"email": "alex.rivera@example.com", "zip_code": "64111"}, ctx)
    out = await execute("issue_refund", {"invoice_id": "INV-1043", "amount_usd": 12, "reason": "duplicate"}, ctx)
    assert out.output["status"] == "policy_not_checked"


async def test_account_tools_require_verification(ctx):
    out = await execute("list_invoices", {}, ctx)
    assert not out.ok and out.error == "identity_not_verified"


async def test_wrong_zip_does_not_verify_or_leak(ctx):
    out = await verify(ctx, zip_code="00000")
    assert out.output["verified"] is False and ctx.session.customer is None
    unknown = await verify(ctx, email="nobody@example.com")
    assert unknown.output["message"] == out.output["message"]


async def test_duplicate_charge_refund_then_idempotent(ctx):
    await verify(ctx)
    first = await execute("issue_refund", {"invoice_id": "INV-1043", "amount_usd": 12, "reason": "duplicate"}, ctx)
    assert first.output["status"] == "refunded"
    again = await execute("issue_refund", {"invoice_id": "INV-1043", "amount_usd": 12, "reason": "duplicate"}, ctx)
    assert again.output["status"] == "denied"  # retries can't double-refund


async def test_refund_window_enforced_in_code(ctx):
    await verify(ctx, "sam.lee@example.com", "10001")
    out = await execute("issue_refund", {"invoice_id": "INV-3100", "amount_usd": 120, "reason": "asked"}, ctx)
    assert out.output["status"] == "denied" and "30 days" in out.output["reason"]


async def test_business_limit_requires_human(ctx):
    await verify(ctx, "priya.shah@example.com", "94107")
    big = await execute("issue_refund", {"invoice_id": "INV-2201", "amount_usd": 12000, "reason": "asked"}, ctx)
    assert big.output["status"] == "requires_human"
    small = await execute("issue_refund", {"invoice_id": "INV-2202", "amount_usd": 300, "reason": "billed in error"}, ctx)
    assert small.output["status"] == "refunded"


async def test_invalid_arguments_are_returned_not_raised(ctx):
    await verify(ctx)
    out = await execute("issue_refund", '{"invoice_id": "INV-1043"}', ctx)
    assert not out.ok and out.error == "invalid_arguments"


async def test_flaky_tool_recovers_on_retry(ctx):
    ctx.simulate = "tool_flaky"
    retries = []

    async def on_retry(name, attempt, msg):
        retries.append(attempt)

    out = await execute("verify_customer", {"email": "alex.rivera@example.com", "zip_code": "64111"}, ctx, on_retry=on_retry)
    assert out.ok and out.attempts == 2 and retries == [1]


async def test_down_system_fails_cleanly(ctx):
    await verify(ctx)
    ctx.simulate = "tool_down"
    out = await execute("list_invoices", {}, ctx, max_retries=1)
    assert not out.ok and out.attempts == 2 and "instruction" in out.output
