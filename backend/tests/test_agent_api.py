from app.agent.verifier import citation_check


def names(events, kind):
    return [d["name"] for e, d in events if e == kind]


def test_refund_chain_streams_status_and_grounds(chat):
    ev = chat("I was charged twice this month, refund please. I'm alex.rivera@example.com, ZIP 64111.")
    assert names(ev, "tool_start") == ["verify_customer", "search_policies", "list_invoices", "issue_refund"]
    assert all(d["ok"] for e, d in ev if e == "tool_end")
    final = next(d for e, d in ev if e == "final")
    assert "refunded $12.00" in final["text"] and final["replaced"] is False
    verification = next(d for e, d in ev if e == "verification")
    assert verification["status"] == "grounded" and verification["citations"]
    trace = next(d for e, d in ev if e == "trace")
    assert trace["outcome"] == "answered" and trace["ttft_ms"] is not None
    assert {s["kind"] for s in trace["spans"]} >= {"llm", "tool"}
    assert ev[-1][0] == "done"


def test_memory_carries_across_turns(chat):
    chat("I was charged twice. I'm alex.rivera@example.com, ZIP 64111.")
    ev = chat("Can you also refund INV-0987?")
    # already verified: no second verify_customer
    assert "verify_customer" not in names(ev, "tool_start")


def test_unrelated_question_declines_without_ticket(chat):
    ev = chat("Can you recommend a laptop for college?")
    assert names(ev, "tool_start") == ["search_policies"]
    assert not any(e == "escalation" for e, _ in ev)
    assert "only help with Orbit Notes" in next(d for e, d in ev if e == "final")["text"]


def test_in_domain_question_missing_from_kb_escalates(chat):
    ev = chat("Do you have a public API for developers?")
    assert names(ev, "tool_start") == ["search_policies", "create_escalation_ticket"]
    assert any(e == "escalation" for e, _ in ev)
    assert next(d for e, d in ev if e == "trace")["outcome"] == "escalated"


def test_rate_limit_is_retried_with_visible_status(chat):
    ev = chat("What happens if my payment fails?", simulate="llm_rate_limit")
    assert any(e == "status" and d["state"] == "retrying" for e, d in ev)
    assert any(e == "final" for e, _ in ev)


def test_llm_timeout_ends_with_typed_error_and_rollback(client, chat):
    ev = chat("How do I export my notes?", simulate="llm_timeout")
    err = next(d for e, d in ev if e == "error")
    assert err["retryable"] is True
    state = client.get(f"/api/sessions/{chat.session_id}").json()
    assert state["transcript"] == []  # rolled back; a retry starts clean


def test_citation_check_catches_invented_sources():
    cited, invented = citation_check("Refunds take 3 days [refunds:9].", {"refunds:1"})
    assert invented == ["refunds:9"]


def test_rate_limiter_returns_429(client):
    sid = client.post("/api/sessions").json()["session_id"]
    codes = [client.post("/api/chat", json={"session_id": sid, "message": "hi"}).status_code for _ in range(22)]
    assert 429 in codes


def test_voice_disabled_without_key(client):
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post("/api/voice/session", json={"session_id": sid})
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["message"]


def test_voice_tools_share_session_state(client):
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post("/api/voice/tool", json={"session_id": sid, "call_id": "c1", "name": "verify_customer",
                                             "arguments": '{"email":"alex.rivera@example.com","zip_code":"64111"}'})
    assert r.json()["ok"]
    client.post("/api/voice/transcript", json={"session_id": sid, "role": "user", "text": "I was charged twice"})
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["customer"]["name"] == "Alex Rivera"
    assert state["transcript"][-1] == {"role": "user", "text": "I was charged twice", "mode": "voice"}


def test_unknown_session_is_404(client):
    r = client.post("/api/chat", json={"session_id": "nope", "message": "hi"})
    assert r.status_code == 404
