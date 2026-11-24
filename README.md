# Voice + Chat Customer Support Agent

A fullstack customer-support AI agent for a fictional note-taking SaaS, **Orbit Notes**. One agent core serves two interfaces, **streaming chat (SSE)** and **live voice (OpenAI Realtime over WebRTC)**. Both share the same session memory, tools and guardrails. The agent chains tools autonomously, refuses to guess when retrieval confidence is low, and streams every step to an observability drawer.

**Live demo:** _<add Render URL>_ · **Walkthrough:** _<add Loom URL>_


---

## What to try

| Scenario | Say or type | What it shows |
|---|---|---|
| Duplicate charge | "I was charged twice this month. I'm alex.rivera@example.com, ZIP 64111." | 4-tool chain: verify → search policies → list invoices → refund, with live step status |
| Policy question | "What happens to my notes if my payment fails?" | RAG answer with clickable citations, grounding badge |
| Out of scope | "Can you help me pick a laptop?" | Retrieval below threshold → passages withheld → escalation ticket |
| Above the agent's limit | "I'm priya.shah@example.com, ZIP 94107. Refund INV-2201, the full $12,000." | Code-enforced limit → `requires_human` → high-priority ticket |
| Outside refund window | "I'm sam.lee@example.com, ZIP 10001. Refund my annual plan." | Policy denial with the reason and the policy cited |
| Failure drills | Sidebar → *Simulate failure* | Flaky tool (retry succeeds), billing system down, LLM 429, LLM timeout |
| Mode switch | Start in chat, then press 🎙 Voice and continue | Voice agent already knows the verified customer and the conversation |

---

## Architecture

```
 Browser (React + TypeScript, Vite)
 ├─ Chat ── POST /api/chat ──────────── SSE: status · tool_start/end · token · verification · trace
 ├─ Voice ─ WebRTC audio ⇄ OpenAI Realtime (direct; no audio relay hop)
 │           └─ data channel events → function calls → POST /api/voice/tool  (executed on OUR server)
 └─ Observability drawer (per-turn traces: prompts, latency waterfall, tokens, cost, guardrail decisions)

 FastAPI backend
 ├─ agent/orchestrator.py   state machine around a tool-calling loop (max N steps), retries, rollback
 ├─ agent/verifier.py       post-generation grounding check (citation check + LLM judge, strict JSON)
 ├─ tools/registry.py       6 tools, pydantic-validated args, identity gate, refund limits, timeouts/retries
 ├─ rag/                    structure-aware chunker → embeddings → Qdrant; confidence-gated retriever
 ├─ voice.py                mints ephemeral Realtime secrets, primes voice with shared history, bridges tools
 ├─ session.py              one session = transcript + verified customer + traces (shared by chat & voice)
 ├─ observability.py        spans, LLM calls with exact prompts, token usage, cost estimates
 └─ ratelimit.py            per-IP sliding windows for chat, voice sessions, voice events
```

The modules map to the rubric's separation: **UI** (`frontend/`) · **orchestrator** (`agent/`) · **tools** (`tools/`) · **vector store** (`rag/`). Providers sit behind small interfaces (`llm/base.py`, `llm/embeddings.py`). Without an API key the app runs a deterministic **mock provider** and a hashing embedder, so the UI, tool chain, guardrails and all tests work offline.

### Agent loop

```
THINKING ──tool calls──▶ RUNNING_TOOLS ──▶ THINKING … (≤ 6 steps)
    └──text──▶ VERIFYING ──grounded──▶ DONE
                   └──ungrounded──▶ ESCALATING ──▶ DONE (handoff message + ticket)
```

* The model decides **which** tool to call next. The **rules** live in code, where the model can't talk its way past them:
  * Account tools return `identity_not_verified` until `verify_customer` succeeds. A wrong ZIP and an unknown email get the same message, so the agent never reveals whether an account exists.
  * Refund windows and approval limits are computed from the invoice and plan (Pro $150, Business $500, 14 / 30 / 90-day windows). The tool answers `refunded`, `denied` (with the reason) or `requires_human`.
  * Refunds are idempotent: a retry can't double-refund.
* Each transition streams to the UI ("Verifying account…", "Searching knowledge base…", "Issuing refund…"), including retries ("timed out, retrying").

### RAG and chunking

* **Corpus:** 7 Markdown policy docs (`backend/data/policies/`), 28 chunks.
* **Chunking choices:**
  * **Split on headings first.** A chunk never crosses a section. Fixed windows straddled "Pro monthly" and "Pro annual" and retrieved the wrong plan's rule with high similarity.
  * **Pack whole paragraphs** up to 900 characters; a paragraph is split by sentence only if it alone is too long.
  * **Paragraph overlap:** the previous paragraph carries forward when it's ≤ 220 characters, so a rule and its exception stay together.
  * **Breadcrumb prefix** ("Refunds and Billing Disputes > Pro annual plans") is embedded with the text. This separates near-identical sentences that belong to different plans.
  * **Plan-tier metadata** comes from front matter and can be overridden per section (`<!-- tiers: pro -->`). Retrieval **filters by the verified customer's plan**, which covers "retrieve policy docs based on the user's tier".
* **Store:** Qdrant. Embedded local mode by default (in-process, zero extra infrastructure for the demo). Set `QDRANT_URL` / `QDRANT_API_KEY` to use a Qdrant server or Qdrant Cloud with the same code.
* **Embeddings:** `text-embedding-3-small`, cached on disk by content hash.

### Confidence guardrails (two layers)

1. **Retrieval gate** (before generation), on top-1 cosine similarity:
   * **high** (≥ answer threshold): passages go to the model.
   * **borderline**: passages go to the model, but the answer is **buffered, not streamed**, until it passes verification.
   * **low** (< escalate threshold): passages are **withheld**. The tool returns only an instruction to hand off, so there's nothing half-relevant for the model to paraphrase into a confident wrong answer.
2. **Grounding verifier** (after generation):
   * **Deterministic citation check:** every `[doc:n]` cited must have been retrieved this turn.
   * **LLM judge:** temperature 0, strict JSON schema, returns `grounded` and `unsupported_claims`.
   * **On failure:** the draft is retracted in the UI, an escalation ticket is created, and the customer gets a handoff message. A verifier outage marks the answer `unverified`, and borderline answers fail closed.

Thresholds are a property of the embedding model, so they're calibrated rather than hand-picked. `scripts/calibrate.py` embeds labelled in-scope and out-of-scope questions and suggests the escalate/answer cut-offs (see `app/config.py: DEFAULT_THRESHOLDS`).

### Voice

* **Transport:** OpenAI Realtime API over **WebRTC**. Browser audio goes straight to OpenAI (lowest latency, no relay through our server). The server mints a **60-second ephemeral client secret**, so the API key never reaches the browser.
* **Turn-taking:** server VAD with `interrupt_response: true`. When the customer talks over the agent, the response is cancelled, local playback is cleared, and the bubble is marked *interrupted*.
* **Live feedback:** live transcripts for both sides (`gpt-4o-mini-transcribe` for the customer, audio transcript deltas for the agent), plus a two-colour waveform from Web Audio analysers.
* **Same brain:** function calls arrive on the data channel and are executed **by the backend** through the same registry, so identity checks, refund limits and the retrieval gate are identical in voice.
* **Shared memory:** voice sessions are primed with the shared chat history, and voice transcripts are written back to it, so you can switch modes mid-conversation.
* **Voice guardrail:** spoken audio can't be retracted, so voice relies on the **retrieval gate** (low confidence returns no passages and an instruction to escalate) rather than post-hoc verification. That's a deliberate latency/safety trade-off (see below).

### Observability

Every chat turn and every voice response produces a trace, streamed to the drawer:

* the **exact prompt** sent for each LLM call;
* a **latency waterfall** of LLM calls, tools and verification, with time-to-first-token or, for voice, speech-end → first audio;
* **token usage** (including audio vs text tokens for Realtime);
* **estimated cost** from a pricing table in `llm/pricing.py`;
* **retrieval scores** plotted against the thresholds, and the guardrail decision.

### Resilience and production readiness

* **LLM 429 / timeouts:**
  * bounded exponential backoff, shown to the user as a "retrying" state;
  * after the last retry, a typed `error` event, a **history rollback** so *Retry* starts clean, and a Retry button.
* **Tool failures:** per-tool timeout, one retry for transient errors, then a structured error the model must not paper over. The failure drill shows the agent escalating instead of inventing invoice data.
* **Rate limiting:** per-IP sliding windows on chat (20/min), voice session minting (6 per 10 min) and voice events.
* **Abuse and cost limits:**
  * 5-minute cap on voice sessions;
  * message length limit;
  * one in-flight turn per session (409);
  * session TTL and LRU eviction.
* **Deployment:** one Docker image (UI built in a Node stage, served by FastAPI) and a Render blueprint with `/api/health` as the health check.

---

## Key trade-offs

| Decision | Why | Cost |
|---|---|---|
| WebRTC straight to OpenAI for audio | Lowest latency; no audio relay to scale | Tool calls make a browser → server round trip (~50–150 ms) |
| Retrieval gate as the voice guardrail (no post-hoc verify) | Spoken words can't be un-said; a pre-generation gate adds no voice latency | Weaker than chat's two layers; a voice answer from a *high* match isn't double-checked |
| Stream first, verify at the end (chat, high confidence) | Fast first token on the common path | A bad draft is visible briefly before retraction. Borderline matches are buffered instead |
| Business rules in code, not prompt | The model can't be prompt-injected past refund limits or identity checks | More code per policy change |
| Embedded Qdrant, in-memory sessions | Zero extra infrastructure; one free instance | Single instance; Redis and a Qdrant server are the scale-out path (both behind interfaces) |
| Custom state machine instead of LangGraph | ~250 readable lines, explicit states, easy to trace | Fewer built-ins (checkpointing, graph viz) |

## Run locally

```bash
# backend
cd backend
python -m venv .venv && . .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp ../.env.example ../.env                          # add OPENAI_API_KEY (or leave empty for offline mock mode)
uvicorn app.main:app --reload --port 8000

# frontend (another terminal)
cd frontend && npm install && npm run dev            # http://localhost:5173 (proxies /api to :8000)
```

Or build once and let FastAPI serve the UI: `cd frontend && npm run build`, then open http://localhost:8000.

**Tests:** `cd backend && pytest`. 24 tests cover chunking, tier filtering, confidence gating, tool policies (identity, windows, limits, idempotency), retries, LLM failure rollback, rate limiting, the SSE event contract and shared chat/voice state. All of them run offline.

## Deploy (Render)

1. New → **Blueprint** → select this repo (uses `render.yaml`).
2. Set `OPENAI_API_KEY` in the service's environment.
3. Done. The free plan sleeps when idle, so the first request after a while takes about 50 s.

## What I'd do next

* Redis-backed sessions and a Qdrant server for horizontal scale; OpenTelemetry export for traces.
* An offline eval set (golden conversations) in CI that scores tool-chain correctness, escalation precision/recall and groundedness per release.
* Async transcript verification for voice, which flags ungrounded spoken answers for human review.
* Hybrid retrieval (BM25 + vectors) with a reranker for long-tail phrasing.

---

_Built with AI-assisted development (Claude Code) as a pair programmer. The architecture, policies and trade-offs were reviewed and are explained in the walkthrough._
