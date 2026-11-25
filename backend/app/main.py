"""HTTP surface: SSE chat, voice session + tool bridge, session state, static UI."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from .agent.orchestrator import AgentOrchestrator
from .config import Settings, get_settings
from .llm.embeddings import HashEmbedder, OpenAIEmbedder
from .llm.mock_provider import MockProvider
from .llm.openai_provider import OpenAIProvider
from .rag.chunker import load_policy_chunks
from .rag.retriever import Retriever
from .rag.store import VectorStore
from .ratelimit import RateLimiter, client_ip
from .schemas import ChatIn, TranscriptIn, UsageIn, VoiceSessionIn, VoiceToolIn
from .session import SessionStore
from .tools.mock_data import DEMO_ACCOUNTS
from .voice import VoiceError, VoiceService

log = logging.getLogger("support_agent")

SAMPLE_PROMPTS = [
    {"label": "Charged twice (full tool chain)", "text": "Hi, I think I was charged twice this month. Can I get a refund? I'm alex.rivera@example.com, ZIP 64111."},
    {"label": "Policy question", "text": "What happens to my notes if my payment fails?"},
    {"label": "Not in the knowledge base", "text": "Do you have a public API for developers?"},
    {"label": "Unrelated question", "text": "Can you help me pick a good laptop for college?"},
    {"label": "Above agent limit", "text": "I'm priya.shah@example.com, ZIP 94107. Please refund invoice INV-2201, the full $12,000."},
    {"label": "Outside refund window", "text": "I'm sam.lee@example.com, ZIP 10001. I want a refund for my annual plan."},
]


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sessions = SessionStore(settings.session_ttl_s, settings.max_sessions)
        self.limiter = RateLimiter()
        if settings.provider == "openai":
            client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)
            self.llm = OpenAIProvider(client, settings.chat_model, settings.verifier_model, settings.llm_timeout_s)
            embedder = OpenAIEmbedder(client, settings.embedding_model, settings.embedding_cache_dir)
        else:
            self.llm = MockProvider()
            embedder = HashEmbedder()
        self.store = VectorStore(settings.qdrant_collection, settings.qdrant_url, settings.qdrant_api_key)
        self.retriever = Retriever(embedder, self.store, settings.thresholds, settings.retrieval_top_k)
        self.agent = AgentOrchestrator(self.llm, self.retriever, settings)
        self.voice = VoiceService(settings, self.retriever)
        self.ready = False

    async def startup(self) -> None:
        chunks = load_policy_chunks(self.settings.policies_dir, self.settings.chunk_max_chars, self.settings.chunk_overlap_chars)
        await self.retriever.index(chunks)
        self.ready = True
        log.info("indexed %d chunks into %s (provider=%s)", len(chunks), self.store.backend, self.settings.provider)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    services = Services(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await services.startup()
        yield

    app = FastAPI(title="Customer Support AI Agent", lifespan=lifespan)
    app.state.services = services
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","), allow_methods=["*"], allow_headers=["*"])

    def limit(request: Request, group: str, n: int, window: float) -> None:
        wait = services.limiter.check(client_ip(request), group, n, window)
        if wait is not None:
            raise HTTPException(429, detail={"error": "rate_limited", "message": f"Too many requests. Try again in {wait:.0f}s.", "retry_after": wait}, headers={"Retry-After": str(int(wait) + 1)})

    def get_session(session_id: str):
        session = services.sessions.get(session_id)
        if not session:
            raise HTTPException(404, detail={"error": "session_not_found", "message": "Session expired. Refresh to start a new one."})
        return session

    # -- meta ---------------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": services.ready, "provider": settings.provider, "chunks": services.retriever.chunk_count}

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        low, high = settings.thresholds
        return {
            "provider": settings.provider,
            "chat_model": services.llm.chat_model,
            "voice_enabled": settings.voice_enabled,
            "realtime_model": settings.realtime_model if settings.voice_enabled else None,
            "vector_store": services.store.backend,
            "embedding_model": services.retriever.embedder.name,
            "thresholds": {"escalate_below": low, "answer_at": high},
            "demo_accounts": DEMO_ACCOUNTS,
            "sample_prompts": SAMPLE_PROMPTS,
            "voice_max_session_s": settings.voice_max_session_s,
            "rate_limits": {"chat_per_min": settings.rate_limit_chat_per_min, "voice_sessions_per_10min": settings.rate_limit_voice_sessions_per_10min},
        }

    # -- sessions -------------------------------------------------------------------

    @app.post("/api/sessions")
    async def create_session() -> dict[str, str]:
        return {"session_id": services.sessions.create().id}

    @app.get("/api/sessions/{session_id}")
    async def session_state(session_id: str) -> dict[str, Any]:
        s = get_session(session_id)
        return {
            "session_id": s.id,
            "customer": s.customer.profile() if s.customer else None,
            "transcript": s.transcript(limit=100),
            "escalations": s.escalations,
            "traces": s.traces,
        }

    # -- chat (SSE) -----------------------------------------------------------------


    @app.post("/api/chat")
    async def chat(body: ChatIn, request: Request) -> StreamingResponse:
        limit(request, "chat", settings.rate_limit_chat_per_min, 60)
        if len(body.message) > settings.max_message_chars:
            raise HTTPException(413, detail={"error": "message_too_long", "message": f"Messages are limited to {settings.max_message_chars} characters."})
        session = get_session(body.session_id)
        if session.lock.locked():
            raise HTTPException(409, detail={"error": "turn_in_progress", "message": "Still answering your last message."})

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def emit(event: str, data: dict[str, Any]) -> None:
            await queue.put(f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n")

        async def run() -> None:
            async with session.lock:
                try:
                    await services.agent.run_turn(session, body.message.strip(), emit, body.simulate)
                except Exception:  # noqa: BLE001 - last-resort guard: never leave the stream hanging
                    log.exception("turn failed")
                    await emit("error", {"code": "internal_error", "message": "Something went wrong on our side. Please try again.", "retryable": True})
                finally:
                    await emit("done", {})
                    await queue.put(None)

        task = asyncio.create_task(run())

        async def stream() -> AsyncIterator[str]:
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # -- voice ----------------------------------------------------------------------


    @app.post("/api/voice/session")
    async def voice_session(body: VoiceSessionIn, request: Request) -> dict[str, Any]:
        limit(request, "voice_session", settings.rate_limit_voice_sessions_per_10min, 600)
        session = get_session(body.session_id)
        try:
            return await services.voice.mint(session)
        except VoiceError as err:
            raise HTTPException(err.status, detail={"error": "voice_unavailable", "message": err.message}) from err


    @app.post("/api/voice/tool")
    async def voice_tool(body: VoiceToolIn, request: Request) -> dict[str, Any]:
        limit(request, "voice_events", settings.rate_limit_voice_events_per_min, 60)
        return await services.voice.run_tool(get_session(body.session_id), body.call_id, body.name, body.arguments)


    @app.post("/api/voice/transcript")
    async def voice_transcript(body: TranscriptIn, request: Request) -> dict[str, bool]:
        limit(request, "voice_events", settings.rate_limit_voice_events_per_min, 60)
        services.voice.record_transcript(get_session(body.session_id), body.role, body.text, body.interrupted)
        return {"ok": True}


    @app.post("/api/voice/usage")
    async def voice_usage(body: UsageIn, request: Request) -> dict[str, Any]:
        limit(request, "voice_events", settings.rate_limit_voice_events_per_min, 60)
        return services.voice.record_usage(get_session(body.session_id), body.usage, body.latency, body.user_text, body.status)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": "http_error", "message": str(exc.detail)}
        return JSONResponse(detail, status_code=exc.status_code, headers=exc.headers)

    # -- static UI ------------------------------------------------------------------

    static = settings.static_dir
    if (static / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            candidate = static / path
            if path and candidate.is_file() and static.resolve() in candidate.resolve().parents:
                return FileResponse(candidate)
            return FileResponse(static / "index.html")

    return app


app = create_app()
