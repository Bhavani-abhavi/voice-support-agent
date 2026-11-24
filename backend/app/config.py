"""Runtime configuration.

Every value has a working default except the OpenAI key. Without a key the app runs
in ``mock`` mode: a deterministic scripted agent plus a hashing embedder, so the UI,
the tool chain, the guardrails and the tests all work offline.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Retrieval thresholds are properties of the embedding model, not of the app: cosine
# scores from text-embedding-3-small sit in a different range from the offline
# hashing embedder. Calibrated with scripts/calibrate.py (see README).
DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    # backend: (escalate_below, answer_at_or_above)
    "openai": (0.30, 0.42),
    "hash": (0.165, 0.20),  # offline bag-of-words embedder; overlaps more, so it escalates more
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        extra="ignore",
    )

    # --- providers -------------------------------------------------------------
    llm_provider: str = "auto"  # auto | openai | mock
    openai_api_key: str = ""
    chat_model: str = "gpt-4.1-mini"
    verifier_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    realtime_model: str = "gpt-realtime-mini"
    realtime_voice: str = "marin"
    transcribe_model: str = "gpt-4o-mini-transcribe"

    # --- vector store ----------------------------------------------------------
    qdrant_url: str = ""  # empty -> embedded Qdrant (local mode, in memory)
    qdrant_api_key: str = ""
    qdrant_collection: str = "support_policies"
    policies_dir: Path = BACKEND_DIR / "data" / "policies"
    embedding_cache_dir: Path = BACKEND_DIR / "data" / ".cache"

    # --- RAG + guardrails ------------------------------------------------------
    chunk_max_chars: int = 900
    chunk_overlap_chars: int = 220
    retrieval_top_k: int = 4
    escalate_threshold: float | None = None
    answer_threshold: float | None = None

    # --- agent loop + resilience -----------------------------------------------
    max_agent_steps: int = 6
    llm_timeout_s: float = 20.0
    llm_max_retries: int = 2
    tool_timeout_s: float = 4.0
    tool_max_retries: int = 1

    # --- abuse protection --------------------------------------------------------
    rate_limit_chat_per_min: int = 20
    rate_limit_voice_sessions_per_10min: int = 6
    rate_limit_voice_events_per_min: int = 240
    voice_max_session_s: int = 300
    max_message_chars: int = 2000
    session_ttl_s: int = 7200
    max_sessions: int = 2000

    # --- serving ---------------------------------------------------------------
    static_dir: Path = BACKEND_DIR / "static"
    cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"

    @property
    def provider(self) -> str:
        if self.llm_provider in ("openai", "mock"):
            return self.llm_provider
        return "openai" if self.openai_api_key else "mock"

    @property
    def embedding_backend(self) -> str:
        return "openai" if self.provider == "openai" else "hash"

    @property
    def thresholds(self) -> tuple[float, float]:
        low, high = DEFAULT_THRESHOLDS[self.embedding_backend]
        return (
            self.escalate_threshold if self.escalate_threshold is not None else low,
            self.answer_threshold if self.answer_threshold is not None else high,
        )

    @property
    def voice_enabled(self) -> bool:
        return self.provider == "openai"


@lru_cache
def get_settings() -> Settings:
    return Settings()
