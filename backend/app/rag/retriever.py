"""Retrieval with a confidence gate (guardrail layer 1 of 2).

Every knowledge-base lookup is scored before the model sees it:

* ``high``       top score >= answer threshold: chunks go to the model.
* ``borderline`` between the thresholds: chunks go to the model, but the answer is
                 buffered (not streamed) until the grounding verifier passes it.
* ``low``        below the escalate threshold: the chunks are WITHHELD. The model
                 gets an instruction to hand off instead, so there is nothing
                 half-relevant for it to paraphrase into a confident wrong answer.

Layer 2 is the post-generation grounding verifier in ``agent/verifier.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..llm.embeddings import Embedder
from .chunker import Chunk
from .store import Hit, VectorStore


@dataclass
class RetrievalResult:
    query: str
    tiers: list[str]
    hits: list[Hit]
    confidence: str  # high | borderline | low
    top_score: float
    thresholds: tuple[float, float]
    latency_ms: float
    extra: dict = field(default_factory=dict)

    def to_tool_output(self) -> dict:
        if self.confidence == "low":
            return {
                "confidence": "low",
                "top_score": round(self.top_score, 3),
                "results": [],
                "instruction": (
                    "No policy matched with enough confidence. Do not answer from general "
                    "knowledge. Tell the customer you will connect them with a specialist "
                    "and call create_escalation_ticket."
                ),
            }
        return {
            "confidence": self.confidence,
            "top_score": round(self.top_score, 3),
            "results": [
                {
                    "id": h.chunk_id,
                    "source": f"{h.doc_title} > {h.section}",
                    "score": round(h.score, 3),
                    "text": h.body,
                }
                for h in self.hits
            ],
            "instruction": "Answer only from these results and cite the ids you used, like [refunds:2].",
        }


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, thresholds: tuple[float, float], top_k: int = 4) -> None:
        self.embedder = embedder
        self.store = store
        self.escalate_below, self.answer_at = thresholds
        self.top_k = top_k
        self.chunk_count = 0

    async def index(self, chunks: list[Chunk]) -> None:
        vectors = await self.embedder.embed([c.embed_text for c in chunks])
        self.store.rebuild(chunks, vectors)
        self.chunk_count = len(chunks)

    def classify(self, top_score: float) -> str:
        if top_score >= self.answer_at:
            return "high"
        if top_score >= self.escalate_below:
            return "borderline"
        return "low"

    async def search(self, query: str, tier: str | None) -> RetrievalResult:
        started = time.perf_counter()
        tiers = ["all", tier] if tier else ["all", "free", "pro", "business"]
        [vector] = await self.embedder.embed([query])
        hits = self.store.search(vector, tiers, self.top_k)
        top = hits[0].score if hits else 0.0
        confidence = self.classify(top)
        # Keep only hits that clear the floor, so a strong first hit doesn't drag
        # unrelated neighbours into the prompt.
        kept = [h for h in hits if h.score >= self.escalate_below][:3]
        return RetrievalResult(
            query=query,
            tiers=tiers,
            hits=kept,
            confidence=confidence,
            top_score=top,
            thresholds=(self.escalate_below, self.answer_at),
            latency_ms=(time.perf_counter() - started) * 1000,
            extra={"all_scores": [round(h.score, 3) for h in hits]},
        )
