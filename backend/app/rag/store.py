"""Qdrant vector store.

Defaults to Qdrant's embedded local mode (in-process, in memory), which needs no
extra service for a demo deploy. Set ``QDRANT_URL`` (and ``QDRANT_API_KEY``) to
point the same code at a Qdrant server or Qdrant Cloud.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .chunker import Chunk


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_title: str
    section: str
    body: str
    tiers: tuple[str, ...]
    score: float


class VectorStore:
    def __init__(self, collection: str, url: str = "", api_key: str = "") -> None:
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key or None) if url else QdrantClient(":memory:")
        self.backend = "qdrant-server" if url else "qdrant-embedded"

    def rebuild(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        dim = len(vectors[0])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        if self.backend == "qdrant-server":  # payload indexes are a no-op in local mode
            self.client.create_payload_index(self.collection, "tiers", qm.PayloadSchemaType.KEYWORD)
        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                vector=v,
                payload={
                    "chunk_id": c.id,
                    "doc_title": c.doc_title,
                    "section": c.section,
                    "body": c.body,
                    "tiers": list(c.tiers),
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, vector: list[float], tiers: list[str], limit: int) -> list[Hit]:
        flt = qm.Filter(must=[qm.FieldCondition(key="tiers", match=qm.MatchAny(any=tiers))])
        result = self.client.query_points(
            self.collection, query=vector, query_filter=flt, limit=limit, with_payload=True
        )
        hits = []
        for p in result.points:
            pl = p.payload or {}
            hits.append(
                Hit(
                    chunk_id=pl["chunk_id"],
                    doc_title=pl["doc_title"],
                    section=pl["section"],
                    body=pl["body"],
                    tiers=tuple(pl.get("tiers", [])),
                    score=float(p.score),
                )
            )
        return hits
