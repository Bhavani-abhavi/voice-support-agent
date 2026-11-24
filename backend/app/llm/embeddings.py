"""Embedding backends behind one small interface.

``OpenAIEmbedder`` is the production path. ``HashEmbedder`` is a deterministic
bag-of-words embedder (feature hashing over stemmed unigrams + bigrams) used in mock
mode and in tests, so retrieval and the confidence guardrail can be exercised with
no network and no key.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Protocol

from openai import AsyncOpenAI

_STOP = set(
    "a an and are as at be but by can do does for from has have how i if in is it its "
    "me my of on or our so that the their them then there this to was we what when "
    "where which who why will with you your".split()
)


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


class HashEmbedder:
    name = "hash-bow-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        tokens = [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for feat in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str, cache_dir: Path | None = None) -> None:
        self.client = client
        self.name = model
        self.cache_path = (cache_dir / f"embeddings-{model}.json") if cache_dir else None
        self._cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except (OSError, ValueError):
                self._cache = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if self._key(t) not in self._cache]
        for start in range(0, len(missing), 96):
            batch = missing[start : start + 96]
            resp = await self.client.embeddings.create(model=self.name, input=batch)
            for text, item in zip(batch, resp.data):
                self._cache[self._key(text)] = item.embedding
        if missing and self.cache_path and len(texts) > 1:  # persist corpus embeddings only
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self._cache))
            except OSError:
                pass
        return [self._cache[self._key(t)] for t in texts]
