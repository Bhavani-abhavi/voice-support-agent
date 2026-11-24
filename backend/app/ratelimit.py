"""Sliding-window rate limiter keyed by client IP and route group.

In-process, which matches the single-instance deploy. Behind a proxy (Render,
Railway) the client IP is the first hop of X-Forwarded-For.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.requests import Request


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def check(self, key: str, group: str, limit: int, window_s: float) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until a slot frees up."""
        now = time.monotonic()
        hits = self._hits[(key, group)]
        while hits and now - hits[0] > window_s:
            hits.popleft()
        if len(hits) >= limit:
            return round(window_s - (now - hits[0]), 1)
        hits.append(now)
        return None
