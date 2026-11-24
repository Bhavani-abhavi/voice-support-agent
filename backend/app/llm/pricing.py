"""Estimated API cost, USD per 1M tokens (OpenAI pricing page, checked 2026-11-24).

Estimates only: update this table when prices change. Unknown models cost 0 and
are flagged in the trace rather than guessed.
"""

from __future__ import annotations

# model: (input, cached_input, output)
TEXT_PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
    "mock-scripted-agent": (0.0, 0.0, 0.0),
}

# realtime model: (text_in, audio_in, cached_text_in, cached_audio_in, text_out, audio_out)
REALTIME_PRICES: dict[str, tuple[float, float, float, float, float, float]] = {
    "gpt-realtime-mini": (0.60, 10.00, 0.06, 0.30, 2.40, 20.00),
    "gpt-realtime": (4.00, 32.00, 0.40, 0.40, 16.00, 64.00),
}


def text_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> tuple[float, bool]:
    price = TEXT_PRICES.get(model)
    if not price:
        return 0.0, False
    p_in, p_cached, p_out = price
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1_000_000, True


def realtime_cost(model: str, usage: dict) -> tuple[float, bool]:
    price = REALTIME_PRICES.get(model)
    if not price:
        return 0.0, False
    t_in, a_in, ct_in, ca_in, t_out, a_out = price
    ind = usage.get("input_token_details") or {}
    outd = usage.get("output_token_details") or {}
    cached = ind.get("cached_tokens_details") or {}
    c_text, c_audio = cached.get("text_tokens", 0), cached.get("audio_tokens", 0)
    text_in = max(ind.get("text_tokens", 0) - c_text, 0)
    audio_in = max(ind.get("audio_tokens", 0) - c_audio, 0)
    cost = (
        text_in * t_in + audio_in * a_in + c_text * ct_in + c_audio * ca_in
        + outd.get("text_tokens", 0) * t_out + outd.get("audio_tokens", 0) * a_out
    )
    return cost / 1_000_000, True
