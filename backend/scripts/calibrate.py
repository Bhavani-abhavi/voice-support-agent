"""Calibrate the retrieval confidence thresholds for the active embedding model.

Embeds a small labelled set of in-scope and out-of-scope questions, prints the
top-1 similarity for each, and suggests thresholds: escalate below the highest
out-of-scope score, answer at or above the lowest in-scope score. Run it once per
embedding model and put the numbers in DEFAULT_THRESHOLDS (app/config.py).

    python scripts/calibrate.py            # uses OpenAI if OPENAI_API_KEY is set
    LLM_PROVIDER=mock python scripts/calibrate.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.main import Services  # noqa: E402

IN_SCOPE = [
    "I was charged twice this month",
    "can I get a refund on my annual plan",
    "what happens if my card payment fails",
    "how do I cancel my pro subscription",
    "my notes are not syncing on my phone",
    "how much does pro cost",
    "how do I export all my notes",
    "I lost my two factor device",
    "how fast does support reply on business",
    "can you delete my personal data",
    "what is a conflict copy",
    "how many devices can I use on free",
]
OUT_OF_SCOPE = [
    "recommend a good laptop for college",
    "what's the weather in Chicago tomorrow",
    "write me a poem about the ocean",
    "how do I file my taxes",
    "who won the world series",
    "can you book me a flight to Denver",
    "what is the capital of Australia",
    "help me with my calculus homework",
]


async def main() -> None:
    settings = get_settings()
    services = Services(settings)
    await services.startup()
    r = services.retriever

    async def top(q: str) -> float:
        res = await r.search(q, None)
        return res.top_score

    ins = [(q, await top(q)) for q in IN_SCOPE]
    outs = [(q, await top(q)) for q in OUT_OF_SCOPE]
    print(f"embedding model: {r.embedder.name}\n")
    for label, rows in (("IN SCOPE", ins), ("OUT OF SCOPE", outs)):
        print(label)
        for q, s in sorted(rows, key=lambda x: x[1]):
            print(f"  {s:.3f}  {q}")
    lo_in, hi_out = min(s for _, s in ins), max(s for _, s in outs)
    print(f"\nlowest in-scope {lo_in:.3f} | highest out-of-scope {hi_out:.3f} | margin {lo_in - hi_out:+.3f}")
    print(f"suggested: escalate_below={hi_out + 0.01:.2f}, answer_at={max(lo_in, hi_out + 0.05):.2f}")
    print(f"current:   escalate_below={r.escalate_below:.2f}, answer_at={r.answer_at:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
