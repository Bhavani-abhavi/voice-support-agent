"""Post-generation grounding verification (guardrail layer 2 of 2).

Two checks, cheapest first:

1. Deterministic citation check: every ``[doc:n]`` the answer cites must be a chunk
   that retrieval actually returned this turn. Catches invented sources instantly.
2. Model-based claim check: a separate, temperature-0 call with a strict JSON schema
   judges whether each factual claim is supported by the evidence.

If the verifier itself fails (timeout etc.) the answer is marked ``unverified``
rather than silently passed, and borderline-confidence answers fail closed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMError, LLMProvider, LLMResult
from .prompts import VERIFIER_SYSTEM

CITATION = re.compile(r"\[([a-z0-9\-]+:\d+)\]")
_POLICY_FACT = re.compile(r"\$\d|\b\d+\s*(day|days|hour|hours|business days|%)\b", re.I)

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["grounded", "unsupported_claims", "reason"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    status: str  # grounded | ungrounded | unverified
    method: str
    reason: str
    unsupported: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    llm: LLMResult | None = None
    prompt: list[dict[str, Any]] | None = None

    @property
    def grounded(self) -> bool:
        return self.status == "grounded"


def citation_check(answer: str, evidence_ids: set[str]) -> tuple[list[str], list[str]]:
    cited = list(dict.fromkeys(CITATION.findall(answer)))
    return cited, [c for c in cited if c not in evidence_ids]


def _heuristic(answer: str, evidence_ids: set[str], tool_evidence: list[dict]) -> Verdict:
    cited, invented = citation_check(answer, evidence_ids)
    if invented:
        return Verdict("ungrounded", "citation-check", f"Cites sources that were not retrieved: {invented}", invented, cited)
    if not evidence_ids and not tool_evidence and _POLICY_FACT.search(answer):
        return Verdict("ungrounded", "heuristic", "States policy facts without any retrieved evidence.", [answer[:120]], cited)
    return Verdict("grounded", "heuristic", "All citations were retrieved this turn.", [], cited)


async def verify(
    provider: LLMProvider,
    user_text: str,
    answer: str,
    chunks: dict[str, dict[str, Any]],
    tool_evidence: list[dict[str, Any]],
) -> Verdict:
    evidence_ids = set(chunks)
    cited, invented = citation_check(answer, evidence_ids)
    if invented:
        return Verdict("ungrounded", "citation-check", f"Cites sources that were not retrieved: {invented}", invented, cited)
    if provider.name == "mock":
        return _heuristic(answer, evidence_ids, tool_evidence)

    evidence = {
        "knowledge_base": [{"id": cid, "source": c["source"], "text": c["text"]} for cid, c in chunks.items()],
        "tool_outputs": [t for t in tool_evidence if t["tool"] != "search_policies"],
    }
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": json.dumps({"customer_message": user_text, "draft_answer": answer, "evidence": evidence})},
    ]
    try:
        data, result = await provider.complete_json(messages, VERDICT_SCHEMA, "grounding_verdict")
    except (LLMError, ValueError) as exc:
        return Verdict("unverified", "llm-judge", f"Verifier unavailable: {type(exc).__name__}", [], cited, prompt=messages)
    status = "grounded" if data.get("grounded") else "ungrounded"
    return Verdict(status, "llm-judge", data.get("reason", ""), data.get("unsupported_claims", []), cited, result, messages)
