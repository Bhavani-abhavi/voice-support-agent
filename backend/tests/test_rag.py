import pytest

from app.config import Settings
from app.llm.embeddings import HashEmbedder
from app.rag.chunker import chunk_markdown, load_policy_chunks
from app.rag.retriever import Retriever
from app.rag.store import VectorStore

DOC = """---
title: Refunds
tiers: [free, pro, business]
---

# Refunds

## Duplicate charges

Duplicate charges are refunded in full.

## Pro monthly
<!-- tiers: pro -->

Pro monthly can be refunded within 14 days.

## Business
<!-- tiers: business -->

Business refunds up to $500.
"""


def test_chunks_never_cross_sections_and_carry_tiers():
    chunks = chunk_markdown("refunds", DOC)
    assert [c.section for c in chunks] == ["Duplicate charges", "Pro monthly", "Business"]
    assert chunks[1].tiers == ("pro",)
    assert chunks[0].tiers == ("free", "pro", "business")
    assert chunks[1].embed_text.startswith("Refunds > Pro monthly")
    assert [c.id for c in chunks] == ["refunds:1", "refunds:2", "refunds:3"]


def test_long_section_splits_with_paragraph_overlap():
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(6))
    chunks = chunk_markdown("doc", f"# T\n\n## Long\n\n{body}", max_chars=400, overlap_chars=220)
    assert len(chunks) > 1
    assert all(len(c.body) <= 400 + 220 for c in chunks)
    # the last paragraph of one chunk opens the next
    assert chunks[1].body.split("\n\n")[0] == chunks[0].body.split("\n\n")[-1]


def test_all_policy_files_chunk():
    s = Settings(llm_provider="mock")
    chunks = load_policy_chunks(s.policies_dir, s.chunk_max_chars, s.chunk_overlap_chars)
    assert len(chunks) >= 20
    assert len({c.id for c in chunks}) == len(chunks)


@pytest.fixture()
async def retriever():
    s = Settings(llm_provider="mock")
    r = Retriever(HashEmbedder(), VectorStore("t"), s.thresholds, top_k=4)
    await r.index(load_policy_chunks(s.policies_dir, s.chunk_max_chars, s.chunk_overlap_chars))
    return r


async def test_in_scope_question_is_confident(retriever):
    res = await retriever.search("refund for a duplicate charge", "pro")
    assert res.confidence == "high"
    assert res.hits[0].chunk_id == "refunds:1"


async def test_out_of_scope_question_withholds_passages(retriever):
    res = await retriever.search("recommend a good laptop for college", None)
    assert res.confidence == "low"
    out = res.to_tool_output()
    assert out["results"] == [] and "escalation" in out["instruction"]


async def test_tier_filter_hides_other_plans(retriever):
    res = await retriever.search("Business plan refund limit account manager", "pro")
    assert all("business" not in h.tiers or "pro" in h.tiers for h in res.hits)
