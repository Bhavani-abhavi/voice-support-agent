"""Structure-aware chunking for Markdown support policies.

Why not fixed-size windows: a policy answer lives inside one section ("Pro annual
plans"), and a window that straddles two sections retrieves the wrong plan's rule
with high similarity. So the chunker:

1. splits on headings first, so a chunk never crosses a section boundary;
2. packs whole paragraphs up to ``max_chars``, splitting a paragraph by sentence
   only when it alone is too long;
3. carries the previous paragraph forward as overlap when it is short enough,
   so a rule and its exception stay together;
4. prefixes every chunk with its breadcrumb ("Refunds > Pro annual plans") before
   embedding, which disambiguates near-identical sentences across plans;
5. attaches plan-tier metadata (document front matter, overridable per section with
   ``<!-- tiers: pro -->``) so retrieval can filter by the verified customer's plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
_TIERS_COMMENT = re.compile(r"^<!--\s*tiers:\s*(.*?)\s*-->$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    section: str
    body: str
    tiers: tuple[str, ...]
    order: int

    @property
    def embed_text(self) -> str:
        return f"{self.doc_title} > {self.section}\n\n{self.body}"


@dataclass
class _Section:
    title: str
    tiers: tuple[str, ...]
    lines: list[str] = field(default_factory=list)


def _parse_list(value: str) -> tuple[str, ...]:
    value = value.strip().strip("[]")
    return tuple(t.strip().lower() for t in value.split(",") if t.strip())


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def _sections(body: str, doc_title: str, doc_tiers: tuple[str, ...]) -> list[_Section]:
    sections: list[_Section] = []
    current = _Section(title="Overview", tiers=doc_tiers)
    for raw in body.splitlines():
        line = raw.rstrip()
        heading = _HEADING.match(line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            if level == 1:
                continue  # the document title; already in the breadcrumb
            if any(l.strip() for l in current.lines):
                sections.append(current)
            current = _Section(title=title, tiers=doc_tiers)
            continue
        tiers = _TIERS_COMMENT.match(line.strip())
        if tiers:
            current.tiers = _parse_list(tiers.group(1))
            continue
        current.lines.append(line)
    if any(l.strip() for l in current.lines):
        sections.append(current)
    return sections


def _paragraphs(lines: list[str]) -> list[str]:
    paras, buf = [], []
    for line in lines:
        if line.strip():
            buf.append(line.strip())
        elif buf:
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    return paras


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces, buf = [], ""
    for sentence in _SENTENCE_END.split(paragraph):
        if buf and len(buf) + 1 + len(sentence) > max_chars:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def chunk_markdown(doc_id: str, text: str, max_chars: int = 900, overlap_chars: int = 220) -> list[Chunk]:
    meta, body = parse_front_matter(text)
    doc_title = meta.get("title", doc_id.replace("-", " ").title())
    doc_tiers = _parse_list(meta.get("tiers", "all")) or ("all",)

    chunks: list[Chunk] = []
    for section in _sections(body, doc_title, doc_tiers):
        units = [p for para in _paragraphs(section.lines) for p in _split_long(para, max_chars)]
        packed: list[str] = []
        buf: list[str] = []
        for unit in units:
            if buf and sum(len(u) + 1 for u in buf) + len(unit) > max_chars:
                packed.append(" ".join(buf) if len(buf) == 1 else "\n\n".join(buf))
                carry = buf[-1] if len(buf[-1]) <= overlap_chars else None
                buf = [carry] if carry else []
            buf.append(unit)
        if buf:
            packed.append("\n\n".join(buf))
        for body_text in packed:
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{len(chunks) + 1}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section=section.title,
                    body=body_text,
                    tiers=section.tiers,
                    order=len(chunks),
                )
            )
    return chunks


def load_policy_chunks(policies_dir: Path, max_chars: int, overlap_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(policies_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8"), max_chars, overlap_chars))
    return chunks
