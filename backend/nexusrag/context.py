"""Context-window management: turn ranked hits into a numbered, token-budgeted
set of sources for the prompt.

- Adjacent chunks of the same page are merged back into one passage (and the
  block they overlap on is de-duplicated), so the model reads continuous text.
- Sources are added in rank order until the token budget is spent; a passage
  that does not fit is skipped and smaller lower-ranked ones can still fill the
  remaining space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape

from .chunking import Chunk
from .retriever import Hit
from .textproc import estimate_tokens


@dataclass
class Source:
    n: int
    chunks: list[Chunk]
    best_rank: int
    # Where the source's best chunk ranked in each retrieval stage (for the UI).
    ranks: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.chunks[0].doc_id

    @property
    def title(self) -> str:
        c = self.chunks[0]
        return " > ".join(p for p in [c.title, *c.heading_path] if p)

    @property
    def url(self) -> str | None:
        return self.chunks[0].url

    @property
    def text(self) -> str:
        ordered = sorted(self.chunks, key=lambda c: c.ord)
        blocks = ordered[0].text.split("\n\n")
        for c in ordered[1:]:
            nxt = c.text.split("\n\n")
            # Drop the leading block(s) the chunker repeated as overlap.
            while nxt and blocks and nxt[0] == blocks[-1]:
                nxt = nxt[1:]
            blocks.extend(nxt)
        return "\n\n".join(blocks)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "doc_id": self.doc_id,
            "title": self.title,
            "url": self.url,
            "chunk_ids": [c.id for c in self.chunks],
            "text": self.text,
            "ranks": self.ranks,
        }


@dataclass
class PackedContext:
    sources: list[Source]
    tokens: int
    dropped: int = 0
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = []
        for s in self.sources:
            attrs = f'n="{s.n}" title="{escape(s.title, quote=True)}"'
            if s.url:
                attrs += f' url="{escape(s.url, quote=True)}"'
            parts.append(f"<source {attrs}>\n{s.text}\n</source>")
        return "\n\n".join(parts)


def pack_context(hits: list[Hit], budget_tokens: int = 6000) -> PackedContext:
    sources: list[Source] = []
    used = 0
    dropped = 0
    for rank, hit in enumerate(hits, start=1):
        c = hit.chunk
        # Merge into an existing source when it is the next/previous chunk of the same page.
        target = next(
            (s for s in sources if s.doc_id == c.doc_id and any(abs(x.ord - c.ord) == 1 for x in s.chunks)),
            None,
        )
        if target is not None:
            before = estimate_tokens(target.text)
            target.chunks.append(c)
            delta = estimate_tokens(target.text) - before
            if used + delta > budget_tokens:
                target.chunks.pop()
                dropped += 1
                continue
            used += delta
            continue
        cost = estimate_tokens(c.text) + estimate_tokens(c.title) + 20  # tag overhead
        if used + cost > budget_tokens:
            dropped += 1
            continue
        ranks = {"bm25": hit.bm25_rank, "dense": hit.dense_rank, "fused": hit.fused_rank}
        sources.append(Source(n=len(sources) + 1, chunks=[c], best_rank=rank, ranks=ranks))
        used += cost
    return PackedContext(sources=sources, tokens=used, dropped=dropped)


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) source numbers cited as [n] or [n, m] in an answer."""
    cited: list[int] = []
    for m in _CITE_RE.finditer(answer):
        for x in m.group(1).split(","):
            v = int(x)
            if v not in cited:
                cited.append(v)
    valid = [v for v in cited if 1 <= v <= n_sources]
    invalid = [v for v in cited if not 1 <= v <= n_sources]
    return valid, invalid
