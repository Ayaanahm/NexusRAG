"""Hybrid retrieval: BM25 + dense vectors -> fusion -> optional re-rank.

Every search returns, per hit, where it ranked in each stage and how long each
stage took. The API exposes this so the UI can show *why* a chunk was chosen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunking import Chunk
from .fusion import rrf, weighted
from .index import CorpusIndex
from .rerank import NoReranker, Reranker

MODES = ("hybrid", "bm25", "dense")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        c = self.chunk
        return {
            "id": c.id,
            "doc_id": c.doc_id,
            "title": c.title,
            "heading_path": c.heading_path,
            "url": c.url,
            "text": c.text,
            "score": round(self.score, 6),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "fused_rank": self.fused_rank,
            "rerank_score": self.rerank_score,
        }


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    mode: str = "hybrid"
    reranker: str = "none"


class HybridRetriever:
    def __init__(
        self,
        index: CorpusIndex,
        reranker: Reranker | None = None,
        fusion: str = "weighted",
        rrf_k: int = 60,
        dense_weight: float = 0.4,
        rrf_dense_weight: float = 1.0,
        bm25_candidates: int = 50,
        dense_candidates: int = 50,
        rerank_candidates: int = 30,
    ) -> None:
        self.index = index
        self.reranker = reranker or NoReranker()
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.rrf_dense_weight = rrf_dense_weight
        self.bm25_candidates = bm25_candidates
        self.dense_candidates = dense_candidates
        self.rerank_candidates = rerank_candidates

    @classmethod
    def from_settings(cls, index: CorpusIndex, reranker: Reranker, s) -> HybridRetriever:
        return cls(
            index,
            reranker,
            fusion=s.fusion,
            rrf_k=s.rrf_k,
            dense_weight=s.dense_weight,
            rrf_dense_weight=s.rrf_dense_weight,
            bm25_candidates=s.bm25_candidates,
            dense_candidates=s.dense_candidates,
            rerank_candidates=s.rerank_candidates,
        )

    def search(self, query: str, k: int = 8, mode: str = "hybrid", rerank: bool = True) -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        kw: list[tuple[str, float]] = []
        de: list[tuple[str, float]] = []
        if mode in ("hybrid", "bm25"):
            s = time.perf_counter()
            kw = self.index.keyword_search(query, self.bm25_candidates)
            t["bm25"] = (time.perf_counter() - s) * 1000
        if mode in ("hybrid", "dense"):
            s = time.perf_counter()
            qv = self.index.embedder.embed_query(query)
            t["embed"] = (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            de = self.index.dense_search(qv, self.dense_candidates)
            t["vector_search"] = (time.perf_counter() - s) * 1000

        bm25_rank = {cid: r for r, (cid, _) in enumerate(kw, start=1)}
        dense_rank = {cid: r for r, (cid, _) in enumerate(de, start=1)}

        s = time.perf_counter()
        if mode == "bm25":
            fused = kw
        elif mode == "dense":
            fused = de
        elif self.fusion == "weighted":
            fused = weighted(kw, de, self.dense_weight)
        else:
            fused = rrf(
                [[c for c, _ in kw], [c for c, _ in de]],
                k=self.rrf_k,
                weights=[1.0, self.rrf_dense_weight],
            )
        t["fusion"] = (time.perf_counter() - s) * 1000

        hits = [
            Hit(
                chunk=self.index.get(cid),
                score=score,
                bm25_rank=bm25_rank.get(cid),
                dense_rank=dense_rank.get(cid),
                fused_rank=r,
            )
            for r, (cid, score) in enumerate(fused, start=1)
            if self.index.get(cid) is not None
        ]

        use_rerank = rerank and not isinstance(self.reranker, NoReranker)
        if use_rerank and hits:
            s = time.perf_counter()
            head = hits[: self.rerank_candidates]
            by_id = {h.chunk.id: h for h in head}
            reranked = self.reranker.rerank(query, [h.chunk for h in head])
            new_head = []
            for chunk, score in reranked:
                h = by_id[chunk.id]
                h.rerank_score = round(score, 6)
                h.score = score
                new_head.append(h)
            hits = new_head + hits[self.rerank_candidates :]
            t["rerank"] = (time.perf_counter() - s) * 1000

        t["total"] = (time.perf_counter() - t0) * 1000
        return SearchResult(
            hits=hits[:k],
            timings_ms={key: round(v, 3) for key, v in t.items()},
            mode=mode,
            reranker=self.reranker.name if use_rerank else "none",
        )
