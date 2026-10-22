"""Combining keyword and vector result lists.

BM25 scores are unbounded and cosine similarities sit in [-1, 1], so they can't
be added raw. Two ways to combine them:

- Reciprocal Rank Fusion uses only ranks: a chunk ranked 1st by BM25 and 8th by
  vectors scores 1/(k+1) + 1/(k+8). It is the common default because it needs no
  score calibration.
- Weighted fusion min-max normalises each list's scores and mixes them.

On held-out SQuAD questions, equal-weight RRF scored slightly *below* BM25 alone
on MRR@10, while weighted fusion (vector weight 0.4, chosen on a separate
half of the data) beat both retrievers. See bench/README.md and
eval/tune_fusion.py. Both methods are available; ``fusion`` picks one.
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """(Weighted) Reciprocal Rank Fusion. Plain RRF trusts every list equally,
    which hurts when one retriever is much weaker than the other; ``weights``
    scales each list's contribution: score = sum(w_i / (k + rank_i))."""
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("need one weight per ranked list")
    scores: dict[str, float] = {}
    for w, ranking in zip(weights, ranked_lists):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def weighted(
    keyword: Sequence[tuple[str, float]],
    dense: Sequence[tuple[str, float]],
    dense_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """Convex combination of min-max normalised scores. A chunk missing from one
    list gets 0 from it."""

    def norm(lst: Sequence[tuple[str, float]]) -> dict[str, float]:
        if not lst:
            return {}
        vals = [s for _, s in lst]
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1.0
        return {i: (s - lo) / span for i, s in lst}

    kw, de = norm(keyword), norm(dense)
    ids = set(kw) | set(de)
    combined = {i: (1 - dense_weight) * kw.get(i, 0.0) + dense_weight * de.get(i, 0.0) for i in ids}
    return sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
