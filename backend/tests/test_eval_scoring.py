import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from run_answers import Claim, Judgement, aggregate, score  # noqa: E402
from run_retrieval import paired_bootstrap, per_query_metrics  # noqa: E402


def test_retrieval_metrics():
    m = per_query_metrics(["x", "gold", "y"], {"gold"})
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["mrr@10"] == 0.5
    assert per_query_metrics(["x"], {"gold"})["mrr@10"] == 0


def test_bootstrap_interval_contains_mean():
    mean, lo, hi = paired_bootstrap([1, 0, 1, 1], [0, 0, 1, 0])
    assert lo <= mean <= hi and mean == 0.5


def test_answer_scoring_and_aggregate():
    j = Judgement(
        claims=[Claim(claim="a", supported=True, source_numbers=[1]), Claim(claim="b", supported=False)],
        answers_the_question=True,
        says_sources_insufficient=False,
    )
    row = score(j, "A [1]. B [4].", n_sources=2)
    assert row["faithfulness"] == 0.5 and row["invalid_citations"] == 1 and row["has_citation"]
    row.update(first_token_ms=100, total_ms=900)
    agg = aggregate([row])
    assert agg["claims_supported_pct"] == 50.0 and agg["invalid_citations_total"] == 1
