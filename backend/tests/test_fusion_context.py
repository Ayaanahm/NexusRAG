import pytest

from nexusrag.chunking import Chunk
from nexusrag.context import extract_citations, pack_context
from nexusrag.fusion import rrf, weighted
from nexusrag.retriever import Hit


def test_rrf_rewards_agreement():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["a"] == fused["b"]
    assert fused["a"] > fused["c"] > 0


def test_weighted_rrf_zero_weight_reproduces_first_list():
    fused = [x for x, _ in rrf([["a", "b"], ["c", "a"]], weights=[1.0, 0.0])]
    assert fused[:2] == ["a", "b"]


def test_rrf_needs_one_weight_per_list():
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


def test_weighted_fusion_normalises_scores():
    out = dict(weighted([("a", 30.0), ("b", 10.0)], [("b", 0.9), ("c", 0.1)], dense_weight=0.5))
    assert out["a"] == pytest.approx(0.5)
    assert out["b"] == pytest.approx(0.5)
    assert out["c"] == pytest.approx(0.0)


def _chunk(doc, ord_, text, title="T"):
    return Chunk(id=f"{doc}{ord_}", doc_id=doc, title=title, heading_path=[], text=text, ord=ord_)


def test_adjacent_chunks_merge_and_overlap_is_removed():
    a0 = _chunk("d", 0, "first\n\nshared")
    a1 = _chunk("d", 1, "shared\n\nsecond")
    ctx = pack_context([Hit(a1, 1.0), Hit(a0, 0.9)], budget_tokens=1000)
    assert len(ctx.sources) == 1
    assert ctx.sources[0].text == "first\n\nshared\n\nsecond"


def test_budget_skips_large_passage_but_keeps_smaller_ones():
    big = _chunk("big", 0, "x" * 4000)
    small = _chunk("small", 0, "tiny")
    ctx = pack_context([Hit(big, 1.0), Hit(small, 0.5)], budget_tokens=200)
    assert [s.doc_id for s in ctx.sources] == ["small"]
    assert ctx.dropped == 1


def test_render_escapes_attributes():
    c = _chunk("d", 0, "body", title='Say "hi" <b>')
    out = pack_context([Hit(c, 1.0)]).render()
    assert 'title="Say &quot;hi&quot; &lt;b&gt;"' in out


def test_extract_citations():
    valid, invalid = extract_citations("Use X [1]. Also Y [2, 3] and [9].", n_sources=3)
    assert valid == [1, 2, 3]
    assert invalid == [9]
