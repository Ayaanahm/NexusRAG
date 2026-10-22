import httpx

from nexusrag.chunking import Chunk
from nexusrag.index import CorpusIndex
from nexusrag.rerank import ClaudeReranker, CohereReranker, _Judgement, _Ranking
from nexusrag.retriever import HybridRetriever
from nexusrag.stores.memory import MemoryStore


def _chunks(n=3):
    return [
        Chunk(id=f"{i:016x}", doc_id=f"d{i}", title="", heading_path=[], text=f"passage {i}", ord=0) for i in range(n)
    ]


def test_cohere_reranker_maps_indices():
    def handler(request: httpx.Request):
        body = request.read()
        assert b'"query":"q"' in body.replace(b" ", b"")
        return httpx.Response(
            200, json={"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}
        )

    r = CohereReranker("key")
    r._client = httpx.Client(transport=httpx.MockTransport(handler))
    out = r.rerank("q", _chunks())
    assert [c.doc_id for c, _ in out] == ["d2", "d0"]


class _FakeMessages:
    def __init__(self, ranking):
        self.ranking = ranking
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs

        class Msg:
            parsed_output = self.ranking

        return Msg()


def _claude(ranking):
    r = ClaudeReranker.__new__(ClaudeReranker)
    r.model, r.name, r.max_chars = "m", "claude/m", 1500
    r._client = type("C", (), {"messages": _FakeMessages(ranking)})()
    return r


def test_claude_reranker_orders_by_score_and_ignores_bad_ids():
    ranking = _Ranking(judgements=[_Judgement(id=1, score=9), _Judgement(id=0, score=2), _Judgement(id=42, score=10)])
    r = _claude(ranking)
    out = r.rerank("q", _chunks())
    assert [c.doc_id for c, _ in out] == ["d1", "d0", "d2"]  # d2 was not graded -> last
    sent = r._client.messages.kwargs["messages"][0]["content"]
    assert "<passage>" in sent and "Question: q" in sent


def test_claude_reranker_falls_back_to_first_stage_order_when_unparsed():
    out = _claude(None).rerank("q", _chunks())
    assert [c.doc_id for c, _ in out] == ["d0", "d1", "d2"]


class _ReverseReranker:
    name = "reverse"

    def rerank(self, query, chunks):
        return [(c, float(i)) for i, c in enumerate(chunks)][::-1]


def test_retriever_applies_reranker_to_head_only(embedder, sample_chunks):
    index = CorpusIndex(embedder, MemoryStore(embedder.dim))
    index.add(sample_chunks)
    ret = HybridRetriever(index, _ReverseReranker(), rerank_candidates=2)
    base = HybridRetriever(index).search("HTTPException headers", k=5).hits
    hits = ret.search("HTTPException headers", k=5).hits
    assert [h.chunk.id for h in hits[:2]] == [base[1].chunk.id, base[0].chunk.id]
    assert [h.chunk.id for h in hits[2:]] == [h.chunk.id for h in base[2:]]
    assert hits[0].rerank_score is not None
