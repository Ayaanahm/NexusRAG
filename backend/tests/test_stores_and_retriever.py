import os

import numpy as np
import pytest

from nexusrag.index import CorpusIndex
from nexusrag.retriever import HybridRetriever
from nexusrag.stores.memory import MemoryStore
from nexusrag.stores.qdrant import QdrantStore


def _stores(dim):
    yield MemoryStore(dim)
    yield QdrantStore(dim, collection="test")
    dsn = os.environ.get("NEXUS_TEST_PG_DSN")
    if dsn:
        from nexusrag.stores.pgvector import PgVectorStore

        s = PgVectorStore(dim, dsn=dsn, table="test_vectors")
        s.reset()
        yield s


def test_stores_agree_on_nearest_neighbours():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(200, 32)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"{i:016x}" for i in range(200)]
    q = vecs[17] + 0.01
    q /= np.linalg.norm(q)
    results = []
    for store in _stores(32):
        store.upsert(ids, vecs)
        assert store.count() == 200
        results.append([i for i, _ in store.search(q, 5)])
        store.delete(ids[:10])
        assert store.count() == 190
    assert all(r[0] == ids[17] for r in results)
    assert all(r == results[0] for r in results)


def test_memory_store_persists(tmp_path):
    p = tmp_path / "v.npz"
    s = MemoryStore(4, path=p)
    s.upsert(["000000000000000a"], np.eye(4, dtype=np.float32)[:1])
    assert MemoryStore(4, path=p).count() == 1


def test_hybrid_search_finds_identifier_and_reports_stage_ranks(embedder, sample_chunks):
    index = CorpusIndex(embedder, MemoryStore(embedder.dim))
    index.add(sample_chunks)
    r = HybridRetriever(index).search("HTTPException custom headers", k=3)
    top = r.hits[0]
    assert top.chunk.doc_id == "errors.md"
    assert top.bm25_rank is not None and top.fused_rank == 1
    assert {"bm25", "embed", "vector_search", "fusion", "total"} <= set(r.timings_ms)


def test_modes_and_bad_mode(embedder, sample_chunks):
    index = CorpusIndex(embedder, MemoryStore(embedder.dim))
    index.add(sample_chunks)
    ret = HybridRetriever(index)
    assert all(h.dense_rank is None for h in ret.search("cors", mode="bm25").hits)
    assert all(h.bm25_rank is None for h in ret.search("cors", mode="dense").hits)
    with pytest.raises(ValueError):
        ret.search("cors", mode="magic")


def test_reingesting_a_document_replaces_its_chunks(embedder, sample_chunks, tmp_path):
    index = CorpusIndex(embedder, MemoryStore(embedder.dim), data_dir=tmp_path)
    index.add(sample_chunks)
    n = index.stats()["chunks"]
    index.add([c for c in sample_chunks if c.doc_id == "cors.md"])
    assert index.stats()["chunks"] == n
    assert index.store.count() == n
    # reload from disk
    again = CorpusIndex(embedder, MemoryStore(embedder.dim, path=tmp_path / "vectors.npz"), data_dir=tmp_path)
    assert again.stats()["chunks"] == n
    assert index.remove_document("cors.md") > 0
    assert all(c.doc_id != "cors.md" for c in index.chunks)
