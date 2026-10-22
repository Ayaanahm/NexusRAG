"""The searchable corpus: chunk catalog + BM25 + a dense vector store, kept in sync."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk
from .embeddings import Embedder
from .stores import VectorStore


class CorpusIndex:
    def __init__(self, embedder: Embedder, store: VectorStore, data_dir: Path | None = None) -> None:
        self.embedder = embedder
        self.store = store
        self.data_dir = data_dir
        self.chunks: list[Chunk] = []
        self.pos: dict[str, int] = {}
        self.bm25 = BM25Index()
        self.version = 0  # bumped on every write; part of the search cache key
        self._lock = threading.RLock()
        if data_dir and (data_dir / "chunks.jsonl").exists():
            self._load()

    # -- persistence -----------------------------------------------------------
    @property
    def _chunks_path(self) -> Path | None:
        return self.data_dir / "chunks.jsonl" if self.data_dir else None

    def _load(self) -> None:
        with self._chunks_path.open() as f:
            self.chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        self._rebuild_keyword_index()
        # Re-embed if the vector store is out of sync (new store, or switched embedder).
        missing = self.store.count() != len(self.chunks)
        meta = self.data_dir / "embedder.txt"
        if missing or not meta.exists() or meta.read_text().strip() != self.embedder.name:
            self.store.reset()
            self._embed_and_store(self.chunks)
            meta.write_text(self.embedder.name)

    def _save(self) -> None:
        if not self._chunks_path:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._chunks_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self._chunks_path)
        (self.data_dir / "embedder.txt").write_text(self.embedder.name)

    # -- writes ----------------------------------------------------------------
    def _rebuild_keyword_index(self) -> None:
        self.pos = {c.id: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25Index().fit([c.index_text for c in self.chunks])

    def _embed_and_store(self, chunks: list[Chunk], batch: int = 1024) -> None:
        for s in range(0, len(chunks), batch):
            part = chunks[s : s + batch]
            vecs = self.embedder.embed_documents([c.index_text for c in part])
            self.store.upsert([c.id for c in part], vecs)

    def add(self, chunks: list[Chunk]) -> int:
        """Add or replace whole documents. Any existing chunks of the same
        ``doc_id`` are removed first, so re-ingesting a changed page is safe."""
        with self._lock:
            doc_ids = {c.doc_id for c in chunks}
            stale = [c.id for c in self.chunks if c.doc_id in doc_ids]
            if stale:
                self.store.delete(stale)
            self.chunks = [c for c in self.chunks if c.doc_id not in doc_ids] + chunks
            self._embed_and_store(chunks)
            self._rebuild_keyword_index()
            self._save()
            self.version += 1
            return len(chunks)

    def remove_document(self, doc_id: str) -> int:
        with self._lock:
            stale = [c.id for c in self.chunks if c.doc_id == doc_id]
            if stale:
                self.store.delete(stale)
                self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
                self._rebuild_keyword_index()
                self._save()
                self.version += 1
            return len(stale)

    # -- reads -----------------------------------------------------------------
    def get(self, chunk_id: str) -> Chunk | None:
        i = self.pos.get(chunk_id)
        return self.chunks[i] if i is not None else None

    def neighbours(self, chunk: Chunk) -> tuple[Chunk | None, Chunk | None]:
        i = self.pos.get(chunk.id)
        if i is None:
            return None, None
        prev = self.chunks[i - 1] if i > 0 and self.chunks[i - 1].doc_id == chunk.doc_id else None
        nxt = self.chunks[i + 1] if i + 1 < len(self.chunks) and self.chunks[i + 1].doc_id == chunk.doc_id else None
        return prev, nxt

    def keyword_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return [(self.chunks[i].id, s) for i, s in self.bm25.search(query, k)]

    def dense_search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        return [(i, s) for i, s in self.store.search(query_vector, k) if i in self.pos]

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "documents": len({c.doc_id for c in self.chunks}),
            "embedder": self.embedder.name,
            "vector_store": self.store.name,
            "version": self.version,
        }
