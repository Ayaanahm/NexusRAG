"""Qdrant: HNSW index, server mode (``url``) or embedded local mode (``path``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


def _point_id(chunk_id: str) -> int:
    # Chunk ids are 16 hex chars = an unsigned 64-bit int, which Qdrant accepts as an id.
    return int(chunk_id, 16)


class QdrantStore:
    name = "qdrant"

    def __init__(
        self,
        dim: int,
        collection: str = "nexusrag",
        url: str | None = None,
        path: Path | None = None,
        hnsw_ef: int = 128,
    ) -> None:
        self.dim = dim
        self.collection = collection
        self.hnsw_ef = hnsw_ef
        self._server = bool(url)
        if url:
            self.client = QdrantClient(url=url, timeout=30)
        # Embedded local mode is exact brute-force search (fine for tests and small
        # corpora); HNSW parameters only apply to a Qdrant server.
        elif path:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(location=":memory:")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
            )

    def upsert(self, ids: list[str], vectors: np.ndarray, batch: int = 512) -> None:
        for s in range(0, len(ids), batch):
            self.client.upsert(
                self.collection,
                points=models.Batch(
                    ids=[_point_id(i) for i in ids[s : s + batch]],
                    vectors=vectors[s : s + batch].tolist(),
                    payloads=[{"chunk_id": i} for i in ids[s : s + batch]],
                ),
                wait=True,
            )

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        res = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=k,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=self.hnsw_ef) if self._server else None,
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=[_point_id(i) for i in ids]),
            )

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def reset(self) -> None:
        self.client.delete_collection(self.collection)
        self._ensure_collection()
