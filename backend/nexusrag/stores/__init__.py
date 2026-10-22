"""Dense vector stores. All share one small interface so the retriever, the API
and the benchmark can swap them with a config value."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    name: str

    def upsert(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...


def make_store(settings, dim: int) -> VectorStore:
    kind = settings.vector_store
    if kind == "memory":
        from .memory import MemoryStore

        return MemoryStore(dim, path=settings.data_dir / "vectors.npz")
    if kind == "qdrant":
        from .qdrant import QdrantStore

        return QdrantStore(
            dim,
            collection=settings.qdrant_collection,
            url=settings.qdrant_url,
            path=None if settings.qdrant_url else settings.data_dir / "qdrant",
        )
    if kind == "pgvector":
        from .pgvector import PgVectorStore

        return PgVectorStore(dim, dsn=settings.pg_dsn)
    raise ValueError(f"Unknown vector store {kind!r}")
