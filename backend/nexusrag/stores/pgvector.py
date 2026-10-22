"""Postgres + pgvector with an HNSW index on cosine distance."""

from __future__ import annotations

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


class PgVectorStore:
    name = "pgvector"

    def __init__(self, dim: int, dsn: str, table: str = "chunk_vectors", ef_search: int = 128) -> None:
        if not table.isidentifier():
            raise ValueError("table must be a plain identifier")
        self.dim = dim
        self.table = table
        self.ef_search = ef_search
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._ensure_table()

    def _ensure_table(self) -> None:
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} (id text PRIMARY KEY, embedding vector({self.dim}) NOT NULL)"
        )
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)"
        )

    def upsert(self, ids: list[str], vectors: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (id, embedding) VALUES (%s, %s) "
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
                list(zip(ids, (np.asarray(v, dtype=np.float32) for v in vectors))),
            )

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        with self.conn.transaction():
            self.conn.execute(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}")
            rows = self.conn.execute(
                f"SELECT id, 1 - (embedding <=> %s) AS score FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
                (vector.astype(np.float32), vector.astype(np.float32), k),
            ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.conn.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (ids,))

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def reset(self) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        self._ensure_table()
