"""Exact (brute-force) cosine search in numpy. The reference the approximate
stores are measured against, and plenty fast below ~100k chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MemoryStore:
    name = "memory"

    def __init__(self, dim: int, path: Path | None = None) -> None:
        self.dim = dim
        self.path = path
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._m = np.zeros((0, dim), dtype=np.float32)
        if path and path.exists():
            data = np.load(path, allow_pickle=False)
            self._ids = [str(i) for i in data["ids"]]
            self._m = data["vectors"].astype(np.float32)
            self._pos = {i: n for n, i in enumerate(self._ids)}

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, ids=np.array(self._ids), vectors=self._m)

    def upsert(self, ids: list[str], vectors: np.ndarray) -> None:
        new_rows, new_ids = [], []
        for i, v in zip(ids, vectors):
            if i in self._pos:
                self._m[self._pos[i]] = v
            else:
                new_ids.append(i)
                new_rows.append(v)
        if new_rows:
            start = len(self._ids)
            self._m = np.vstack([self._m, np.asarray(new_rows, dtype=np.float32)])
            for n, i in enumerate(new_ids):
                self._pos[i] = start + n
            self._ids.extend(new_ids)
        self._save()

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        s = self._m @ vector.astype(np.float32)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(self._ids[i], float(s[i])) for i in top]

    def delete(self, ids: list[str]) -> None:
        drop = {self._pos[i] for i in ids if i in self._pos}
        if not drop:
            return
        keep = [n for n in range(len(self._ids)) if n not in drop]
        self._ids = [self._ids[n] for n in keep]
        self._m = self._m[keep]
        self._pos = {i: n for n, i in enumerate(self._ids)}
        self._save()

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids, self._pos = [], {}
        self._m = np.zeros((0, self.dim), dtype=np.float32)
        self._save()
