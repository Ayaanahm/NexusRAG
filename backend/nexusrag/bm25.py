"""Okapi BM25 over a sparse term-document matrix.

Scoring a query is one sparse row-slice and a sum, so it stays in the low
milliseconds for tens of thousands of chunks without a search server.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import sparse

from .textproc import code_tokens


class BM25Index:
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = code_tokens,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.vocab: dict[str, int] = {}
        self._weights: sparse.csr_matrix | None = None  # terms x docs, BM25 term weights
        self.n_docs = 0

    def fit(self, texts: Sequence[str]) -> BM25Index:
        rows, cols, vals = [], [], []
        doc_len = np.zeros(len(texts), dtype=np.float32)
        for d, text in enumerate(texts):
            counts: dict[int, int] = {}
            toks = self.tokenizer(text)
            doc_len[d] = len(toks)
            for t in toks:
                tid = self.vocab.setdefault(t, len(self.vocab))
                counts[tid] = counts.get(tid, 0) + 1
            for tid, c in counts.items():
                rows.append(tid)
                cols.append(d)
                vals.append(c)
        self.n_docs = len(texts)
        tf = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(len(self.vocab), self.n_docs),
        )
        df = np.diff(tf.indptr).astype(np.float32)  # docs containing each term
        idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg_len = float(doc_len.mean()) if self.n_docs else 1.0
        norm = self.k1 * (1 - self.b + self.b * doc_len / max(avg_len, 1e-9))

        # Precompute per-(term, doc) BM25 weights so a query is a row gather + sum.
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.col]) * idf[tf.row]
        self._weights = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("BM25Index.fit() has not been called")
        ids = [self.vocab[t] for t in self.tokenizer(query) if t in self.vocab]
        if not ids:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self._weights[ids].sum(axis=0)).ravel()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        if k >= len(s):
            top = np.argsort(-s)
        else:
            top = np.argpartition(-s, k)[:k]
            top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]
