"""Text embedders. All return L2-normalised float32 rows, so cosine similarity
is a dot product in every vector store."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class WordLlamaEmbedder:
    """Static token embeddings distilled from LLM input layers
    (github.com/dleemiller/WordLlama). Weights ship inside the pip wheel, so this
    works offline and embeds thousands of chunks per second on one CPU core.
    Quality is well below transformer encoders; it is the zero-setup default."""

    def __init__(self, dim: int = 256) -> None:
        from wordllama import WordLlama

        self._ensure_bundled_tokenizer(WordLlama)
        self._model = WordLlama.load(dim=dim, disable_download=True)
        self.dim = dim
        self.name = f"wordllama-l2-supercat-{dim}"

    @staticmethod
    def _ensure_bundled_tokenizer(cls) -> None:
        # wordllama 0.4 bundles its tokenizer under ``tokenizers/`` but looks for it
        # under ``tokenizer/`` and then tries to download it. Copy it to the cache
        # location it checks so loading never needs the network.
        import wordllama

        name = "l2_supercat_tokenizer_config.json"
        bundled = Path(wordllama.__file__).parent / "tokenizers" / name
        cache_dir = cls.get_file_path("tokenizer", None)
        if bundled.exists() and not (cache_dir / name).exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(bundled, cache_dir / name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(self._model.embed(texts, norm=True, batch_size=256))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    """Any sentence-transformers model, e.g. BAAI/bge-small-en-v1.5.
    Requires ``pip install nexusrag[local-models]`` and Hugging Face access."""

    # Models trained with an instruction prefix for queries.
    QUERY_PREFIXES = {
        "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
        "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
        "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    }

    def __init__(self, model_id: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model_id
        self._query_prefix = self.QUERY_PREFIXES.get(model_id, "")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._model.encode(texts, batch_size=16, normalize_embeddings=True))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self._query_prefix + text])[0]


class FastEmbedEmbedder:
    """ONNX Runtime models via Qdrant's fastembed (no PyTorch), e.g.
    BAAI/bge-small-en-v1.5. Downloads the model from Hugging Face on first use."""

    # Memory grows with batch size x passage length: on the FastAPI docs, batches of
    # 64 long passages peaked at 2.3 GB RSS, batches of 16 at 0.75 GB.
    def __init__(self, model_id: str, threads: int | None = None, batch_size: int = 16) -> None:
        from fastembed import TextEmbedding

        self.batch_size = batch_size

        self._model = TextEmbedding(model_id, threads=threads)
        self.name = model_id
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))
        # fastembed does not add BGE's retrieval instruction to queries; we do.
        self._query_prefix = SentenceTransformerEmbedder.QUERY_PREFIXES.get(model_id, "")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack(list(self._model.embed(texts, batch_size=self.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self._query_prefix + text])[0]


def make_embedder(spec: str) -> Embedder:
    if spec == "wordllama":
        return WordLlamaEmbedder()
    if spec.startswith("fastembed:"):
        return FastEmbedEmbedder(spec.split(":", 1)[1])
    if spec.startswith("sentence-transformers:"):
        return SentenceTransformerEmbedder(spec.split(":", 1)[1])
    raise ValueError(f"Unknown embedder {spec!r}")
