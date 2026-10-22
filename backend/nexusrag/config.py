"""Runtime configuration, read from environment variables (prefix ``NEXUS_``) or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEXUS_", env_file=".env", extra="ignore")

    # Where chunks and the in-process indexes are persisted.
    data_dir: Path = Path("./var")

    # Embeddings: "fastembed:<model>" (ONNX, downloaded from Hugging Face on first use),
    # "sentence-transformers:<model>", or "wordllama" (tiny static model bundled in
    # its pip wheel, for fully offline use; much weaker, see bench/README.md).
    embedder: str = "fastembed:BAAI/bge-small-en-v1.5"

    # Dense store: "memory" (exact numpy search), "qdrant" or "pgvector".
    vector_store: str = "memory"
    qdrant_url: str | None = None  # e.g. http://qdrant:6333; unset = embedded local mode
    qdrant_collection: str = "nexusrag"
    pg_dsn: str = "postgresql://rag:rag@localhost:5432/rag"

    # Re-ranking: "none", "fastembed:<model>" (e.g. fastembed:BAAI/bge-reranker-base),
    # "cross-encoder:<model>" (sentence-transformers), "cohere" or "claude".
    reranker: str = "none"
    rerank_candidates: int = 30

    # Retrieval. Defaults were chosen on held-out SQuAD questions with
    # eval/tune_fusion.py for bge-small-en-v1.5; retune if you change the embedder.
    fusion: str = "weighted"  # "weighted" (normalised scores) or "rrf" (ranks)
    dense_weight: float = 0.4  # fusion="weighted": share of the score from vectors
    rrf_k: int = 60
    rrf_dense_weight: float = 1.0  # fusion="rrf": weight of the vector list (BM25 = 1.0)
    bm25_candidates: int = 50
    dense_candidates: int = 50

    # Generation
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    cohere_api_key: str | None = Field(default=None, validation_alias="COHERE_API_KEY")
    generation_model: str = "claude-sonnet-5-5"
    rerank_model: str = "claude-haiku-5-5"
    max_answer_tokens: int = 1024
    context_token_budget: int = 6000

    # Serving
    redis_url: str | None = None  # e.g. redis://redis:6379/0; unset = no cache, no rate limit
    cache_ttl_seconds: int = 3600
    rate_limit_per_minute: int = 30
    cors_origins: list[str] = ["http://localhost:3000"]
    admin_token: str | None = None  # required for POST /documents when set


@lru_cache
def get_settings() -> Settings:
    return Settings()
