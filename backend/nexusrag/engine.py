"""Wires the configured components together. One Engine per process."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Settings
from .embeddings import make_embedder
from .generate import AnswerGenerator
from .index import CorpusIndex
from .rerank import make_reranker
from .retriever import HybridRetriever
from .stores import make_store

log = logging.getLogger(__name__)


@dataclass
class Engine:
    settings: Settings
    index: CorpusIndex
    retriever: HybridRetriever
    generator: AnswerGenerator | None


def build_engine(settings: Settings) -> Engine:
    embedder = make_embedder(settings.embedder)
    store = make_store(settings, embedder.dim)
    index = CorpusIndex(embedder, store, data_dir=settings.data_dir)
    reranker = make_reranker(settings)
    retriever = HybridRetriever.from_settings(index, reranker, settings)
    generator = (
        AnswerGenerator(settings.anthropic_api_key, settings.generation_model, settings.max_answer_tokens)
        if settings.anthropic_api_key
        else None
    )
    if generator is None:
        log.warning("ANTHROPIC_API_KEY is not set: /chat is disabled, /search still works")
    log.info("engine ready: %s, reranker=%s", index.stats(), reranker.name)
    return Engine(settings, index, retriever, generator)
