"""HTTP API.

uvicorn nexusrag.api:app --port 8000
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from .cache import Cache
from .chunking import chunk_markdown, chunk_plain
from .config import Settings, get_settings
from .context import extract_citations, pack_context
from .engine import Engine, build_engine

log = logging.getLogger("nexusrag")

MAX_UPLOAD_BYTES = 2_000_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else get_settings()
    if not hasattr(app.state, "engine"):
        app.state.engine = await run_in_threadpool(build_engine, settings)
    app.state.cache = Cache(settings.redis_url, settings.cache_ttl_seconds, settings.rate_limit_per_minute)
    yield


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="NexusRAG", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    if engine is not None:
        app.state.engine = engine
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    _routes(app)
    return app


# -- request models --------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=8, ge=1, le=50)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"
    rerank: bool = True


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    k: int = Field(default=8, ge=1, le=20)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"


def _engine(request: Request) -> Engine:
    return request.app.state.engine


def _cache(request: Request) -> Cache:
    return request.app.state.cache


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def _rate_limit(request: Request, cache: Cache = Depends(_cache)) -> None:
    if not await cache.allow(_client_id(request)):
        raise HTTPException(429, "Rate limit exceeded, try again in a minute")


def _routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health(engine: Engine = Depends(_engine), cache: Cache = Depends(_cache)):
        return {
            "status": "ok",
            "index": engine.index.stats(),
            "reranker": engine.retriever.reranker.name,
            "chat_enabled": engine.generator is not None,
            "cache": "up" if await cache.ping() else ("down" if cache.enabled else "disabled"),
        }

    @app.post("/search", dependencies=[Depends(_rate_limit)])
    async def search(req: SearchRequest, engine: Engine = Depends(_engine), cache: Cache = Depends(_cache)):
        key = Cache.key("search", req.model_dump(), engine.index.version, engine.retriever.reranker.name)
        cached = await cache.get_json(key)
        if cached is not None:
            return {**cached, "cached": True}
        res = await run_in_threadpool(engine.retriever.search, req.query, req.k, req.mode, req.rerank)
        body = {
            "hits": [h.to_dict() for h in res.hits],
            "timings_ms": res.timings_ms,
            "mode": res.mode,
            "reranker": res.reranker,
        }
        await cache.set_json(key, body)
        return {**body, "cached": False}

    @app.post("/chat", dependencies=[Depends(_rate_limit)])
    async def chat(req: ChatRequest, request: Request, engine: Engine = Depends(_engine)):
        if engine.generator is None:
            raise HTTPException(503, "Chat is disabled: set ANTHROPIC_API_KEY on the server")

        async def events():
            t0 = time.perf_counter()
            try:
                res = await run_in_threadpool(engine.retriever.search, req.question, max(req.k, 12), req.mode, True)
                ctx = pack_context(res.hits, engine.settings.context_token_budget)
                retrieval_ms = (time.perf_counter() - t0) * 1000
                yield {
                    "event": "sources",
                    "data": json.dumps(
                        {
                            "sources": [s.to_dict() for s in ctx.sources],
                            "context_tokens": ctx.tokens,
                            "dropped": ctx.dropped,
                            "timings_ms": res.timings_ms,
                        }
                    ),
                }
                answer: list[str] = []
                first_token_ms = None
                usage: dict = {}
                async for kind, payload in engine.generator.stream(
                    req.question, ctx, [t.model_dump() for t in req.history]
                ):
                    if await request.is_disconnected():
                        log.info("client disconnected mid-stream")
                        return
                    if kind == "token":
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - t0) * 1000
                        answer.append(payload["text"])
                        yield {"event": "token", "data": json.dumps(payload)}
                    else:
                        usage = payload
                valid, invalid = extract_citations("".join(answer), len(ctx.sources))
                yield {"event": "citations", "data": json.dumps({"cited": valid, "invalid": invalid})}
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "retrieval_ms": round(retrieval_ms, 1),
                            "first_token_ms": round(first_token_ms or 0, 1),
                            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                            "usage": usage,
                        }
                    ),
                }
            except Exception as e:  # noqa: BLE001 - report errors in-band; headers are already sent
                log.exception("chat failed")
                yield {"event": "error", "data": json.dumps({"message": type(e).__name__})}

        return EventSourceResponse(events(), ping=15)

    @app.get("/documents")
    async def documents(engine: Engine = Depends(_engine)):
        counts: dict[str, dict] = {}
        for c in engine.index.chunks:
            d = counts.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "url": c.url, "chunks": 0})
            d["chunks"] += 1
        return {"documents": sorted(counts.values(), key=lambda d: d["doc_id"])}

    def _require_admin(request: Request, x_admin_token: str | None = Header(default=None)):
        expected = request.app.state.settings.admin_token
        if not expected:
            raise HTTPException(403, "Uploads are disabled: set NEXUS_ADMIN_TOKEN on the server")
        if x_admin_token != expected:
            raise HTTPException(401, "Invalid admin token")

    @app.post("/documents", dependencies=[Depends(_require_admin)])
    async def upload(files: list[UploadFile] = File(...), engine: Engine = Depends(_engine)):
        added = []
        for f in files:
            name = (f.filename or "upload.md").replace("\\", "/").split("/")[-1]
            if not name.lower().endswith((".md", ".markdown", ".txt")):
                raise HTTPException(415, f"{name}: only .md and .txt are supported")
            raw = await f.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name}: larger than {MAX_UPLOAD_BYTES} bytes")
            text = raw.decode("utf-8", errors="replace")
            doc_id = f"uploads/{name}"
            chunks = (
                chunk_markdown(text, doc_id)
                if name.lower().endswith((".md", ".markdown"))
                else chunk_plain(text, doc_id, title=name)
            )
            n = await run_in_threadpool(engine.index.add, chunks)
            added.append({"doc_id": doc_id, "chunks": n})
        return {"added": added, "index": engine.index.stats()}

    @app.delete("/documents/{doc_id:path}", dependencies=[Depends(_require_admin)])
    async def delete(doc_id: str, engine: Engine = Depends(_engine)):
        n = await run_in_threadpool(engine.index.remove_document, doc_id)
        if not n:
            raise HTTPException(404, "No such document")
        return {"removed_chunks": n}


app = create_app()
