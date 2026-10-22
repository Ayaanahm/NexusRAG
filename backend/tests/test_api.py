import json

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from nexusrag.api import create_app
from nexusrag.config import Settings
from nexusrag.engine import Engine
from nexusrag.index import CorpusIndex
from nexusrag.retriever import HybridRetriever
from nexusrag.stores.memory import MemoryStore


class FakeGenerator:
    def __init__(self, reply="Add CORSMiddleware [1]. See also [7]."):
        self.reply = reply
        self.seen = None

    async def stream(self, question, ctx, history=None):
        self.seen = (question, ctx, history)
        for word in self.reply.split(" "):
            yield "token", {"text": word + " "}
        yield "usage", {"input_tokens": 10, "output_tokens": 5}


def _client(embedder, chunks, tmp_path, generator=None, **overrides):
    settings = Settings(data_dir=tmp_path, admin_token="s3cret", **overrides)
    index = CorpusIndex(embedder, MemoryStore(embedder.dim))
    index.add(chunks)
    engine = Engine(settings, index, HybridRetriever(index), generator)
    app = create_app(settings, engine)
    return TestClient(app), engine


def _sse(text):
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        ev = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                ev["event"] = line[6:].strip()
            elif line.startswith("data:"):
                ev["data"] = json.loads(line[5:].strip())
        if "event" in ev:
            events.append(ev)
    return events


def test_health_and_search(embedder, sample_chunks, tmp_path):
    client, _ = _client(embedder, sample_chunks, tmp_path)
    with client:
        h = client.get("/health").json()
        assert h["index"]["documents"] == 3 and h["chat_enabled"] is False and h["cache"] == "disabled"
        r = client.post("/search", json={"query": "CORSMiddleware allow_origins", "k": 2}).json()
        assert r["hits"][0]["doc_id"] == "cors.md"
        assert r["hits"][0]["url"].startswith("https://docs.example.com/cors/")
        assert r["cached"] is False


def test_search_validation(embedder, sample_chunks, tmp_path):
    client, _ = _client(embedder, sample_chunks, tmp_path)
    with client:
        assert client.post("/search", json={"query": ""}).status_code == 422
        assert client.post("/search", json={"query": "x", "k": 500}).status_code == 422
        assert client.post("/search", json={"query": "x", "mode": "magic"}).status_code == 422


def test_chat_disabled_without_key(embedder, sample_chunks, tmp_path):
    client, _ = _client(embedder, sample_chunks, tmp_path)
    with client:
        assert client.post("/chat", json={"question": "cors?"}).status_code == 503


def test_chat_streams_sources_tokens_and_checks_citations(embedder, sample_chunks, tmp_path):
    gen = FakeGenerator()
    client, _ = _client(embedder, sample_chunks, tmp_path, generator=gen)
    with client:
        r = client.post(
            "/chat",
            json={"question": "How do I allow my frontend origin?", "history": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        events = _sse(r.text)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "sources" and kinds[-2:] == ["citations", "done"]
    assert "".join(e["data"]["text"] for e in events if e["event"] == "token").startswith("Add CORSMiddleware")
    sources = events[0]["data"]["sources"]
    assert sources and sources[0]["n"] == 1
    cites = next(e for e in events if e["event"] == "citations")["data"]
    assert cites == {"cited": [1], "invalid": [7]}
    assert gen.seen[2] == [{"role": "user", "content": "hi"}]


def test_upload_requires_admin_token_and_is_searchable(embedder, sample_chunks, tmp_path):
    client, _ = _client(embedder, sample_chunks, tmp_path)
    files = {
        "files": ("runbook.md", b"# Runbook\n\nRestart the flux capacitor with fluxctl restart.\n", "text/markdown")
    }
    with client:
        assert client.post("/documents", files=files).status_code == 401
        r = client.post("/documents", files=files, headers={"x-admin-token": "s3cret"})
        assert r.status_code == 200 and r.json()["added"][0]["doc_id"] == "uploads/runbook.md"
        hit = client.post("/search", json={"query": "fluxctl restart"}).json()["hits"][0]
        assert hit["doc_id"] == "uploads/runbook.md"
        bad = {"files": ("x.exe", b"MZ", "application/octet-stream")}
        assert client.post("/documents", files=bad, headers={"x-admin-token": "s3cret"}).status_code == 415
        assert client.delete("/documents/uploads/runbook.md", headers={"x-admin-token": "s3cret"}).status_code == 200
        assert client.delete("/documents/uploads/runbook.md", headers={"x-admin-token": "s3cret"}).status_code == 404


def test_uploads_disabled_without_admin_token(embedder, sample_chunks, tmp_path):
    client, engine = _client(embedder, sample_chunks, tmp_path)
    engine.settings.admin_token = None
    client.app.state.settings.admin_token = None
    with client:
        files = {"files": ("a.md", b"# A\n\nb\n", "text/markdown")}
        assert client.post("/documents", files=files, headers={"x-admin-token": "x"}).status_code == 403


@pytest.fixture
def fake_redis(monkeypatch):
    server = fakeredis.FakeServer()
    monkeypatch.setattr("redis.asyncio.from_url", lambda *a, **k: fakeredis.aioredis.FakeRedis(server=server))


def test_redis_cache_and_rate_limit(embedder, sample_chunks, tmp_path, fake_redis):
    client, _ = _client(embedder, sample_chunks, tmp_path, redis_url="redis://fake", rate_limit_per_minute=3)
    with client:
        body = {"query": "background tasks email"}
        assert client.post("/search", json=body).json()["cached"] is False
        assert client.post("/search", json=body).json()["cached"] is True
        assert client.post("/search", json=body).status_code == 200
        assert client.post("/search", json=body).status_code == 429
        assert client.get("/health").json()["cache"] == "up"
