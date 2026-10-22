"""Runs the real Anthropic SDK against a canned streaming response, so the
generator's use of the streaming API is checked without a network call."""

import json

import anthropic

try:  # anthropic>=2 is built on httpx2; older releases on httpx
    import httpx2 as httpx
except ImportError:
    import httpx

from nexusrag.chunking import Chunk
from nexusrag.context import pack_context
from nexusrag.generate import AnswerGenerator, build_messages
from nexusrag.retriever import Hit


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


CANNED = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 42, "output_tokens": 0},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Use CORSMiddleware"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " [1]."}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 7},
    },
    {"type": "message_stop"},
]


def _ctx():
    c = Chunk(
        id="0000000000000001",
        doc_id="cors.md",
        title="CORS",
        heading_path=["Use CORSMiddleware"],
        text="Add CORSMiddleware.",
        url="https://x/cors/",
    )
    return pack_context([Hit(c, 1.0)])


async def test_generator_streams_tokens_and_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(CANNED))

    client = anthropic.AsyncAnthropic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    gen = AnswerGenerator("test", "claude-test", client=client)
    out = [e async for e in gen.stream("How do I enable CORS?", _ctx())]
    assert [p["text"] for k, p in out if k == "token"] == ["Use CORSMiddleware", " [1]."]
    assert out[-1] == (
        "usage",
        {"model": "claude-test", "input_tokens": 42, "output_tokens": 7, "stop_reason": "end_turn"},
    )
    body = seen["body"]
    assert body["stream"] is True and body["model"] == "claude-test"
    assert (
        '<source n="1" title="CORS &gt; Use CORSMiddleware" url="https://x/cors/">' in body["messages"][-1]["content"]
    )


def test_history_is_trimmed_and_alternates():
    hist = [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "dangling"},
    ]
    msgs = build_messages("q2", _ctx(), hist)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "q1" and msgs[-1]["content"].endswith("Question: q2")
