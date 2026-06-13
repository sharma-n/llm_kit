"""invoke_stream: SSE decoding, tool-call reassembly, retry, validation."""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from llm_kit.errors import ValidationError
from llm_kit.llm.client import LLMClient
from llm_kit.llm.streaming import (
    StreamAssembler,
    StreamEnd,
    TextChunk,
    ToolCallStarted,
)
from llm_kit.messages.streaming import (
    FinishFragment,
    MetaFragment,
    TextFragment,
    ToolCallFragment,
    UsageFragment,
)
from llm_kit.messages.tools import ToolDefinition
from llm_kit.messages.types import Message

from llm_kit.config.llm import MessageFormat

from conftest import (
    anthropic_sse,
    anthropic_text_stream,
    gemini_sse,
    gemini_text_stream,
    make_config,
    mock_client,
    openai_text_stream,
    sse_bytes,
    stream_mock_client,
)


def _make_llm(client: httpx.AsyncClient) -> LLMClient:
    return LLMClient(make_config(), client=client, owns_client=True)


def _make_anthropic_llm(client: httpx.AsyncClient) -> LLMClient:
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.ANTHROPIC
    return LLMClient(cfg, client=client, owns_client=True)


def _make_gemini_llm(client: httpx.AsyncClient) -> LLMClient:
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.GEMINI
    cfg.llm.chat_completions_path = "/v1beta/models/gemini-test:generateContent"
    return LLMClient(cfg, client=client, owns_client=True)


async def _collect(stream):
    return [event async for event in stream]


# ----------------------------------------------------------------- text stream


async def test_text_stream_yields_chunks_then_end() -> None:
    body = openai_text_stream(["Hel", "lo ", "world"], prompt_tokens=11, completion_tokens=5)
    llm = _make_llm(stream_mock_client(body))
    async with llm:
        events = await _collect(llm.invoke_stream([Message.user("hi")]))

    texts = [e.text for e in events if isinstance(e, TextChunk)]
    assert texts == ["Hel", "lo ", "world"]

    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "Hello world"
    assert end.response.model == "test-model"
    assert end.response.finish_reason == "stop"
    assert end.response.usage.total_tokens == 16


# ------------------------------------------------------------ tool-call stream


async def test_tool_call_fragments_reassembled() -> None:
    # name arrives first; arguments are split across several chunks.
    chunks = [
        {
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "book", "arguments": ""}}
                        ]
                    },
                }
            ],
        },
        {
            "model": "test-model",
            "choices": [
                {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"fli'}}]}}
            ],
        },
        {
            "model": "test-model",
            "choices": [
                {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ght": "UA1"}'}}]}}
            ],
        },
        {"model": "test-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {
            "model": "test-model",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9},
        },
    ]
    llm = _make_llm(stream_mock_client(sse_bytes(chunks)))
    async with llm:
        events = await _collect(llm.invoke_stream([Message.user("book")], tools=[]))

    started = [e for e in events if isinstance(e, ToolCallStarted)]
    assert len(started) == 1
    assert started[0].name == "book"
    assert started[0].id == "call_1"

    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert len(end.response.tool_calls) == 1
    assert end.response.tool_calls[0].arguments == {"flight": "UA1"}
    assert end.response.finish_reason == "tool_calls"
    assert end.response.usage.total_tokens == 9


# ----------------------------------------------------------- anthropic stream


async def test_anthropic_text_stream_yields_chunks_then_end() -> None:
    body = anthropic_text_stream(["Hel", "lo ", "world"], input_tokens=11, output_tokens=5)
    llm = _make_anthropic_llm(stream_mock_client(body))
    async with llm:
        events = await _collect(llm.invoke_stream([Message.user("hi")]))

    texts = [e.text for e in events if isinstance(e, TextChunk)]
    assert texts == ["Hel", "lo ", "world"]

    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "Hello world"
    assert end.response.model == "claude-test"
    assert end.response.finish_reason == "end_turn"
    assert end.response.usage.total_tokens == 16


async def test_anthropic_tool_call_stream_reassembled() -> None:
    # tool_use block announced at content_block_start; args split across deltas.
    events = [
        {
            "type": "message_start",
            "message": {"model": "claude-test", "usage": {"input_tokens": 3, "output_tokens": 1}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "book", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"fli'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'ght": "UA1"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 6},
        },
        {"type": "message_stop"},
    ]
    llm = _make_anthropic_llm(stream_mock_client(anthropic_sse(events)))
    async with llm:
        collected = await _collect(
            llm.invoke_stream([Message.user("book")], tools=[])
        )

    started = [e for e in collected if isinstance(e, ToolCallStarted)]
    assert len(started) == 1
    assert started[0].name == "book"
    assert started[0].id == "toolu_1"

    end = collected[-1]
    assert isinstance(end, StreamEnd)
    assert len(end.response.tool_calls) == 1
    assert end.response.tool_calls[0].arguments == {"flight": "UA1"}
    assert end.response.finish_reason == "tool_use"
    assert end.response.usage.total_tokens == 9  # input 3 + output 6


# ------------------------------------------------------------------ gemini


async def test_gemini_text_stream_yields_chunks_then_end() -> None:
    body = gemini_text_stream(["Hel", "lo ", "world"], prompt_tokens=11, completion_tokens=5)
    llm = _make_gemini_llm(stream_mock_client(body))
    async with llm:
        events = await _collect(llm.invoke_stream([Message.user("hi")]))

    texts = [e.text for e in events if isinstance(e, TextChunk)]
    assert texts == ["Hel", "lo ", "world"]

    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "Hello world"
    assert end.response.model == "gemini-test"
    assert end.response.finish_reason == "STOP"
    assert end.response.usage.total_tokens == 16


async def test_gemini_tool_call_stream_parsed() -> None:
    # Gemini streams a functionCall whole in one chunk (args is a complete dict).
    chunks = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "book", "args": {"flight": "UA1"}}}
                        ],
                    }
                }
            ],
            "modelVersion": "gemini-test",
        },
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP"}
            ],
            "modelVersion": "gemini-test",
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 6,
                "totalTokenCount": 9,
            },
        },
    ]
    llm = _make_gemini_llm(stream_mock_client(gemini_sse(chunks)))
    async with llm:
        collected = await _collect(llm.invoke_stream([Message.user("book")], tools=[]))

    started = [e for e in collected if isinstance(e, ToolCallStarted)]
    assert len(started) == 1
    assert started[0].name == "book"

    end = collected[-1]
    assert isinstance(end, StreamEnd)
    assert len(end.response.tool_calls) == 1
    assert end.response.tool_calls[0].arguments == {"flight": "UA1"}
    assert end.response.finish_reason == "STOP"
    assert end.response.usage.total_tokens == 9


async def test_gemini_streaming_hits_sse_endpoint() -> None:
    seen: dict[str, str] = {}
    body = gemini_text_stream(["ok"])

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    llm = _make_gemini_llm(mock_client(handler))
    async with llm:
        await _collect(llm.invoke_stream([Message.user("hi")]))

    assert "streamGenerateContent" in seen["url"]
    assert "alt=sse" in seen["url"]


# --------------------------------------------------------------- response_model


class _Answer(BaseModel):
    answer: int


async def test_response_model_validates_streamed_text() -> None:
    body = openai_text_stream(['{"ans', 'wer": ', "42}"])
    llm = _make_llm(stream_mock_client(body))
    async with llm:
        events = await _collect(llm.invoke_stream([Message.user("q")], response_model=_Answer))

    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.parsed == _Answer(answer=42)


async def test_response_model_mismatch_raises_at_end() -> None:
    body = openai_text_stream(['{"answer": "not-an-int"}'])
    llm = _make_llm(stream_mock_client(body))
    with pytest.raises(ValidationError):
        async with llm:
            await _collect(llm.invoke_stream([Message.user("q")], response_model=_Answer))


# ---------------------------------------------------------------------- retry


async def test_retryable_status_before_stream_is_retried() -> None:
    calls = {"n": 0}
    good = openai_text_stream(["ok"])

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, content=b"overloaded")
        return httpx.Response(
            200, content=good, headers={"content-type": "text/event-stream"}
        )

    llm = _make_llm(mock_client(handler))
    async with llm:
        events = await _collect(llm.invoke_stream([Message.user("hi")]))

    assert calls["n"] == 2  # one retry
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "ok"


# ------------------------------------------------------------- assembler unit


def test_assembler_orders_tool_calls_by_first_seen_index() -> None:
    asm = StreamAssembler()
    asm.ingest(MetaFragment(model="m"))
    asm.ingest(ToolCallFragment(index=0, id="a", name="first", args_delta="{}"))
    asm.ingest(ToolCallFragment(index=1, id="b", name="second", args_delta=""))
    asm.ingest(ToolCallFragment(index=1, args_delta='{"k": 1}'))
    asm.ingest(UsageFragment(prompt_tokens=1, completion_tokens=2, total_tokens=3))
    asm.ingest(FinishFragment(reason="tool_calls"))

    end = asm.finish()
    assert [c.name for c in end.response.tool_calls] == ["first", "second"]
    assert end.response.tool_calls[1].arguments == {"k": 1}
    assert end.response.model == "m"
    assert end.response.usage.completion_tokens == 2


def test_assembler_emits_text_chunk_on_text_fragment() -> None:
    asm = StreamAssembler()
    out = asm.ingest(TextFragment(text="hi"))
    assert out == [TextChunk("hi")]
    assert asm.ingest(TextFragment(text="")) == []
