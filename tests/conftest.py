"""Shared test fixtures: mock httpx transport, sample configs, fake LLM."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest

from llm_kit.config.app import AppConfig
from llm_kit.config.batch import BatchConfig
from llm_kit.config.embed import EmbeddingConfig
from llm_kit.config.http import HTTPConfig, RetryConfig
from llm_kit.config.image import ImageConfig
from llm_kit.config.llm import LLMConfig
from llm_kit.config.provider_batch import ProviderBatchConfig
from llm_kit.config.rate_limit import RateLimitConfig
from llm_kit.llm.base import BaseLLM
from llm_kit.llm.client import LLMClient
from llm_kit.llm.response import BatchItem, BatchResult, LLMResponse, TokenUsage
from llm_kit.messages.types import Message


def make_config(**overrides: Any) -> AppConfig:
    """Build a fast, deterministic AppConfig suitable for tests."""
    cfg = AppConfig(
        llm=LLMConfig(
            base_url="https://test.invalid",
            model="test-model",
            api_key="test-key",
            api_key_env="",
        ),
        embed=EmbeddingConfig(
            base_url="https://test.invalid",
            model="test-embed-model",
            api_key="test-key",
            api_key_env="",
        ),
        http=HTTPConfig(
            llm_retry=RetryConfig(
                max_retries=2,
                backoff_base_seconds=0.001,
                backoff_max_seconds=0.01,
                jitter_seconds=0.0,
            )
        ),
        rate_limit=RateLimitConfig(
            max_concurrent_requests=8,
            requests_per_minute=None,
            tokens_per_minute=None,
        ),
        image=ImageConfig(
            max_concurrent_downloads=4,
            cache_size=16,
            retry=RetryConfig(
                max_retries=1,
                backoff_base_seconds=0.001,
                backoff_max_seconds=0.01,
                jitter_seconds=0.0,
            ),
        ),
        batch=BatchConfig(worker_concurrency=4, input_buffer_size=8, progress_log_every=0),
        provider_batch=ProviderBatchConfig(poll_interval_seconds=0.0),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def app_config() -> AppConfig:
    return make_config()


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """Wrap a handler in a MockTransport-backed AsyncClient."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def chat_response(
    *,
    content: str = "hello",
    model: str = "test-model",
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build a minimal OpenAI-shape chat-completion response body."""
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def embedding_response(
    *,
    vectors: list[list[float]],
    model: str = "test-embed-model",
    prompt_tokens: int = 10,
    out_of_order: bool = False,
) -> dict[str, Any]:
    """Build a minimal OpenAI-shape embeddings response body.

    ``out_of_order`` reverses the ``data`` array (keeping each entry's ``index``)
    to exercise the client's index-based re-sort.
    """
    data = [
        {"object": "embedding", "index": i, "embedding": vec}
        for i, vec in enumerate(vectors)
    ]
    if out_of_order:
        data = list(reversed(data))
    return {
        "object": "list",
        "model": model,
        "data": data,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


def tool_use_response(*, arguments_json: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "X", "arguments": arguments_json},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def tool_call_response(
    *,
    tool_calls: list[dict[str, Any]],
    content: str | None = None,
    finish_reason: str = "tool_calls",
    prompt_tokens: int = 5,
    completion_tokens: int = 4,
) -> dict[str, Any]:
    """OpenAI-shape chat response carrying one or more ``tool_calls``.

    Each entry in ``tool_calls`` is ``{"id", "name", "arguments"}`` where
    ``arguments`` is the JSON *string* the provider would send.
    """
    return {
        "id": "chatcmpl-test",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse_bytes(chunks: list[dict[str, Any]], *, done: bool = True) -> bytes:
    """Serialize chunk dicts as an OpenAI-style ``text/event-stream`` body."""
    parts = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if done:
        parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


def openai_text_stream(
    text_pieces: list[str],
    *,
    model: str = "test-model",
    finish_reason: str = "stop",
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
) -> bytes:
    """A simple text-only OpenAI stream: deltas, a finish chunk, a usage chunk."""
    chunks: list[dict[str, Any]] = [
        {"model": model, "choices": [{"index": 0, "delta": {"content": piece}}]}
        for piece in text_pieces
    ]
    chunks.append(
        {"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
    )
    chunks.append(
        {
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )
    return sse_bytes(chunks)


def stream_mock_client(body: bytes, *, status: int = 200) -> httpx.AsyncClient:
    """An AsyncClient whose every request returns ``body`` as a stream."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    return mock_client(handler)


class FakeLLM(BaseLLM):
    """Drop-in BaseLLM for batch-driver tests. No HTTP, no httpx, deterministic."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        fail_ids: set[str] | None = None,
        delay_seconds: float = 0.0,
        responder: Callable[[list[Message]], str] | None = None,
        stream_script: list[Any] | None = None,
    ) -> None:
        self._config = config or make_config()
        self._fail_ids = fail_ids or set()
        self._delay = delay_seconds
        self._responder = responder or (lambda msgs: f"echo: {msgs[-1].text or ''}")
        self._stream_script = stream_script
        self.invoke_calls: list[list[Message]] = []
        self.stream_calls: list[list[Message]] = []
        self.fail_now: set[str] = set(self._fail_ids)

    async def invoke(
        self,
        messages: list[Message],
        *,
        response_model: type | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        import asyncio

        self.invoke_calls.append(messages)
        if self._delay:
            await asyncio.sleep(self._delay)
        # Treat last user text after "id=" as the item id (set by tests)
        text = (messages[-1].text or "")
        for fid in list(self.fail_now):
            if fid in text:
                from llm_kit.errors import ProviderError

                raise ProviderError(f"simulated failure for {fid}", status_code=400)
        return LLMResponse(
            text=self._responder(messages),
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def invoke_stream(
        self,
        messages: list[Message],
        *,
        response_model: type | None = None,
        overrides: dict[str, Any] | None = None,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> AsyncIterator[Any]:
        """Scripted stream for agent-style tests.

        If ``stream_script`` was supplied, yields those events verbatim.
        Otherwise yields one ``TextChunk`` per word of the responder text and a
        terminal ``StreamEnd`` built from the assembled text.
        """
        from llm_kit.llm.streaming import StreamEnd, TextChunk
        from llm_kit.llm.response import LLMResponse, TokenUsage

        self.stream_calls.append(messages)
        if self._stream_script is not None:
            for event in self._stream_script:
                yield event
            return
        text = self._responder(messages)
        for word in text.split():
            yield TextChunk(text=word + " ")
        yield StreamEnd(
            LLMResponse(
                text=text,
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        )

    def batch_stream(self, items, *, completed_ids=None):
        from llm_kit.batch.processor import run_batch_stream

        return run_batch_stream(
            llm=self,
            items=items,
            config=self._config.batch,
            completed_ids=completed_ids,
        )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


def make_items(n: int, *, prefix: str = "id") -> list[BatchItem]:
    return [
        BatchItem(id=f"{prefix}-{i:04d}", messages=[Message.user(f"q for id={prefix}-{i:04d}")])
        for i in range(n)
    ]


__all__ = [
    "FakeLLM",
    "app_config",
    "chat_response",
    "embedding_response",
    "fake_llm",
    "make_config",
    "make_items",
    "mock_client",
    "openai_text_stream",
    "sse_bytes",
    "stream_mock_client",
    "tool_call_response",
    "tool_use_response",
]
