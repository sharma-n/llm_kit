"""Shape tests for the scaffolded Anthropic + Gemini batch backends.

These assert envelope wrap (submit) and unwrap (fetch_results) against canned
mock responses only. They make NO claim that the backends work against the live
provider APIs — those are unverified scaffolds (see the module docstrings).
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_kit.config.llm import MessageFormat
from llm_kit.provider_batch.anthropic import AnthropicBatchBackend
from llm_kit.provider_batch.gemini import GeminiBatchBackend
from llm_kit.provider_batch.types import BatchRequest, BatchState

from conftest import make_config, mock_client


def _anthropic_config():
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.ANTHROPIC
    cfg.llm.base_url = "https://test.invalid"
    return cfg


async def test_anthropic_submit_wraps_requests_inline() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages/batches" and request.method == "POST":
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"id": "msgbatch-1"})
        return httpx.Response(404, json={})

    backend = AnthropicBatchBackend(mock_client(handler), _anthropic_config(), "key")
    batch_id = await backend.submit(
        [BatchRequest(custom_id="a", body={"model": "m", "messages": []})]
    )
    assert batch_id == "msgbatch-1"
    assert captured["body"]["requests"] == [
        {"custom_id": "a", "params": {"model": "m", "messages": []}}
    ]


async def test_anthropic_poll_and_fetch_unwrap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages/batches/msgbatch-1":
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch-1",
                    "processing_status": "ended",
                    "results_url": "https://test.invalid/results/msgbatch-1",
                },
            )
        if request.url.path == "/results/msgbatch-1":
            lines = [
                json.dumps(
                    {
                        "custom_id": "a",
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "content": [{"type": "text", "text": "hi"}],
                                "usage": {"input_tokens": 1, "output_tokens": 1},
                            },
                        },
                    }
                ),
                json.dumps(
                    {"custom_id": "b", "result": {"type": "errored", "error": "boom"}}
                ),
            ]
            return httpx.Response(200, text="\n".join(lines))
        return httpx.Response(404, json={})

    backend = AnthropicBatchBackend(mock_client(handler), _anthropic_config(), "key")
    status = await backend.poll("msgbatch-1")
    assert status.state is BatchState.COMPLETED
    raws = [r async for r in backend.fetch_results(status)]
    by_id = {r.custom_id: r for r in raws}
    assert by_id["a"].inner_body == {
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    assert by_id["b"].error is not None


def _gemini_config():
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.GEMINI
    cfg.llm.base_url = "https://test.invalid"
    cfg.llm.chat_completions_path = "/v1beta/models/gemini-2.0-flash:generateContent"
    return cfg


async def test_gemini_submit_wraps_requests_with_metadata_key() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if ":batchGenerateContent" in request.url.path:
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"name": "operations/op-1"})
        return httpx.Response(404, json={})

    backend = GeminiBatchBackend(mock_client(handler), _gemini_config(), "key")
    batch_id = await backend.submit(
        [BatchRequest(custom_id="a", body={"contents": []})]
    )
    assert batch_id == "operations/op-1"
    inlined = captured["body"]["batch"]["inputConfig"]["requests"]["requests"]
    assert inlined == [{"request": {"contents": []}, "metadata": {"key": "a"}}]


async def test_gemini_poll_done_and_fetch_unwrap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1beta/operations/op-1":
            return httpx.Response(
                200,
                json={
                    "name": "operations/op-1",
                    "done": True,
                    "response": {
                        "inlinedResponses": {
                            "inlinedResponses": [
                                {
                                    "metadata": {"key": "a"},
                                    "response": {
                                        "candidates": [
                                            {"content": {"parts": [{"text": "hi"}]}}
                                        ]
                                    },
                                }
                            ]
                        }
                    },
                },
            )
        return httpx.Response(404, json={})

    backend = GeminiBatchBackend(mock_client(handler), _gemini_config(), "key")
    status = await backend.poll("operations/op-1")
    assert status.state is BatchState.COMPLETED
    raws = [r async for r in backend.fetch_results(status)]
    assert len(raws) == 1
    assert raws[0].custom_id == "a"
    assert raws[0].inner_body["candidates"][0]["content"]["parts"][0]["text"] == "hi"
