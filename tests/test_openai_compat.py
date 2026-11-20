"""Tests for LLMClient against an httpx.MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.errors import AuthError, ProviderError, ValidationError
from llm_kit.llm.client import LLMClient
from llm_kit.messages.types import ImagePart, Message

from conftest import chat_response, make_config, mock_client, tool_use_response


class _Capture:
    """Recorder for the last request that hit the mock transport."""

    def __init__(self) -> None:
        self.request: httpx.Request | None = None
        self.body: dict[str, Any] | None = None


def _make_llm(handler, **cfg_overrides) -> tuple[LLMClient, httpx.AsyncClient]:
    client = mock_client(handler)
    cfg = make_config(**cfg_overrides)
    llm = LLMClient(cfg, client=client, owns_client=True)
    return llm, client


async def test_happy_path_returns_text_and_usage() -> None:
    capture = _Capture()

    def handler(request: httpx.Request) -> httpx.Response:
        capture.request = request
        capture.body = json.loads(request.content.decode())
        return httpx.Response(200, json=chat_response(content="bonjour"))

    llm, _ = _make_llm(handler)
    async with llm:
        response = await llm.invoke([Message.user("hi")])

    assert response.text == "bonjour"
    assert response.usage.total_tokens == 10
    assert response.model == "test-model"
    assert response.finish_reason == "stop"
    assert response.latency_seconds >= 0.0


async def test_auth_header_injected_from_config() -> None:
    capture = _Capture()

    def handler(request: httpx.Request) -> httpx.Response:
        capture.request = request
        return httpx.Response(200, json=chat_response())

    llm, _ = _make_llm(handler)
    async with llm:
        await llm.invoke([Message.user("hi")])

    assert capture.request is not None
    assert capture.request.headers["authorization"] == "Bearer test-key"
    assert capture.request.headers["content-type"] == "application/json"


async def test_request_body_includes_model_and_messages() -> None:
    capture = _Capture()

    def handler(request: httpx.Request) -> httpx.Response:
        capture.body = json.loads(request.content.decode())
        return httpx.Response(200, json=chat_response())

    llm, _ = _make_llm(handler)
    async with llm:
        await llm.invoke([Message.system("be brief"), Message.user("hi")])

    assert capture.body is not None
    assert capture.body["model"] == "test-model"
    assert capture.body["messages"][0]["role"] == "system"
    assert capture.body["messages"][1]["role"] == "user"


async def test_overrides_take_precedence() -> None:
    capture = _Capture()

    def handler(request: httpx.Request) -> httpx.Response:
        capture.body = json.loads(request.content.decode())
        return httpx.Response(200, json=chat_response())

    llm, _ = _make_llm(handler)
    async with llm:
        await llm.invoke([Message.user("hi")], overrides={"temperature": 0.99})

    assert capture.body is not None
    assert capture.body["temperature"] == 0.99


async def test_response_model_validation_succeeds() -> None:
    class Out(BaseModel):
        sentiment: str
        confidence: float

    capture = _Capture()

    def handler(request: httpx.Request) -> httpx.Response:
        capture.body = json.loads(request.content.decode())
        payload = json.dumps({"sentiment": "positive", "confidence": 0.91})
        return httpx.Response(200, json=chat_response(content=payload))

    llm, _ = _make_llm(handler)
    async with llm:
        response = await llm.invoke([Message.user("how is it?")], response_model=Out)

    assert isinstance(response.parsed, Out)
    assert response.parsed.sentiment == "positive"
    assert response.parsed.confidence == pytest.approx(0.91)
    # Request must carry the json_schema response_format.
    assert capture.body is not None
    assert capture.body["response_format"]["type"] == "json_schema"
    assert capture.body["response_format"]["json_schema"]["name"] == "Out"


async def test_response_model_validation_failure_raises_validation_error() -> None:
    class Out(BaseModel):
        n: int

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(content='{"n": "not-an-int"}'))

    llm, _ = _make_llm(handler)
    async with llm:
        with pytest.raises(ValidationError) as info:
            await llm.invoke([Message.user("hi")], response_model=Out)
    assert info.value.raw_content == '{"n": "not-an-int"}'


async def test_tool_use_style_extracts_from_tool_call() -> None:
    class Out(BaseModel):
        action: str

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=tool_use_response(arguments_json='{"action": "go"}')
        )

    llm, _ = _make_llm(
        handler,
        llm=make_config().llm.__class__(
            base_url="https://test.invalid",
            model="test-model",
            api_key="test-key",
            api_key_env="",
            structured_output_style=StructuredOutputStyle.TOOL_USE,
        ),
    )
    async with llm:
        response = await llm.invoke([Message.user("go?")], response_model=Out)
    assert isinstance(response.parsed, Out)
    assert response.parsed.action == "go"


async def test_401_response_raises_auth_error_without_retry() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="invalid key")

    llm, _ = _make_llm(handler)
    async with llm:
        with pytest.raises(AuthError):
            await llm.invoke([Message.user("hi")])
    assert calls == 1


async def test_500_retried_then_succeeds() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 2:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=chat_response(content="ok"))

    llm, _ = _make_llm(handler)
    async with llm:
        response = await llm.invoke([Message.user("hi")])
    assert response.text == "ok"
    assert calls == 2


async def test_500_exhausts_then_raises_provider_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="permanent")

    llm, _ = _make_llm(handler)
    async with llm:
        with pytest.raises(ProviderError) as info:
            await llm.invoke([Message.user("hi")])
    assert info.value.status_code == 500


async def test_non_json_response_becomes_provider_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    llm, _ = _make_llm(handler)
    async with llm:
        with pytest.raises(ProviderError):
            await llm.invoke([Message.user("hi")])


async def test_image_in_message_is_base64_encoded_in_request() -> None:
    """End-to-end: image URL → fetched → base64 → embedded in request body."""
    captured_body: dict[str, Any] | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        url = str(request.url)
        if "image" in url:
            return httpx.Response(
                200, content=b"PNG-DATA", headers={"content-type": "image/png"}
            )
        captured_body = json.loads(request.content.decode())
        return httpx.Response(200, json=chat_response(content="seen"))

    llm, _ = _make_llm(handler)
    async with llm:
        await llm.invoke(
            [Message.user("look:", ImagePart.from_url("http://x/image.png"))]
        )

    assert captured_body is not None
    content = captured_body["messages"][0]["content"]
    assert isinstance(content, list)
    image_part = next(p for p in content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
