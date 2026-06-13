"""Tool definitions in + parsed tool calls out (non-streaming path)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.llm.client import LLMClient
from llm_kit.messages.formatter import (
    AnthropicFormatter,
    GeminiFormatter,
    OpenAIFormatter,
)
from llm_kit.messages.tools import ToolCall, ToolDefinition
from llm_kit.messages.types import Message

from conftest import make_config, mock_client, tool_call_response

WEATHER = ToolDefinition(
    name="get_weather",
    description="Look up the weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _make_llm(handler, **cfg) -> LLMClient:
    return LLMClient(make_config(**cfg), client=mock_client(handler), owns_client=True)


async def test_invoke_with_tools_returns_parsed_tool_calls() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=tool_call_response(
                tool_calls=[
                    {"id": "call_1", "name": "get_weather", "arguments": '{"city": "Paris"}'}
                ]
            ),
        )

    llm = _make_llm(handler)
    async with llm:
        resp = await llm.invoke([Message.user("weather?")], tools=[WEATHER])

    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call == ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    assert resp.finish_reason == "tool_calls"


async def test_tools_rendered_into_request_body() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content.decode()))
        return httpx.Response(200, json=tool_call_response(
            tool_calls=[{"id": "c1", "name": "get_weather", "arguments": "{}"}]
        ))

    llm = _make_llm(handler)
    async with llm:
        await llm.invoke([Message.user("hi")], tools=[WEATHER], tool_choice="auto")

    assert seen["tool_choice"] == "auto"
    assert seen["tools"][0]["type"] == "function"
    assert seen["tools"][0]["function"]["name"] == "get_weather"
    assert seen["tools"][0]["function"]["parameters"] == WEATHER.parameters


async def test_multiple_tool_calls_parsed_in_order() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=tool_call_response(
                tool_calls=[
                    {"id": "a", "name": "f", "arguments": '{"x": 1}'},
                    {"id": "b", "name": "g", "arguments": '{"y": 2}'},
                ]
            ),
        )

    llm = _make_llm(handler)
    async with llm:
        resp = await llm.invoke([Message.user("go")], tools=[WEATHER])

    assert [c.id for c in resp.tool_calls] == ["a", "b"]
    assert resp.tool_calls[1].arguments == {"y": 2}


async def test_malformed_arguments_preserved_as_raw() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=tool_call_response(
                tool_calls=[{"id": "c1", "name": "f", "arguments": "{not json"}]
            ),
        )

    llm = _make_llm(handler)
    async with llm:
        resp = await llm.invoke([Message.user("go")], tools=[WEATHER])

    assert resp.tool_calls[0].arguments == {"_raw": "{not json"}


def test_message_tool_calls_render_to_openai_wire() -> None:
    fmt = OpenAIFormatter(structured_output_style=StructuredOutputStyle.NONE)
    msgs = [
        Message.user("book it"),
        Message.assistant_tool_calls(
            [ToolCall(id="call_1", name="book", arguments={"flight": "UA1"})]
        ),
        Message.tool_result("call_1", "booked"),
    ]
    wire = fmt.format_messages(msgs, {})

    assert wire[1]["role"] == "assistant"
    tc = wire[1]["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "book"
    assert json.loads(tc["function"]["arguments"]) == {"flight": "UA1"}

    assert wire[2] == {"role": "tool", "tool_call_id": "call_1", "content": "booked"}


@pytest.mark.parametrize("fmt", [AnthropicFormatter(), GeminiFormatter()])
def test_non_openai_tools_not_implemented(fmt) -> None:
    with pytest.raises(NotImplementedError):
        fmt.build_body(
            model="m",
            messages=[],
            system=None,
            temperature=None,
            top_p=None,
            max_tokens=None,
            extra_sampling={},
            response_model=None,
            tools=[WEATHER],
        )


async def test_invoke_without_tools_still_empty_tool_calls() -> None:
    from conftest import chat_response

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(content="plain"))

    llm = _make_llm(handler)
    async with llm:
        resp = await llm.invoke([Message.user("hi")])

    assert resp.text == "plain"
    assert resp.tool_calls == []
