"""Tool definitions in + parsed tool calls out (non-streaming path)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.llm.client import LLMClient
from llm_kit.messages.formatter import (
    AnthropicFormatter,
    GeminiFormatter,
    OpenAIFormatter,
)
from llm_kit.messages.tools import ToolCall, ToolDefinition
from llm_kit.messages.types import Message

from llm_kit.config.llm import MessageFormat

from conftest import (
    anthropic_tool_response,
    gemini_tool_response,
    make_config,
    mock_client,
    tool_call_response,
)

WEATHER = ToolDefinition(
    name="get_weather",
    description="Look up the weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _make_llm(handler, **cfg) -> LLMClient:
    return LLMClient(make_config(**cfg), client=mock_client(handler), owns_client=True)


def _make_anthropic_llm(handler) -> LLMClient:
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.ANTHROPIC
    return LLMClient(cfg, client=mock_client(handler), owns_client=True)


def _make_gemini_llm(handler) -> LLMClient:
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.GEMINI
    return LLMClient(cfg, client=mock_client(handler), owns_client=True)


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


# ----------------------------------------------------------------- anthropic


async def test_anthropic_invoke_with_tools_returns_parsed_tool_calls() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=anthropic_tool_response(
                tool_calls=[
                    {"id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
                ]
            ),
        )

    llm = _make_anthropic_llm(handler)
    async with llm:
        resp = await llm.invoke([Message.user("weather?")], tools=[WEATHER])

    assert resp.tool_calls == [
        ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Paris"})
    ]
    assert resp.finish_reason == "tool_use"


async def test_anthropic_tools_rendered_into_request_body() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json=anthropic_tool_response(
                tool_calls=[{"id": "t1", "name": "get_weather", "input": {}}]
            ),
        )

    llm = _make_anthropic_llm(handler)
    async with llm:
        await llm.invoke([Message.user("hi")], tools=[WEATHER], tool_choice="required")

    # Anthropic shape: flat name/description/input_schema, mapped tool_choice.
    assert seen["tools"][0] == {
        "name": "get_weather",
        "description": "Look up the weather",
        "input_schema": WEATHER.parameters,
    }
    assert seen["tool_choice"] == {"type": "any"}


def test_anthropic_message_tool_calls_render_to_wire() -> None:
    fmt = AnthropicFormatter()
    msgs = [
        Message.user("book it"),
        Message.assistant_tool_calls(
            [ToolCall(id="toolu_1", name="book", arguments={"flight": "UA1"})],
            text="on it",
        ),
        Message.tool_result("toolu_1", "booked"),
    ]
    wire = fmt.format_messages(msgs, {})

    # Assistant turn: text block + tool_use block (input is a dict).
    assert wire[1]["role"] == "assistant"
    assert wire[1]["content"][0] == {"type": "text", "text": "on it"}
    assert wire[1]["content"][1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "book",
        "input": {"flight": "UA1"},
    }
    # Tool result becomes a user message carrying a tool_result block.
    assert wire[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "booked"}
        ],
    }


def test_anthropic_coalesces_adjacent_tool_results() -> None:
    fmt = AnthropicFormatter()
    msgs = [
        Message.assistant_tool_calls(
            [
                ToolCall(id="a", name="f", arguments={}),
                ToolCall(id="b", name="g", arguments={}),
            ]
        ),
        Message.tool_result("a", "ra"),
        Message.tool_result("b", "rb"),
    ]
    wire = fmt.format_messages(msgs, {})

    # The two tool results collapse into ONE user turn (Anthropic requirement).
    assert len(wire) == 2
    assert wire[1]["role"] == "user"
    assert [b["tool_use_id"] for b in wire[1]["content"]] == ["a", "b"]


def test_anthropic_tool_choice_specific_tool_name() -> None:
    fmt = AnthropicFormatter()
    body = fmt.build_body(
        model="m",
        messages=[],
        system=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        extra_sampling={},
        response_model=None,
        tools=[WEATHER],
        tool_choice="get_weather",
    )
    assert body["tool_choice"] == {"type": "tool", "name": "get_weather"}


# ------------------------------------------------------------------- gemini


async def test_gemini_invoke_with_tools_returns_parsed_tool_calls() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=gemini_tool_response(
                tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}}]
            ),
        )

    llm = _make_gemini_llm(handler)
    async with llm:
        resp = await llm.invoke([Message.user("weather?")], tools=[WEATHER])

    assert resp.tool_calls == [
        ToolCall(id="call_0", name="get_weather", arguments={"city": "Paris"})
    ]
    assert resp.finish_reason == "STOP"


async def test_gemini_tools_rendered_into_request_body() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json=gemini_tool_response(
                tool_calls=[{"name": "get_weather", "args": {}}]
            ),
        )

    llm = _make_gemini_llm(handler)
    async with llm:
        await llm.invoke([Message.user("hi")], tools=[WEATHER], tool_choice="required")

    # Gemini shape: functionDeclarations nest under a single tools entry.
    assert seen["tools"][0]["functionDeclarations"][0] == {
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": WEATHER.parameters,
    }
    # "required" maps to functionCallingConfig mode ANY.
    assert seen["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}


def test_gemini_message_tool_calls_render_to_wire() -> None:
    fmt = GeminiFormatter()
    msgs = [
        Message.user("book it"),
        Message.assistant_tool_calls(
            [ToolCall(id="call_0", name="book", arguments={"flight": "UA1"})]
        ),
        Message.tool_result("call_0", "booked"),
    ]
    wire = fmt.format_messages(msgs, {})

    # Assistant turn: a model-role content with a functionCall part (args is a dict).
    assert wire[1]["role"] == "model"
    assert wire[1]["parts"] == [
        {"functionCall": {"name": "book", "args": {"flight": "UA1"}}}
    ]
    # Tool result becomes a user content with a functionResponse part; the name
    # is recovered from the preceding assistant call (Gemini matches by name).
    assert wire[2] == {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": "book", "response": {"result": "booked"}}}
        ],
    }


def test_gemini_coalesces_adjacent_tool_results() -> None:
    fmt = GeminiFormatter()
    msgs = [
        Message.assistant_tool_calls(
            [
                ToolCall(id="call_0", name="f", arguments={}),
                ToolCall(id="call_1", name="g", arguments={}),
            ]
        ),
        Message.tool_result("call_0", "ra"),
        Message.tool_result("call_1", "rb"),
    ]
    wire = fmt.format_messages(msgs, {})

    # The two function responses collapse into ONE user turn.
    assert len(wire) == 2
    assert wire[1]["role"] == "user"
    assert [p["functionResponse"]["name"] for p in wire[1]["parts"]] == ["f", "g"]


def test_gemini_tool_choice_specific_tool_name() -> None:
    fmt = GeminiFormatter()
    body = fmt.build_body(
        model="m",
        messages=[],
        system=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        extra_sampling={},
        response_model=None,
        tools=[WEATHER],
        tool_choice="get_weather",
    )
    assert body["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}
    }


def test_gemini_streaming_endpoint_path() -> None:
    fmt = GeminiFormatter()
    path = "/v1beta/models/gemini-2.0-flash:generateContent"
    assert fmt.endpoint_path(path, stream=False) == path
    assert (
        fmt.endpoint_path(path, stream=True)
        == "/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
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
