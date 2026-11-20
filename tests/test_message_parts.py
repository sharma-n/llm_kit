"""Ordered text/image parts: Message API, formatters, and end-to-end resolution."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from llm_kit.config.image import ImageFailurePolicy
from llm_kit.config.llm import MessageFormat, StructuredOutputStyle
from llm_kit.errors import ImageFetchError
from llm_kit.llm.client import LLMClient
from llm_kit.messages.formatter import AnthropicFormatter, GeminiFormatter, OpenAIFormatter
from llm_kit.messages.types import ImagePart, Message, Role, TextPart

from conftest import chat_response, make_config, mock_client, tool_use_response  # noqa: F401

RAW = bytes(range(256)) * 3  # not valid text/PNG on purpose
RAW_B64 = base64.b64encode(RAW).decode("ascii")
DATA_URL = "data:image/jpeg;base64,QUJD"  # b"ABC"


def _b64_data_url(img: ImagePart) -> str:
    assert img.data is not None
    return f"data:{img.mime_type};base64,{base64.b64encode(img.data).decode()}"


# ------------------------------------------------------------------ types


def test_constructors_build_parts_in_order() -> None:
    img1 = ImagePart.from_url("http://x/a.png")
    img2 = ImagePart.from_bytes(b"\x00", "image/png")
    msg = Message.user("look", img1, "and", img2)
    assert msg.role is Role.USER
    assert msg.parts == [TextPart("look"), img1, TextPart("and"), img2]
    assert Message.system("s").parts == [TextPart("s")]
    assert Message.assistant("a").parts == [TextPart("a")]
    assert Message.tool_result("id", "obs").parts == [TextPart("obs")]
    assert Message.assistant_tool_calls([]).parts == []
    assert not hasattr(Message.user("x"), "text")
    assert not hasattr(Message.user("x"), "image_urls")


def test_image_part_validation() -> None:
    with pytest.raises(ValueError):
        ImagePart()
    with pytest.raises(ValueError):
        ImagePart(url="http://x", data=b"x", mime_type="image/png")
    with pytest.raises(ValueError):
        ImagePart(data=b"x")
    with pytest.raises(ValueError):
        ImagePart(url="http://x", mime_type="image/png")


# ------------------------------------------------------------------ formatters


def _interleaved() -> tuple[list[Message], dict[ImagePart, str]]:
    a = ImagePart.from_url("http://x/a.png")
    b = ImagePart.from_bytes(RAW, "image/webp")
    msgs = [Message.user("one", a, "two", b)]
    return msgs, {a: "data:image/png;base64,AAA=", b: _b64_data_url(b)}


def test_openai_interleaved_order() -> None:
    msgs, enc = _interleaved()
    content = OpenAIFormatter(
        structured_output_style=StructuredOutputStyle.NONE
    ).format_messages(msgs, enc)[0]["content"]
    assert [c["type"] for c in content] == ["text", "image_url", "text", "image_url"]
    assert [c.get("text") for c in content if c["type"] == "text"] == ["one", "two"]
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AAA="
    url = content[3]["image_url"]["url"]
    assert url.startswith("data:image/webp;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == RAW


def test_anthropic_interleaved_order_and_bytes_roundtrip() -> None:
    msgs, enc = _interleaved()
    content = AnthropicFormatter().format_messages(msgs, enc)[0]["content"]
    assert [c["type"] for c in content] == ["text", "image", "text", "image"]
    assert content[0]["text"] == "one" and content[2]["text"] == "two"
    src = content[3]["source"]
    assert src["media_type"] == "image/webp"
    assert base64.b64decode(src["data"]) == RAW


def test_gemini_interleaved_order_and_bytes_roundtrip() -> None:
    msgs, enc = _interleaved()
    parts = GeminiFormatter().format_messages(msgs, enc)[0]["parts"]
    assert [next(iter(p)) for p in parts] == ["text", "inlineData", "text", "inlineData"]
    assert parts[0]["text"] == "one" and parts[2]["text"] == "two"
    inline = parts[3]["inlineData"]
    assert inline["mimeType"] == "image/webp"
    assert base64.b64decode(inline["data"]) == RAW


def test_two_text_parts_stay_two_blocks() -> None:
    msgs = [Message.user("a", "b")]
    o = OpenAIFormatter(structured_output_style=StructuredOutputStyle.NONE)
    assert o.format_messages(msgs, {})[0]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
    assert AnthropicFormatter().format_messages(msgs, {})[0]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
    assert GeminiFormatter().format_messages(msgs, {})[0]["parts"] == [
        {"text": "a"},
        {"text": "b"},
    ]


def test_single_text_part_stays_plain_string() -> None:
    o = OpenAIFormatter(structured_output_style=StructuredOutputStyle.NONE)
    assert o.format_messages([Message.user("a")], {})[0]["content"] == "a"
    assert AnthropicFormatter().format_messages([Message.user("a")], {})[0]["content"] == "a"


def test_system_prompt_joins_text_parts() -> None:
    msgs = [Message.system("a", "b"), Message.system("c"), Message.user("u")]
    assert AnthropicFormatter().system_prompt(msgs) == "a\nb\n\nc"
    assert GeminiFormatter().system_prompt(msgs) == "a\nb\n\nc"


def test_data_url_part_accepted_by_formatters() -> None:
    img = ImagePart.from_url(DATA_URL)
    enc = {img: DATA_URL}
    msgs = [Message.user("x", img)]
    o = OpenAIFormatter(structured_output_style=StructuredOutputStyle.NONE)
    assert o.format_messages(msgs, enc)[0]["content"][1]["image_url"]["url"] == DATA_URL
    a = AnthropicFormatter().format_messages(msgs, enc)[0]["content"][1]["source"]
    assert (a["media_type"], a["data"]) == ("image/jpeg", "QUJD")
    g = GeminiFormatter().format_messages(msgs, enc)[0]["parts"][1]["inlineData"]
    assert (g["mimeType"], g["data"]) == ("image/jpeg", "QUJD")


# ------------------------------------------------------------------ end to end


def _llm(fmt: MessageFormat, handler: Any, **image: Any) -> LLMClient:
    cfg = make_config()
    cfg.llm.message_format = fmt
    for k, v in image.items():
        setattr(cfg.image, k, v)
    return LLMClient(cfg, client=mock_client(handler), owns_client=True)


def _llm_only_handler(captured: list[dict[str, Any]], fmt: MessageFormat):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "test.invalid":
            pytest.fail(f"unexpected non-LLM request: {request.url}")
        captured.append(json.loads(request.content))
        if fmt is MessageFormat.OPENAI:
            return httpx.Response(200, json=chat_response(content="ok"))
        if fmt is MessageFormat.ANTHROPIC:
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "end_turn",
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )

    return handler


def _extract(body: dict[str, Any], fmt: MessageFormat) -> list[tuple[str, str, bytes | None]]:
    """Normalize to (kind, text-or-mime, decoded bytes) per block."""
    out: list[tuple[str, str, bytes | None]] = []
    if fmt is MessageFormat.OPENAI:
        for c in body["messages"][0]["content"]:
            if c["type"] == "text":
                out.append(("text", c["text"], None))
            else:
                head, b64 = c["image_url"]["url"].split(",", 1)
                out.append(("image", head[5:].split(";")[0], base64.b64decode(b64)))
    elif fmt is MessageFormat.ANTHROPIC:
        for c in body["messages"][0]["content"]:
            if c["type"] == "text":
                out.append(("text", c["text"], None))
            else:
                s = c["source"]
                out.append(("image", s["media_type"], base64.b64decode(s["data"])))
    else:
        for p in body["contents"][0]["parts"]:
            if "text" in p:
                out.append(("text", p["text"], None))
            else:
                i = p["inlineData"]
                out.append(("image", i["mimeType"], base64.b64decode(i["data"])))
    return out


@pytest.mark.parametrize("fmt", list(MessageFormat))
async def test_end_to_end_bytes_and_data_url_never_fetch(fmt: MessageFormat) -> None:
    captured: list[dict[str, Any]] = []
    llm = _llm(fmt, _llm_only_handler(captured, fmt))
    msg = Message.user(
        "first",
        ImagePart.from_bytes(RAW, "image/png"),
        "second",
        ImagePart.from_url(DATA_URL),
    )
    async with llm:
        resp = await llm.invoke([msg])
    assert resp.text == "ok"
    assert _extract(captured[0], fmt) == [
        ("text", "first", None),
        ("image", "image/png", RAW),
        ("text", "second", None),
        ("image", "image/jpeg", b"ABC"),
    ]
    assert len(llm._images._cache) == 0  # bytes / data: parts bypass the cache


async def test_end_to_end_url_image_still_fetched_and_cached() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "img.invalid":
            return httpx.Response(200, content=b"PNG", headers={"content-type": "image/png"})
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response())

    llm = _llm(MessageFormat.OPENAI, handler)
    async with llm:
        await llm.invoke([Message.user("a", ImagePart.from_url("http://img.invalid/a.png"), "b")])
    assert _extract(captured[0], MessageFormat.OPENAI) == [
        ("text", "a", None),
        ("image", "image/png", b"PNG"),
        ("text", "b", None),
    ]
    assert "http://img.invalid/a.png" in llm._images._cache


async def test_large_bytes_encode_off_loop_and_roundtrip() -> None:
    captured: list[dict[str, Any]] = []
    llm = _llm(
        MessageFormat.OPENAI,
        _llm_only_handler(captured, MessageFormat.OPENAI),
        threadpool_encode_threshold_bytes=16,
    )
    async with llm:
        await llm.invoke([Message.user(ImagePart.from_bytes(RAW, "image/png"))])
    assert _extract(captured[0], MessageFormat.OPENAI)[0][2] == RAW


@pytest.mark.parametrize(
    "bad",
    ["data:image/png,rawnotbase64", "data:;base64,AAAA", "data:image/png;base64,@@@@", "data:image/png;base64,QUJ"],
)
async def test_malformed_data_url_raises_by_default(bad: str) -> None:
    llm = _llm(MessageFormat.OPENAI, _llm_only_handler([], MessageFormat.OPENAI))
    async with llm:
        with pytest.raises(ImageFetchError):
            await llm.invoke([Message.user("x", ImagePart.from_url(bad))])


async def test_malformed_data_url_follows_skip_and_placeholder_policies() -> None:
    bad = ImagePart.from_url("data:image/png;base64,@@@@")
    captured: list[dict[str, Any]] = []

    llm = _llm(
        MessageFormat.OPENAI,
        _llm_only_handler(captured, MessageFormat.OPENAI),
        on_failure=ImageFailurePolicy.SKIP,
    )
    async with llm:
        await llm.invoke([Message.user("x", bad)])
    assert _extract(captured[0], MessageFormat.OPENAI) == [("text", "x", None)]

    captured.clear()
    llm = _llm(
        MessageFormat.OPENAI,
        _llm_only_handler(captured, MessageFormat.OPENAI),
        on_failure=ImageFailurePolicy.PLACEHOLDER,
    )
    async with llm:
        await llm.invoke([Message.user("x", bad)])
    kinds = [k for k, _, _ in _extract(captured[0], MessageFormat.OPENAI)]
    assert kinds == ["text", "image"]


async def test_streaming_path_uses_same_resolution() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "test.invalid":
            pytest.fail("unexpected fetch")
        seen.append(json.loads(request.content))
        sse = (
            'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    llm = _llm(MessageFormat.OPENAI, handler)
    async with llm:
        async for _ in llm.invoke_stream(
            [Message.user("a", ImagePart.from_bytes(RAW, "image/png"))]
        ):
            pass
    assert seen[0]["stream"] is True
    assert _extract(seen[0], MessageFormat.OPENAI)[1][2] == RAW
