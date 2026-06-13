"""Tests for OpenAIFormatter: message shaping and response_format injection."""

from __future__ import annotations

from pydantic import BaseModel

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.messages.formatter import OpenAIFormatter
from llm_kit.messages.types import Message


def _fmt(style: StructuredOutputStyle = StructuredOutputStyle.OPENAI_JSON_SCHEMA) -> OpenAIFormatter:
    return OpenAIFormatter(structured_output_style=style)


class _Foo(BaseModel):
    name: str
    count: int


def test_text_only_message_uses_string_content() -> None:
    out = _fmt().format_messages([Message.system("hi")], {})
    assert out == [{"role": "system", "content": "hi"}]


def test_message_with_image_uses_content_array() -> None:
    msgs = [Message.user("look:", image_urls=["http://x/a.png"])]
    out = _fmt().format_messages(msgs, {"http://x/a.png": "data:image/png;base64,AAA"})
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look:"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAA"},
    }


def test_image_only_message_omits_text_part() -> None:
    msgs = [Message.user(None, image_urls=["http://x/a.png"])]
    out = _fmt().format_messages(msgs, {"http://x/a.png": "data:image/png;base64,AAA"})
    content = out[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "image_url"


def test_missing_encoded_image_is_skipped() -> None:
    """If the fetcher's SKIP policy dropped a URL, the formatter mirrors that."""
    msgs = [Message.user("text", image_urls=["http://x/a.png", "http://x/b.png"])]
    encoded = {"http://x/a.png": "data:image/png;base64,AAA"}  # b.png missing
    out = _fmt().format_messages(msgs, encoded)
    content = out[0]["content"]
    image_urls = [p["image_url"]["url"] for p in content if p["type"] == "image_url"]
    assert image_urls == ["data:image/png;base64,AAA"]


def test_response_format_json_schema_style() -> None:
    rf = _fmt(StructuredOutputStyle.OPENAI_JSON_SCHEMA).response_format(_Foo)
    assert rf is not None
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "_Foo"
    assert rf["json_schema"]["strict"] is True
    assert "properties" in rf["json_schema"]["schema"]


def test_response_format_tool_use_style_carries_marker() -> None:
    rf = _fmt(StructuredOutputStyle.TOOL_USE).response_format(_Foo)
    assert rf is not None
    assert rf["_style"] == "tool_use"
    assert rf["tool"]["function"]["name"] == "_Foo"


def test_response_format_none_style_returns_none() -> None:
    assert _fmt(StructuredOutputStyle.NONE).response_format(_Foo) is None


def test_response_format_no_model_returns_none() -> None:
    assert _fmt().response_format(None) is None
