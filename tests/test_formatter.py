"""Tests for OpenAIFormatter and AnthropicFormatter."""

from __future__ import annotations

from pydantic import BaseModel

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.messages.formatter import AnthropicFormatter, OpenAIFormatter
from llm_kit.messages.types import Message


def _fmt(style: StructuredOutputStyle = StructuredOutputStyle.OPENAI_JSON_SCHEMA) -> OpenAIFormatter:
    return OpenAIFormatter(structured_output_style=style)


def _afmt() -> AnthropicFormatter:
    return AnthropicFormatter()


class _Foo(BaseModel):
    name: str
    count: int


# ============================================================ OpenAIFormatter — request


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


def test_openai_system_prompt_returns_none() -> None:
    msgs = [Message.system("be helpful"), Message.user("hi")]
    assert _fmt().system_prompt(msgs) is None


def test_structured_output_json_schema_style() -> None:
    fields = _fmt(StructuredOutputStyle.OPENAI_JSON_SCHEMA).structured_output_fields(_Foo)
    assert "response_format" in fields
    rf = fields["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "_Foo"
    assert rf["json_schema"]["strict"] is True
    assert "properties" in rf["json_schema"]["schema"]


def test_structured_output_tool_use_style() -> None:
    fields = _fmt(StructuredOutputStyle.TOOL_USE).structured_output_fields(_Foo)
    assert "tools" in fields and "tool_choice" in fields
    assert fields["tools"][0]["function"]["name"] == "_Foo"
    assert fields["tool_choice"]["function"]["name"] == "_Foo"


def test_structured_output_none_style_returns_empty_dict() -> None:
    assert _fmt(StructuredOutputStyle.NONE).structured_output_fields(_Foo) == {}


def test_structured_output_no_model_returns_empty_dict() -> None:
    assert _fmt().structured_output_fields(None) == {}


# ============================================================ OpenAIFormatter — response


def test_openai_extract_text_string_content() -> None:
    data = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    assert _fmt().extract_text(data) == "hello"


def test_openai_extract_text_list_content() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
                },
                "finish_reason": "stop",
            }
        ]
    }
    assert _fmt().extract_text(data) == "foobar"


def test_openai_extract_text_tool_use_style() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"function": {"name": "X", "arguments": '{"x": 1}'}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    assert _fmt(StructuredOutputStyle.TOOL_USE).extract_text(data) == '{"x": 1}'


def test_openai_extract_usage() -> None:
    data = {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    assert _fmt().extract_usage(data) == (7, 3, 10)


def test_openai_finish_reason() -> None:
    data = {"choices": [{"finish_reason": "stop"}]}
    assert _fmt().finish_reason(data) == "stop"


def test_openai_finish_reason_missing_returns_none() -> None:
    assert _fmt().finish_reason({}) is None


# ============================================================ AnthropicFormatter — request


def test_anthropic_system_excluded_from_messages() -> None:
    msgs = [Message.system("be helpful"), Message.user("hi")]
    out = _afmt().format_messages(msgs, {})
    assert len(out) == 1
    assert out[0]["role"] == "user"


def test_anthropic_system_prompt_extracted() -> None:
    msgs = [Message.system("be helpful"), Message.user("hi")]
    assert _afmt().system_prompt(msgs) == "be helpful"


def test_anthropic_system_prompt_returns_none_when_absent() -> None:
    assert _afmt().system_prompt([Message.user("hi")]) is None


def test_anthropic_multiple_system_messages_joined() -> None:
    msgs = [Message.system("part1"), Message.system("part2"), Message.user("hi")]
    assert _afmt().system_prompt(msgs) == "part1\n\npart2"


def test_anthropic_text_message_uses_string_content() -> None:
    out = _afmt().format_messages([Message.user("hello")], {})
    assert out == [{"role": "user", "content": "hello"}]


def test_anthropic_image_format_uses_base64_source() -> None:
    msgs = [Message.user("look:", image_urls=["http://x/a.png"])]
    out = _afmt().format_messages(msgs, {"http://x/a.png": "data:image/png;base64,AAA="})
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look:"}
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAA="},
    }


def test_anthropic_structured_output_fields() -> None:
    fields = _afmt().structured_output_fields(_Foo)
    fmt = fields["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "properties" in fmt["schema"]
    assert "tools" not in fields
    assert "tool_choice" not in fields


def test_anthropic_structured_output_no_model_returns_empty_dict() -> None:
    assert _afmt().structured_output_fields(None) == {}


def test_anthropic_default_headers_include_version_and_key() -> None:
    headers = _afmt().default_headers("sk-ant-test")
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["x-api-key"] == "sk-ant-test"
    assert "authorization" not in headers


def test_anthropic_default_headers_no_key_omits_key_header() -> None:
    headers = _afmt().default_headers(None)
    assert "anthropic-version" in headers
    assert "x-api-key" not in headers


# ============================================================ AnthropicFormatter — response


def test_anthropic_extract_text_from_text_block() -> None:
    data = {"content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn"}
    assert _afmt().extract_text(data) == "hello"


def test_anthropic_extract_text_concatenates_multiple_text_blocks() -> None:
    data = {"content": [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]}
    assert _afmt().extract_text(data) == "foobar"



def test_anthropic_extract_usage() -> None:
    data = {"usage": {"input_tokens": 10, "output_tokens": 5}}
    assert _afmt().extract_usage(data) == (10, 5, 15)


def test_anthropic_finish_reason() -> None:
    assert _afmt().finish_reason({"stop_reason": "end_turn"}) == "end_turn"


def test_anthropic_finish_reason_missing_returns_none() -> None:
    assert _afmt().finish_reason({}) is None
