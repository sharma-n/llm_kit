"""Tests for OpenAIFormatter, AnthropicFormatter, and GeminiFormatter."""

from __future__ import annotations

from pydantic import BaseModel

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.messages.formatter import AnthropicFormatter, GeminiFormatter, OpenAIFormatter
from llm_kit.messages.types import ImagePart, Message, TextPart


def _fmt(style: StructuredOutputStyle = StructuredOutputStyle.OPENAI_JSON_SCHEMA) -> OpenAIFormatter:
    return OpenAIFormatter(structured_output_style=style)


def _afmt() -> AnthropicFormatter:
    return AnthropicFormatter()


def _gfmt() -> GeminiFormatter:
    return GeminiFormatter()


class _Foo(BaseModel):
    name: str
    count: int


# ============================================================ OpenAIFormatter — request


def test_text_only_message_uses_string_content() -> None:
    out = _fmt().format_messages([Message.system("hi")], {})
    assert out == [{"role": "system", "content": "hi"}]


def test_message_with_image_uses_content_array() -> None:
    msgs = [Message.user("look:", ImagePart.from_url("http://x/a.png"))]
    out = _fmt().format_messages(msgs, {ImagePart.from_url("http://x/a.png"): "data:image/png;base64,AAA"})
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "look:"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAA"},
    }


def test_image_only_message_omits_text_part() -> None:
    msgs = [Message.user(ImagePart.from_url("http://x/a.png"))]
    out = _fmt().format_messages(msgs, {ImagePart.from_url("http://x/a.png"): "data:image/png;base64,AAA"})
    content = out[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "image_url"


def test_missing_encoded_image_is_skipped() -> None:
    """If the fetcher's SKIP policy dropped a URL, the formatter mirrors that."""
    msgs = [Message.user("text", ImagePart.from_url("http://x/a.png"), ImagePart.from_url("http://x/b.png"))]
    encoded = {ImagePart.from_url("http://x/a.png"): "data:image/png;base64,AAA"}  # b.png missing
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
    schema = rf["json_schema"]["schema"]
    assert "properties" in schema
    # OpenAI strict mode requires these on every object, which Pydantic omits.
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["count", "name"]


def test_structured_output_strict_patches_nested_objects() -> None:
    class _Inner(BaseModel):
        x: int

    class _Outer(BaseModel):
        inner: _Inner
        tags: list[_Inner]

    fields = _fmt(StructuredOutputStyle.OPENAI_JSON_SCHEMA).structured_output_fields(_Outer)
    schema = fields["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    # Nested model definitions live under $defs and must be patched too.
    inner = schema["$defs"]["_Inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["x"]


def test_structured_output_tool_use_style() -> None:
    fields = _fmt(StructuredOutputStyle.TOOL_USE).structured_output_fields(_Foo)
    assert "tools" in fields and "tool_choice" in fields
    assert fields["tools"][0]["function"]["name"] == "_Foo"
    assert fields["tool_choice"]["function"]["name"] == "_Foo"


def test_structured_output_dict_field_additionalproperties_false() -> None:
    """dict[str, str] fields must not produce additionalProperties as a schema object."""

    class _Facts(BaseModel):
        facts: dict[str, str]

    fields = _fmt(StructuredOutputStyle.OPENAI_JSON_SCHEMA).structured_output_fields(_Facts)
    schema = fields["response_format"]["json_schema"]["schema"]
    ap = schema["properties"]["facts"].get("additionalProperties")
    assert ap is False, f"expected False, got {ap!r}"


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
    assert _fmt().extract_usage(data) == (7, 3, 10, 0)


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
    msgs = [Message.user("look:", ImagePart.from_url("http://x/a.png"))]
    out = _afmt().format_messages(msgs, {ImagePart.from_url("http://x/a.png"): "data:image/png;base64,AAA="})
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
    schema = fmt["schema"]
    assert "properties" in schema
    assert schema["additionalProperties"] is False
    assert "tools" not in fields
    assert "tool_choice" not in fields


def test_anthropic_structured_output_patches_nested_objects() -> None:
    class _Inner(BaseModel):
        x: int

    class _Outer(BaseModel):
        inner: _Inner

    fields = _afmt().structured_output_fields(_Outer)
    schema = fields["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    inner = schema["$defs"]["_Inner"]
    assert inner["additionalProperties"] is False


def test_anthropic_structured_output_dict_field_additionalproperties_false() -> None:
    """dict[str, str] fields must not produce additionalProperties as a schema object."""

    class _Facts(BaseModel):
        facts: dict[str, str]

    fields = _afmt().structured_output_fields(_Facts)
    schema = fields["output_config"]["format"]["schema"]
    ap = schema["properties"]["facts"].get("additionalProperties")
    assert ap is False, f"expected False, got {ap!r}"


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
    assert _afmt().extract_usage(data) == (10, 5, 15, 0)


def test_anthropic_finish_reason() -> None:
    assert _afmt().finish_reason({"stop_reason": "end_turn"}) == "end_turn"


def test_anthropic_finish_reason_missing_returns_none() -> None:
    assert _afmt().finish_reason({}) is None


# ============================================================ GeminiFormatter — request


def test_gemini_system_excluded_from_contents() -> None:
    msgs = [Message.system("be helpful"), Message.user("hi")]
    out = _gfmt().format_messages(msgs, {})
    assert len(out) == 1
    assert out[0]["role"] == "user"


def test_gemini_assistant_role_mapped_to_model() -> None:
    msgs = [Message.assistant("hello")]
    out = _gfmt().format_messages(msgs, {})
    assert out[0]["role"] == "model"


def test_gemini_system_prompt_extracted() -> None:
    msgs = [Message.system("be helpful"), Message.user("hi")]
    assert _gfmt().system_prompt(msgs) == "be helpful"


def test_gemini_system_prompt_returns_none_when_absent() -> None:
    assert _gfmt().system_prompt([Message.user("hi")]) is None


def test_gemini_text_message_uses_parts() -> None:
    out = _gfmt().format_messages([Message.user("hello")], {})
    assert out == [{"role": "user", "parts": [{"text": "hello"}]}]


def test_gemini_image_format_uses_inline_data() -> None:
    msgs = [Message.user("look:", ImagePart.from_url("http://x/a.png"))]
    out = _gfmt().format_messages(msgs, {ImagePart.from_url("http://x/a.png"): "data:image/png;base64,AAA="})
    parts = out[0]["parts"]
    assert parts[0] == {"text": "look:"}
    assert parts[1] == {"inlineData": {"mimeType": "image/png", "data": "AAA="}}


def test_gemini_empty_message_gets_placeholder_part() -> None:
    out = _gfmt().format_messages([Message.user()], {})
    assert out[0]["parts"] == [{"text": ""}]


def test_gemini_structured_output_fields() -> None:
    fields = _gfmt().structured_output_fields(_Foo)
    assert fields["responseMimeType"] == "application/json"
    assert "properties" in fields["responseJsonSchema"]
    assert "output_config" not in fields


def test_gemini_structured_output_no_model_returns_empty_dict() -> None:
    assert _gfmt().structured_output_fields(None) == {}


def test_gemini_default_headers_include_api_key() -> None:
    headers = _gfmt().default_headers("my-key")
    assert headers == {"x-goog-api-key": "my-key"}
    assert "authorization" not in headers


def test_gemini_default_headers_no_key_returns_empty() -> None:
    assert _gfmt().default_headers(None) == {}


def test_gemini_build_body_structure() -> None:
    msgs = [Message.system("sys"), Message.user("hi")]
    formatted = _gfmt().format_messages(msgs, {})
    body = _gfmt().build_body(
        model="gemini-2.0-flash",
        messages=formatted,
        system=_gfmt().system_prompt(msgs),
        temperature=0.5,
        top_p=0.9,
        max_tokens=100,
        extra_sampling={},
        response_model=None,
    )
    assert "contents" in body
    assert "model" not in body  # model goes in the URL
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert body["generationConfig"]["temperature"] == 0.5
    assert body["generationConfig"]["topP"] == 0.9
    assert body["generationConfig"]["maxOutputTokens"] == 100


def test_gemini_build_body_structured_output_in_gen_config() -> None:
    body = _gfmt().build_body(
        model="gemini-2.0-flash",
        messages=[],
        system=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        extra_sampling={},
        response_model=_Foo,
    )
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "properties" in body["generationConfig"]["responseJsonSchema"]


def test_gemini_build_body_no_gen_config_when_all_none() -> None:
    body = _gfmt().build_body(
        model="gemini-2.0-flash",
        messages=[],
        system=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        extra_sampling={},
        response_model=None,
    )
    assert "generationConfig" not in body


# ============================================================ GeminiFormatter — response


def test_gemini_extract_text() -> None:
    data = {
        "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}]
    }
    assert _gfmt().extract_text(data) == "hello"


def test_gemini_extract_text_concatenates_parts() -> None:
    data = {
        "candidates": [{"content": {"parts": [{"text": "foo"}, {"text": "bar"}]}}]
    }
    assert _gfmt().extract_text(data) == "foobar"


def test_gemini_extract_usage() -> None:
    data = {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}
    assert _gfmt().extract_usage(data) == (10, 5, 15, 0)


def test_gemini_extract_usage_missing_returns_zeros() -> None:
    assert _gfmt().extract_usage({}) == (0, 0, 0, 0)


def test_gemini_finish_reason() -> None:
    data = {"candidates": [{"finishReason": "STOP"}]}
    assert _gfmt().finish_reason(data) == "STOP"


def test_gemini_finish_reason_missing_returns_none() -> None:
    assert _gfmt().finish_reason({}) is None
