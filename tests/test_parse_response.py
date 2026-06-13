"""Tests for the shared ``parse_response`` used by both invoke and the batch driver."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.errors import ValidationError
from llm_kit.llm.parsing import parse_response
from llm_kit.messages.formatter import AnthropicFormatter, GeminiFormatter, OpenAIFormatter

from conftest import chat_response


class _Schema(BaseModel):
    x: int


def _openai() -> OpenAIFormatter:
    return OpenAIFormatter(structured_output_style=StructuredOutputStyle.OPENAI_JSON_SCHEMA)


def test_openai_text_usage_and_finish_reason() -> None:
    data = chat_response(content="bonjour", prompt_tokens=7, completion_tokens=3)
    resp = parse_response(data, None, _openai(), latency_seconds=1.5)
    assert resp.text == "bonjour"
    assert resp.parsed is None
    assert resp.usage.total_tokens == 10
    assert resp.model == "test-model"
    assert resp.finish_reason == "stop"
    assert resp.latency_seconds == 1.5
    assert resp.raw is data


def test_openai_structured_output_validates() -> None:
    data = chat_response(content='{"x": 5}')
    resp = parse_response(data, _Schema, _openai())
    assert isinstance(resp.parsed, _Schema)
    assert resp.parsed.x == 5


def test_validation_error_on_schema_mismatch() -> None:
    data = chat_response(content='{"x": "not-an-int"}')
    with pytest.raises(ValidationError) as exc:
        parse_response(data, _Schema, _openai())
    assert exc.value.raw_content == '{"x": "not-an-int"}'


def test_anthropic_shape() -> None:
    data = {
        "content": [{"type": "text", "text": "hi there"}],
        "usage": {"input_tokens": 5, "output_tokens": 2},
        "model": "claude-test",
        "stop_reason": "end_turn",
    }
    resp = parse_response(data, None, AnthropicFormatter())
    assert resp.text == "hi there"
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 2
    assert resp.usage.total_tokens == 7
    assert resp.finish_reason == "end_turn"


def test_gemini_shape() -> None:
    data = {
        "candidates": [
            {"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1},
        "model": "gemini-test",
    }
    resp = parse_response(data, None, GeminiFormatter())
    assert resp.text == "hi"
    assert resp.usage.total_tokens == 5
    assert resp.finish_reason == "STOP"
