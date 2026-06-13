"""LLM endpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MessageFormat(StrEnum):
    """Which wire format to use for message encoding and response parsing."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class StructuredOutputStyle(StrEnum):
    """How the provider expects structured-output schemas to be passed.

    Only applies to the OpenAI formatter; Anthropic and Gemini use their own
    native structured-output shapes and ignore this setting.
    """

    OPENAI_JSON_SCHEMA = "openai_json_schema"
    TOOL_USE = "tool_use"
    NONE = "none"


@dataclass(slots=True)
class LLMConfig:
    """Connection + default sampling parameters for the LLM endpoint."""

    base_url: str = "https://api.openai.com"
    chat_completions_path: str = "/v1/chat/completions"
    model: str = "gpt-4o-mini"

    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None  # resolved at client init; never persisted to YAML

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    extra_sampling: dict[str, object] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)

    message_format: MessageFormat = MessageFormat.OPENAI
    structured_output_style: StructuredOutputStyle = StructuredOutputStyle.OPENAI_JSON_SCHEMA

    prompt_cost_per_1m_tokens: float = 0.0
    completion_cost_per_1m_tokens: float = 0.0
