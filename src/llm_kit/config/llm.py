"""LLM endpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class StructuredOutputStyle(StrEnum):
    """How the provider expects structured-output schemas to be passed.

    Why: not every OpenAI-compatible endpoint accepts ``response_format`` in the
    same shape. Override per provider so swapping endpoints is config-only.
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

    structured_output_style: StructuredOutputStyle = StructuredOutputStyle.OPENAI_JSON_SCHEMA
