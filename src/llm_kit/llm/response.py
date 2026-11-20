"""Response, BatchItem, and BatchResult dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_kit.errors import LLMError
from llm_kit.messages.schema import ResponseSchema
from llm_kit.messages.settings import RequestSettings
from llm_kit.messages.tools import ToolCall, ToolDefinition
from llm_kit.messages.types import Message

__all__ = [
    "BatchItem",
    "BatchResult",
    "BatchSummary",
    "LLMResponse",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
]


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Thinking tokens, when the provider reports them. ``total_tokens`` always
    # includes them. OpenAI already counts them inside ``completion_tokens``;
    # Gemini reports them outside ``candidatesTokenCount`` (so outside
    # ``completion_tokens``); Anthropic does not report them separately (0).
    reasoning_tokens: int = 0

    @property
    def output_tokens(self) -> int:
        """Billable output tokens (thinking included, never double counted).

        ``total - prompt`` is right for every provider: OpenAI total =
        prompt + completion (completion already holds reasoning); Anthropic
        total = prompt + output; Gemini total = prompt + candidates + thoughts.
        ``max`` guards a provider that omits ``total_tokens``.
        """
        return max(self.completion_tokens, self.total_tokens - self.prompt_tokens)


@dataclass(slots=True)
class LLMResponse:
    """One successful LLM invocation."""

    text: str
    # BaseModel for a Pydantic response_model; decoded JSON (dict/list/...) for a dict schema.
    parsed: Any = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0


@dataclass(slots=True)
class BatchItem:
    """One unit of work for the batch driver.

    ``response_model`` overrides whatever the caller passed to ``invoke`` — each
    item can validate against its own schema (a Pydantic class, or a JSON schema
    dict). ``settings`` carries per-item max_tokens/temperature/top_p/reasoning;
    ``overrides`` is the raw body escape hatch (deep-merged last).
    """

    id: str
    messages: list[Message]
    response_model: ResponseSchema | None = None
    settings: RequestSettings | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchResult:
    """One completed batch unit. Exactly one of ``response`` or ``error`` is set."""

    id: str
    response: LLMResponse | None = None
    error: LLMError | None = None
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class BatchSummary:
    """Aggregate token usage and estimated cost for a completed batch run."""

    total_items: int = 0
    ok_count: int = 0
    error_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    # Billable output: per item max(completion, total - prompt), summed. Covers
    # thinking tokens whether or not the provider's completion count includes them.
    output_tokens: int = 0
    estimated_cost: float = 0.0
