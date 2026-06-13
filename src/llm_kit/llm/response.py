"""Response, BatchItem, and BatchResult dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from llm_kit.errors import LLMError
from llm_kit.messages.types import Message


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class LLMResponse:
    """One successful LLM invocation."""

    text: str
    parsed: BaseModel | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0


@dataclass(slots=True)
class BatchItem:
    """One unit of work for the batch driver.

    ``response_model`` overrides whatever the caller passed to ``invoke`` — each
    item can validate against its own schema. ``overrides`` lets a single item
    bump max_tokens, change temperature, etc.
    """

    id: str
    messages: list[Message]
    response_model: type[BaseModel] | None = None
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
    estimated_cost: float = 0.0
