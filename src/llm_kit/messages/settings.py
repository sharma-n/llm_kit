"""Per-request sampling and reasoning settings (provider-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReasoningEffort(StrEnum):
    """Coarse reasoning/thinking intensity, mapped per provider by the formatter."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class Reasoning:
    """How hard the model should think. Set ``effort``, ``budget_tokens``, or both.

    Each formatter renders whichever the provider understands and converts the
    other (see ``messages/formatter/_shared.py`` for the tables). When both are
    given, OpenAI and Gemini use ``effort`` and Anthropic uses ``budget_tokens``.
    """

    effort: ReasoningEffort | None = None
    budget_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class RequestSettings:
    """Per-request overrides of the client-wide ``LLMConfig`` sampling defaults.

    ``None`` fields inherit from ``LLMConfig``. Precedence is
    ``LLMConfig < settings < overrides``.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    reasoning: Reasoning | None = None


__all__ = ["Reasoning", "ReasoningEffort", "RequestSettings"]
