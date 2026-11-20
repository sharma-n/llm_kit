"""Helpers shared across every provider formatter."""

from __future__ import annotations

import json
from typing import Any

from llm_kit.messages.settings import Reasoning, ReasoningEffort
from llm_kit.messages.types import Message, TextPart


def parse_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Coerce a tool call's ``arguments`` (JSON string or dict) to a dict.

    Preserves the original text under ``_raw`` if it is not valid JSON, so a
    malformed argument blob is surfaced rather than silently dropped.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_raw": raw}


def message_text(msg: Message) -> str:
    """A message's text parts joined with a newline (images ignored).

    Used only where the wire format wants a single string: tool-result content
    and system prompts. Ordinary turns keep one wire block per part.
    """
    return "\n".join(p.text for p in msg.parts if isinstance(p, TextPart))


# Reasoning mapping tables -------------------------------------------------

# budget_tokens upper bound (inclusive) -> effort; anything above is HIGH.
_BUDGET_THRESHOLDS: tuple[tuple[int, ReasoningEffort], ...] = (
    (1024, ReasoningEffort.MINIMAL),
    (4096, ReasoningEffort.LOW),
    (16384, ReasoningEffort.MEDIUM),
)

# effort -> Anthropic thinking budget_tokens.
EFFORT_TO_BUDGET: dict[ReasoningEffort, int] = {
    ReasoningEffort.MINIMAL: 1024,
    ReasoningEffort.LOW: 2048,
    ReasoningEffort.MEDIUM: 8192,
    ReasoningEffort.HIGH: 24576,
}


def effort_from_budget(budget_tokens: int) -> ReasoningEffort:
    """Map a token budget onto the nearest effort level (<=1024 minimal, <=4096
    low, <=16384 medium, else high)."""
    for limit, effort in _BUDGET_THRESHOLDS:
        if budget_tokens <= limit:
            return effort
    return ReasoningEffort.HIGH


def resolve_effort(reasoning: Reasoning | None) -> ReasoningEffort | None:
    """Effort to send to effort-based providers (OpenAI, Gemini). ``effort`` wins
    over ``budget_tokens``; ``None`` if neither is set."""
    if reasoning is None:
        return None
    if reasoning.effort is not None:
        return ReasoningEffort(reasoning.effort)
    if reasoning.budget_tokens is not None:
        return effort_from_budget(reasoning.budget_tokens)
    return None


def resolve_budget(reasoning: Reasoning | None) -> int | None:
    """Token budget to send to budget-based providers (Anthropic). ``budget_tokens``
    wins over ``effort``; ``None`` if neither is set."""
    if reasoning is None:
        return None
    if reasoning.budget_tokens is not None:
        return reasoning.budget_tokens
    if reasoning.effort is not None:
        return EFFORT_TO_BUDGET[ReasoningEffort(reasoning.effort)]
    return None
