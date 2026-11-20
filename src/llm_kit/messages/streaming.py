"""Normalized stream fragments — what a formatter decodes the wire into.

These live in the ``messages/`` layer (alongside the formatter that produces
them) so the formatter never has to import upward into ``llm/``. The ``llm/``
layer's ``StreamAssembler`` consumes these fragments and assembles them into an
``LLMResponse`` — the streaming analog of how ``parse_response`` sits above the
formatter's ``extract_*`` methods.

The set is deliberately minimal and provider-agnostic; each formatter's
``stream_events`` maps its provider's SSE shape onto these.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TextFragment:
    """A text delta."""

    text: str


@dataclass(slots=True)
class ToolCallFragment:
    """A piece of a tool call. ``index`` groups fragments of the same call;
    ``id``/``name`` arrive once, ``args_delta`` accumulates across fragments."""

    index: int
    id: str | None = None
    name: str | None = None
    args_delta: str = ""


@dataclass(slots=True)
class UsageFragment:
    """Token usage, typically on the final chunk."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0


@dataclass(slots=True)
class FinishFragment:
    """The stop/finish reason."""

    reason: str | None


@dataclass(slots=True)
class MetaFragment:
    """Stream metadata (e.g. the resolved model name)."""

    model: str | None


StreamFragment = (
    TextFragment | ToolCallFragment | UsageFragment | FinishFragment | MetaFragment
)


__all__ = [
    "FinishFragment",
    "MetaFragment",
    "StreamFragment",
    "TextFragment",
    "ToolCallFragment",
    "UsageFragment",
]
