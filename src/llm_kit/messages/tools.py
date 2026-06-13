"""Provider-agnostic tool-calling types.

A leaf module (no internal deps) so both ``messages/types.py`` and
``llm/response.py`` can reference these without an import cycle.

The split mirrors the request/response halves the formatter owns:
  - ``ToolDefinition`` is what the caller *sends* (rendered into the provider's
    native tool slot by the formatter).
  - ``ToolCall`` is what the model *returns* (decoded from the provider's wire
    shape by the formatter). Decoding lives in ``llm_kit``; *executing* the call
    is the consumer's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolDefinition:
    """A tool offered to the model. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    """A tool call the model emitted, decoded to a provider-agnostic shape.

    ``arguments`` is the parsed argument object. For streamed calls the formatter
    accumulates the raw argument fragments and parses them once complete; if the
    accumulated text is not valid JSON it is preserved under the ``_raw`` key so
    the consumer can still inspect it rather than silently losing it.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


__all__ = ["ToolCall", "ToolDefinition"]
