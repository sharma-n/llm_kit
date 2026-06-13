"""Streaming events and the fragment assembler.

``invoke_stream`` yields a small set of **public** events the consumer reacts to:

  - ``TextChunk``       — a text delta; forward to the user immediately (TTFT).
  - ``ToolCallStarted`` — a UI hint, emitted the first time a tool call's name is
    known (its arguments are still arriving).
  - ``StreamEnd``       — terminal; carries the fully-assembled ``LLMResponse``
    (text, parsed tool calls, usage, finish reason, optional ``response_model``).

The per-provider SSE decoding lives in the formatter, which emits the **internal**
normalized fragments below. ``StreamAssembler`` turns those fragments into public
events and the final ``LLMResponse`` — the streaming analog of
``parse_response``: provider-agnostic, shared across every formatter, so the
hard fragment-reassembly logic exists once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from llm_kit.llm.parsing import validate_text
from llm_kit.llm.response import LLMResponse, TokenUsage
from llm_kit.messages.streaming import (
    FinishFragment,
    MetaFragment,
    StreamFragment,
    TextFragment,
    ToolCallFragment,
    UsageFragment,
)
from llm_kit.messages.tools import ToolCall

# --------------------------------------------------------------------- public


@dataclass(slots=True)
class TextChunk:
    text: str


@dataclass(slots=True)
class ToolCallStarted:
    index: int
    id: str | None
    name: str


@dataclass(slots=True)
class StreamEnd:
    response: LLMResponse


StreamEvent = TextChunk | ToolCallStarted | StreamEnd


# --------------------------------------------------------------------- assembler


@dataclass(slots=True)
class _ToolBuffer:
    id: str | None = None
    name: str | None = None
    args: str = ""
    started: bool = False


class StreamAssembler:
    """Accumulate normalized fragments into public events + a final response."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._tools: dict[int, _ToolBuffer] = {}
        self._order: list[int] = []
        self._usage = TokenUsage()
        self._finish: str | None = None
        self._model: str | None = None

    def ingest(self, fragment: StreamFragment) -> list[StreamEvent]:
        """Fold one fragment into state, returning any public events to yield."""
        if isinstance(fragment, TextFragment):
            if fragment.text:
                self._text.append(fragment.text)
                return [TextChunk(fragment.text)]
            return []
        if isinstance(fragment, ToolCallFragment):
            return self._ingest_tool(fragment)
        if isinstance(fragment, UsageFragment):
            self._usage = TokenUsage(
                prompt_tokens=fragment.prompt_tokens,
                completion_tokens=fragment.completion_tokens,
                total_tokens=fragment.total_tokens,
            )
            return []
        if isinstance(fragment, FinishFragment):
            self._finish = fragment.reason
            return []
        if isinstance(fragment, MetaFragment):
            if fragment.model:
                self._model = fragment.model
            return []
        return []

    def _ingest_tool(self, frag: ToolCallFragment) -> list[StreamEvent]:
        buf = self._tools.get(frag.index)
        if buf is None:
            buf = _ToolBuffer()
            self._tools[frag.index] = buf
            self._order.append(frag.index)
        if frag.id is not None:
            buf.id = frag.id
        if frag.name is not None:
            buf.name = frag.name
        if frag.args_delta:
            buf.args += frag.args_delta
        # Emit the UI hint the first time we know the call's name.
        if not buf.started and buf.name:
            buf.started = True
            return [ToolCallStarted(index=frag.index, id=buf.id, name=buf.name)]
        return []

    def finish(
        self,
        *,
        response_model: type[BaseModel] | None = None,
        latency_seconds: float = 0.0,
    ) -> StreamEnd:
        text = "".join(self._text)
        tool_calls = [
            ToolCall(
                id=buf.id or f"call_{idx}",
                name=buf.name or "",
                arguments=_parse_args(buf.args),
            )
            for idx in self._order
            if (buf := self._tools[idx]).name is not None
        ]
        parsed = validate_text(text, response_model) if response_model else None
        return StreamEnd(
            LLMResponse(
                text=text,
                parsed=parsed,
                usage=self._usage,
                model=self._model,
                finish_reason=self._finish,
                tool_calls=tool_calls,
                latency_seconds=latency_seconds,
            )
        )


def _parse_args(raw: str) -> dict[str, Any]:
    """Parse accumulated argument text to a dict; preserve raw if not valid JSON."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_raw": raw}


__all__ = [
    "StreamAssembler",
    "StreamEnd",
    "StreamEvent",
    "TextChunk",
    "ToolCallStarted",
]
