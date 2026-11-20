"""Internal, provider-agnostic message representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from llm_kit.messages.tools import ToolCall


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class TextPart:
    """A run of text inside a message."""

    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An image inside a message: either a ``url`` or raw ``data`` bytes.

    Exactly one of ``url`` / ``data`` is set. ``url`` may be an http(s) URL
    (fetched + cached at request time) or a ``data:...;base64,...`` URL (used as
    is, no network). ``data`` requires ``mime_type`` and is base64-encoded once
    at request time; it never goes through the fetcher or its cache.
    """

    url: str | None = None
    data: bytes | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("ImagePart requires exactly one of `url` or `data`")
        if self.data is not None and not self.mime_type:
            raise ValueError("ImagePart with `data` requires `mime_type`")
        if self.url is not None and self.mime_type is not None:
            raise ValueError("ImagePart `mime_type` is only valid with `data`")

    @classmethod
    def from_url(cls, url: str) -> ImagePart:
        return cls(url=url)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> ImagePart:
        return cls(data=bytes(data), mime_type=mime_type)


Part = TextPart | ImagePart


def _as_parts(items: tuple[str | Part, ...]) -> list[Part]:
    return [TextPart(i) if isinstance(i, str) else i for i in items]


@dataclass(slots=True)
class Message:
    """One conversation turn: a role plus an ordered list of content ``parts``.

    Parts are rendered by the formatters in exactly the order given, one wire
    block per part (text/image interleaving is preserved). Image bytes are
    resolved to base64 at request time; this struct stays small so 1M of them
    fit in memory comfortably.

    Two optional fields support replaying a tool-calling exchange back to the
    model (the agent path); both default empty, so the batch path pays nothing:
      - ``tool_calls`` — set on an ``ASSISTANT`` turn that asked to call tools.
      - ``tool_call_id`` — set on a ``TOOL`` turn carrying one tool's result
        (its text parts hold the observation content).
    """

    role: Role
    parts: list[Part] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def user(cls, *parts: str | Part) -> Message:
        """``Message.user("look", ImagePart.from_url(u), "and", ...)``; ``str`` -> TextPart."""
        return cls(role=Role.USER, parts=_as_parts(parts))

    @classmethod
    def system(cls, *parts: str | Part) -> Message:
        return cls(role=Role.SYSTEM, parts=_as_parts(parts))

    @classmethod
    def assistant(cls, *parts: str | Part) -> Message:
        return cls(role=Role.ASSISTANT, parts=_as_parts(parts))

    @classmethod
    def assistant_tool_calls(
        cls, tool_calls: list[ToolCall], text: str | None = None
    ) -> Message:
        """An assistant turn that requested one or more tool calls."""
        parts: list[Part] = [TextPart(text)] if text else []
        return cls(role=Role.ASSISTANT, parts=parts, tool_calls=list(tool_calls))

    @classmethod
    def tool_result(cls, call_id: str, content: str) -> Message:
        """A tool's result, fed back to the model as an observation."""
        return cls(role=Role.TOOL, parts=[TextPart(content)], tool_call_id=call_id)
