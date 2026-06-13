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


@dataclass(slots=True)
class Message:
    """One conversation turn. ``image_urls`` may be empty or arbitrarily long.

    Image URLs are resolved to base64 data URLs by the image fetcher at request
    time; this struct stays small so 1M of them fit in memory comfortably.

    Two optional fields support replaying a tool-calling exchange back to the
    model (the agent path); both default empty, so the batch path pays nothing:
      - ``tool_calls`` — set on an ``ASSISTANT`` turn that asked to call tools.
      - ``tool_call_id`` — set on a ``TOOL`` turn carrying one tool's result
        (``text`` holds the observation content).
    """

    role: Role
    text: str | None = None
    image_urls: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def user(cls, text: str | None = None, image_urls: list[str] | None = None) -> Message:
        return cls(role=Role.USER, text=text, image_urls=image_urls or [])

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role=Role.SYSTEM, text=text)

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.ASSISTANT, text=text)

    @classmethod
    def assistant_tool_calls(
        cls, tool_calls: list[ToolCall], text: str | None = None
    ) -> Message:
        """An assistant turn that requested one or more tool calls."""
        return cls(role=Role.ASSISTANT, text=text, tool_calls=list(tool_calls))

    @classmethod
    def tool_result(cls, call_id: str, content: str) -> Message:
        """A tool's result, fed back to the model as an observation."""
        return cls(role=Role.TOOL, text=content, tool_call_id=call_id)
