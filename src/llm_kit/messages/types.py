"""Internal, provider-agnostic message representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True)
class Message:
    """One conversation turn. ``image_urls`` may be empty or arbitrarily long.

    Image URLs are resolved to base64 data URLs by the image fetcher at request
    time; this struct stays small so 1M of them fit in memory comfortably.
    """

    role: Role
    text: str | None = None
    image_urls: list[str] = field(default_factory=list)

    @classmethod
    def user(cls, text: str | None = None, image_urls: list[str] | None = None) -> Message:
        return cls(role=Role.USER, text=text, image_urls=image_urls or [])

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role=Role.SYSTEM, text=text)

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.ASSISTANT, text=text)
