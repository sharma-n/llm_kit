"""Helpers shared across every provider formatter."""

from __future__ import annotations

import json
from typing import Any

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
