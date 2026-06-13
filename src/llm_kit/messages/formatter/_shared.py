"""Helpers shared across every provider formatter."""

from __future__ import annotations

import json
from typing import Any


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
