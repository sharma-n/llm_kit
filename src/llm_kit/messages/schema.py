"""Structured-output schema type and the helpers formatters share.

A ``response_model`` may be a Pydantic class (validated on the way back) or a
plain JSON schema ``dict`` (the decoded JSON is returned unvalidated).
"""

from __future__ import annotations

import copy
import re
from typing import Any

from pydantic import BaseModel

ResponseSchema = type[BaseModel] | dict[str, Any]


def json_schema(schema: ResponseSchema) -> dict[str, Any]:
    """A fresh JSON schema dict for ``schema``; never aliases the caller's dict.

    Formatters mutate/transform the result freely (strictify, enforce passes).
    """
    if isinstance(schema, dict):
        return copy.deepcopy(schema)
    return schema.model_json_schema()


def schema_name(schema: ResponseSchema) -> str:
    """Name for providers that need one: class name, dict ``title``, else ``"response"``.

    Sanitised to ``[A-Za-z0-9_-]{1,64}`` (OpenAI's name rules).
    """
    if isinstance(schema, dict):
        raw = schema.get("title")
        name = raw if isinstance(raw, str) and raw else "response"
    else:
        name = schema.__name__
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64]
    return name or "response"


__all__ = ["ResponseSchema", "json_schema", "schema_name"]
