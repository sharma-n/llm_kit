"""Shared response parsing: raw provider JSON → ``LLMResponse``.

Both the live client (``LLMClient.invoke``) and the offline provider-batch driver
parse a provider response body identically — extract text, optionally validate it
against a pydantic ``response_model``, pull usage + finish_reason — and assemble
the same ``LLMResponse``. That logic lives here so the two paths can never drift,
and a schema mismatch always becomes the same typed ``ValidationError``.

The formatter is duck-typed via the ``ResponseParser`` protocol so this module
stays decoupled from the concrete ``OpenAIFormatter`` / ``AnthropicFormatter`` /
``GeminiFormatter`` classes.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import ValidationError as PydanticValidationError

from llm_kit.errors import ValidationError
from llm_kit.llm.response import LLMResponse, TokenUsage, ToolCall
from llm_kit.messages.schema import ResponseSchema, schema_name


class ResponseParser(Protocol):
    """The response-side slice of a formatter that ``parse_response`` needs."""

    def extract_text(self, data: dict[str, Any]) -> str: ...
    def extract_usage(self, data: dict[str, Any]) -> tuple[int, int, int, int]: ...
    def finish_reason(self, data: dict[str, Any]) -> str | None: ...
    def extract_tool_calls(self, data: dict[str, Any]) -> list[ToolCall]: ...


def parse_response(
    data: dict[str, Any],
    response_model: ResponseSchema | None,
    formatter: ResponseParser,
    *,
    latency_seconds: float = 0.0,
) -> LLMResponse:
    """Build an ``LLMResponse`` from a parsed provider response body.

    Raises ``ValidationError`` if ``response_model`` is supplied and the response
    text does not satisfy it.
    """
    text = formatter.extract_text(data)
    parsed = validate_text(text, response_model)
    prompt, completion, total, reasoning = formatter.extract_usage(data)
    return LLMResponse(
        text=text,
        parsed=parsed,
        usage=TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=reasoning,
        ),
        model=data.get("model"),
        finish_reason=formatter.finish_reason(data),
        tool_calls=formatter.extract_tool_calls(data),
        raw=data,
        latency_seconds=latency_seconds,
    )


def validate_text(text: str, response_model: ResponseSchema | None) -> Any:
    """Validate ``text`` against ``response_model``; ``None`` if no model.

    A Pydantic class validates and returns the model instance. A JSON schema
    ``dict`` is *not* enforced client-side: the text is just decoded
    (``json.loads``) and returned as plain Python data. Either way a failure
    becomes the same typed ``ValidationError``; shared by the non-streaming
    ``parse_response`` and the streaming assembler.
    """
    if response_model is None:
        return None
    if isinstance(response_model, dict):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"response is not valid JSON for schema {schema_name(response_model)!r}: {exc}",
                raw_content=text,
                cause=exc,
            ) from exc
    try:
        return response_model.model_validate_json(text)
    except PydanticValidationError as exc:
        raise ValidationError(
            f"response did not match {response_model.__name__}: {exc.errors()}",
            raw_content=text,
            cause=exc,
        ) from exc
