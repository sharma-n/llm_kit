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

from typing import Any, Protocol

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from llm_kit.errors import ValidationError
from llm_kit.llm.response import LLMResponse, TokenUsage


class ResponseParser(Protocol):
    """The response-side slice of a formatter that ``parse_response`` needs."""

    def extract_text(self, data: dict[str, Any]) -> str: ...
    def extract_usage(self, data: dict[str, Any]) -> tuple[int, int, int]: ...
    def finish_reason(self, data: dict[str, Any]) -> str | None: ...


def parse_response(
    data: dict[str, Any],
    response_model: type[BaseModel] | None,
    formatter: ResponseParser,
    *,
    latency_seconds: float = 0.0,
) -> LLMResponse:
    """Build an ``LLMResponse`` from a parsed provider response body.

    Raises ``ValidationError`` if ``response_model`` is supplied and the response
    text does not satisfy it.
    """
    text = formatter.extract_text(data)
    parsed = _validate(text, response_model)
    prompt, completion, total = formatter.extract_usage(data)
    return LLMResponse(
        text=text,
        parsed=parsed,
        usage=TokenUsage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
        ),
        model=data.get("model"),
        finish_reason=formatter.finish_reason(data),
        raw=data,
        latency_seconds=latency_seconds,
    )


def _validate(text: str, response_model: type[BaseModel] | None) -> BaseModel | None:
    if response_model is None:
        return None
    try:
        return response_model.model_validate_json(text)
    except PydanticValidationError as exc:
        raise ValidationError(
            f"response did not match {response_model.__name__}: {exc.errors()}",
            raw_content=text,
            cause=exc,
        ) from exc
