"""OpenAI-compatible chat-completions client.

Targets any endpoint that speaks ``POST /v1/chat/completions`` with the OpenAI
request/response shape — OpenAI itself, Together, vLLM, Ollama, etc. Provider
quirks are handled via ``LLMConfig.structured_output_style`` and
``LLMConfig.extra_headers``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, Self

import httpx
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from llm_kit.batch.processor import run_batch_stream
from llm_kit.config.app import AppConfig
from llm_kit.config.llm import LLMConfig, StructuredOutputStyle
from llm_kit.errors import AuthError, LLMError, ProviderError, ValidationError
from llm_kit.http.retry import retry_request
from llm_kit.http.session import build_async_client
from llm_kit.images.fetcher import ImageFetcher
from llm_kit.llm.base import BaseLLM
from llm_kit.llm.response import BatchItem, BatchResult, LLMResponse, TokenUsage
from llm_kit.messages.formatter import OpenAIFormatter
from llm_kit.messages.types import Message
from llm_kit.rate_limit.limiter import RateLimiter, TokenEstimator

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """HTTP client for any OpenAI-shaped chat-completions endpoint."""

    def __init__(
        self,
        config: AppConfig,
        *,
        client: httpx.AsyncClient | None = None,
        token_estimator: TokenEstimator | None = None,
        owns_client: bool | None = None,
    ) -> None:
        self._config = config
        self._client = client or build_async_client(config.http)
        # If we built the client, we close it. If it was injected, the caller
        # owns its lifecycle and we leave it alone.
        self._owns_client = client is None if owns_client is None else owns_client

        self._formatter = OpenAIFormatter(
            structured_output_style=config.llm.structured_output_style
        )
        self._images = ImageFetcher(self._client, config.image)
        self._limiter = RateLimiter(config.rate_limit, token_estimator=token_estimator)
        self._api_key = self._resolve_api_key(config.llm)

    # ------------------------------------------------------------------ public

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        return cls(AppConfig.from_yaml(path))

    async def invoke(
        self,
        messages: list[Message],
        *,
        response_model: type[BaseModel] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        encoded = await self._images.fetch_many(
            [u for m in messages for u in m.image_urls]
        )
        body = self._build_body(messages, encoded, response_model, overrides)

        estimated = self._limiter.estimate_tokens(messages)
        started = time.perf_counter()

        async with self._limiter.acquire(estimated_tokens=estimated):
            response = await retry_request(
                lambda: self._client.post(
                    self._endpoint(),
                    headers=self._headers(),
                    json=body,
                ),
                retry=self._config.http.llm_retry,
                operation="chat_completion",
            )

        elapsed = time.perf_counter() - started
        data = self._parse_json(response)
        parsed_obj, text = self._extract_content(data, response_model)
        usage = self._extract_usage(data)
        self._limiter.true_up_tokens(estimated=estimated, actual=usage.total_tokens)

        return LLMResponse(
            text=text,
            parsed=parsed_obj,
            usage=usage,
            model=data.get("model"),
            finish_reason=_safe_finish_reason(data),
            raw=data,
            latency_seconds=elapsed,
        )

    def batch_stream(
        self,
        items: AsyncIterable[BatchItem] | list[BatchItem],
        *,
        completed_ids: set[str] | None = None,
    ) -> AsyncIterator[BatchResult]:
        return run_batch_stream(
            llm=self,
            items=items,
            config=self._config.batch,
            completed_ids=completed_ids,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---------------------------------------------------------------- internal

    def _resolve_api_key(self, config: LLMConfig) -> str | None:
        if config.api_key:
            return config.api_key
        if config.api_key_env:
            return os.environ.get(config.api_key_env)
        return None

    def _endpoint(self) -> str:
        base = self._config.llm.base_url.rstrip("/")
        path = self._config.llm.chat_completions_path
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        headers.update(self._config.llm.extra_headers)
        return headers

    def _build_body(
        self,
        messages: list[Message],
        encoded_images: dict[str, str],
        response_model: type[BaseModel] | None,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        cfg = self._config.llm
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": self._formatter.format_messages(messages, encoded_images),
        }
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            body["top_p"] = cfg.top_p
        if cfg.max_tokens is not None:
            body["max_tokens"] = cfg.max_tokens
        body.update(cfg.extra_sampling)

        rf = self._formatter.response_format(response_model)
        if rf is not None:
            if rf.get("_style") == "tool_use":
                tool = rf["tool"]
                body["tools"] = [tool]
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool["function"]["name"]},
                }
            else:
                body["response_format"] = rf

        if overrides:
            body.update(overrides)
        return body

    def _parse_json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"non-JSON response from provider: {response.text[:200]!r}",
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(f"unexpected response shape: {type(data).__name__}")
        return data

    def _extract_content(
        self,
        data: dict[str, Any],
        response_model: type[BaseModel] | None,
    ) -> tuple[BaseModel | None, str]:
        text = _extract_text(data, self._config.llm.structured_output_style)
        if response_model is None:
            return None, text
        try:
            parsed = response_model.model_validate_json(text)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"response did not match {response_model.__name__}: {exc.errors()}",
                raw_content=text,
                cause=exc,
            ) from exc
        return parsed, text

    def _extract_usage(self, data: dict[str, Any]) -> TokenUsage:
        usage = data.get("usage") or {}
        return TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )


def _safe_finish_reason(data: dict[str, Any]) -> str | None:
    try:
        return data["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def _extract_text(data: dict[str, Any], style: StructuredOutputStyle) -> str:
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"missing choices[0].message in response: {data!r}", cause=exc) from exc

    if style is StructuredOutputStyle.TOOL_USE:
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            args = tool_calls[0].get("function", {}).get("arguments")
            if isinstance(args, str):
                return args
            if isinstance(args, dict):
                return json.dumps(args)

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some providers return content as an array of typed parts even for
        # text-only responses; concatenate the text parts.
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(parts)
    return content if content is not None else ""


# Re-export for symmetry with `from llm_kit.llm import AuthError` if someone wants it.
__all__ = ["AuthError", "LLMError", "OpenAICompatibleLLM"]
