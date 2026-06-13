"""LLM client for any endpoint that speaks the OpenAI or Anthropic wire format.

Provider-specific logic (message encoding, response parsing, auth headers) is
fully delegated to the active formatter — the client itself is wire-format agnostic.
Select the formatter via ``LLMConfig.message_format``.
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
from llm_kit.config.llm import MessageFormat
from llm_kit.errors import AuthError, LLMError, ProviderError, ValidationError
from llm_kit.http.retry import retry_request
from llm_kit.http.session import build_async_client
from llm_kit.images.fetcher import ImageFetcher
from llm_kit.llm.base import BaseLLM
from llm_kit.llm.response import BatchItem, BatchResult, LLMResponse, TokenUsage
from llm_kit.messages.formatter import AnthropicFormatter, GeminiFormatter, OpenAIFormatter
from llm_kit.messages.types import Message
from llm_kit.rate_limit.limiter import RateLimiter, TokenEstimator

logger = logging.getLogger(__name__)


class LLMClient(BaseLLM):
    """HTTP client for OpenAI-compatible and Anthropic chat endpoints."""

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

        if config.llm.message_format is MessageFormat.ANTHROPIC:
            self._formatter: AnthropicFormatter | GeminiFormatter | OpenAIFormatter = AnthropicFormatter()
        elif config.llm.message_format is MessageFormat.GEMINI:
            self._formatter = GeminiFormatter()
        else:
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
            finish_reason=self._formatter.finish_reason(data),
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

    def _resolve_api_key(self, config: Any) -> str | None:
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
        headers.update(self._formatter.default_headers(self._api_key))
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
        body = self._formatter.build_body(
            model=cfg.model,
            messages=self._formatter.format_messages(messages, encoded_images),
            system=self._formatter.system_prompt(messages),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            extra_sampling=cfg.extra_sampling,
            response_model=response_model,
        )
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
        text = self._formatter.extract_text(data)
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
        prompt, completion, total = self._formatter.extract_usage(data)
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

__all__ = ["AuthError", "LLMClient", "LLMError"]
