"""LLM client for any endpoint that speaks the OpenAI or Anthropic wire format.

Provider-specific logic (message encoding, response parsing, auth headers) is
fully delegated to the active formatter — the client itself is wire-format agnostic.
Select the formatter via ``LLMConfig.message_format``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any, Self

import httpx
from pydantic import BaseModel

from llm_kit.batch.processor import run_batch_stream
from llm_kit.config.app import AppConfig
from llm_kit.config.llm import MessageFormat
from llm_kit.errors import AuthError, LLMError, ProviderError
from llm_kit.http.retry import retry_request
from llm_kit.http.session import build_async_client
from llm_kit.images.fetcher import ImageFetcher
from llm_kit.llm.base import BaseLLM
from llm_kit.llm.parsing import parse_response
from llm_kit.llm.response import BatchItem, BatchResult, BatchSummary, LLMResponse
from llm_kit.messages.formatter import AnthropicFormatter, GeminiFormatter, OpenAIFormatter
from llm_kit.messages.types import Message
from llm_kit.provider_batch.driver import build_backend, reconcile, serialize_items
from llm_kit.provider_batch.job import BatchJob
from llm_kit.provider_batch.types import BatchJobStatus
from llm_kit.rate_limit.limiter import RateLimiter, TokenEstimator

logger = logging.getLogger(__name__)


async def _skip_completed(
    results: AsyncIterator[BatchResult], completed: set[str]
) -> AsyncIterator[BatchResult]:
    """Drop results whose id is already present in the output jsonl."""
    async for result in results:
        if result.id in completed:
            continue
        yield result


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
        self._batch_backend = build_backend(config, self._client, self._api_key)

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
        body = await self._build_request_body(messages, response_model, overrides)

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
        result = parse_response(
            data, response_model, self._formatter, latency_seconds=elapsed
        )
        self._limiter.true_up_tokens(
            estimated=estimated, actual=result.usage.total_tokens
        )
        return result

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

    async def batch_to_jsonl(
        self,
        items: AsyncIterable[BatchItem] | list[BatchItem],
        output_path: str | Path,
        *,
        resume: bool = True,
    ) -> BatchSummary:
        summary = await super().batch_to_jsonl(items, output_path, resume=resume)
        llm_cfg = self._config.llm
        summary.estimated_cost = (
            summary.prompt_tokens / 1_000_000 * llm_cfg.prompt_cost_per_1m_tokens
            + summary.completion_tokens / 1_000_000 * llm_cfg.completion_cost_per_1m_tokens
        )
        return summary

    # ------------------------------------------------------ provider batch API

    async def submit_batch(self, items: list[BatchItem]) -> BatchJob:
        """Serialize ``items`` and submit them to the provider's batch endpoint.

        Returns a ``BatchJob`` handle. Call ``handle.save(path)`` to persist it
        for cross-process resume — the manifest carries each item's metadata, but
        ``response_model`` (a Python type) is not persisted, so cross-process
        result validation requires re-supplying ``items`` to ``fetch_batch_results``.
        """
        cfg = self._config.provider_batch
        requests, manifest = await serialize_items(
            self._build_request_body, items, max_requests=cfg.max_requests_per_batch
        )
        batch_id = await self._batch_backend.submit(requests)
        return BatchJob(
            provider=str(self._config.llm.message_format),
            batch_id=batch_id,
            manifest=manifest,
        )

    async def poll_batch(self, job: BatchJob) -> BatchJobStatus:
        """Return a normalized status snapshot for the submitted job."""
        return await self._batch_backend.poll(job.batch_id)

    async def fetch_batch_results(
        self, job: BatchJob, *, items: list[BatchItem] | None = None
    ) -> AsyncIterator[BatchResult]:
        """Download and reconcile results for ``job`` into ``BatchResult``s.

        ``response_model`` validation is applied only for ids present in
        ``items``; otherwise results come back as raw text. Metadata is always
        restored from the job manifest.
        """
        status = await self._batch_backend.poll(job.batch_id)
        response_models = (
            {it.id: it.response_model for it in items} if items else {}
        )
        metadata = {cid: job.metadata_for(cid) for cid in job.manifest}
        raw = self._batch_backend.fetch_results(status)
        async for result in reconcile(
            raw, self._formatter, response_models=response_models, metadata=metadata
        ):
            yield result

    async def run_batch_api(
        self,
        items: list[BatchItem],
        output_path: str | Path,
        *,
        poll_interval: float | None = None,
        resume_job: BatchJob | None = None,
    ) -> int:
        """Submit, poll to completion, and write reconciled results to jsonl.

        A convenience that holds the process open across the job's lifetime. Pass
        ``resume_job`` (e.g. ``BatchJob.load(path)``) to reconnect to an already
        submitted job instead of submitting again. Two-level resume: the saved
        handle reconnects to the in-flight job, and ids already present in
        ``output_path`` are skipped on write.
        """
        from llm_kit.batch.jsonl import load_completed_ids, write_results_to_jsonl

        interval = (
            poll_interval
            if poll_interval is not None
            else self._config.provider_batch.poll_interval_seconds
        )
        job = resume_job or await self.submit_batch(items)

        status = await self._batch_backend.poll(job.batch_id)
        while not status.state.is_terminal:
            await asyncio.sleep(interval)
            status = await self._batch_backend.poll(job.batch_id)

        path = Path(output_path)
        completed = load_completed_ids(path) if path.exists() else set()
        if completed:
            logger.info("resuming: %d items already in %s", len(completed), path)
        response_models = {it.id: it.response_model for it in items}
        metadata = {cid: job.metadata_for(cid) for cid in job.manifest}
        raw = self._batch_backend.fetch_results(status)
        results = reconcile(
            raw, self._formatter, response_models=response_models, metadata=metadata
        )
        return await write_results_to_jsonl(path, _skip_completed(results, completed))

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

    async def _build_request_body(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Fetch+inline images and assemble the provider request body.

        Shared by the live ``invoke`` path and the provider-batch serializer so
        both produce byte-identical request bodies for the same item.
        """
        encoded = await self._images.fetch_many(
            [u for m in messages for u in m.image_urls]
        )
        return self._build_body(messages, encoded, response_model, overrides)

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


__all__ = ["AuthError", "LLMClient", "LLMError"]
