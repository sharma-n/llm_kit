"""Offline batch orchestration: serialize → submit → reconcile.

This is the provider-batch counterpart to ``batch/processor.py``. It shares the
input contract (``list[BatchItem]``) and the output contract (``BatchResult``)
but none of the middle — there is no worker pool and no rate limiter. The body of
each request is built by the same client helper the live path uses, and each
downloaded result is parsed by the same ``parse_response``; the per-item failure
handling mirrors ``_process_item`` in the live processor.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Protocol

import httpx
from llm_kit.config.app import AppConfig
from llm_kit.config.llm import MessageFormat
from llm_kit.errors import LLMError
from llm_kit.llm.parsing import ResponseParser, parse_response
from llm_kit.llm.response import BatchItem, BatchResult
from llm_kit.messages.schema import ResponseSchema
from llm_kit.messages.settings import RequestSettings
from llm_kit.messages.types import Message
from llm_kit.provider_batch.anthropic import AnthropicBatchBackend
from llm_kit.provider_batch.backend import BatchBackend
from llm_kit.provider_batch.gemini import GeminiBatchBackend
from llm_kit.provider_batch.openai import OpenAIBatchBackend
from llm_kit.provider_batch.types import BatchJobStatus, BatchRequest, RawResult

class BodyBuilder(Protocol):
    """Keyword-based so per-item fields (settings, schema) flow by name."""

    def __call__(
        self,
        messages: list[Message],
        *,
        response_model: ResponseSchema | None = None,
        settings: RequestSettings | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Awaitable[dict[str, Any]]: ...


def build_backend(
    config: AppConfig, client: httpx.AsyncClient, api_key: str | None
) -> BatchBackend:
    """Select the transport backend for the configured provider.

    Mirrors the formatter selection in ``LLMClient`` — keyed by ``message_format``.
    """
    fmt = config.llm.message_format
    if fmt is MessageFormat.ANTHROPIC:
        return AnthropicBatchBackend(client, config, api_key)
    if fmt is MessageFormat.GEMINI:
        return GeminiBatchBackend(client, config, api_key)
    return OpenAIBatchBackend(client, config, api_key)


async def serialize_items(
    build_request_body: BodyBuilder,
    items: list[BatchItem],
    *,
    max_requests: int,
) -> tuple[list[BatchRequest], dict[str, dict[str, Any]]]:
    """Build one ``BatchRequest`` per item plus a custom_id→metadata manifest.

    Images are fetched and inlined here (via ``build_request_body``) since the
    provider's batch workers cannot reach back into our fetcher.
    """
    requests: list[BatchRequest] = []
    manifest: dict[str, dict[str, Any]] = {}
    for item in items:
        body = await build_request_body(
            item.messages,
            response_model=item.response_model,
            settings=item.settings,
            overrides=item.overrides or None,
        )
        requests.append(BatchRequest(custom_id=item.id, body=body))
        entry: dict[str, Any] = {"metadata": dict(item.metadata)}
        if isinstance(item.response_model, dict):
            # JSON-serialisable, so persist it: a fresh process can validate
            # (decode) results without re-supplying the items.
            entry["response_schema"] = copy.deepcopy(item.response_model)
        manifest[item.id] = entry
    if len(requests) > max_requests:
        raise ValueError(
            f"batch of {len(requests)} requests exceeds max_requests_per_batch={max_requests}"
        )
    return requests, manifest


async def reconcile(
    raw_results: AsyncIterator[RawResult],
    formatter: ResponseParser,
    *,
    response_models: dict[str, ResponseSchema | None],
    metadata: dict[str, dict[str, Any]],
) -> AsyncIterator[BatchResult]:
    """Turn raw downloaded results into ``BatchResult``s (out-of-order safe)."""
    async for raw in raw_results:
        meta = metadata.get(raw.custom_id, {})
        if raw.error is not None or raw.inner_body is None:
            yield BatchResult(id=raw.custom_id, error=_as_llm_error(raw), metadata=meta)
            continue
        try:
            response = parse_response(
                raw.inner_body, response_models.get(raw.custom_id), formatter
            )
        except LLMError as err:  # e.g. ValidationError against the response_model
            yield BatchResult(id=raw.custom_id, error=err, metadata=meta)
            continue
        yield BatchResult(id=raw.custom_id, response=response, metadata=meta)


def _as_llm_error(raw: RawResult) -> LLMError:
    err = raw.error
    if isinstance(err, LLMError):
        return err
    if err is not None:
        return LLMError(str(err), cause=err)
    return LLMError(f"batch result for {raw.custom_id!r} had neither body nor error")


def fetch_results(backend: BatchBackend, status: BatchJobStatus) -> AsyncIterator[RawResult]:
    """Thin pass-through so callers don't import the backend module directly."""
    return backend.fetch_results(status)
