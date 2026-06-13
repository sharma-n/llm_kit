"""Dedicated micro-batching driver for embeddings.

Why a separate driver instead of reusing ``batch/processor.py``: the chat
pipeline's unit is *one item → one request*. Embeddings invert that — the
endpoint takes an array, so the unit is *one request → many inputs*. The packer
stage below accumulates inputs into request-sized chunks (bounded by both an
input count and an estimated-token budget) before any HTTP call is made. That
packing is the dominant throughput lever and doesn't fit the per-item processor.

Shape mirrors the processor otherwise: a producer/packer, N workers, and a
single draining consumer, all inside a TaskGroup with sentinel shutdown. Memory
stays O(worker_concurrency × max_inputs_per_request + buffers), not O(N).

Failure granularity is coarser than chat: an HTTP failure fails the whole chunk,
so every input in that request gets the same error result (under the RECORD
policy) or aborts the stream (under RAISE).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import TYPE_CHECKING, Protocol

from llm_kit.config.batch import BatchConfig, BatchFailurePolicy
from llm_kit.config.embed import EmbeddingConfig
from llm_kit.embed.response import EmbeddingResponse, EmbeddingResult, EmbedItem, split_usage
from llm_kit.errors import LLMError
from llm_kit.llm.response import TokenUsage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_SENTINEL: object = object()

TextEstimator = Callable[[list[str]], int]


class _EmbedBackend(Protocol):
    """The slice of the embedder the driver needs (duck-typed, no import cycle)."""

    async def embed_request(
        self, texts: list[str]
    ) -> tuple[list[list[float]], TokenUsage, str | None]: ...


async def run_embed_stream(
    *,
    embedder: _EmbedBackend,
    items: AsyncIterable[EmbedItem] | list[EmbedItem],
    batch_config: BatchConfig,
    embed_config: EmbeddingConfig,
    estimator: TextEstimator,
    completed_ids: set[str] | None = None,
) -> AsyncIterator[EmbeddingResult]:
    """Async iterator yielding one ``EmbeddingResult`` per input as chunks land."""
    completed = completed_ids or set()
    max_inputs = embed_config.max_inputs_per_request
    max_tokens = embed_config.max_tokens_per_request
    workers = batch_config.worker_concurrency

    # Chunks can be large (up to max_inputs items), so bound the chunk queue by
    # the worker count rather than input_buffer_size to keep memory sane.
    chunk_q: asyncio.Queue[list[EmbedItem] | object] = asyncio.Queue(maxsize=max(workers, 1))
    output_q: asyncio.Queue[EmbeddingResult | object] = asyncio.Queue(
        maxsize=max(batch_config.input_buffer_size, workers)
    )

    stats = _Stats()

    async def packer() -> None:
        """Read items, pack into request-sized chunks, push to the chunk queue."""
        try:
            current: list[EmbedItem] = []
            current_tokens = 0
            async for item in _aiter(items):
                if item.id in completed:
                    stats.skipped += 1
                    continue
                est = estimator([item.text])
                # Flush before adding if this item would breach either cap. An
                # oversized lone input (est > max_tokens) still goes out as its
                # own chunk — the provider, not us, rejects truly-too-big inputs.
                if current and (
                    len(current) >= max_inputs or current_tokens + est > max_tokens
                ):
                    await chunk_q.put(current)
                    current = []
                    current_tokens = 0
                current.append(item)
                current_tokens += est
            if current:
                await chunk_q.put(current)
        finally:
            for _ in range(workers):
                await chunk_q.put(_SENTINEL)

    async def worker() -> None:
        while True:
            obj = await chunk_q.get()
            if obj is _SENTINEL:
                return
            assert isinstance(obj, list)
            for result in await _process_chunk(embedder, obj, estimator):
                await output_q.put(result)
                stats.completed += 1
                if (
                    batch_config.progress_log_every
                    and stats.completed % batch_config.progress_log_every == 0
                ):
                    logger.info(
                        "embed progress: %d completed (%d ok, %d errors, %d skipped)",
                        stats.completed,
                        stats.ok,
                        stats.errors,
                        stats.skipped,
                    )

    async def main_task() -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(packer(), name="embed-packer")
                for i in range(workers):
                    tg.create_task(worker(), name=f"embed-worker-{i}")
        finally:
            await output_q.put(_SENTINEL)

    driver = asyncio.create_task(main_task(), name="embed-driver")

    try:
        while True:
            obj = await output_q.get()
            if obj is _SENTINEL:
                break
            assert isinstance(obj, EmbeddingResult)
            if obj.ok:
                stats.ok += 1
            else:
                stats.errors += 1
                if batch_config.on_failure is BatchFailurePolicy.RAISE:
                    driver.cancel()
                    assert obj.error is not None
                    raise obj.error
            yield obj
        await driver  # surface programming errors from inside the pipeline
    finally:
        if not driver.done():
            driver.cancel()
            try:
                await driver
            except (asyncio.CancelledError, BaseException):  # noqa: BLE001
                pass


async def _process_chunk(
    embedder: _EmbedBackend,
    chunk: list[EmbedItem],
    estimator: TextEstimator,
) -> list[EmbeddingResult]:
    texts = [it.text for it in chunk]
    try:
        vectors, usage, model = await embedder.embed_request(texts)
    except LLMError as err:
        logger.warning("embed chunk of %d failed: %s", len(chunk), err)
        return [EmbeddingResult(id=it.id, error=err, metadata=it.metadata) for it in chunk]
    except Exception as exc:  # programming bug — surface loudly per input
        logger.exception("embed chunk raised unexpected exception")
        wrapped = LLMError(f"unexpected error: {type(exc).__name__}: {exc}", cause=exc)
        return [EmbeddingResult(id=it.id, error=wrapped, metadata=it.metadata) for it in chunk]

    per_usage = split_usage(usage, [estimator([t]) for t in texts])
    return [
        EmbeddingResult(
            id=it.id,
            response=EmbeddingResponse(
                vector=vectors[i], index=i, usage=per_usage[i], model=model
            ),
            metadata=it.metadata,
        )
        for i, it in enumerate(chunk)
    ]


async def _aiter(
    items: AsyncIterable[EmbedItem] | list[EmbedItem],
) -> AsyncIterator[EmbedItem]:
    if hasattr(items, "__aiter__"):
        async for item in items:  # type: ignore[union-attr]
            yield item
        return
    for item in items:  # type: ignore[union-attr]
        yield item


class _Stats:
    __slots__ = ("completed", "ok", "errors", "skipped")

    def __init__(self) -> None:
        self.completed = 0
        self.ok = 0
        self.errors = 0
        self.skipped = 0
