"""Batch processor: bounded producer/consumer pipeline over an item stream.

Design (sized to 1M+ inputs):

- Producer reads input items one at a time and pushes onto a bounded queue.
- N workers pull items, call ``llm.invoke(...)``, and push ``BatchResult``s
  onto a bounded output queue.
- A single consumer drains the output queue and yields to the caller.

Bounded queues are what keep memory at O(workers + buffer) rather than O(N).
Each per-item failure becomes a ``BatchResult`` with ``error`` set — only a
``raise`` policy (or a programming bug) cancels the whole pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator
from typing import TYPE_CHECKING

from llm_kit.config.batch import BatchConfig, BatchFailurePolicy
from llm_kit.errors import LLMError
from llm_kit.llm.response import BatchItem, BatchResult

if TYPE_CHECKING:
    from llm_kit.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_SENTINEL: object = object()


async def run_batch_stream(
    *,
    llm: BaseLLM,
    items: AsyncIterable[BatchItem] | list[BatchItem],
    config: BatchConfig,
    completed_ids: set[str] | None = None,
) -> AsyncIterator[BatchResult]:
    """Async iterator yielding ``BatchResult``s as they complete."""
    completed = completed_ids or set()
    input_q: asyncio.Queue[BatchItem | object] = asyncio.Queue(maxsize=config.input_buffer_size)
    output_q: asyncio.Queue[BatchResult | object] = asyncio.Queue(
        maxsize=max(config.input_buffer_size, config.worker_concurrency)
    )

    stats = _Stats()

    async def producer() -> None:
        try:
            async for item in _aiter(items):
                if item.id in completed:
                    stats.skipped += 1
                    continue
                await input_q.put(item)
        finally:
            for _ in range(config.worker_concurrency):
                await input_q.put(_SENTINEL)

    async def worker(worker_id: int) -> None:
        while True:
            obj = await input_q.get()
            if obj is _SENTINEL:
                return
            assert isinstance(obj, BatchItem)
            result = await _process_item(llm, obj)
            await output_q.put(result)
            stats.completed += 1
            if config.progress_log_every and stats.completed % config.progress_log_every == 0:
                logger.info(
                    "batch progress: %d completed (%d ok, %d errors, %d skipped)",
                    stats.completed,
                    stats.ok,
                    stats.errors,
                    stats.skipped,
                )

    async def main_task() -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(producer(), name="batch-producer")
                for i in range(config.worker_concurrency):
                    tg.create_task(worker(i), name=f"batch-worker-{i}")
        finally:
            await output_q.put(_SENTINEL)

    driver = asyncio.create_task(main_task(), name="batch-driver")

    try:
        while True:
            obj = await output_q.get()
            if obj is _SENTINEL:
                break
            assert isinstance(obj, BatchResult)
            if obj.ok:
                stats.ok += 1
            else:
                stats.errors += 1
                if config.on_failure is BatchFailurePolicy.RAISE:
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


async def _process_item(llm: BaseLLM, item: BatchItem) -> BatchResult:
    try:
        response = await llm.invoke(
            item.messages,
            response_model=item.response_model,
            overrides=item.overrides or None,
        )
    except LLMError as err:
        logger.warning("batch item %s failed: %s", item.id, err)
        return BatchResult(id=item.id, error=err, metadata=item.metadata)
    except Exception as exc:  # programming bug — surface loudly via a result
        logger.exception("batch item %s raised unexpected exception", item.id)
        return BatchResult(
            id=item.id,
            error=LLMError(f"unexpected error: {type(exc).__name__}: {exc}", cause=exc),
            metadata=item.metadata,
        )
    return BatchResult(id=item.id, response=response, metadata=item.metadata)


async def _aiter(items: AsyncIterable[BatchItem] | list[BatchItem]) -> AsyncIterator[BatchItem]:
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
