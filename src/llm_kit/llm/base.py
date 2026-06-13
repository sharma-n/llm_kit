"""Abstract LLM client contract.

Concrete subclasses implement the wire-level details. Everything above this
layer (the batch driver, your application code) talks to ``BaseLLM`` and never
sees a provider-specific shape.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_kit.errors import LLMError
from llm_kit.llm.response import BatchItem, BatchResult, LLMResponse
from llm_kit.messages.types import Message

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Async LLM client. Provider-agnostic surface."""

    @abstractmethod
    async def invoke(
        self,
        messages: list[Message],
        *,
        response_model: type[BaseModel] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Issue a single chat-completion request.

        Raises ``LLMError`` on terminal failure after retries.
        """

    @abstractmethod
    def batch_stream(
        self,
        items: AsyncIterable[BatchItem] | list[BatchItem],
        *,
        completed_ids: set[str] | None = None,
    ) -> AsyncIterator[BatchResult]:
        """Yield ``BatchResult`` objects as items complete. Out-of-order safe."""

    async def batch_to_jsonl(
        self,
        items: AsyncIterable[BatchItem] | list[BatchItem],
        output_path: str | Path,
        *,
        resume: bool = True,
    ) -> int:
        """Stream results to ``output_path`` (jsonl). Returns total items written.

        If ``resume`` and the file exists, completed IDs are loaded from it and
        skipped. The file is opened in append mode; corrupt trailing lines from
        a prior crash are tolerated when loading IDs.
        """
        from llm_kit.batch.jsonl import (  # local import to avoid cycle
            load_completed_ids,
            write_results_to_jsonl,
        )

        path = Path(output_path)
        completed = load_completed_ids(path) if resume and path.exists() else set()
        if completed:
            logger.info("resuming: %d items already in %s", len(completed), path)

        return await write_results_to_jsonl(
            path,
            self.batch_stream(items, completed_ids=completed),
        )

    async def batch_to_callback(
        self,
        items: AsyncIterable[BatchItem] | list[BatchItem],
        callback: Callable[[BatchResult], Any],
        *,
        completed_ids: set[str] | None = None,
    ) -> int:
        """Invoke ``callback(result)`` for every completed item. Returns count.

        ``callback`` may be sync or async — both are awaited correctly.
        """
        import inspect

        count = 0
        async for result in self.batch_stream(items, completed_ids=completed_ids):
            outcome = callback(result)
            if inspect.isawaitable(outcome):
                await outcome
            count += 1
        return count

    # Subclasses that hold sockets implement these. The base no-ops so simple
    # in-memory stubs don't need to.
    async def aclose(self) -> None:  # pragma: no cover - trivial
        pass

    async def __aenter__(self) -> BaseLLM:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


__all__ = ["BaseLLM", "LLMError"]
