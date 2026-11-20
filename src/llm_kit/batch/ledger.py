"""UsageLedger: accumulate TokenUsage across a stream of BatchResults."""

from __future__ import annotations

from llm_kit.llm.response import BatchResult, BatchSummary


class UsageLedger:
    """Mutable accumulator for token usage across a batch run.

    Use directly in a streaming loop::

        ledger = UsageLedger()
        async for result in llm.batch_stream(items):
            ledger.add(result)
            ...
        summary = ledger.to_summary(prompt_cost_per_1m=0.15, completion_cost_per_1m=0.60)
    """

    def __init__(self) -> None:
        self._total_items = 0
        self._ok_count = 0
        self._error_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._reasoning_tokens = 0
        self._output_tokens = 0

    def add(self, result: BatchResult) -> None:
        """Accumulate one result's token usage."""
        self._total_items += 1
        if result.ok:
            self._ok_count += 1
            if result.response is not None:
                u = result.response.usage
                self._prompt_tokens += u.prompt_tokens
                self._completion_tokens += u.completion_tokens
                self._total_tokens += u.total_tokens
                self._reasoning_tokens += u.reasoning_tokens
                self._output_tokens += u.output_tokens
        else:
            self._error_count += 1

    def to_summary(
        self,
        *,
        prompt_cost_per_1m: float = 0.0,
        completion_cost_per_1m: float = 0.0,
    ) -> BatchSummary:
        """Return a BatchSummary, computing estimated_cost from configured prices."""
        cost = (
            self._prompt_tokens / 1_000_000 * prompt_cost_per_1m
            + self._output_tokens / 1_000_000 * completion_cost_per_1m
        )
        return BatchSummary(
            total_items=self._total_items,
            ok_count=self._ok_count,
            error_count=self._error_count,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._total_tokens,
            reasoning_tokens=self._reasoning_tokens,
            output_tokens=self._output_tokens,
            estimated_cost=cost,
        )
