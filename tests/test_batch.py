"""Tests for the batch driver: pipeline, sinks, resume, failure modes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llm_kit.batch.ledger import UsageLedger
from llm_kit.config.batch import BatchConfig, BatchFailurePolicy
from llm_kit.errors import LLMError
from llm_kit.llm.response import BatchItem, BatchResult, BatchSummary, LLMResponse, TokenUsage
from llm_kit.messages.types import Message

from conftest import FakeLLM, make_config, make_items


async def test_batch_stream_yields_every_item() -> None:
    items = make_items(20)
    llm = FakeLLM()
    results = [r async for r in llm.batch_stream(items)]
    assert len(results) == 20
    assert {r.id for r in results} == {i.id for i in items}
    assert all(r.ok for r in results)


async def test_batch_stream_preserves_ids_across_workers() -> None:
    """Out-of-order completion is fine, but every ID must appear exactly once."""
    cfg = make_config()
    cfg.batch = BatchConfig(worker_concurrency=8, progress_log_every=0)
    llm = FakeLLM(config=cfg, delay_seconds=0.005)
    items = make_items(50)
    results = [r async for r in llm.batch_stream(items)]
    seen_ids = [r.id for r in results]
    assert sorted(seen_ids) == sorted([i.id for i in items])
    assert len(set(seen_ids)) == len(items)


async def test_completed_ids_filter_skips_inputs() -> None:
    items = make_items(10)
    skip = {items[0].id, items[5].id, items[9].id}
    llm = FakeLLM()
    results = [r async for r in llm.batch_stream(items, completed_ids=skip)]
    assert {r.id for r in results} == {i.id for i in items} - skip


async def test_per_item_failure_is_recorded_default() -> None:
    items = make_items(10, prefix="rec")
    fail = {items[3].id, items[7].id}
    llm = FakeLLM(fail_ids=fail)
    results = [r async for r in llm.batch_stream(items)]
    assert len(results) == 10
    failed = [r for r in results if not r.ok]
    assert {r.id for r in failed} == fail
    for r in failed:
        assert isinstance(r.error, LLMError)


async def test_per_item_failure_with_raise_policy_aborts() -> None:
    cfg = make_config()
    cfg.batch = BatchConfig(
        worker_concurrency=2, on_failure=BatchFailurePolicy.RAISE, progress_log_every=0
    )
    items = make_items(20, prefix="raise")
    llm = FakeLLM(config=cfg, fail_ids={items[2].id})

    with pytest.raises(LLMError):
        async for _ in llm.batch_stream(items):
            pass


async def test_batch_to_jsonl_writes_one_line_per_item(tmp_path: Path) -> None:
    items = make_items(15, prefix="j")
    llm = FakeLLM()
    output = tmp_path / "out.jsonl"
    summary = await llm.batch_to_jsonl(items, output)
    assert summary.total_items == 15

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 15
    ids_in_file = {json.loads(line)["id"] for line in lines}
    assert ids_in_file == {i.id for i in items}


async def test_batch_to_jsonl_resumes_from_existing_file(tmp_path: Path) -> None:
    items = make_items(10, prefix="resume")
    llm = FakeLLM()
    output = tmp_path / "out.jsonl"

    first = await llm.batch_to_jsonl(items, output)
    assert first.total_items == 10

    # Second pass on same items: every ID is already in the file.
    second = await llm.batch_to_jsonl(items, output, resume=True)
    assert second.total_items == 0

    # Add new items, only those should be processed.
    extra = make_items(3, prefix="extra")
    third = await llm.batch_to_jsonl(items + extra, output, resume=True)
    assert third.total_items == 3


async def test_batch_to_callback_invokes_for_each_result() -> None:
    items = make_items(8, prefix="cb")
    llm = FakeLLM()
    seen: list[str] = []
    count = await llm.batch_to_callback(items, lambda r: seen.append(r.id))
    assert count == 8
    assert set(seen) == {i.id for i in items}


async def test_batch_to_callback_awaits_async_callbacks() -> None:
    items = make_items(5, prefix="acb")
    llm = FakeLLM()
    seen: list[str] = []

    async def cb(result) -> None:
        await asyncio.sleep(0.001)
        seen.append(result.id)

    count = await llm.batch_to_callback(items, cb)
    assert count == 5
    assert set(seen) == {i.id for i in items}


async def test_unexpected_exception_inside_invoke_is_captured() -> None:
    """A non-LLMError exception must not crash the pipeline; it becomes an error result."""

    class ExplodingLLM(FakeLLM):
        async def invoke(self, messages, *, response_model=None, settings=None, overrides=None):
            raise RuntimeError("programming bug")

    llm = ExplodingLLM()
    items = make_items(3, prefix="bug")
    results = [r async for r in llm.batch_stream(items)]
    assert len(results) == 3
    assert all(not r.ok for r in results)
    for r in results:
        assert isinstance(r.error, LLMError)
        assert "RuntimeError" in str(r.error) or "programming bug" in str(r.error)


async def test_async_iterable_input_works() -> None:
    """The driver should accept an async-iterable input, not only a list."""

    async def gen():
        for i in range(5):
            yield BatchItem(id=f"async-{i}", messages=[Message.user(f"q{i}")])

    llm = FakeLLM()
    results = [r async for r in llm.batch_stream(gen())]
    assert {r.id for r in results} == {f"async-{i}" for i in range(5)}


# ----------------------------------------------------------------- UsageLedger


def _ok_result(id: str, prompt: int = 1, completion: int = 1) -> BatchResult:
    return BatchResult(
        id=id,
        response=LLMResponse(
            text="ok",
            usage=TokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            ),
        ),
    )


def _err_result(id: str) -> BatchResult:
    from llm_kit.errors import LLMError

    return BatchResult(id=id, error=LLMError("boom"))


def test_ledger_accumulates_ok_results() -> None:
    ledger = UsageLedger()
    ledger.add(_ok_result("a", prompt=10, completion=5))
    ledger.add(_ok_result("b", prompt=20, completion=8))
    s = ledger.to_summary()
    assert s.total_items == 2
    assert s.ok_count == 2
    assert s.error_count == 0
    assert s.prompt_tokens == 30
    assert s.completion_tokens == 13
    assert s.total_tokens == 43
    assert s.estimated_cost == 0.0


def test_ledger_accumulates_error_results() -> None:
    ledger = UsageLedger()
    ledger.add(_ok_result("a"))
    ledger.add(_err_result("b"))
    ledger.add(_err_result("c"))
    s = ledger.to_summary()
    assert s.total_items == 3
    assert s.ok_count == 1
    assert s.error_count == 2
    assert s.prompt_tokens == 1
    assert s.completion_tokens == 1


def test_ledger_cost_calculation() -> None:
    ledger = UsageLedger()
    ledger.add(_ok_result("a", prompt=500_000, completion=250_000))
    s = ledger.to_summary(prompt_cost_per_1m=1.0, completion_cost_per_1m=2.0)
    # 500_000/1e6 * 1.0 + 250_000/1e6 * 2.0 = 0.5 + 0.5 = 1.0
    assert abs(s.estimated_cost - 1.0) < 1e-9


def test_ledger_cost_zero_by_default() -> None:
    ledger = UsageLedger()
    ledger.add(_ok_result("a", prompt=1_000_000, completion=1_000_000))
    assert ledger.to_summary().estimated_cost == 0.0


async def test_batch_to_jsonl_returns_batch_summary(tmp_path: Path) -> None:
    # FakeLLM returns TokenUsage(prompt=1, completion=1, total=2) per item.
    items = make_items(10, prefix="sum")
    llm = FakeLLM()
    output = tmp_path / "out.jsonl"
    summary = await llm.batch_to_jsonl(items, output)

    assert isinstance(summary, BatchSummary)
    assert summary.total_items == 10
    assert summary.ok_count == 10
    assert summary.error_count == 0
    assert summary.prompt_tokens == 10
    assert summary.completion_tokens == 10
    assert summary.total_tokens == 20
    assert summary.estimated_cost == 0.0


async def test_batch_to_jsonl_summary_counts_errors(tmp_path: Path) -> None:
    items = make_items(5, prefix="errsum")
    fail_ids = {items[1].id, items[3].id}
    llm = FakeLLM(fail_ids=fail_ids)
    output = tmp_path / "out.jsonl"
    summary = await llm.batch_to_jsonl(items, output)

    assert summary.total_items == 5
    assert summary.ok_count == 3
    assert summary.error_count == 2
    assert summary.prompt_tokens == 3
    assert summary.completion_tokens == 3
