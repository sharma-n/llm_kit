"""Terminal-state contract for fetching provider-batch results (issue #7)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from llm_kit import (
    BatchItem,
    BatchJobFailedError,
    BatchJobNotFinishedError,
    BatchState,
    LLMClient,
    Message,
    ProviderError,
)
from llm_kit.provider_batch.anthropic import AnthropicBatchBackend
from llm_kit.provider_batch.gemini import GeminiBatchBackend
from llm_kit.testing import FakeBatchBackend

from conftest import make_config, mock_client
from test_provider_batch_openai import FakeOpenAIBatch, _items as _oa_items, _make_llm


def _no_network() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected network request")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _client(fake: FakeBatchBackend) -> LLMClient:
    return LLMClient(make_config(), client=_no_network(), batch_backend=fake)


def _items(n: int = 2) -> list[BatchItem]:
    return [BatchItem(id=f"i{k}", messages=[Message.user(f"q{k}")]) for k in range(n)]


# ------------------------------------------------------------ not finished


async def test_fetch_on_running_job_raises_not_finished() -> None:
    fake = FakeBatchBackend(polls_until_done=99)
    llm = _client(fake)
    job = await llm.submit_batch(_items())
    with pytest.raises(BatchJobNotFinishedError) as ei:
        _ = [r async for r in llm.fetch_batch_results(job)]
    e = ei.value
    assert e.batch_id == job.batch_id
    assert e.state is BatchState.IN_PROGRESS and e.raw_state == "in_progress"
    assert "in_progress" in str(e) and e.retryable is False
    with pytest.raises(BatchJobNotFinishedError):
        _ = [r async for r in llm.fetch_raw_results(job)]


async def test_paused_is_not_fetchable_but_waited_through() -> None:
    fake = FakeBatchBackend(states=[BatchState.PAUSED, BatchState.PAUSED])
    llm = _client(fake)
    job = await llm.submit_batch(_items())
    with pytest.raises(BatchJobNotFinishedError) as ei:
        _ = [r async for r in llm.fetch_batch_results(job)]
    assert ei.value.state is BatchState.PAUSED and ei.value.raw_state == "paused"
    assert not BatchState.PAUSED.is_terminal
    results = [r async for r in llm.fetch_batch_results(job, wait=True, poll_interval=0)]
    assert len(results) == 2 and all(r.error is None for r in results)


async def test_wait_true_returns_results() -> None:
    fake = FakeBatchBackend(polls_until_done=4)
    llm = _client(fake)
    job = await llm.submit_batch(_items(3))
    results = [r async for r in llm.fetch_batch_results(job, wait=True, poll_interval=0)]
    assert sorted(r.id for r in results) == ["i0", "i1", "i2"]
    assert fake.poll_counts[job.batch_id] >= 4
    raws = [r async for r in llm.fetch_raw_results(job, wait=True, poll_interval=0)]
    assert len(raws) == 3


# ------------------------------------------------------------- failed ends


@pytest.mark.parametrize(
    ("state", "raw"),
    [
        (BatchState.FAILED, "JOB_STATE_FAILED"),
        (BatchState.EXPIRED, "timed_out"),
        (BatchState.CANCELLED, "cancelled_by_user"),
    ],
)
async def test_terminal_failures_raise_with_raw_state_and_detail(
    state: BatchState, raw: str
) -> None:
    fake = FakeBatchBackend(fail_job=state, raw_state=raw, error_detail="quota exceeded")
    llm = _client(fake)
    job = await llm.submit_batch(_items())
    for call in (
        lambda: llm.fetch_batch_results(job),
        lambda: llm.fetch_raw_results(job),
    ):
        with pytest.raises(BatchJobFailedError) as ei:
            _ = [r async for r in call()]
        e = ei.value
        assert e.state is state and e.raw_state == raw and e.detail == "quota exceeded"
        assert e.batch_id == job.batch_id
        assert raw in str(e) and "quota exceeded" in str(e)
        assert e.retryable is False and isinstance(e, ProviderError)


async def test_detail_optional() -> None:
    fake = FakeBatchBackend(fail_job=BatchState.FAILED)
    llm = _client(fake)
    job = await llm.submit_batch(_items())
    with pytest.raises(BatchJobFailedError) as ei:
        _ = [r async for r in llm.fetch_batch_results(job, wait=True, poll_interval=0)]
    assert ei.value.detail is None and ei.value.raw_state == "failed"


async def test_allow_partial_yields_partial_results() -> None:
    fake = FakeBatchBackend(fail_job=BatchState.EXPIRED, partial_output=True)
    llm = _client(fake)
    job = await llm.submit_batch(_items(2))
    results = [r async for r in llm.fetch_batch_results(job, allow_partial=True)]
    assert sorted(r.id for r in results) == ["i0", "i1"]
    raws = [r async for r in llm.fetch_raw_results(job, allow_partial=True)]
    assert len(raws) == 2


async def test_allow_partial_without_output_yields_nothing() -> None:
    fake = FakeBatchBackend(fail_job=BatchState.FAILED)
    llm = _client(fake)
    job = await llm.submit_batch(_items())
    assert [r async for r in llm.fetch_batch_results(job, allow_partial=True)] == []


async def test_allow_partial_still_requires_finished() -> None:
    llm = _client(FakeBatchBackend(polls_until_done=99))
    job = await llm.submit_batch(_items())
    with pytest.raises(BatchJobNotFinishedError):
        _ = [r async for r in llm.fetch_batch_results(job, allow_partial=True)]


# -------------------------------------------------------------- run_batch_api


async def test_run_batch_api_failed_job_raises_and_writes_nothing(tmp_path: Path) -> None:
    fake = FakeBatchBackend(fail_job=BatchState.FAILED, raw_state="boom", error_detail="bad")
    llm = _client(fake)
    out = tmp_path / "sub" / "out.jsonl"
    with pytest.raises(BatchJobFailedError) as ei:
        await llm.run_batch_api(_items(), out, poll_interval=0)
    assert ei.value.raw_state == "boom"
    assert not out.exists() and not out.parent.exists()


async def test_run_batch_api_failed_job_leaves_existing_file_untouched(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    original = json.dumps({"id": "i0", "response": None, "error": None}) + "\n"
    out.write_text(original)
    llm = _client(FakeBatchBackend(fail_job=BatchState.CANCELLED))
    with pytest.raises(BatchJobFailedError):
        await llm.run_batch_api(_items(), out, poll_interval=0)
    assert out.read_text() == original


async def test_run_batch_api_allow_partial_writes_partial(tmp_path: Path) -> None:
    fake = FakeBatchBackend(fail_job=BatchState.EXPIRED, partial_output=True)
    out = tmp_path / "out.jsonl"
    n = await _client(fake).run_batch_api(_items(2), out, poll_interval=0, allow_partial=True)
    assert n == 2 and out.exists()


async def test_run_batch_api_waits_through_paused(tmp_path: Path) -> None:
    fake = FakeBatchBackend(states=[BatchState.PAUSED, BatchState.IN_PROGRESS, BatchState.PAUSED])
    out = tmp_path / "out.jsonl"
    n = await _client(fake).run_batch_api(_items(3), out, poll_interval=0)
    assert n == 3
    (batch_id,) = fake.submitted
    assert fake.poll_counts[batch_id] >= 4


# ------------------------------------------------------------ real backends


def _gemini_cfg():
    cfg = make_config()
    cfg.llm.base_url = "https://gemini.test"
    cfg.llm.chat_completions_path = "/v1beta/models/g:generateContent"
    return cfg


async def test_gemini_failed_job_raises_with_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"name": "batches/g1"})
        return httpx.Response(
            200,
            json={
                "name": "batches/g1",
                "done": True,
                "metadata": {"state": "JOB_STATE_FAILED"},
                "error": {"code": 8, "message": "resource exhausted"},
            },
        )

    cfg = _gemini_cfg()
    llm = LLMClient(
        cfg, client=mock_client(handler), batch_backend=GeminiBatchBackend(mock_client(handler), cfg, "k")
    )
    job = await llm.submit_batch(_items(1))
    with pytest.raises(BatchJobFailedError) as ei:
        _ = [r async for r in llm.fetch_batch_results(job)]
    e = ei.value
    assert e.state is BatchState.FAILED and e.raw_state == "JOB_STATE_FAILED"
    assert e.detail == "resource exhausted"
    assert [r async for r in llm.fetch_batch_results(job, allow_partial=True)] == []


@pytest.mark.parametrize(
    ("raw", "state"),
    [
        ("JOB_STATE_PAUSED", BatchState.PAUSED),
        ("BATCH_STATE_PAUSED", BatchState.PAUSED),
        ("BATCH_STATE_FAILED", BatchState.FAILED),
        ("BATCH_STATE_EXPIRED", BatchState.EXPIRED),
        ("JOB_STATE_CANCELLED", BatchState.CANCELLED),
        ("BATCH_STATE_SUCCEEDED", BatchState.COMPLETED),
    ],
)
async def test_gemini_state_map(raw: str, state: BatchState) -> None:
    handler = lambda r: httpx.Response(200, json={"metadata": {"state": raw}})  # noqa: E731
    backend = GeminiBatchBackend(mock_client(handler), _gemini_cfg(), "k")
    assert (await backend.poll("batches/x")).state is state


async def test_openai_failed_batch_raises_with_detail() -> None:
    class FailedOpenAI(FakeOpenAIBatch):
        def _poll(self) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "batch-abc",
                    "status": "failed",
                    "errors": {
                        "data": [
                            {"code": "invalid_request", "message": "line 1 bad"},
                            {"code": "x", "message": "line 2 bad"},
                        ]
                    },
                },
            )

    llm = _make_llm(FailedOpenAI())
    async with llm:
        job = await llm.submit_batch(_oa_items(2))
        with pytest.raises(BatchJobFailedError) as ei:
            _ = [r async for r in llm.fetch_batch_results(job)]
    e = ei.value
    assert e.state is BatchState.FAILED and e.raw_state == "failed"
    assert e.detail == "line 1 bad; line 2 bad"


async def test_openai_cancelled_with_partial_output() -> None:
    class Cancelled(FakeOpenAIBatch):
        def _poll(self) -> httpx.Response:
            return httpx.Response(
                200,
                json={"id": "batch-abc", "status": "cancelled", "output_file_id": "file-out"},
            )

    llm = _make_llm(Cancelled())
    async with llm:
        job = await llm.submit_batch(_oa_items(2))
        with pytest.raises(BatchJobFailedError):
            _ = [r async for r in llm.fetch_batch_results(job)]
        partial = [r async for r in llm.fetch_batch_results(job, allow_partial=True)]
    assert len(partial) == 2


async def test_openai_cancelling_is_in_progress() -> None:
    from llm_kit.provider_batch.openai import _STATE_MAP

    assert _STATE_MAP["cancelling"] is BatchState.IN_PROGRESS
    assert _STATE_MAP["failed"] is BatchState.FAILED
    assert _STATE_MAP["expired"] is BatchState.EXPIRED
    assert _STATE_MAP["cancelled"] is BatchState.CANCELLED


async def test_anthropic_ended_vs_cancelled() -> None:
    def make(counts: dict, cancel: str | None):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "b",
                    "processing_status": "ended",
                    "cancel_initiated_at": cancel,
                    "request_counts": counts,
                    "results_url": "https://x.test/results",
                },
            )

        return AnthropicBatchBackend(mock_client(handler), make_config(), "k")

    done = await make({"succeeded": 2, "canceled": 0}, None).poll("b")
    assert done.state is BatchState.COMPLETED
    late = await make({"succeeded": 2, "canceled": 0}, "2026-01-01T00:00:00Z").poll("b")
    assert late.state is BatchState.COMPLETED
    cancelled = await make({"succeeded": 1, "canceled": 1}, "2026-01-01T00:00:00Z").poll("b")
    assert cancelled.state is BatchState.CANCELLED
    assert cancelled.raw_state == "ended" and cancelled.output_ref == "https://x.test/results"
    assert "2026-01-01" in (cancelled.error_detail or "")
