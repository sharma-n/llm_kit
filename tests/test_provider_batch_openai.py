"""End-to-end tests for the OpenAI provider-batch path against a stateful mock.

The handler below emulates the OpenAI Batch API lifecycle: multipart file upload
→ batch create → poll (in_progress then completed) → download output/error files.
It parses the uploaded jsonl to learn the custom_ids it must echo back, so the
round-trip reconciliation by custom_id is genuinely exercised.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import BaseModel

from llm_kit.llm.client import LLMClient
from llm_kit.llm.response import BatchItem
from llm_kit.messages.types import Message
from llm_kit.provider_batch.job import BatchJob

from conftest import chat_response, make_config, mock_client


class _Schema(BaseModel):
    x: int


class FakeOpenAIBatch:
    """Stateful MockTransport handler emulating the OpenAI Batch API."""

    def __init__(
        self,
        *,
        polls_until_complete: int = 1,
        fail_ids: set[str] | None = None,
        content_for: Callable[[str], str] | None = None,
    ) -> None:
        self.custom_ids: list[str] = []
        self.poll_count = 0
        self.polls_until_complete = polls_until_complete
        self.fail_ids = fail_ids or set()
        self.content_for = content_for or (lambda cid: f"echo:{cid}")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if method == "POST" and path == "/v1/files":
            self._capture_uploaded(request)
            return httpx.Response(200, json={"id": "file-in", "object": "file"})
        if method == "POST" and path == "/v1/batches":
            return httpx.Response(200, json={"id": "batch-abc", "status": "validating"})
        if method == "GET" and path == "/v1/batches/batch-abc":
            return self._poll()
        if method == "GET" and path == "/v1/files/file-out/content":
            return httpx.Response(200, text=self._output_jsonl())
        if method == "GET" and path == "/v1/files/file-err/content":
            return httpx.Response(200, text=self._error_jsonl())
        return httpx.Response(404, json={"error": f"unexpected {method} {path}"})

    def _capture_uploaded(self, request: httpx.Request) -> None:
        body = request.content.decode("utf-8", "replace")
        for line in body.splitlines():
            line = line.strip()
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "custom_id" in rec:
                self.custom_ids.append(rec["custom_id"])

    def _poll(self) -> httpx.Response:
        self.poll_count += 1
        counts = {"total": len(self.custom_ids)}
        if self.poll_count < self.polls_until_complete:
            return httpx.Response(
                200,
                json={"id": "batch-abc", "status": "in_progress", "request_counts": counts},
            )
        return httpx.Response(
            200,
            json={
                "id": "batch-abc",
                "status": "completed",
                "output_file_id": "file-out",
                "error_file_id": "file-err" if self.fail_ids else None,
                "request_counts": counts,
            },
        )

    def _output_jsonl(self) -> str:
        lines = []
        for cid in self.custom_ids:
            if cid in self.fail_ids:
                continue
            body = chat_response(content=self.content_for(cid))
            lines.append(
                json.dumps(
                    {
                        "custom_id": cid,
                        "response": {"status_code": 200, "request_id": "r", "body": body},
                        "error": None,
                    }
                )
            )
        return "\n".join(lines) + "\n"

    def _error_jsonl(self) -> str:
        lines = [
            json.dumps(
                {
                    "custom_id": cid,
                    "response": None,
                    "error": {"message": "model unavailable", "code": 400},
                }
            )
            for cid in self.custom_ids
            if cid in self.fail_ids
        ]
        return ("\n".join(lines) + "\n") if lines else ""


def _make_llm(handler) -> LLMClient:
    return LLMClient(make_config(), client=mock_client(handler), owns_client=True)


def _items(n: int) -> list[BatchItem]:
    return [
        BatchItem(id=f"item-{i}", messages=[Message.user(f"q{i}")], metadata={"i": i})
        for i in range(n)
    ]


async def test_submit_builds_manifest_and_returns_handle() -> None:
    llm = _make_llm(FakeOpenAIBatch())
    items = _items(3)
    async with llm:
        job = await llm.submit_batch(items)
    assert job.batch_id == "batch-abc"
    assert job.provider == "openai"
    assert set(job.manifest) == {"item-0", "item-1", "item-2"}
    assert job.metadata_for("item-1") == {"i": 1}


async def test_round_trip_reconciles_by_custom_id() -> None:
    llm = _make_llm(FakeOpenAIBatch())
    items = _items(3)
    async with llm:
        job = await llm.submit_batch(items)
        results = [r async for r in llm.fetch_batch_results(job, items=items)]
    by_id = {r.id: r for r in results}
    assert set(by_id) == {"item-0", "item-1", "item-2"}
    assert all(r.ok for r in results)
    assert by_id["item-2"].response is not None
    assert by_id["item-2"].response.text == "echo:item-2"
    assert by_id["item-1"].metadata == {"i": 1}


async def test_response_model_validates_on_fetch() -> None:
    handler = FakeOpenAIBatch(content_for=lambda cid: '{"x": 7}')
    llm = _make_llm(handler)
    items = [BatchItem(id="a", messages=[Message.user("q")], response_model=_Schema)]
    async with llm:
        job = await llm.submit_batch(items)
        results = [r async for r in llm.fetch_batch_results(job, items=items)]
    assert results[0].ok
    assert isinstance(results[0].response.parsed, _Schema)
    assert results[0].response.parsed.x == 7


async def test_failed_item_from_error_file_maps_to_error_result() -> None:
    handler = FakeOpenAIBatch(fail_ids={"item-1"})
    llm = _make_llm(handler)
    items = _items(3)
    async with llm:
        job = await llm.submit_batch(items)
        results = {r.id: r async for r in llm.fetch_batch_results(job, items=items)}
    assert results["item-0"].ok
    assert not results["item-1"].ok
    assert results["item-1"].error is not None
    assert "model unavailable" in str(results["item-1"].error)


async def test_run_batch_api_polls_then_writes_jsonl(tmp_path) -> None:
    handler = FakeOpenAIBatch(polls_until_complete=2)
    llm = _make_llm(handler)
    items = _items(3)
    out = tmp_path / "out.jsonl"
    async with llm:
        written = await llm.run_batch_api(items, out, poll_interval=0.0)
    assert written == 3
    assert handler.poll_count >= 2  # exercised the poll loop
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    ids = {json.loads(line)["id"] for line in lines}
    assert ids == {"item-0", "item-1", "item-2"}


async def test_run_batch_api_skips_already_written_ids(tmp_path) -> None:
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"id": "item-0", "ok": True}) + "\n", encoding="utf-8")
    llm = _make_llm(FakeOpenAIBatch())
    async with llm:
        written = await llm.run_batch_api(_items(3), out, poll_interval=0.0)
    assert written == 2  # item-0 already present, skipped
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # 1 pre-existing + 2 new


async def test_job_save_load_round_trip_then_fetch(tmp_path) -> None:
    llm = _make_llm(FakeOpenAIBatch())
    items = _items(2)
    handle_path = tmp_path / "job.json"
    async with llm:
        job = await llm.submit_batch(items)
        job.save(handle_path)
        reloaded = BatchJob.load(handle_path)
        assert reloaded.batch_id == job.batch_id
        assert reloaded.manifest == job.manifest
        results = [r async for r in llm.fetch_batch_results(reloaded, items=items)]
    assert {r.id for r in results} == {"item-0", "item-1"}
    # metadata restored from the reloaded manifest even though it came from disk
    assert {r.id: r.metadata["i"] for r in results} == {"item-0": 0, "item-1": 1}
