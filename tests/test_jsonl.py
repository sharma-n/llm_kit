"""Tests for jsonl writer + completed-id loader (the checkpoint)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

from llm_kit.batch.jsonl import load_completed_ids, write_results_to_jsonl
from llm_kit.errors import ProviderError, RateLimitError
from llm_kit.llm.response import BatchResult, LLMResponse, TokenUsage


async def _aiter(items: list[BatchResult]) -> AsyncIterator[BatchResult]:
    for item in items:
        yield item


async def test_round_trip_ok_results(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    results = [
        BatchResult(
            id=f"id-{i}",
            response=LLMResponse(text=f"r{i}", usage=TokenUsage(1, 1, 2)),
        )
        for i in range(3)
    ]
    n = await write_results_to_jsonl(path, _aiter(results))
    assert n == 3

    lines = path.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert [p["id"] for p in parsed] == ["id-0", "id-1", "id-2"]
    assert all(p["ok"] for p in parsed)
    assert parsed[0]["response"]["text"] == "r0"
    assert parsed[0]["response"]["usage"]["total_tokens"] == 2

    assert load_completed_ids(path) == {"id-0", "id-1", "id-2"}


async def test_error_result_serialization_carries_metadata(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    results = [
        BatchResult(id="a", error=ProviderError("server down", status_code=503)),
        BatchResult(id="b", error=RateLimitError("slow", retry_after_seconds=2.5)),
    ]
    await write_results_to_jsonl(path, _aiter(results))
    parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert parsed[0]["ok"] is False
    assert parsed[0]["error"]["type"] == "ProviderError"
    assert parsed[0]["error"]["status_code"] == 503

    assert parsed[1]["error"]["type"] == "RateLimitError"
    assert parsed[1]["error"]["retry_after_seconds"] == 2.5


def test_load_completed_ids_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_completed_ids(tmp_path / "does-not-exist.jsonl") == set()


def test_load_completed_ids_tolerates_malformed_tail(tmp_path: Path) -> None:
    """A crash mid-write can leave a truncated final line; we skip it."""
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"id":"a","ok":true}\n'
        '{"id":"b","ok":true}\n'
        '{"id":"c", BROKEN\n',
        encoding="utf-8",
    )
    assert load_completed_ids(path) == {"a", "b"}


def test_load_completed_ids_skips_records_without_string_id(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"id":"a","ok":true}\n'
        '{"ok":true}\n'  # no id at all
        '{"id":123,"ok":true}\n',  # id wrong type
        encoding="utf-8",
    )
    assert load_completed_ids(path) == {"a"}


async def test_writer_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "out.jsonl"
    await write_results_to_jsonl(path, _aiter([BatchResult(id="x", response=LLMResponse(text="t"))]))
    assert path.exists()
