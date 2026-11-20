"""JSONL output + ID-based resume.

The output file *is* the checkpoint. On resume we scan it for completed IDs
and skip those inputs. No second file to keep in sync. Corrupt trailing lines
(e.g. from a crash mid-write) are tolerated when loading IDs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_kit.llm.response import BatchResult, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


async def write_results_to_jsonl(
    path: Path,
    results: AsyncIterator[BatchResult],
) -> int:
    """Append ``results`` to ``path`` as jsonl. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # Open in the worker thread once and reuse, flushing after each write so
    # crashes don't lose more than the in-flight line.
    handle = await asyncio.to_thread(path.open, "a", encoding="utf-8")
    try:
        async for result in results:
            line = json.dumps(_result_to_dict(result), ensure_ascii=False)
            await asyncio.to_thread(_write_line, handle, line)
            count += 1
    finally:
        await asyncio.to_thread(handle.close)
    return count


def load_completed_ids(path: Path) -> set[str]:
    """Scan ``path`` for already-completed IDs. Tolerates a corrupt last line."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("skipping malformed jsonl line %d in %s", lineno, path)
                continue
            item_id = record.get("id")
            if isinstance(item_id, str):
                ids.add(item_id)
    return ids


def _write_line(handle: Any, line: str) -> None:
    handle.write(line)
    handle.write("\n")
    handle.flush()


def _result_to_dict(result: BatchResult) -> dict[str, Any]:
    out: dict[str, Any] = {"id": result.id, "ok": result.ok}
    if result.metadata:
        out["metadata"] = result.metadata
    if result.response is not None:
        out["response"] = _response_to_dict(result.response)
    if result.error is not None:
        out["error"] = _error_to_dict(result.error)
    return out


def _response_to_dict(response: LLMResponse) -> dict[str, Any]:
    return {
        "text": response.text,
        "parsed": _parsed_to_jsonable(response.parsed),
        "usage": _usage_to_dict(response.usage),
        "model": response.model,
        "finish_reason": response.finish_reason,
        "latency_seconds": response.latency_seconds,
    }


def _parsed_to_jsonable(parsed: Any) -> Any:
    """Pydantic models are dumped; dict-schema results (dicts/lists/scalars) pass as-is."""
    if isinstance(parsed, BaseModel):
        return parsed.model_dump()
    return parsed


def _usage_to_dict(usage: TokenUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def _error_to_dict(error: BaseException) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    for attr in ("status_code", "retry_after_seconds", "url", "raw_content"):
        if hasattr(error, attr):
            value = getattr(error, attr)
            if value is not None:
                out[attr] = value
    return out
