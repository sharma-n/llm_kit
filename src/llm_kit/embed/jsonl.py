"""JSONL output for embedding results.

Same contract as the chat-side writer ([batch/jsonl.py]): the file IS the
checkpoint, one record per input keyed by ``id``. Resume reuses
``batch.jsonl.load_completed_ids`` since it only scans the ``id`` field.

Vectors are written inline — a 1M-row file of 1536-dim vectors is large by
design. Use a smaller embedding ``dimensions`` or a different sink if that
matters for your storage.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from llm_kit.batch.jsonl import _error_to_dict, _usage_to_dict, _write_line
from llm_kit.embed.response import EmbeddingResult


async def write_embed_results_to_jsonl(
    path: Path,
    results: AsyncIterator[EmbeddingResult],
) -> int:
    """Append ``results`` to ``path`` as jsonl. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    handle = await asyncio.to_thread(path.open, "a", encoding="utf-8")
    try:
        async for result in results:
            line = json.dumps(_result_to_dict(result), ensure_ascii=False)
            await asyncio.to_thread(_write_line, handle, line)
            count += 1
    finally:
        await asyncio.to_thread(handle.close)
    return count


def _result_to_dict(result: EmbeddingResult) -> dict[str, Any]:
    out: dict[str, Any] = {"id": result.id, "ok": result.ok}
    if result.metadata:
        out["metadata"] = result.metadata
    if result.response is not None:
        r = result.response
        out["response"] = {
            "vector": r.vector,
            "index": r.index,
            "usage": _usage_to_dict(r.usage),
            "model": r.model,
        }
    if result.error is not None:
        out["error"] = _error_to_dict(result.error)
    return out
