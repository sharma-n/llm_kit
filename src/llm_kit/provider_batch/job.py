"""The persisted batch-job handle — the offline driver's checkpoint.

Unlike the live pipeline (where the output jsonl *is* the checkpoint, CLAUDE.md
invariant #6), an offline batch has no output to checkpoint against between
submit and completion. The handle below is therefore a small, necessary sidecar:
it records the provider batch id and a manifest mapping each ``custom_id`` back to
its caller metadata, so a *fresh process* can reconnect to an in-flight job and
reconcile results.

Note what is *not* persisted: each item's ``response_model`` is a Python type and
is not JSON-serializable, so cross-process result validation requires the caller
to re-supply the original items at fetch time (the in-process ``run_batch_api``
path keeps them in memory). Metadata, which is JSON, is always restored from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self


@dataclass(slots=True)
class BatchJob:
    """A handle to a submitted provider batch job."""

    provider: str  # the LLMConfig.message_format value, e.g. "openai"
    batch_id: str  # the provider's batch identifier
    input_file_id: str | None = None  # OpenAI uploaded-file id (None for inline providers)
    # custom_id -> {"metadata": {...}}; reconciliation lookup on fetch.
    manifest: dict[str, dict[str, Any]] = field(default_factory=dict)

    def metadata_for(self, custom_id: str) -> dict[str, Any]:
        entry = self.manifest.get(custom_id) or {}
        meta = entry.get("metadata")
        return dict(meta) if isinstance(meta, dict) else {}

    def save(self, path: str | Path) -> None:
        """Write the handle to ``path`` as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reload a handle previously written by ``save``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            provider=data["provider"],
            batch_id=data["batch_id"],
            input_file_id=data.get("input_file_id"),
            manifest=data.get("manifest") or {},
        )

    def _to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "batch_id": self.batch_id,
            "input_file_id": self.input_file_id,
            "manifest": self.manifest,
        }
