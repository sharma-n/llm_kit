"""The persisted batch-job handle — the offline driver's checkpoint.

Unlike the live pipeline (where the output jsonl *is* the checkpoint, CLAUDE.md
invariant #6), an offline batch has no output to checkpoint against between
submit and completion. The handle below is therefore a small, necessary sidecar:
it records the provider batch id and a manifest mapping each ``custom_id`` back to
its caller metadata, so a *fresh process* can reconnect to an in-flight job and
reconcile results.

What is persisted per item: its metadata and, if its ``response_model`` is a JSON
schema ``dict``, that schema (``response_schema``) — so a fresh process can decode
results without the original items. A Pydantic ``response_model`` is a Python type
and is not JSON-serializable: validating against one cross-process requires the
caller to re-supply the original items at fetch time. Manifests written before
``response_schema`` existed still load (the key is simply absent).
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
    # custom_id -> {"metadata": {...}, "response_schema": {...}?}; reconciliation lookup.
    manifest: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Registered name of the backend that submitted the job. Differs from
    # ``provider`` when ``ProviderBatchConfig.backend`` overrides the default or a
    # backend was injected. Empty means "same as provider" (old handles).
    backend: str = ""
    # Backend-specific JSON-serialisable state (from ``SubmitResult.extra``).
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend:
            self.backend = self.provider

    def metadata_for(self, custom_id: str) -> dict[str, Any]:
        entry = self.manifest.get(custom_id) or {}
        meta = entry.get("metadata")
        return dict(meta) if isinstance(meta, dict) else {}

    def response_schema_for(self, custom_id: str) -> dict[str, Any] | None:
        entry = self.manifest.get(custom_id) or {}
        schema = entry.get("response_schema")
        return dict(schema) if isinstance(schema, dict) else None

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
            backend=data.get("backend") or data["provider"],
            extra=data.get("extra") or {},
        )

    def _to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "batch_id": self.batch_id,
            "input_file_id": self.input_file_id,
            "manifest": self.manifest,
            "backend": self.backend,
            "extra": self.extra,
        }
