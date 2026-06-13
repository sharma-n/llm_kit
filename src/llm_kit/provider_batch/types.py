"""Value types shared across the provider-batch backends and driver.

These are deliberately transport-agnostic: a ``BatchRequest`` is one item's
``custom_id`` + provider request body, and a ``RawResult`` is one downloaded
result still in raw-dict form (the driver, not the backend, turns it into an
``LLMResponse``). ``BatchState`` collapses each provider's bespoke status
vocabulary into a small lifecycle enum the driver can branch on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BatchState(StrEnum):
    """Normalized lifecycle state for an offline batch job."""

    PENDING = "pending"  # accepted, validating, or queued — not started
    IN_PROGRESS = "in_progress"  # actively running or finalizing
    COMPLETED = "completed"  # finished; results available
    FAILED = "failed"  # terminal failure of the whole job
    EXPIRED = "expired"  # completion window elapsed
    CANCELLED = "cancelled"  # cancelled by the caller

    @property
    def is_terminal(self) -> bool:
        return self in (
            BatchState.COMPLETED,
            BatchState.FAILED,
            BatchState.EXPIRED,
            BatchState.CANCELLED,
        )


@dataclass(slots=True, frozen=True)
class BatchRequest:
    """One item's request: a stable id and the provider request body."""

    custom_id: str
    body: dict[str, Any]


@dataclass(slots=True, frozen=True)
class RawResult:
    """One downloaded result, still raw.

    Exactly one of ``inner_body`` (the provider's per-request response body, to be
    parsed by the formatter) or ``error`` (a per-request failure reported by the
    provider) is set.
    """

    custom_id: str
    inner_body: dict[str, Any] | None = None
    error: Exception | None = None


@dataclass(slots=True)
class BatchJobStatus:
    """A poll snapshot: normalized state plus provider-specific retrieval handles."""

    state: BatchState
    raw_state: str = ""  # the provider's own status string, for logging/debugging
    counts: dict[str, int] = field(default_factory=dict)
    # Provider handles the driver needs to fetch results, kept opaque here.
    output_ref: str | None = None  # e.g. OpenAI output_file_id / Anthropic results_url
    error_ref: str | None = None  # e.g. OpenAI error_file_id
