"""The ``BatchBackend`` abstraction — provider transport for offline batches.

A backend owns everything provider-specific about the *transport*: how requests
are uploaded (a file vs an inline array), the create/poll/results endpoints, the
status vocabulary, and the per-line envelope that wraps each item's body and that
unwraps each downloaded result. It deliberately does **not** know how to build a
request body or parse a response body — those come from the formatter
(``build_body`` and ``parse_response``), keeping CLAUDE.md invariant #2 intact.

The interface is high-level on purpose so file-upload (OpenAI) and inline-array
(Anthropic, Gemini) transports both satisfy it without leaking which one a given
provider uses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from llm_kit.provider_batch.types import BatchJobStatus, BatchRequest, RawResult, SubmitResult


@runtime_checkable
class BatchBackend(Protocol):
    """Provider transport for an offline batch job."""

    async def submit(self, requests: list[BatchRequest]) -> str | SubmitResult:
        """Upload/submit all requests.

        Return the provider batch id, or a ``SubmitResult`` when the backend also
        has JSON-serialisable state to persist on ``BatchJob.extra``.
        """
        ...

    async def poll(self, batch_id: str) -> BatchJobStatus:
        """Return a normalized status snapshot for ``batch_id``."""
        ...

    def fetch_results(self, status: BatchJobStatus) -> AsyncIterator[RawResult]:
        """Stream downloaded results for a completed job. Out-of-order safe."""
        ...

    async def cancel(self, batch_id: str) -> None:
        """Best-effort cancel of an in-flight job."""
        ...
