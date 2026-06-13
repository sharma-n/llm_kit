"""Provider-native (offline) Batch API configuration.

These knobs drive the *offline* batch driver — the one that submits a whole
``list[BatchItem]`` to a provider's 24h asynchronous batch endpoint, polls, and
reconciles. This is distinct from ``BatchConfig`` (the client-side concurrent
pipeline). The endpoint paths below sit alongside ``LLMConfig.base_url`` /
``chat_completions_path``, which are reused for the per-request body shape and
the batch's ``endpoint`` target.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProviderBatchConfig:
    """Connection + lifecycle knobs for the provider-native Batch API driver."""

    # How long the provider may take to finish the batch. OpenAI accepts "24h".
    completion_window: str = "24h"

    # Transport endpoints, joined onto LLMConfig.base_url. OpenAI Files + Batches.
    files_path: str = "/v1/files"
    batches_path: str = "/v1/batches"

    # Default cadence for run_batch_api's poll loop. Callers can override.
    poll_interval_seconds: float = 60.0

    # Soft guard against accidentally submitting an over-large single batch.
    # OpenAI's published ceiling is 50k requests per batch.
    max_requests_per_batch: int = 50_000
