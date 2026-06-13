"""Batch driver configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BatchFailurePolicy(StrEnum):
    """How the batch driver reacts to a per-item terminal failure."""

    RECORD = "record"  # write an error result, keep going (default)
    RAISE = "raise"  # cancel the batch on first failure


@dataclass(slots=True)
class BatchConfig:
    """Driver-level knobs for batch_stream and friends."""

    # Worker pool size for the batch driver. The LLM-side concurrency cap is
    # enforced by RateLimitConfig — this just bounds how many BatchItems are
    # in flight (fetching images, queued for the limiter, etc.) at once.
    worker_concurrency: int = 64

    # How many input items to keep buffered in front of the worker pool.
    # Bounds memory when reading a 1M-item jsonl source.
    input_buffer_size: int = 256

    on_failure: BatchFailurePolicy = BatchFailurePolicy.RECORD

    progress_log_every: int = 100  # 0 disables
