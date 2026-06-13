"""Rate-limit configuration: RPM + TPM + concurrency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RateLimitConfig:
    """Concurrency and per-minute caps for outbound LLM traffic."""

    max_concurrent_requests: int = 32
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None

    # Default token estimator: characters / 4 (rough but dep-free).
    # Callers can swap in a real tokenizer by replacing the estimator on the
    # composed limiter after construction.
    default_chars_per_token: float = 4.0
