"""HTTP transport and retry configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetryConfig:
    """Retry behavior for a single HTTP-bound operation."""

    max_retries: int = 5
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 30.0
    jitter_seconds: float = 0.25
    retry_on_status: tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504)


@dataclass(slots=True)
class HTTPConfig:
    """Shared transport settings for the underlying httpx.AsyncClient."""

    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 120.0
    write_timeout_seconds: float = 30.0
    pool_timeout_seconds: float = 10.0

    max_connections: int = 256
    max_keepalive_connections: int = 64
    keepalive_expiry_seconds: float = 30.0
    http2: bool = True

    llm_retry: RetryConfig = field(default_factory=RetryConfig)
