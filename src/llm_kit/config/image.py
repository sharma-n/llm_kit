"""Image fetcher configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from llm_kit.config.http import RetryConfig


class ImageFailurePolicy(StrEnum):
    """What to do when an image cannot be retrieved after retries."""

    SKIP = "skip"  # drop the image, keep the rest of the message
    RAISE = "raise"  # surface ImageFetchError up to the LLM call
    PLACEHOLDER = "placeholder"  # replace with a 1x1 transparent PNG


@dataclass(slots=True)
class ImageConfig:
    """Async image download + base64 encoding behavior."""

    max_concurrent_downloads: int = 16
    fetch_timeout_seconds: float = 30.0
    max_bytes: int = 20 * 1024 * 1024  # 20 MB per image
    default_mime_type: str = "image/png"

    cache_size: int = 1024  # LRU entries; per-process, in-memory
    threadpool_encode_threshold_bytes: int = 1 * 1024 * 1024  # offload b64 above this

    on_failure: ImageFailurePolicy = ImageFailurePolicy.RAISE
    retry: RetryConfig = field(
        default_factory=lambda: RetryConfig(
            max_retries=3,
            backoff_base_seconds=0.25,
            backoff_max_seconds=10.0,
        )
    )
