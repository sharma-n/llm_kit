"""Tests for the token bucket + composed rate limiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from llm_kit.config.rate_limit import RateLimitConfig
from llm_kit.messages.types import Message
from llm_kit.rate_limit.limiter import RateLimiter, TokenBucket


async def test_token_bucket_immediate_when_capacity_available() -> None:
    bucket = TokenBucket(capacity_per_minute=600)  # 10/sec
    start = time.perf_counter()
    await bucket.acquire(1)
    await bucket.acquire(1)
    assert time.perf_counter() - start < 0.05


async def test_token_bucket_throttles_when_drained() -> None:
    """A bucket sized at 600/min refills at 10/sec; draining + reacquiring waits."""
    bucket = TokenBucket(capacity_per_minute=600)
    await bucket.acquire(600)  # drain
    start = time.perf_counter()
    await bucket.acquire(2)  # should wait ~0.2s
    elapsed = time.perf_counter() - start
    assert 0.1 <= elapsed < 0.6


async def test_token_bucket_oversize_request_clamps_to_capacity() -> None:
    """An over-capacity request must not deadlock — it drains the bucket fully."""
    bucket = TokenBucket(capacity_per_minute=60)  # 1/sec
    start = time.perf_counter()
    await bucket.acquire(120)  # 2x capacity — clamped to 60
    assert time.perf_counter() - start < 0.1


def test_token_bucket_refund_capped_at_capacity() -> None:
    bucket = TokenBucket(capacity_per_minute=100)
    # Already at capacity; refund must not push beyond.
    bucket.refund(50)
    assert bucket._tokens == 100  # type: ignore[attr-defined]


async def test_rate_limiter_semaphore_caps_concurrency() -> None:
    """Three coroutines, concurrency=2 → at most 2 active at once."""
    limiter = RateLimiter(RateLimitConfig(max_concurrent_requests=2))
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def task() -> None:
        nonlocal in_flight, peak
        async with limiter.acquire():
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            async with lock:
                in_flight -= 1

    await asyncio.gather(*(task() for _ in range(6)))
    assert peak == 2


async def test_rate_limiter_no_tpm_means_no_token_wait() -> None:
    limiter = RateLimiter(RateLimitConfig(max_concurrent_requests=4, tokens_per_minute=None))
    start = time.perf_counter()
    async with limiter.acquire(estimated_tokens=1_000_000):
        pass
    assert time.perf_counter() - start < 0.05


def test_default_token_estimator_uses_chars_per_token() -> None:
    limiter = RateLimiter(RateLimitConfig(default_chars_per_token=4.0))
    est = limiter.estimate_tokens([Message.user("a" * 40)])
    assert est == 10  # 40 chars / 4


def test_default_token_estimator_charges_for_images() -> None:
    limiter = RateLimiter(RateLimitConfig(default_chars_per_token=4.0))
    est_text = limiter.estimate_tokens([Message.user("hi")])
    est_with_image = limiter.estimate_tokens([Message.user("hi", image_urls=["http://x"])])
    assert est_with_image > est_text + 100  # images add a chunky baseline


def test_custom_token_estimator_takes_precedence() -> None:
    limiter = RateLimiter(
        RateLimitConfig(),
        token_estimator=lambda msgs: 42,
    )
    assert limiter.estimate_tokens([Message.user("anything")]) == 42


async def test_true_up_refunds_overestimate() -> None:
    """If we estimated 100 but actually used 60, the bucket gets 40 back."""
    cfg = RateLimitConfig(max_concurrent_requests=4, tokens_per_minute=600)  # 10/sec
    limiter = RateLimiter(cfg)
    # Drain ~all of capacity through one large acquire.
    async with limiter.acquire(estimated_tokens=600):
        pass
    # Refund 600 → bucket back at capacity (clamped).
    limiter.true_up_tokens(estimated=600, actual=0)
    start = time.perf_counter()
    async with limiter.acquire(estimated_tokens=10):
        pass
    # If refund worked, this is instant.
    assert time.perf_counter() - start < 0.1


async def test_true_up_debits_underestimate() -> None:
    """Bucket capacity equals tokens_per_minute; refill is capacity / 60s.

    Sized at 60_000/min (1000/sec) → draining 60_000 zeros the bucket, then a
    debit of 100 forces acquire(1) to wait ~101/1000 ≈ 0.1s for refill.
    """
    cfg = RateLimitConfig(max_concurrent_requests=4, tokens_per_minute=60_000)
    limiter = RateLimiter(cfg)
    async with limiter.acquire(estimated_tokens=60_000):  # fully drain
        pass
    limiter.true_up_tokens(estimated=0, actual=100)  # debit 100 → tokens ≈ -100
    start = time.perf_counter()
    async with limiter.acquire(estimated_tokens=1):
        pass
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.05  # tolerant lower bound; actual is ~0.1s


def test_tpm_enabled_flag() -> None:
    assert RateLimiter(RateLimitConfig(tokens_per_minute=None)).tpm_enabled is False
    assert RateLimiter(RateLimitConfig(tokens_per_minute=1000)).tpm_enabled is True
