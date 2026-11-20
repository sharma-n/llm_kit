"""Composed rate limiter: concurrency + RPM + (optional) TPM.

Usage:

    async with limiter.acquire(estimated_tokens=est):
        response = await call_llm(...)
        limiter.true_up_tokens(estimated=est, actual=response.usage.total_tokens)

The concurrency semaphore caps in-flight requests. The RPM bucket caps the
issuance rate. The TPM bucket (if configured) caps token throughput; the caller
provides an estimate up front, then trues it up when the response arrives so
the bucket stays honest under sustained load.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from llm_kit.config.rate_limit import RateLimitConfig
from llm_kit.messages.types import Message, TextPart

TokenEstimator = Callable[[list[Message]], int]


class TokenBucket:
    """Classic refilling token bucket. ``capacity`` per minute → per-second rate."""

    __slots__ = ("_capacity", "_refill_per_second", "_tokens", "_last_refill", "_lock")

    def __init__(self, capacity_per_minute: float) -> None:
        self._capacity = float(capacity_per_minute)
        self._refill_per_second = self._capacity / 60.0
        self._tokens = self._capacity
        # time.monotonic is the right clock for refill math: it does not require
        # a running event loop (so construction outside async context works) and
        # is immune to wall-clock adjustments.
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float) -> None:
        amount = max(amount, 0.0)
        if amount == 0.0:
            return
        # Allow oversized single requests through (else we'd deadlock); they
        # just drain the bucket fully and pay the corresponding wait.
        amount = min(amount, self._capacity)
        while True:
            async with self._lock:
                self._refill_locked()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                deficit = amount - self._tokens
                wait = deficit / self._refill_per_second
            await asyncio.sleep(wait)

    def refund(self, amount: float) -> None:
        """Return tokens to the bucket. Used by ``true_up_tokens`` for overshoot."""
        if amount <= 0:
            return
        # No lock needed for a single float assignment, but we keep semantics
        # simple by clamping at capacity.
        self._tokens = min(self._capacity, self._tokens + amount)

    def debit(self, amount: float) -> None:
        """Take tokens without waiting. Used to true up under-estimates."""
        if amount <= 0:
            return
        self._tokens -= amount  # may go negative; bucket recovers naturally

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
            self._last_refill = now


@dataclass(slots=True)
class _Acquisition:
    estimated_tokens: int


class RateLimiter:
    """Coordinates the concurrency semaphore + RPM bucket + optional TPM bucket."""

    def __init__(
        self,
        config: RateLimitConfig,
        *,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._rpm = (
            TokenBucket(config.requests_per_minute)
            if config.requests_per_minute is not None
            else None
        )
        self._tpm = (
            TokenBucket(config.tokens_per_minute)
            if config.tokens_per_minute is not None
            else None
        )
        self._token_estimator = token_estimator or self._default_estimator

    @property
    def tpm_enabled(self) -> bool:
        return self._tpm is not None

    def estimate_tokens(self, messages: list[Message]) -> int:
        return self._token_estimator(messages)

    @asynccontextmanager
    async def acquire(self, *, estimated_tokens: int = 0) -> AsyncIterator[_Acquisition]:
        """Acquire all gates needed to issue one LLM request."""
        async with self._semaphore:
            if self._rpm is not None:
                await self._rpm.acquire(1.0)
            if self._tpm is not None and estimated_tokens > 0:
                await self._tpm.acquire(float(estimated_tokens))
            yield _Acquisition(estimated_tokens=estimated_tokens)

    def true_up_tokens(self, *, estimated: int, actual: int) -> None:
        """Reconcile the TPM bucket using the response's reported token count.

        If we over-estimated, refund the difference; if we under-estimated,
        debit it. Keeps the bucket from drifting over a long batch.
        """
        if self._tpm is None or estimated == actual:
            return
        delta = actual - estimated
        if delta > 0:
            self._tpm.debit(float(delta))
        else:
            self._tpm.refund(float(-delta))

    def _default_estimator(self, messages: list[Message]) -> int:
        chars = 0
        for m in messages:
            for part in m.parts:
                if isinstance(part, TextPart):
                    chars += len(part.text)
                else:
                    # Images: rough heuristic, since per-provider tile counts
                    # vary. 1000 tokens per image is the right order of
                    # magnitude for vision models; under-estimates will be
                    # corrected by true_up_tokens.
                    chars += 4000
        return max(1, int(chars / self._config.default_chars_per_token))
