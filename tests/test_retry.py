"""Tests for the shared retry wrapper.

Covers: classification of statuses into typed errors, backoff progression,
honoring Retry-After, terminal vs transient, exhaustion behavior, and
transport-exception mapping.
"""

from __future__ import annotations

import time

import httpx
import pytest

from llm_kit.config.http import RetryConfig
from llm_kit.errors import (
    AuthError,
    NetworkError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from llm_kit.http.retry import retry_request


def _fast_retry(max_retries: int = 3) -> RetryConfig:
    return RetryConfig(
        max_retries=max_retries,
        backoff_base_seconds=0.001,
        backoff_max_seconds=0.01,
        jitter_seconds=0.0,
    )


async def test_first_success_no_retry() -> None:
    calls = 0

    async def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"ok": True})

    resp = await retry_request(send, retry=_fast_retry(), operation="op")
    assert resp.status_code == 200
    assert calls == 1


async def test_transient_503_then_success() -> None:
    calls = 0

    async def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"ok": True})

    resp = await retry_request(send, retry=_fast_retry(), operation="op")
    assert resp.status_code == 200
    assert calls == 3


async def test_terminal_401_no_retry() -> None:
    calls = 0

    async def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="nope")

    with pytest.raises(AuthError):
        await retry_request(send, retry=_fast_retry(), operation="op")
    assert calls == 1  # auth errors are terminal


async def test_terminal_400_no_retry() -> None:
    """400 with retryable=False (the default for 4xx other than 408/409/425/429)."""
    calls = 0

    async def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad")

    with pytest.raises(ProviderError) as info:
        await retry_request(send, retry=_fast_retry(), operation="op")
    assert info.value.status_code == 400
    assert calls == 1


async def test_exhaustion_raises_last_error() -> None:
    calls = 0

    async def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="boom")

    with pytest.raises(ProviderError) as info:
        await retry_request(send, retry=_fast_retry(max_retries=2), operation="op")
    assert info.value.status_code == 500
    assert calls == 3  # initial + 2 retries


async def test_rate_limit_honors_retry_after() -> None:
    """The Retry-After header should bound the next sleep duration."""
    calls = 0

    async def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0.05"}, text="slow down")
        return httpx.Response(200, json={"ok": True})

    start = time.perf_counter()
    resp = await retry_request(
        send,
        retry=RetryConfig(
            max_retries=2,
            backoff_base_seconds=10.0,  # large — proves Retry-After overrides
            backoff_max_seconds=10.0,
            jitter_seconds=0.0,
        ),
        operation="op",
    )
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    assert elapsed < 1.0  # would be ~10s without Retry-After honoring


async def test_429_without_retry_after_uses_backoff() -> None:
    calls = 0

    async def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 2:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"ok": True})

    resp = await retry_request(send, retry=_fast_retry(), operation="op")
    assert resp.status_code == 200


async def test_timeout_exception_becomes_typed_timeout() -> None:
    calls = 0

    async def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("read timed out", request=None)

    with pytest.raises(TimeoutError):
        await retry_request(send, retry=_fast_retry(max_retries=1), operation="op")
    assert calls == 2


async def test_transport_error_becomes_network_error() -> None:
    async def send() -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(NetworkError):
        await retry_request(send, retry=_fast_retry(max_retries=0), operation="op")


async def test_rate_limit_error_after_exhaustion_propagates() -> None:
    async def send() -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0.001"}, text="busy")

    with pytest.raises(RateLimitError) as info:
        await retry_request(send, retry=_fast_retry(max_retries=1), operation="op")
    assert info.value.retry_after_seconds == 0.001
