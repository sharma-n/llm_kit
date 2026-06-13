"""Retry wrapper used by both the LLM client and the image fetcher.

The wrapper expects the inner ``send`` coroutine to either return an
``httpx.Response`` (which we then classify) or raise an httpx transport
exception. Classification produces an ``LLMError`` subclass with the right
``retryable`` bit, and the loop applies exponential backoff with jitter,
honoring ``Retry-After`` when the server provided one.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx

from llm_kit.config.http import RetryConfig
from llm_kit.errors import (
    AuthError,
    LLMError,
    NetworkError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)

logger = logging.getLogger(__name__)


async def retry_request(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    retry: RetryConfig,
    operation: str,
) -> httpx.Response:
    """Drive ``send`` with retries, returning the first successful response.

    Raises an ``LLMError`` subclass once retries are exhausted or on the first
    terminal error.
    """
    last_error: LLMError | None = None
    for attempt in range(retry.max_retries + 1):
        try:
            response = await send()
        except httpx.TimeoutException as exc:
            last_error = TimeoutError(f"{operation}: timeout", cause=exc)
        except httpx.HTTPError as exc:
            last_error = NetworkError(f"{operation}: transport error: {exc}", cause=exc)
        else:
            if response.status_code < 400:
                return response
            last_error = _classify(response, operation)

        if not last_error.retryable or attempt == retry.max_retries:
            assert last_error is not None
            raise last_error

        delay = _backoff_delay(attempt, retry, last_error)
        logger.warning(
            "%s failed (attempt %d/%d): %s — retrying in %.2fs",
            operation,
            attempt + 1,
            retry.max_retries + 1,
            last_error,
            delay,
        )
        await asyncio.sleep(delay)

    # Unreachable: the loop either returns a response or raises.
    assert last_error is not None
    raise last_error


@asynccontextmanager
async def retry_stream(
    open_stream: Callable[[], httpx.Response | Any],
    *,
    retry: RetryConfig,
    operation: str,
) -> AsyncIterator[httpx.Response]:
    """Open a streaming response with retries, yielding the validated response.

    ``open_stream`` returns an httpx streaming context manager
    (``client.stream(...)``). Retries cover only **establishing** the stream —
    transport errors before the first byte and retryable error statuses. Once a
    successful response is yielded and the caller begins consuming the body, a
    mid-stream failure propagates and is **not** retried (the bytes already
    flushed cannot be replayed).
    """
    last_error: LLMError | None = None
    for attempt in range(retry.max_retries + 1):
        cm = open_stream()
        try:
            response = await cm.__aenter__()
        except httpx.TimeoutException as exc:
            last_error = TimeoutError(f"{operation}: timeout", cause=exc)
        except httpx.HTTPError as exc:
            last_error = NetworkError(f"{operation}: transport error: {exc}", cause=exc)
        else:
            if response.status_code < 400:
                try:
                    yield response
                finally:
                    await cm.__aexit__(None, None, None)
                return
            # Error status: read the body for classification, then close.
            try:
                await response.aread()
            except Exception:  # pragma: no cover - defensive
                pass
            last_error = _classify(response, operation)
            await cm.__aexit__(None, None, None)

        if not last_error.retryable or attempt == retry.max_retries:
            raise last_error

        delay = _backoff_delay(attempt, retry, last_error)
        logger.warning(
            "%s failed (attempt %d/%d): %s — retrying in %.2fs",
            operation,
            attempt + 1,
            retry.max_retries + 1,
            last_error,
            delay,
        )
        await asyncio.sleep(delay)

    assert last_error is not None  # pragma: no cover - loop raises first
    raise last_error


def _classify(response: httpx.Response, operation: str) -> LLMError:
    status = response.status_code
    body_preview = _safe_preview(response)
    message = f"{operation}: HTTP {status}: {body_preview}"

    if status in (401, 403):
        return AuthError(message)
    if status == 429:
        return RateLimitError(message, retry_after_seconds=_retry_after(response))
    return ProviderError(
        message,
        status_code=status,
        retryable=status >= 500 or status in (408, 409, 425),
    )


def _safe_preview(response: httpx.Response, limit: int = 512) -> str:
    try:
        text = response.text
    except Exception:  # pragma: no cover - defensive against decode errors
        return "<undecodable body>"
    return text[:limit]


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _backoff_delay(attempt: int, retry: RetryConfig, error: LLMError) -> float:
    if isinstance(error, RateLimitError) and error.retry_after_seconds is not None:
        return min(error.retry_after_seconds, retry.backoff_max_seconds)
    base = retry.backoff_base_seconds * (2**attempt)
    jitter = random.uniform(0.0, retry.jitter_seconds)
    return min(base + jitter, retry.backoff_max_seconds)
