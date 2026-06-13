"""Tests for the typed exception hierarchy and retryability flags."""

from __future__ import annotations

import pytest

from llm_kit.errors import (
    AuthError,
    ImageFetchError,
    LLMError,
    NetworkError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)


@pytest.mark.parametrize(
    ("cls", "retryable"),
    [
        (NetworkError, True),
        (TimeoutError, True),
        (RateLimitError, True),
        (AuthError, False),
        (ImageFetchError, False),
        (ValidationError, False),
    ],
)
def test_retryability_flags(cls: type[LLMError], retryable: bool) -> None:
    err = cls("boom") if cls is not RateLimitError else cls("boom", retry_after_seconds=1.0)
    assert err.retryable is retryable
    assert isinstance(err, LLMError)


def test_provider_error_carries_status_and_retryable_flag() -> None:
    err = ProviderError("server down", status_code=503, retryable=True)
    assert err.status_code == 503
    assert err.retryable is True

    terminal = ProviderError("bad request", status_code=400, retryable=False)
    assert terminal.status_code == 400
    assert terminal.retryable is False


def test_rate_limit_error_carries_retry_after() -> None:
    err = RateLimitError("slow down", retry_after_seconds=5.5)
    assert err.retry_after_seconds == 5.5


def test_image_fetch_error_carries_url() -> None:
    err = ImageFetchError("bad image", url="https://x/y.png")
    assert err.url == "https://x/y.png"


def test_validation_error_carries_raw_content() -> None:
    err = ValidationError("bad json", raw_content='{"x":')
    assert err.raw_content == '{"x":'


def test_cause_chaining_preserved() -> None:
    inner = ValueError("origin")
    err = NetworkError("wrapped", cause=inner)
    assert err.cause is inner
