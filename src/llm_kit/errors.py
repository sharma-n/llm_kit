"""Typed exception hierarchy.

The batch driver catches ``LLMError`` and turns it into a per-item ``BatchResult``
with the error attached. Non-``LLMError`` exceptions propagate (those are bugs).
"""

from __future__ import annotations


class LLMError(Exception):
    """Base for every recoverable, per-item failure."""

    retryable: bool = False

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class NetworkError(LLMError):
    """Connection reset, DNS, TLS — anything below HTTP."""

    retryable = True


class TimeoutError(LLMError):  # noqa: A001 - intentional shadow within package
    """Request timed out before the server responded."""

    retryable = True


class RateLimitError(LLMError):
    """HTTP 429 from the provider. ``retry_after_seconds`` is the server hint."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.retry_after_seconds = retry_after_seconds


class AuthError(LLMError):
    """HTTP 401/403. Terminal — retrying will not help."""

    retryable = False


class ProviderError(LLMError):
    """Other 4xx/5xx surfaces. ``status_code`` is the raw HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.status_code = status_code
        self.retryable = retryable


class ImageFetchError(LLMError):
    """An image in the message could not be retrieved after retries."""

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.url = url


class ValidationError(LLMError):
    """The response did not satisfy the supplied pydantic schema."""

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        raw_content: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.raw_content = raw_content


__all__ = [
    "AuthError",
    "ImageFetchError",
    "LLMError",
    "NetworkError",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "ValidationError",
]
