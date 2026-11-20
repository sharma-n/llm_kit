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


class BatchJobNotFinishedError(LLMError):
    """Results were requested from a provider batch job that is still running.

    Pass ``wait=True`` to ``fetch_batch_results`` / ``fetch_raw_results`` (or poll
    until ``BatchState.is_terminal``) first. ``raw_state`` is the provider's own
    status string.
    """

    retryable = False

    def __init__(self, *, batch_id: str, state: object, raw_state: str = "") -> None:
        value = getattr(state, "value", state)
        super().__init__(
            f"batch job {batch_id} has not finished (state={value}, "
            f"raw_state={raw_state!r}); pass wait=True or poll until it is terminal"
        )
        self.batch_id = batch_id
        self.state = state
        self.raw_state = raw_state


class BatchJobFailedError(ProviderError):
    """A provider batch job ended failed, expired or cancelled (job-level failure).

    Never retryable. ``state`` is the normalized ``BatchState``, ``raw_state`` the
    provider's own status string, ``detail`` the provider's error message if any.
    Pass ``allow_partial=True`` to fetch whatever partial output the provider kept.
    """

    def __init__(
        self,
        *,
        batch_id: str,
        state: object,
        raw_state: str = "",
        detail: str | None = None,
    ) -> None:
        value = getattr(state, "value", state)
        msg = f"batch job {batch_id} ended {value} (raw_state={raw_state!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg, retryable=False)
        self.batch_id = batch_id
        self.state = state
        self.raw_state = raw_state
        self.detail = detail


__all__ = [
    "AuthError",
    "BatchJobFailedError",
    "BatchJobNotFinishedError",
    "ImageFetchError",
    "LLMError",
    "NetworkError",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "ValidationError",
]
