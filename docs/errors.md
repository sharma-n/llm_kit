# Errors

[← back to README](../README.md)

All recoverable failures inherit from `LLMError`:

| Class | When | Retryable |
|---|---|---|
| `NetworkError` | Below-HTTP transport error | yes |
| `TimeoutError` | Connect/read timeout | yes |
| `RateLimitError` | HTTP 429 (carries `retry_after_seconds`) | yes |
| `AuthError` | HTTP 401 / 403 | no |
| `ProviderError` | Other 4xx/5xx (carries `status_code`) | depends on class |
| `BatchJobNotFinishedError` | Fetched results from a provider batch job still running (carries `batch_id`, `state`, `raw_state`) | no |
| `BatchJobFailedError` | Provider batch job ended failed/expired/cancelled; subclass of `ProviderError` (carries `batch_id`, `state`, `raw_state`, `detail`) | no |
| `ImageFetchError` | Image download failed after retries | no |
| `ValidationError` | Response didn't match pydantic schema | no |

In a batch, these become `BatchResult.error`. Outside a batch, they're raised from `invoke()` and `invoke_stream()`.
