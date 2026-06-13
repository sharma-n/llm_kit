# llm_kit

A provider-agnostic, async, high-throughput toolkit for multimodal LLM applications.

Bring your own endpoint (anything that speaks OpenAI's `/v1/chat/completions` shape), your own API key, and a YAML config — `llm_kit` handles the rate limiting, image fetching + base64 encoding, retries, structured outputs, and 1M-scale batching.

## Why this exists

Most LLM client libraries assume one provider, one model, and synchronous calls. When you're building agentic systems or running large multimodal batches, you usually need:

- **Provider freedom** — swap OpenAI for vLLM / Together / Anthropic / a local Ollama instance with a config change, not a code change.
- **Multimodal by default** — every message can carry zero, one, or many image URLs, and they should be fetched, encoded, and inlined transparently.
- **Real throughput** — async, persistent HTTPS connections, RPM and TPM rate limits respected, hundreds of concurrent requests.
- **Big batches without OOM** — run a million prompts through and stream results to disk as they complete.

That's what this is.

## Install

This is a `uv`-managed project on Python 3.13+.

```bash
uv sync               # production deps only
uv sync --extra dev   # adds pytest + pytest-asyncio
```

The package is `llm_kit` (the repo directory happens to be called `harness/`).

## Quick start

### Single request

```python
import asyncio
from llm_kit import Message, OpenAICompatibleLLM

async def main():
    async with OpenAICompatibleLLM.from_yaml("examples/config.yaml") as llm:
        response = await llm.invoke([
            Message.system("You are a terse assistant."),
            Message.user("In one sentence, what is the capital of France?"),
        ])
    print(response.text)
    print(response.usage)        # TokenUsage(prompt=..., completion=..., total=...)

asyncio.run(main())
```

Set `OPENAI_API_KEY` (or whichever env var your config points at) and run with `uv run python examples/single_invoke.py`.

### Single request with images

Any URL goes in `image_urls`; `llm_kit` fetches and base64-encodes it before sending.

```python
response = await llm.invoke([
    Message.system("Describe the image in one sentence."),
    Message.user("Caption this:", image_urls=[
        "https://example.com/cat.jpg",
        "https://example.com/dog.png",
    ]),
])
```

### Structured outputs (one model does double duty)

Pass a pydantic model — it's both injected as `response_format` in the request AND used to validate the response. The validated object lands at `response.parsed`.

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    confidence: float

response = await llm.invoke(
    [Message.user("how do you feel?")],
    response_model=Sentiment,
)
print(response.parsed.label, response.parsed.confidence)
```

If the model rejects the response, you get a typed `ValidationError` (subclass of `LLMError`) with the raw content attached for debugging.

### Batch — three sinks for the same pipeline

```python
from llm_kit import BatchItem, Message

items = [
    BatchItem(
        id=f"item-{i:06d}",
        messages=[Message.user(f"Caption:", image_urls=["https://..."])],
        response_model=Sentiment,    # per-item schemas are supported
    )
    for i in range(1_000_000)
]

# (1) async iterator — yields results as they complete
async for result in llm.batch_stream(items):
    if result.ok:
        do_something_with(result.response)
    else:
        log_error(result.error)

# (2) stream to JSONL, resumable on crash
written = await llm.batch_to_jsonl(items, "out.jsonl", resume=True)

# (3) per-result callback (sync or async)
await llm.batch_to_callback(items, lambda r: save(r))
```

Per-item failures become `BatchResult(error=...)` records — they don't crash the batch. `BatchConfig.on_failure: raise` flips that to fail-fast.

### Resume after a crash

`batch_to_jsonl(..., resume=True)` scans the existing output file for completed item IDs and skips them. There's no separate checkpoint file to keep in sync — the output IS the checkpoint.

## Configuration

Everything is driven by a YAML file loaded into [`AppConfig`](src/llm_kit/config/app.py). Environment-variable interpolation with `${VAR}` and `${VAR:-default}` works inside string values.

The full schema is in [examples/config.yaml](examples/config.yaml). The five sections:

| Section | Knobs |
|---|---|
| `llm` | `base_url`, `model`, `api_key_env`, sampling defaults, `structured_output_style` |
| `http` | connect/read/pool timeouts, pool sizes, HTTP/2, retry policy for the LLM call |
| `rate_limit` | `requests_per_minute`, `tokens_per_minute`, `max_concurrent_requests` |
| `image` | `max_concurrent_downloads`, `cache_size`, `on_failure`, retry policy for image fetches |
| `batch` | `worker_concurrency`, `input_buffer_size`, `on_failure`, progress logging |

A minimal config (only override what you need):

```yaml
llm:
  base_url: https://api.openai.com
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
rate_limit:
  requests_per_minute: 1000
  tokens_per_minute: 1500000
  max_concurrent_requests: 64
```

All other fields take their dataclass defaults.

### Provider portability

For an OpenAI-compatible non-OpenAI endpoint:

```yaml
llm:
  base_url: https://api.together.xyz       # or http://localhost:11434 for Ollama
  model: meta-llama/Llama-3.3-70B-Instruct
  api_key_env: TOGETHER_API_KEY
  structured_output_style: openai_json_schema  # or tool_use, or none
```

`structured_output_style` is the per-provider switch for how `response_format` is wired. Anthropic-style providers may want `tool_use`; pure local servers without structured-output support take `none`.

## Rate limiting

Three gates layered together, all in `RateLimitConfig`:

1. **Concurrency semaphore** (`max_concurrent_requests`) — caps in-flight HTTP requests to the LLM.
2. **RPM token bucket** — refills at `requests_per_minute / 60` per second. Honors `Retry-After` if the server sends one on a 429.
3. **TPM token bucket** (optional) — same shape but token-sized. The default estimator is `chars / 4`; pass a real tokenizer via `OpenAICompatibleLLM(config, token_estimator=fn)` for accuracy. After each response, the bucket is **trued up** with the response's reported `usage.total_tokens` so drift doesn't accumulate.

A request must acquire all configured gates before being issued.

> **The token bucket starts full.** If you set RPM 1000, the first minute can burst up to ~2000 requests (the bucket's 1000-token capacity plus the first minute of refill). This matches how OpenAI/Anthropic actually behave. For steady-state behavior, read the **sliding-60s window median RPM**, not the overall RPM.

## Multimodal images

URLs in `Message.image_urls` are fetched in parallel by a shared `ImageFetcher`:

- **Concurrent**, bounded by `ImageConfig.max_concurrent_downloads`.
- **Cached** in an in-memory LRU keyed by URL — duplicate URLs across a batch resolve once.
- **Retried** independently of the LLM call (`ImageConfig.retry`).
- **Big payloads** (>1 MiB by default) are base64-encoded on a thread to avoid blocking the event loop.

Failure behavior is configurable:

```yaml
image:
  on_failure: raise         # raise ImageFetchError up to the batch result (default)
  # on_failure: skip        # drop the failed image, send the rest of the message
  # on_failure: placeholder # substitute a 1x1 transparent PNG
```

> **Tuning rule:** the fetcher runs *before* the LLM rate limiter. If it can't feed the limiter fast enough, your configured RPM becomes unreachable. Make sure:
>
> `max_concurrent_downloads ≥ ⌈ RPM/60 × avg_images_per_item × avg_fetch_latency_seconds ⌉`
>
> The benchmark below demonstrates this empirically.

## Errors

All recoverable failures inherit from `LLMError`:

| Class | When | Retryable |
|---|---|---|
| `NetworkError` | Below-HTTP transport error | yes |
| `TimeoutError` | Connect/read timeout | yes |
| `RateLimitError` | HTTP 429 (carries `retry_after_seconds`) | yes |
| `AuthError` | HTTP 401 / 403 | no |
| `ProviderError` | Other 4xx/5xx (carries `status_code`) | depends on class |
| `ImageFetchError` | Image download failed after retries | no |
| `ValidationError` | Response didn't match pydantic schema | no |

In a batch, these become `BatchResult.error`. Outside a batch, they're raised from `invoke()`.

## Examples

- [examples/single_invoke.py](examples/single_invoke.py) — one chat call
- [examples/batch_structured.py](examples/batch_structured.py) — small batch with per-item pydantic schemas and image inputs

## Tests

```bash
uv run pytest                                       # all 91 tests, ~1.5s
uv run pytest tests/test_batch.py                   # one module
uv run pytest tests/test_batch.py::test_per_item_failure_is_recorded_default
```

The suite uses `httpx.MockTransport` for wire-level coverage and a `FakeLLM` (in [tests/conftest.py](tests/conftest.py)) for batch-driver coverage. No network access needed.

## Benchmarking

[benchmarks/rpm_throughput.py](benchmarks/rpm_throughput.py) verifies that the configured RPM is actually enforced end-to-end, with realistic image fetching in the loop.

```bash
# Validate RPM enforcement at 1000 RPM, 10K items, ~10 min
uv run python benchmarks/rpm_throughput.py

# Smaller sanity run
uv run python benchmarks/rpm_throughput.py --items 500

# Stress the image fetcher with realistic CDN latency, no cache
uv run python benchmarks/rpm_throughput.py --no-cache --image-latency 0.1

# Force the image fetcher to be the bottleneck
uv run python benchmarks/rpm_throughput.py --no-cache --image-latency 0.1 --max-image-downloads 4
```

Reference result for 10K items at 1000 RPM with a 3s/request simulated LLM latency:

| Scenario | Steady-state RPM | Wall clock |
|---|---:|---:|
| Cached (20 unique URLs reused) | 1001.0 | 543s |
| No cache, 100ms image latency, 64 fetch slots | 1001.0 | 543s |
| No cache, 100ms image latency, **4 fetch slots** | 863.0 | 697s |

The first two are statistically identical — at sensible defaults, image traffic doesn't move the LLM RPM. The third shows what happens when the fetcher becomes the bottleneck: items queue upstream of the rate limiter, and effective RPM drops below the configured cap.

## Project layout

```
src/llm_kit/
  config/        # dataclasses + YAML loader (env-var interpolation)
  http/          # shared httpx.AsyncClient + retry wrapper
  rate_limit/    # token bucket + composed limiter (sem + RPM + TPM)
  images/        # async fetch + LRU cache + base64 encode
  messages/      # Message dataclass + OpenAI wire-format adapter
  llm/           # BaseLLM + OpenAICompatibleLLM + LLMResponse
  batch/         # bounded pipeline + jsonl writer + ID-based resume
  errors.py      # typed exception hierarchy
examples/        # runnable examples
tests/           # pytest suite (91 tests)
benchmarks/      # load tests
```

## License

Add your own.
