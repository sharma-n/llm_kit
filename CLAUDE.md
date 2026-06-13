# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project name gotcha

The directory is `harness/` but the installed package is **`llm_kit`** (under `src/llm_kit/`). Always `import llm_kit`, never `import harness`.

## Tooling

`uv`-managed. Python 3.13 floor.

- Install deps: `uv sync --extra dev` (the `dev` extra brings in pytest + pytest-asyncio)
- Run anything Python: `uv run python ...` (do not call `python` directly — it won't find the package)
- Run an example: `uv run python examples/single_invoke.py`
- Run the test suite: `uv run pytest`
  - One file: `uv run pytest tests/test_batch.py`
  - One test: `uv run pytest tests/test_batch.py::test_per_item_failure_is_recorded_default`
  - `asyncio_mode = "auto"` is preconfigured in `pyproject.toml`, so async tests don't need a decorator.
- Test fixtures + helpers live in [tests/conftest.py](tests/conftest.py): `make_config` (deterministic AppConfig with retries/jitter dialed down), `mock_client` (httpx.MockTransport wrapper), `chat_response` / `tool_use_response` (OpenAI-shaped response bodies), and `FakeLLM` (drop-in BaseLLM for batch-driver tests, no HTTP). Test files import these via `from conftest import ...` (not `tests.conftest`).

## Architecture — the load-bearing invariants

The package is six layers, strictly bottom-up. Each layer depends only on layers below it.

```
config  →  http  →  rate_limit / images / messages  →  llm  →  batch
```

A few invariants that span files and are easy to break without realizing:

1. **One shared `httpx.AsyncClient`.** Built once in [`http/session.py`](src/llm_kit/http/session.py) and passed into both the [`ImageFetcher`](src/llm_kit/images/fetcher.py) and the [`OpenAICompatibleLLM`](src/llm_kit/llm/openai_compat.py). Do not create a second client — pool sizing, HTTP/2 multiplexing, and keepalive economics all depend on sharing. If the LLM client built the client, it owns it (`aclose` on exit); if one was injected, the caller owns lifecycle (`owns_client` flag).

2. **The OpenAI wire format lives only in [`messages/formatter.py`](src/llm_kit/messages/formatter.py).** `OpenAICompatibleLLM` does not know what a content-array looks like. Supporting a new provider shape (Anthropic-style tool-use, etc.) is a new formatter — do not scatter wire-format code into the LLM client. The existing `StructuredOutputStyle` enum on [`LLMConfig`](src/llm_kit/config/llm.py) is the per-provider switch.

3. **`http/retry.py` is shared.** Both the image fetcher and the LLM client call `retry_request(...)` — there is no second retry implementation. Adding retry behavior means editing that one file. Each caller passes its own `RetryConfig` (LLM uses `HTTPConfig.llm_retry`, images use `ImageConfig.retry`).

4. **Pydantic schema does double duty.** When a `response_model` is supplied to `invoke` (or set on a `BatchItem`), the formatter injects its JSON schema as `response_format` in the request *and* the LLM client validates the response against it. A `ValidationError` on the way back becomes a typed `BatchResult.error`, not an exception.

5. **TPM bucket is trued-up after every response.** `OpenAICompatibleLLM.invoke` calls `RateLimiter.true_up_tokens(estimated, actual)` immediately after parsing the response. Skipping this lets the bucket drift over a long batch. The default token estimator is char/4 — pass `token_estimator=fn` to the constructor for a real tokenizer.

6. **The output jsonl IS the checkpoint.** [`batch_to_jsonl`](src/llm_kit/llm/base.py) on resume calls `load_completed_ids(path)` which scans the output file's `id` fields. There is no separate checkpoint file. This is deliberate — do not add one.

7. **Per-item failures never crash the pipeline by default.** The batch processor catches `LLMError` (and unexpected exceptions) per item and emits `BatchResult(error=...)`. The only way a batch aborts mid-stream is `BatchConfig.on_failure = raise`.

8. **The image fetcher runs upstream of the rate limiter.** `OpenAICompatibleLLM.invoke` does `fetch_images → acquire(RPM bucket) → POST`. This means *fetcher throughput is the effective ceiling on LLM RPM* — if the fetcher's semaphore is saturated, items queue before they ever reach the limiter, and the configured RPM cap becomes unreachable. Concretely: `max_concurrent_downloads ≥ ceil(RPM/60 × avg_images_per_item × avg_fetch_latency_sec)`. The benchmark in [benchmarks/rpm_throughput.py](benchmarks/rpm_throughput.py) demonstrates this — dropping fetcher slots from 64 to 4 (well below the workload's need) cuts steady-state RPM from 1001 to 863 even though the LLM limiter would happily allow more.

## The batch pipeline

[`batch/processor.py`](src/llm_kit/batch/processor.py) is the hottest code in the repo. Structure:

- Producer reads the input iterable into a bounded `input_q` (`BatchConfig.input_buffer_size`).
- N workers (`BatchConfig.worker_concurrency`) pull from `input_q`, call `llm.invoke`, push to bounded `output_q`.
- Main coroutine drains `output_q` and yields to the caller.
- Everything runs inside an `asyncio.TaskGroup`; sentinel objects signal shutdown.

Memory is O(`worker_concurrency` + `input_buffer_size`), not O(N) — which is what makes 1M-item batches tractable.

Note that batch concurrency (`BatchConfig.worker_concurrency`) is *not* the same as LLM concurrency (`RateLimitConfig.max_concurrent_requests`). Batch concurrency caps in-flight items (which include image fetches + queueing); LLM concurrency caps in-flight HTTP requests to the model. Batch concurrency should generally be `>=` LLM concurrency.

## Adding a new config section

1. New dataclass in `src/llm_kit/config/your_section.py`.
2. Add a field to `AppConfig` in [`config/app.py`](src/llm_kit/config/app.py).
3. Re-export from `config/__init__.py`.

The YAML loader walks dataclass fields via `get_type_hints` and recursively constructs nested dataclasses, enums (`StrEnum`), tuples, lists, and dicts. `${VAR}` / `${VAR:-default}` interpolation happens **after** YAML parsing and runs only on string values — so `${VAR}` inside a YAML comment is ignored. Putting interpolation back into the pre-parse stage will reintroduce that bug.

## Public surface

`from llm_kit import ...` exposes `AppConfig`, `BaseLLM`, `OpenAICompatibleLLM`, `Message`, `BatchItem`, `BatchResult`, `LLMResponse`, and the full error hierarchy. Treat anything outside that list as internal — feel free to refactor freely.

## Benchmarks

[benchmarks/rpm_throughput.py](benchmarks/rpm_throughput.py) is the load-test for the limiter + image fetcher pipeline. It runs a real `OpenAICompatibleLLM` against an `httpx.MockTransport` whose chat handler sleeps a configurable latency. Use it to validate any changes to `rate_limit/`, `images/`, or the request path in `llm/openai_compat.py`.

Flags worth knowing:
- `--items N` — batch size (default 10000; a 1000-RPM run with this default takes ~10 min)
- `--no-cache` — sets `ImageConfig.cache_size = 0`, forcing every image reference to fetch
- `--image-latency 0.1` — adds 100ms per-image sleep in the mock handler (CDN-like)
- `--max-image-downloads 4` — drops `ImageConfig.max_concurrent_downloads` to force fetcher saturation

The output reports overall RPM, sliding-60s window stats, and inter-arrival percentiles. Use the sliding-60s **median** (not the overall RPM) to read steady-state behavior — the token bucket starts full, so the first minute always bursts above the cap by design.
