# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project name

The directory is `llm_kit/` and the installed package is **`llm_kit`** (under `src/llm_kit/`). Always `import llm_kit`.

## Tooling

`uv`-managed. Python 3.13 floor.

- Install deps: `uv sync --extra dev` (the `dev` extra brings in pytest + pytest-asyncio)
- Run anything Python: `uv run python ...` (do not call `python` directly — it won't find the package)
- Run an example: `uv run python examples/single_invoke.py`
- Run the test suite: `uv run pytest`
  - One file: `uv run pytest tests/test_batch.py`
  - One test: `uv run pytest tests/test_batch.py::test_per_item_failure_is_recorded_default`
  - `asyncio_mode = "auto"` is preconfigured in `pyproject.toml`, so async tests don't need a decorator.
- Test fixtures + helpers live in [tests/conftest.py](tests/conftest.py): `make_config` (deterministic AppConfig with retries/jitter dialed down — includes an `embed` section), `mock_client` (httpx.MockTransport wrapper), `chat_response` / `tool_use_response` / `embedding_response` (OpenAI-shaped response bodies), and `FakeLLM` (drop-in BaseLLM for batch-driver tests, no HTTP). Embed-driver tests use a local `FakeEmbedder` in [tests/test_embed_driver.py](tests/test_embed_driver.py) implementing the `embed_request` contract. Test files import these via `from conftest import ...` (not `tests.conftest`).

## Architecture — the load-bearing invariants

The package is six layers, strictly bottom-up. Each layer depends only on layers below it.

```
config  →  http  →  rate_limit / images / messages  →  llm   →  batch
                                                     →  embed →  (embed/driver)
                                                     →  llm   →  provider_batch
```

`embed/` is a peer of `llm/` — same layer, same dependencies (config, http, rate_limit), no dependency on `images/` or `messages/` (embeddings are text-in, vector-out). It does **not** reuse `batch/processor.py`; see invariant #9.

`provider_batch/` is the **offline** batch driver (the provider runs the 24h job), a sibling of the client-side `batch/` pipeline. It shares the input/output contracts (`list[BatchItem]` → `BatchResult`) but none of the middle — no worker pool, no rate limiter. It depends on `config`, `http`, `images`/`messages`, and `llm` (it reuses `LLMClient._build_request_body` for bodies and `llm/parsing.py::parse_response` for responses), but **not** `rate_limit`. See invariant #10.

A few invariants that span files and are easy to break without realizing:

1. **One shared `httpx.AsyncClient`.** Built once in [`http/session.py`](src/llm_kit/http/session.py) and passed into both the [`ImageFetcher`](src/llm_kit/images/fetcher.py) and the [`LLMClient`](src/llm_kit/llm/client.py). Do not create a second client — pool sizing, HTTP/2 multiplexing, and keepalive economics all depend on sharing. If the LLM client built the client, it owns it (`aclose` on exit); if one was injected, the caller owns lifecycle (`owns_client` flag).

2. **All wire-format logic lives only in [`messages/formatter.py`](src/llm_kit/messages/formatter.py).** `LLMClient` does not know what a content-array, body shape, or response looks like. `OpenAIFormatter`, `AnthropicFormatter`, and `GeminiFormatter` each own both sides of the conversation — full request assembly (`build_body`, which calls `format_messages`, `system_prompt`, `structured_output_fields`, `default_headers`, and renders any `tools`/`tool_choice`) and response parsing (`extract_text`, `extract_usage`, `finish_reason`, `extract_tool_calls`). They also own **stream decoding**: `stream_events(lines)` turns a provider SSE stream into normalized fragments (`messages/streaming.py`: `TextFragment` / `ToolCallFragment` / `UsageFragment` / `FinishFragment` / `MetaFragment`). Supporting a new provider is a new formatter class; nothing in `LLMClient` changes. `MessageFormat` on [`LLMConfig`](src/llm_kit/config/llm.py) selects the formatter (`OPENAI`, `ANTHROPIC`, or `GEMINI`); `StructuredOutputStyle` is a secondary switch that only applies to `OpenAIFormatter`. Tools/streaming are currently implemented for `OpenAIFormatter` only — Anthropic/Gemini raise `NotImplementedError` if `tools`/`stream` are requested (a fast-follow). The response-side assembly (text → optional `response_model` validation → `LLMResponse`) is factored into [`llm/parsing.py::parse_response`](src/llm_kit/llm/parsing.py) so the live `invoke` and the provider-batch driver build responses identically — both call the formatter's `extract_*` through it. The **streaming** analog is [`llm/streaming.py::StreamAssembler`](src/llm_kit/llm/streaming.py): provider-agnostic, it folds the formatter's fragments into public events (`TextChunk`, `ToolCallStarted`, terminal `StreamEnd`) and the same `LLMResponse` (reusing `parsing.validate_text`). **Decode vs. execute:** `llm_kit` *parses* tool calls (wire decoding); *executing* them, the multi-step loop, and feeding observations back are the consumer's job (see SPEC.md / `agent_kit`). Streaming is **single-call only** — `invoke_stream` never touches the batch path (ROADMAP constraint).

3. **`http/retry.py` is shared.** Both the image fetcher and the LLM client call `retry_request(...)` — there is no second retry implementation. Adding retry behavior means editing that one file. Each caller passes its own `RetryConfig` (LLM uses `HTTPConfig.llm_retry`, images use `ImageConfig.retry`). Streaming uses `retry_stream(...)` in the same file — it retries only *establishing* the stream (pre-first-byte transport errors + retryable statuses); once bytes flush, a mid-stream failure propagates rather than replaying.

4. **Pydantic schema does double duty.** When a `response_model` is supplied to `invoke` (or set on a `BatchItem`), the formatter injects its JSON schema into the request using the provider's native mechanism (`response_format` for OpenAI, `output_config.format` for Anthropic, `generationConfig.responseSchema` for Gemini) *and* the LLM client validates the response against it. A `ValidationError` on the way back becomes a typed `BatchResult.error`, not an exception.

5. **TPM bucket is trued-up after every response.** `LLMClient.invoke` calls `RateLimiter.true_up_tokens(estimated, actual)` immediately after parsing the response. Skipping this lets the bucket drift over a long batch. The default token estimator is char/4 — pass `token_estimator=fn` to the constructor for a real tokenizer.

6. **The output jsonl IS the checkpoint** — *for the live pipeline*. [`batch_to_jsonl`](src/llm_kit/llm/base.py) on resume calls `load_completed_ids(path)` which scans the output file's `id` fields. There is no separate checkpoint file. This is deliberate — do not add one. (Invariants #5, #6, and #8 are scoped to the *live* path; they do **not** carry over to `provider_batch/`, which has no per-request pacing and no local output to resume from until the provider's job completes — see #10.)

7. **Per-item failures never crash the pipeline by default.** The batch processor catches `LLMError` (and unexpected exceptions) per item and emits `BatchResult(error=...)`. The only way a batch aborts mid-stream is `BatchConfig.on_failure = raise`.

8. **The image fetcher runs upstream of the rate limiter.** `LLMClient.invoke` does `fetch_images → acquire(RPM bucket) → POST`. This means *fetcher throughput is the effective ceiling on LLM RPM* — if the fetcher's semaphore is saturated, items queue before they ever reach the limiter, and the configured RPM cap becomes unreachable. Concretely: `max_concurrent_downloads ≥ ceil(RPM/60 × avg_images_per_item × avg_fetch_latency_sec)`. The benchmark in [benchmarks/rpm_throughput.py](benchmarks/rpm_throughput.py) demonstrates this — dropping fetcher slots from 64 to 4 (well below the workload's need) cuts steady-state RPM from 1001 to 863 even though the LLM limiter would happily allow more.

9. **Embeddings are chunk-as-unit, not item-as-unit.** The `/v1/embeddings` endpoint takes an *array* of inputs per request, so [`embed/driver.py`](src/llm_kit/embed/driver.py) has a **packer** stage that accumulates inputs into request-sized chunks (bounded by `EmbeddingConfig.max_inputs_per_request` *and* `max_tokens_per_request`) before any POST. This is why it doesn't reuse `batch/processor.py` (whose unit is one-item-one-request). The rate-limit mapping follows from this: **one chunk = one RPM debit**, and the **TPM debit is the sum of the chunk's inputs**, trued up from the response's reported usage. Do not "simplify" this back to one-request-per-input — that throws away the dominant throughput lever for embeddings. Consequence to keep in mind: failures are coarser-grained — one failed request fails its whole chunk, so every input in it gets the same `EmbeddingResult.error` (per-input `usage` is the request total split by `split_usage`, exact in aggregate).

10. **Provider batch splits "what to send" from "how to send it."** [`provider_batch/`](src/llm_kit/provider_batch/) has two axes. The **request/response *body*** is fully reused — the formatter's `build_body` (via `LLMClient._build_request_body`) produces each request body, and `parse_response` parses each result body — so a new provider needs *no* new body code. The **transport** (file-upload vs inline array, create/poll/results endpoints, status vocabulary, and the per-line *envelope* that wraps a `custom_id` around the body and unwraps each result) is a [`BatchBackend`](src/llm_kit/provider_batch/backend.py) implementation, selected by `MessageFormat` exactly like the formatter. The envelope lives in the backend, not the formatter — it is coupled to the transport, not the message shape. All three backends (`OpenAIBatchBackend` file-upload; `Anthropic`/`Gemini` inline-array) are implemented against their providers' documented Batch wire formats and covered by mock-transport tests; there are **no live-key integration tests** in-repo (the OpenAI path is the one most likely exercised against a real endpoint). Because a job runs up to 24h, the [`BatchJob`](src/llm_kit/provider_batch/job.py) handle is a *necessary* sidecar checkpoint (contrast #6) — it persists `batch_id` + a `custom_id`→metadata manifest so a fresh process can reconnect. `response_model` is a Python type and is **not** persisted: cross-process result validation requires re-supplying the original `items` to `fetch_batch_results`; otherwise results come back as raw text (metadata is always restored from the manifest).

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

## Cost & usage accounting

[`batch/ledger.py`](src/llm_kit/batch/ledger.py) provides `UsageLedger`, a mutable accumulator for token usage across a batch run. `BaseLLM.batch_to_jsonl` uses it internally and now returns a `BatchSummary` (total items, ok/error counts, token totals, estimated cost) instead of a bare `int`.

`LLMClient` overrides `batch_to_jsonl` to inject pricing from `LLMConfig.prompt_cost_per_1m_tokens` and `LLMConfig.completion_cost_per_1m_tokens` (both default to `0.0`, so accounting is fully opt-in). Pricing is not in the API response — providers only return token counts — so set these fields to the provider's published per-token rates in your YAML config.

For streaming use (`batch_stream`), construct a `UsageLedger` yourself and call `.add(result)` per iteration, then `.to_summary(prompt_cost_per_1m=..., completion_cost_per_1m=...)` when done.

## Public surface

`from llm_kit import ...` exposes `AppConfig`, `BaseLLM`, `LLMClient`, `OpenAICompatibleEmbedder`, `Message`, `BatchItem`, `BatchResult`, `BatchSummary`, `LLMResponse`, `ToolCall`, `ToolDefinition`, `TextChunk`, `ToolCallStarted`, `StreamEnd`, `StreamEvent`, `EmbedItem`, `EmbeddingResponse`, `EmbeddingResult`, `BatchJob`, `BatchJobStatus`, `BatchState`, `UsageLedger`, and the full error hierarchy. Treat anything outside that list as internal — feel free to refactor freely. (The `provider_batch` backends are internal; `LLMClient.submit_batch` / `poll_batch` / `fetch_batch_results` / `run_batch_api` are the surface.) `LLMClient.invoke` and `invoke_stream` both accept `tools` / `tool_choice` and surface parsed `ToolCall`s; `invoke_stream` yields `StreamEvent`s (`TextChunk*` → optional `ToolCallStarted` → terminal `StreamEnd`).

## Benchmarks

[benchmarks/rpm_throughput.py](benchmarks/rpm_throughput.py) is the load-test for the limiter + image fetcher pipeline. It runs a real `LLMClient` against an `httpx.MockTransport` whose chat handler sleeps a configurable latency. Use it to validate any changes to `rate_limit/`, `images/`, or the request path in `llm/client.py`.

Flags worth knowing:
- `--items N` — batch size (default 10000; a 1000-RPM run with this default takes ~10 min)
- `--no-cache` — sets `ImageConfig.cache_size = 0`, forcing every image reference to fetch
- `--image-latency 0.1` — adds 100ms per-image sleep in the mock handler (CDN-like)
- `--max-image-downloads 4` — drops `ImageConfig.max_concurrent_downloads` to force fetcher saturation

The output reports overall RPM, sliding-60s window stats, and inter-arrival percentiles. Use the sliding-60s **median** (not the overall RPM) to read steady-state behavior — the token bucket starts full, so the first minute always bursts above the cap by design.

[benchmarks/embed_throughput.py](benchmarks/embed_throughput.py) is the embeddings counterpart — a real `OpenAICompatibleEmbedder` against a latency-sleeping mock embeddings handler. Use it to validate changes to `embed/` (the packer especially) or the embeddings request path. It measures **inputs/sec** (the headline) and the realized inputs-per-request distribution, so it directly exercises invariant #9. The key lever is `--max-inputs` (`EmbeddingConfig.max_inputs_per_request`): the default 2048 packs into a handful of requests; `--max-inputs 1` is the un-packed pathological case where RPM binds and inputs/sec collapses ~100×. Same bucket-burst caveat applies — short runs report `overall_rpm` above the cap, so read steady state from the inter-arrival percentiles or a >60s run's window samples.
