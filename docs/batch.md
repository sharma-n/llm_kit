# Batching

[← back to README](../README.md)

Two ways to process many prompts: a **live** concurrent pipeline that your process drives through the rate limiter, and the provider's **offline** 24h Batch API (≈50% cheaper). Both take a `list[BatchItem]` in and yield `BatchResult`s out.

## Live pipeline — three sinks

```python
from llm_kit import BatchItem, ImagePart, Message, RequestSettings

items = [
    BatchItem(
        id=f"item-{i:06d}",
        messages=[Message.user("Caption:", ImagePart.from_url("https://..."))],
        response_model=Sentiment,    # per-item schemas: a Pydantic class or a JSON schema dict
        settings=RequestSettings(temperature=0.0),  # per-item sampling/reasoning (optional)
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

Memory is O(`worker_concurrency` + `input_buffer_size`), not O(N) — which is what makes 1M-item batches tractable.

## Resume after a crash

`batch_to_jsonl(..., resume=True)` scans the existing output file for completed item IDs and skips them. There's no separate checkpoint file to keep in sync — the output IS the checkpoint.

## Provider-native batch API (≈50% cheaper, async)

The three sinks above run a *live* concurrent pipeline — your process holds the connections and paces requests through the rate limiter. The **provider batch API** is the opposite trade: hand the whole `list[BatchItem]` to the provider's 24h asynchronous batch endpoint (OpenAI Files + Batches, Anthropic Message Batches, or Google Gemini Batch Mode), poll until it finishes, then download and reconcile. Roughly half the price, at the cost of latency — and the job outlives your process.

Same `BatchItem` in, same `BatchResult` out — only the middle differs (no worker pool, no rate limiter). The backend is selected from `ProviderBatchConfig.backend` (a registered name) or, when that is `None`, from `message_format`.

```python
from llm_kit import BatchJob

# One-shot: submit, poll to completion, write reconciled results to jsonl.
written = await llm.run_batch_api(items, "out.jsonl", poll_interval=60)

# Or split across processes — the BatchJob handle is the checkpoint:
job = await llm.submit_batch(items)         # serialize + upload + create
job.save("job.json")                        # persist the handle (batch_id + manifest)

# ...later, in a fresh process...
job = BatchJob.load("job.json")
status = await llm.poll_batch(job)          # BatchJobStatus(state=..., counts=...)
if status.state.is_terminal:
    async for result in llm.fetch_batch_results(job, items=items):
        ...
```

A **dict** `response_model` (JSON schema) is persisted in the handle's manifest, so a fresh process decodes results without the original items. A **Pydantic** class is a Python type and isn't persisted — re-supply `items` at fetch time to validate against it (items win over the manifest; metadata is restored from the manifest either way). Per-item `settings` are baked into each request body at submit time. Handles saved before schemas were persisted still load. Results come back as the same `BatchResult` records the live pipeline yields, so the jsonl output and resume-skip behavior are identical. See [examples/provider_batch.py](../examples/provider_batch.py).

### Raw requests, custom and fake backends

`submit_requests` skips `BatchItem` serialization: you hand it `BatchRequest(custom_id, body)` objects and the bodies reach the backend untouched (`ValueError` on duplicate `custom_id` or more than `max_requests_per_batch`). `fetch_raw_results(job)` is the matching un-parsed iterator of `RawResult`s.

```python
from llm_kit import BatchRequest

job = await llm.submit_requests(
    [BatchRequest("a", {...provider body...})], metadata={"a": {"tag": "x"}}
)
async for raw in llm.fetch_raw_results(job):   # raw.inner_body or raw.error
    ...
```

Backends are pluggable. Inject one directly (`LLMClient(cfg, batch_backend=my_backend)`, also `from_yaml(path, batch_backend=...)`), or register a factory and select it by name:

```python
from llm_kit import register_batch_backend

register_batch_backend("mine", lambda config, client, api_key: MyBackend(...))  # replace=True to override
# config: provider_batch: { backend: mine }   (None => the message_format value)
```

An unknown name raises `ValueError` listing the registered ones. A factory may import optional dependencies lazily. `BatchBackend.submit` returns either the batch id (`str`) or `SubmitResult(batch_id, extra)`; `extra` (JSON-serialisable) is stored on `BatchJob.extra`. `BatchJob.backend` records which backend submitted the job (old files fall back to `provider`); resuming with a different backend logs a warning but proceeds. Note `extra` is persisted and available on the job, but `poll`/`fetch_results` are still called with the batch id / status only.

For tests, `llm_kit.testing.FakeBatchBackend` runs the whole offline flow in memory (see the README). Its state lives on the instance, so a second client sharing it can resume a saved job.

> All three backends follow their providers' documented Batch wire formats and are covered by mock-transport tests; there are no live-key integration tests in-repo. The OpenAI path is the most exercised.

## Per-item settings and usage

`BatchItem.settings` (`RequestSettings`: `max_tokens`, `temperature`, `top_p`, `reasoning`) lets items in one batch differ; unset fields inherit from `LLMConfig`. `BatchItem.overrides` is deep-merged into the body last. `BatchSummary.reasoning_tokens` (and each jsonl record's `usage.reasoning_tokens`) reports thinking tokens; `total_tokens` includes them. Dict-schema results are written to jsonl as plain JSON under `parsed`.
