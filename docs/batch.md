# Batching

[← back to README](../README.md)

Two ways to process many prompts: a **live** concurrent pipeline that your process drives through the rate limiter, and the provider's **offline** 24h Batch API (≈50% cheaper). Both take a `list[BatchItem]` in and yield `BatchResult`s out.

## Live pipeline — three sinks

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

Memory is O(`worker_concurrency` + `input_buffer_size`), not O(N) — which is what makes 1M-item batches tractable.

## Resume after a crash

`batch_to_jsonl(..., resume=True)` scans the existing output file for completed item IDs and skips them. There's no separate checkpoint file to keep in sync — the output IS the checkpoint.

## Provider-native batch API (≈50% cheaper, async)

The three sinks above run a *live* concurrent pipeline — your process holds the connections and paces requests through the rate limiter. The **provider batch API** is the opposite trade: hand the whole `list[BatchItem]` to the provider's 24h asynchronous batch endpoint (OpenAI Files + Batches, Anthropic Message Batches, or Google Gemini Batch Mode), poll until it finishes, then download and reconcile. Roughly half the price, at the cost of latency — and the job outlives your process.

Same `BatchItem` in, same `BatchResult` out — only the middle differs (no worker pool, no rate limiter). The backend is selected automatically from `message_format`.

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

`response_model` is a Python type, so it isn't persisted in the handle — re-supply `items` at fetch time and per-item schemas validate as usual (metadata is restored from the manifest either way). Results come back as the same `BatchResult` records the live pipeline yields, so the jsonl output and resume-skip behavior are identical. See [examples/provider_batch.py](../examples/provider_batch.py).

> All three backends follow their providers' documented Batch wire formats and are covered by mock-transport tests; there are no live-key integration tests in-repo. The OpenAI path is the most exercised.
