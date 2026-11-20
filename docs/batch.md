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
if status.state.is_terminal:  # else raises BatchJobNotFinishedError (or pass wait=True)
    async for result in llm.fetch_batch_results(job, items=items):
        ...
```

### Unfinished and failed jobs

Fetching checks the job's state first (in one place, `LLMClient._fetch_raw`, so `fetch_batch_results`, `fetch_raw_results` and `run_batch_api` all behave alike):

- **Still running** (`PENDING`, `IN_PROGRESS`, `PAUSED`): raises `BatchJobNotFinishedError` (`batch_id`, `state`, `raw_state`). Pass `wait=True` (and optionally `poll_interval=`) to poll until terminal instead; `PAUSED` is non-terminal, so waiting continues through it.
- **Ended `FAILED` / `EXPIRED` / `CANCELLED`**: raises `BatchJobFailedError` (`batch_id`, `state`, `raw_state` = the provider's own status string, `detail` = the provider's error message when it exposes one, also on `BatchJobStatus.error_detail`). It is a `ProviderError` and never retryable. Pass `allow_partial=True` to fetch whatever partial output the backend kept instead (OpenAI expired/cancelled and Anthropic cancelled keep it; Gemini failed/expired/cancelled and OpenAI failed usually yield nothing).
- `run_batch_api(..., allow_partial=False)` raises *before* creating or opening `output_path`, so a failed job leaves no file and does not clobber an existing one.

These are raised on first iteration of the async generators, not when they are called. Backend state mapping: OpenAI `cancelling` stays `IN_PROGRESS` until `cancelled`; Anthropic has no job-level failure, so `ended` is `COMPLETED`, except an ended batch with `cancel_initiated_at` set and at least one `canceled` item, which is `CANCELLED`; Gemini accepts both `JOB_STATE_*` and `BATCH_STATE_*` names, with `PAUSED` mapped to `BatchState.PAUSED`.

```python
status = await llm.poll_batch(job)
results = [r async for r in llm.fetch_batch_results(job, items=items, wait=True, poll_interval=60)]
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

## Vertex AI (Gemini) batch backend

Gemini on Vertex AI batch prediction (`batchPredictionJobs`) works differently from the Gemini Developer API backend: requests go in a JSONL file on Google Cloud Storage, results come back as JSONL files on GCS. Request and response *bodies* are still the Gemini formatter's; only the transport differs.

```bash
pip install "llm_kit[vertex]"     # or: uv sync --extra vertex   (google-genai + google-cloud-storage)
gcloud auth application-default login
```

```yaml
llm:
  message_format: gemini            # required: Vertex reuses the Gemini formatter bodies
  model: gemini-2.5-flash           # bare name or publishers/google/models/...
provider_batch:
  backend: vertex
  vertex:
    project: my-project             # null => SDK/ADC default (GOOGLE_CLOUD_PROJECT)
    location: global
    gcs_prefix: gs://my-bucket/llm-kit   # required at submit
    upload_images: false            # true => upload bytes, send fileData gs:// URIs
    cleanup_input: false            # defaults for cleanup_batch()
    cleanup_output: false
    display_name_prefix: llm-kit
```

Everything else is the normal API (`run_batch_api`, `submit_batch`, `poll_batch`, `fetch_batch_results`).

**Auth and the SDK exception.** The backend never builds tokens or headers: `google-genai` (`genai.Client(vertexai=True, ...)`, async `client.aio.batches`) and `google-cloud-storage` (sync, run via `asyncio.to_thread`) resolve Application Default Credentials themselves and own their transport, so this backend does not use the shared `httpx` client (a documented exception to CLAUDE.md invariant #1). The base install has no Google dependency; selecting `vertex` without the extra raises `ImportError` with the install hint. Both SDK clients can be injected: `VertexBatchBackend(config, genai_client=..., storage_client=...)` passed as `LLMClient(batch_backend=...)`.

**IAM and bucket.** The caller needs `roles/aiplatform.user` on the project and object read/write/list/delete (`roles/storage.objectUser` or similar) on the bucket; the bucket should be in a compatible region/multi-region for the chosen location, and the Vertex AI service agent must be able to read the input and write the output prefix.

**Layout.** Each run gets `{gcs_prefix}/llm-kit-<uuid>/` containing `input.jsonl` (rows `{"request": <Gemini body>}`), `keys.jsonl` (sidecar: `custom_id` plus a canonical hash of the body as written), `output/` (Vertex writes `prediction-model-*/predictions.jsonl` below it, plus `incremental_predictions/predictions-chunked-*.jsonl` progress chunks; only the final file is read, the chunks are a fallback when no final file exists, and each `custom_id` is yielded once) and, with `upload_images`, `images/<sha256>.<ext>`. `SubmitResult.extra` (stored on `BatchJob.extra`) records `run_prefix`, `input_uri`, `output_uri`, `keys_uri`, `image_uris`, but nothing needs it: fetch and cleanup derive the run directory from the job resource (`src`/`dest`) alone, so a fresh process reconnects with just the saved handle.

**Matching rows to requests.** Vertex output rows carry no custom id and are not guaranteed to be in input order. Each request gets a label `llm_kit_id` (a label-safe encoding of the custom_id; skipped if it would exceed 63 chars). On fetch a row is matched by (1) that label if echoed in the row's `request`, else (2) the canonical hash (sorted-keys JSON, our label dropped) of the echoed `request` against `keys.jsonl`. Identical requests are interchangeable. A row matching neither is logged and skipped; a request with no output row yields a per-item error. A row with non-empty `status` or no `response` is a per-item error. A live run confirmed that Vertex echoes the label, so matching normally uses it. The hash fallback has not been exercised live; see [vertex_smoke_test.md](vertex_smoke_test.md).

**Job states.** `QUEUED/PENDING/UNSPECIFIED` -> `PENDING`; `RUNNING/CANCELLING/UPDATING` -> `IN_PROGRESS`; `SUCCEEDED/PARTIALLY_SUCCEEDED` -> `COMPLETED` (failed rows become per-item errors); `FAILED` -> `FAILED` (with `error_detail` from the job error); `CANCELLED`, `EXPIRED`, `PAUSED` map to the same-named states (`PAUSED` is waited through); unknown states stay `IN_PROGRESS` with `raw_state` kept.

**Images.** By default image bytes are inlined (`inlineData`) like the other Gemini paths. With `upload_images: true` the backend uploads each distinct image (deduped by sha256) to `{run}images/` and rewrites the part to `fileData: {mimeType, fileUri}`; use it when requests are too large to inline. Doing this in the backend is acceptable under invariant #2 because it relocates bytes (transport), while the formatter still decides what an image part is.

**Cancel and clean up.**

```python
deleted = await llm.cleanup_batch(job)                          # uses cleanup_input / cleanup_output
deleted = await llm.cleanup_batch(job, input=True, output=True) # explicit; returns deleted gs:// URIs
```

`input` covers `input.jsonl`, `keys.jsonl` and uploaded images; `output` covers `output/`. Delete output only after you have fetched results. Backends without `cleanup` raise `NotImplementedError`.

## Per-item settings and usage

`BatchItem.settings` (`RequestSettings`: `max_tokens`, `temperature`, `top_p`, `reasoning`) lets items in one batch differ; unset fields inherit from `LLMConfig`. `BatchItem.overrides` is deep-merged into the body last. `BatchSummary.reasoning_tokens` (and each jsonl record's `usage.reasoning_tokens`) reports thinking tokens; `total_tokens` includes them. Dict-schema results are written to jsonl as plain JSON under `parsed`.
