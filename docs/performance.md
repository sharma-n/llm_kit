# Rate limiting, images & benchmarks

[← back to README](../README.md)

## Rate limiting

Three gates layered together, all in `RateLimitConfig`:

1. **Concurrency semaphore** (`max_concurrent_requests`) — caps in-flight HTTP requests to the LLM.
2. **RPM token bucket** — refills at `requests_per_minute / 60` per second. Honors `Retry-After` if the server sends one on a 429.
3. **TPM token bucket** (optional) — same shape but token-sized. The default estimator is `chars / 4`; pass a real tokenizer via `LLMClient(config, token_estimator=fn)` for accuracy. After each response, the bucket is **trued up** with the response's reported `usage.total_tokens` so drift doesn't accumulate.

A request must acquire all configured gates before being issued.

> **The token bucket starts full.** If you set RPM 1000, the first minute can burst up to ~2000 requests (the bucket's 1000-token capacity plus the first minute of refill). This matches how OpenAI/Anthropic actually behave. For steady-state behavior, read the **sliding-60s window median RPM**, not the overall RPM.

## Multimodal images

A message is an ordered list of `parts` (`TextPart` / `ImagePart`); formatters emit them in exactly that order, one wire block per part. `Message.user("look", ImagePart.from_url(u), "and", ImagePart.from_bytes(png, "image/png"))` builds it.

- `ImagePart.from_bytes(data, mime_type)` is base64-encoded once (on a thread above `threadpool_encode_threshold_bytes`) and never fetched or cached.
- `ImagePart.from_url("data:<mime>;base64,...")` is validated and used as is: no network, not cached. A malformed data URL raises `ImageFetchError`, subject to `image.on_failure` like any fetch failure.
- http(s) URLs are fetched in parallel by a shared `ImageFetcher`:

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

## Benchmarking

[benchmarks/rpm_throughput.py](../benchmarks/rpm_throughput.py) verifies that the configured RPM is actually enforced end-to-end, with realistic image fetching in the loop.

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

### Embeddings throughput

[benchmarks/embed_throughput.py](../benchmarks/embed_throughput.py) demonstrates the embeddings analogue: throughput is governed by how many inputs the packer puts in each request, not by RPM alone.

```bash
# Packed (default): a few fat requests, RPM never binds
uv run python benchmarks/embed_throughput.py

# The pathological un-packed case: one request per input → RPM-bound
uv run python benchmarks/embed_throughput.py --items 3500 --max-inputs 1 --rpm 2000 --latency 0.01

# Force the token cap (not the count cap) to bind the chunk size
uv run python benchmarks/embed_throughput.py --max-tokens 2000
```

Reference result for 20K inputs (~60 tokens each), 20ms/request latency:

| Scenario | inputs/request | inputs/sec | requests |
|---|---:|---:|---:|
| Packed (`max_inputs_per_request` 2048) | 2048 | 9107 | 10 |
| Un-packed (`max_inputs_per_request` 1)* | 1 | 78 | 3500 |

\* *un-packed row run at 3500 inputs / RPM 2000 / 10ms latency so it finishes; it's RPM-bound, the packed row is not.* The ~100× gap is the entire point of the embeddings micro-batching design — `max_inputs_per_request` is the dominant throughput lever for embeddings.
