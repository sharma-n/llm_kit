"""Benchmark: how micro-batch packing drives embedding throughput.

This is the embeddings analogue of [rpm_throughput.py]. Where that one proves the
RPM cap binds the *chat* path, this one demonstrates invariant #9: the embeddings
unit is the *request* (an array of inputs), not the individual input — so the
dominant throughput lever is how many inputs the packer puts in each request.

Setup:
- Real ``OpenAICompatibleEmbedder`` against an ``httpx.MockTransport``.
- The embeddings handler sleeps ``--latency`` per request and returns one
  ``--dim``-length vector per input, with a token ``usage`` derived from the body.
- N inputs are streamed through ``embed_batch``; the packer chunks them under
  ``max_inputs_per_request`` / ``max_tokens_per_request``.

Measurement:
- Each request records ``time.monotonic()`` on handler entry (the moment the
  limiter released it) *and* how many inputs it carried.
- We report inputs/sec (the headline), requests-per-minute (overall + sliding
  60s window), the realized inputs-per-request distribution, and request
  inter-arrival percentiles.

The packing trade-off, made concrete (vary ``--max-inputs``):
- ``--max-inputs 2048`` (default): a few fat requests. RPM never binds; inputs/sec
  is limited only by latency and concurrency.
- ``--max-inputs 1``: one request per input — the array advantage is thrown away.
  RPM becomes the binding constraint and inputs/sec collapses to ≈ RPM/60.

Usage:
    # packed (default): high inputs/sec, few requests
    uv run python benchmarks/embed_throughput.py

    # the pathological un-packed case — RPM-bound, slow (keep --items small)
    uv run python benchmarks/embed_throughput.py --items 3000 --max-inputs 1

    # force the token cap to bind instead of the count cap
    uv run python benchmarks/embed_throughput.py --max-tokens 2000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

import httpx

from llm_kit.config.app import AppConfig
from llm_kit.config.batch import BatchConfig
from llm_kit.config.embed import EmbeddingConfig
from llm_kit.config.http import HTTPConfig, RetryConfig
from llm_kit.config.rate_limit import RateLimitConfig
from llm_kit.embed.client import OpenAICompatibleEmbedder
from llm_kit.embed.response import EmbedItem


DEFAULT_ITEMS = 20_000
DEFAULT_RPM = 3000
DEFAULT_LATENCY = 0.02
DEFAULT_DIM = 256
# A fixed ~240-char input so the char/4 token estimate is a deterministic ~60
# tokens/input — makes the token-cap lever predictable.
INPUT_TEXT = "lorem ipsum dolor sit amet " * 9


def make_handler(stats: "_Stats", latency_seconds: float, dim: int):
    zero_vec = [0.0] * dim

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        inputs = body["input"]
        n = len(inputs)
        stats.request_timestamps.append(time.monotonic())
        stats.request_input_counts.append(n)
        if latency_seconds > 0:
            await asyncio.sleep(latency_seconds)
        # Token usage roughly mirrors the client's char/4 estimate so the TPM
        # true-up is a near no-op (what a real provider would report).
        tokens = sum(len(t) for t in inputs) // 4
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "fake-embed",
                "data": [
                    {"object": "embedding", "index": i, "embedding": zero_vec}
                    for i in range(n)
                ],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )

    return handler


def make_config(
    *,
    rpm: int,
    tpm: int | None,
    max_inputs: int,
    max_tokens: int,
    dim: int,
) -> AppConfig:
    return AppConfig(
        embed=EmbeddingConfig(
            base_url="https://fake.invalid",
            model="fake-embed",
            api_key="bench",
            api_key_env="",
            dimensions=dim,
            max_inputs_per_request=max_inputs,
            max_tokens_per_request=max_tokens,
        ),
        http=HTTPConfig(llm_retry=RetryConfig(max_retries=0)),
        rate_limit=RateLimitConfig(
            max_concurrent_requests=200,  # sized so RPM/TPM bind, not concurrency
            requests_per_minute=rpm,
            tokens_per_minute=tpm,
        ),
        batch=BatchConfig(
            worker_concurrency=200,
            input_buffer_size=512,
            progress_log_every=0,  # we print our own progress
        ),
    )


def make_items(count: int) -> list[EmbedItem]:
    return [EmbedItem(id=f"bench-{i:07d}", text=INPUT_TEXT) for i in range(count)]


class _Stats:
    def __init__(self) -> None:
        self.request_timestamps: list[float] = []
        self.request_input_counts: list[int] = []


def analyze(
    stats: _Stats, inputs_completed: int, started_at: float, ended_at: float, rpm_limit: int
) -> dict[str, Any]:
    ts = stats.request_timestamps
    n_req = len(ts)
    duration = ended_at - started_at
    if n_req == 0 or duration <= 0:
        return {"requests": n_req, "inputs_completed": inputs_completed}

    overall_rpm = n_req * 60.0 / duration
    inputs_per_second = inputs_completed / duration

    # Sliding 60s window over the request stream → requests-per-minute samples.
    window = 60.0
    window_rates: list[float] = []
    left = 0
    next_sample_at = ts[0] + window
    for right in range(n_req):
        t = ts[right]
        while ts[left] < t - window:
            left += 1
        if t >= next_sample_at:
            count_in_window = right - left + 1
            actual_span = max(t - ts[left], 1e-6)
            window_rates.append(count_in_window * 60.0 / actual_span)
            next_sample_at = t + 1.0

    intervals = [ts[i] - ts[i - 1] for i in range(1, n_req)]
    intervals_sorted = sorted(intervals)

    def _pct(p: float) -> float:
        if not intervals_sorted:
            return 0.0
        return intervals_sorted[min(len(intervals_sorted) - 1, int(len(intervals_sorted) * p))]

    counts = stats.request_input_counts
    return {
        "inputs_completed": inputs_completed,
        "requests_issued": n_req,
        "wall_clock_seconds": round(duration, 2),
        "inputs_per_second": round(inputs_per_second, 1),
        "overall_rpm": round(overall_rpm, 2),
        "configured_rpm_limit": rpm_limit,
        "inputs_per_request": {
            "min": min(counts),
            "median": int(statistics.median(counts)),
            "mean": round(statistics.mean(counts), 1),
            "max": max(counts),
        },
        "window_60s_rpm": {
            "samples": len(window_rates),
            "min": round(min(window_rates), 2) if window_rates else None,
            "median": round(statistics.median(window_rates), 2) if window_rates else None,
            "max": round(max(window_rates), 2) if window_rates else None,
        },
        "request_inter_arrival_ms": {
            "min": round(min(intervals) * 1000, 2) if intervals else 0.0,
            "median": round(statistics.median(intervals) * 1000, 2) if intervals else 0.0,
            "p95": round(_pct(0.95) * 1000, 2),
            "p99": round(_pct(0.99) * 1000, 2),
            "max": round(max(intervals) * 1000, 2) if intervals else 0.0,
        },
    }


async def run(
    items_count: int,
    *,
    rpm: int,
    tpm: int | None,
    latency_seconds: float,
    max_inputs: int,
    max_tokens: int,
    dim: int,
) -> dict[str, Any]:
    print(
        f"benchmark: {items_count} inputs, RPM limit {rpm}"
        + (f", TPM limit {tpm}" if tpm else "")
        + f", latency {latency_seconds*1000:.0f}ms/request"
    )
    print(
        f"packing: max_inputs_per_request={max_inputs}, "
        f"max_tokens_per_request={max_tokens}, dim={dim}"
    )
    # An optimistic floor: if perfectly packed, this many requests are needed.
    min_requests = -(-items_count // max_inputs)  # ceil
    print(f"min requests if perfectly packed: {min_requests}")
    print()

    stats = _Stats()
    cfg = make_config(
        rpm=rpm, tpm=tpm, max_inputs=max_inputs, max_tokens=max_tokens, dim=dim
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(make_handler(stats, latency_seconds, dim))
    )
    emb = OpenAICompatibleEmbedder(cfg, client=client, owns_client=True)

    items = make_items(items_count)
    completed = 0
    last_report = 0.0
    report_every = 5.0

    started = time.monotonic()
    async with emb:
        async for _ in emb.embed_batch(items):
            completed += 1
            now = time.monotonic() - started
            if now - last_report >= report_every:
                rate = completed / now if now > 0 else 0.0
                print(
                    f"  t+{now:6.1f}s | inputs {completed:7d} | "
                    f"{rate:8.1f} inputs/s | requests {len(stats.request_timestamps):5d}"
                )
                last_report = now
    ended = time.monotonic()

    print()
    metrics = analyze(stats, completed, started, ended, rpm)
    print("results:")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items", type=int, default=DEFAULT_ITEMS,
        help=f"number of inputs to embed (default {DEFAULT_ITEMS})",
    )
    parser.add_argument(
        "--rpm", type=int, default=DEFAULT_RPM,
        help=f"requests_per_minute cap (default {DEFAULT_RPM})",
    )
    parser.add_argument(
        "--tpm", type=int, default=None,
        help="tokens_per_minute cap (default off; set to watch the token cap bind)",
    )
    parser.add_argument(
        "--latency", type=float, default=DEFAULT_LATENCY,
        help=f"simulated per-request latency in seconds (default {DEFAULT_LATENCY})",
    )
    parser.add_argument(
        "--max-inputs", type=int, default=2048,
        help="EmbeddingConfig.max_inputs_per_request — the packing lever (default 2048)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=300_000,
        help="EmbeddingConfig.max_tokens_per_request (default 300000)",
    )
    parser.add_argument(
        "--dim", type=int, default=DEFAULT_DIM,
        help=f"embedding dimensionality of the returned vectors (default {DEFAULT_DIM})",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.items,
            rpm=args.rpm,
            tpm=args.tpm,
            latency_seconds=args.latency,
            max_inputs=args.max_inputs,
            max_tokens=args.max_tokens,
            dim=args.dim,
        )
    )


if __name__ == "__main__":
    main()
