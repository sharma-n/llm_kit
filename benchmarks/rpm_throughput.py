"""Benchmark: verify the configured RPM cap is actually enforced.

Setup:
- Real ``LLMClient`` against an ``httpx.MockTransport``.
- Chat handler sleeps ``LATENCY_SECONDS`` to simulate a slow LLM.
- Image handler returns a tiny fake PNG (cached after first fetch).
- Batch of N items, each with 1–4 random image URLs drawn from a small pool.

Measurement:
- Every chat request records ``time.monotonic()`` the instant the handler is
  entered — i.e. the moment the limiter released the request.
- We report overall RPM, sliding 60-second window stats (peak / min / mean
  after a warmup), and inter-arrival-time percentiles.

Notes:
- The token bucket starts at full capacity, so the first 60 seconds will burst
  above the cap (up to ``capacity + refill_during_window``). This is by-design
  bucket behavior; the *steady-state* RPM is the metric to read.
- The in-flight semaphore (``max_concurrent_requests``) is sized well above
  ``RPM * latency / 60`` so RPM is the binding constraint, not concurrency.

Usage:
    uv run python benchmarks/rpm_throughput.py            # default: 10000 items, ~10 min
    uv run python benchmarks/rpm_throughput.py --items 500  # quick smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from typing import Any

import httpx

from llm_kit.config.app import AppConfig
from llm_kit.config.batch import BatchConfig
from llm_kit.config.http import HTTPConfig, RetryConfig
from llm_kit.config.image import ImageConfig, ImageFailurePolicy
from llm_kit.config.llm import LLMConfig
from llm_kit.config.rate_limit import RateLimitConfig
from llm_kit.llm.client import LLMClient
from llm_kit.llm.response import BatchItem
from llm_kit.messages.types import ImagePart, Message


LATENCY_SECONDS = 3.0
RPM_LIMIT = 1000
IMAGE_POOL_SIZE = 20
DEFAULT_ITEMS = 10_000

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake image bytes for benchmark"

_CHAT_RESPONSE: dict[str, Any] = {
    "id": "bench",
    "model": "fake-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


def make_handler(stats: "_Stats", image_latency_seconds: float):
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/img/" in url:
            stats.image_fetches += 1
            if image_latency_seconds > 0:
                await asyncio.sleep(image_latency_seconds)
            return httpx.Response(
                200, content=FAKE_PNG, headers={"content-type": "image/png"}
            )
        stats.chat_timestamps.append(time.monotonic())
        await asyncio.sleep(LATENCY_SECONDS)
        return httpx.Response(200, json=_CHAT_RESPONSE)

    return handler


def make_config(*, cache_size: int, max_image_downloads: int) -> AppConfig:
    return AppConfig(
        llm=LLMConfig(
            base_url="https://fake.invalid",
            model="fake-model",
            api_key="bench",
            api_key_env="",
        ),
        http=HTTPConfig(
            llm_retry=RetryConfig(max_retries=0),  # no retries on the happy path
        ),
        rate_limit=RateLimitConfig(
            max_concurrent_requests=200,  # >> RPM*latency/60 = 50, so RPM binds
            requests_per_minute=RPM_LIMIT,
            tokens_per_minute=None,
        ),
        image=ImageConfig(
            max_concurrent_downloads=max_image_downloads,
            cache_size=cache_size,  # 0 disables the LRU cache
            on_failure=ImageFailurePolicy.RAISE,
            retry=RetryConfig(max_retries=0),
        ),
        batch=BatchConfig(
            worker_concurrency=200,
            input_buffer_size=512,
            progress_log_every=0,  # we print our own progress
        ),
    )


def make_items(count: int) -> list[BatchItem]:
    rng = random.Random(0xC0FFEE)
    pool = [f"https://fake.invalid/img/{i}.png" for i in range(IMAGE_POOL_SIZE)]
    items: list[BatchItem] = []
    for i in range(count):
        k = rng.randint(1, 4)
        images = rng.sample(pool, k=k)
        items.append(
            BatchItem(
                id=f"bench-{i:06d}",
                messages=[Message.user(
                        f"describe item {i}", *(ImagePart.from_url(u) for u in images)
                    )],
            )
        )
    return items


class _Stats:
    def __init__(self) -> None:
        self.chat_timestamps: list[float] = []
        self.image_fetches: int = 0


def analyze(stats: _Stats, started_at: float, ended_at: float) -> dict[str, Any]:
    ts = stats.chat_timestamps
    n = len(ts)
    if n == 0:
        return {"count": 0}

    duration = ended_at - started_at
    overall_rpm = n * 60.0 / duration

    # Sliding 60s window over the timestamp stream. Per second, count the
    # requests in the prior 60s and emit a rate sample.
    window = 60.0
    window_rates: list[float] = []
    left = 0
    next_sample_at = ts[0] + window  # only sample once we have a full window
    for right in range(n):
        t = ts[right]
        while ts[left] < t - window:
            left += 1
        if t >= next_sample_at:
            count_in_window = right - left + 1
            actual_span = max(t - ts[left], 1e-6)
            window_rates.append(count_in_window * 60.0 / actual_span)
            next_sample_at = t + 1.0  # one sample per second of elapsed time

    intervals = [ts[i] - ts[i - 1] for i in range(1, n)]
    intervals_sorted = sorted(intervals)

    def _pct(p: float) -> float:
        if not intervals_sorted:
            return 0.0
        return intervals_sorted[min(len(intervals_sorted) - 1, int(len(intervals_sorted) * p))]

    return {
        "items_completed": n,
        "wall_clock_seconds": round(duration, 2),
        "overall_rpm": round(overall_rpm, 2),
        "configured_rpm_limit": RPM_LIMIT,
        "ratio_to_limit": round(overall_rpm / RPM_LIMIT, 3),
        "window_60s_rpm": {
            "samples": len(window_rates),
            "min": round(min(window_rates), 2) if window_rates else None,
            "mean": round(statistics.mean(window_rates), 2) if window_rates else None,
            "median": round(statistics.median(window_rates), 2) if window_rates else None,
            "max": round(max(window_rates), 2) if window_rates else None,
        },
        "inter_arrival_ms": {
            "min": round(min(intervals) * 1000, 2) if intervals else 0.0,
            "median": round(statistics.median(intervals) * 1000, 2) if intervals else 0.0,
            "p95": round(_pct(0.95) * 1000, 2),
            "p99": round(_pct(0.99) * 1000, 2),
            "max": round(max(intervals) * 1000, 2) if intervals else 0.0,
        },
        "image_http_requests": stats.image_fetches,
    }


async def run(
    items_count: int,
    *,
    no_cache: bool,
    image_latency_seconds: float,
    max_image_downloads: int,
) -> dict[str, Any]:
    cache_size = 0 if no_cache else IMAGE_POOL_SIZE * 2
    print(
        f"benchmark: {items_count} items, RPM limit {RPM_LIMIT}, "
        f"latency {LATENCY_SECONDS}s/request"
    )
    print(
        f"image config: cache_size={cache_size} "
        f"({'DISABLED' if cache_size == 0 else 'enabled'}), "
        f"per-fetch latency {image_latency_seconds*1000:.0f}ms, "
        f"max_concurrent_downloads={max_image_downloads}"
    )
    estimated = items_count / RPM_LIMIT * 60 + LATENCY_SECONDS
    print(f"estimated runtime: ~{estimated/60:.1f} min ({estimated:.0f}s)")
    print()

    stats = _Stats()
    cfg = make_config(cache_size=cache_size, max_image_downloads=max_image_downloads)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(make_handler(stats, image_latency_seconds))
    )
    llm = LLMClient(cfg, client=client, owns_client=True)

    items = make_items(items_count)
    completed = 0
    last_report = 0.0
    report_every = max(15.0, estimated / 20)  # ~20 progress lines

    started = time.monotonic()
    async with llm:
        async for _ in llm.batch_stream(items):
            completed += 1
            now = time.monotonic() - started
            if now - last_report >= report_every:
                running_rpm = completed * 60.0 / now if now > 0 else 0.0
                print(
                    f"  t+{now:6.1f}s | completed {completed:6d} | "
                    f"running RPM {running_rpm:7.1f} | "
                    f"image fetches {stats.image_fetches:6d}"
                )
                last_report = now
    ended = time.monotonic()

    print()
    metrics = analyze(stats, started, ended)
    print("results:")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=int,
        default=DEFAULT_ITEMS,
        help=f"number of batch items (default {DEFAULT_ITEMS})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable the image LRU cache (every reference triggers an HTTP fetch + b64 encode)",
    )
    parser.add_argument(
        "--image-latency",
        type=float,
        default=0.0,
        help="simulated per-image download latency in seconds (default 0; try 0.1 for CDN-like)",
    )
    parser.add_argument(
        "--max-image-downloads",
        type=int,
        default=64,
        help="ImageConfig.max_concurrent_downloads (default 64; lower to force fetcher bottleneck)",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.items,
            no_cache=args.no_cache,
            image_latency_seconds=args.image_latency,
            max_image_downloads=args.max_image_downloads,
        )
    )


if __name__ == "__main__":
    main()
