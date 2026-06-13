"""Tests for ImageFetcher: caching, retry, failure policies, oversize handling."""

from __future__ import annotations

import asyncio
import base64
from collections import defaultdict

import httpx
import pytest

from llm_kit.config.http import RetryConfig
from llm_kit.config.image import ImageConfig, ImageFailurePolicy
from llm_kit.errors import ImageFetchError
from llm_kit.images.fetcher import ImageFetcher


def _img_config(**overrides: object) -> ImageConfig:
    cfg = ImageConfig(
        max_concurrent_downloads=4,
        cache_size=8,
        retry=RetryConfig(
            max_retries=2,
            backoff_base_seconds=0.001,
            backoff_max_seconds=0.01,
            jitter_seconds=0.0,
        ),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_basic_fetch_returns_data_url() -> None:
    payload = b"\x89PNG fake"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, headers={"content-type": "image/png"})

    async with _client(handler) as client:
        fetcher = ImageFetcher(client, _img_config())
        out = await fetcher.fetch_many(["http://x/a.png"])
    expected_b64 = base64.b64encode(payload).decode("ascii")
    assert out == {"http://x/a.png": f"data:image/png;base64,{expected_b64}"}


async def test_dedup_within_batch_only_fetches_once() -> None:
    hits = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(200, content=b"x", headers={"content-type": "image/png"})

    async with _client(handler) as client:
        fetcher = ImageFetcher(client, _img_config())
        out = await fetcher.fetch_many(["http://x/a.png"] * 5)
    assert hits == 1
    assert len(out) == 1


async def test_lru_cache_avoids_refetch_across_batches() -> None:
    hits = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(200, content=b"x", headers={"content-type": "image/png"})

    async with _client(handler) as client:
        fetcher = ImageFetcher(client, _img_config())
        await fetcher.fetch_many(["http://x/a.png"])
        await fetcher.fetch_many(["http://x/a.png"])
    assert hits == 1


async def test_raise_policy_propagates_image_fetch_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing")

    async with _client(handler) as client:
        fetcher = ImageFetcher(client, _img_config(on_failure=ImageFailurePolicy.RAISE))
        with pytest.raises(ImageFetchError):
            await fetcher.fetch_many(["http://x/missing.png"])


async def test_skip_policy_omits_failed_url() -> None:
    counts: dict[str, int] = defaultdict(int)

    def handler(request: httpx.Request) -> httpx.Response:
        counts[str(request.url)] += 1
        if "good" in str(request.url):
            return httpx.Response(200, content=b"x", headers={"content-type": "image/png"})
        return httpx.Response(404, text="missing")

    async with _client(handler) as client:
        fetcher = ImageFetcher(client, _img_config(on_failure=ImageFailurePolicy.SKIP))
        out = await fetcher.fetch_many(["http://x/good.png", "http://x/bad.png"])
    assert "http://x/good.png" in out
    assert "http://x/bad.png" not in out


async def test_placeholder_policy_substitutes_data_url() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing")

    async with _client(handler) as client:
        fetcher = ImageFetcher(
            client, _img_config(on_failure=ImageFailurePolicy.PLACEHOLDER)
        )
        out = await fetcher.fetch_many(["http://x/missing.png"])
    assert out["http://x/missing.png"].startswith("data:image/png;base64,")


async def test_oversize_image_raises_under_raise_policy() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * 1000,
            headers={"content-type": "image/png"},
        )

    async with _client(handler) as client:
        fetcher = ImageFetcher(
            client,
            _img_config(max_bytes=100, on_failure=ImageFailurePolicy.RAISE),
        )
        with pytest.raises(ImageFetchError, match="exceeds max_bytes"):
            await fetcher.fetch_many(["http://x/big.png"])


async def test_empty_input_returns_empty_dict() -> None:
    def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        raise AssertionError("fetcher should not issue requests for empty input")

    async with _client(handler) as client:
        fetcher = ImageFetcher(client, _img_config())
        out = await fetcher.fetch_many([])
    assert out == {}


async def test_transient_failure_retried_then_success() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, content=b"x", headers={"content-type": "image/png"})

    async with _client(handler) as client:
        fetcher = ImageFetcher(client, _img_config())
        out = await fetcher.fetch_many(["http://x/a.png"])
    assert "http://x/a.png" in out
    assert calls == 2


async def test_concurrent_downloads_complete() -> None:
    """Many URLs in parallel should all resolve, exercising the semaphore."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=str(request.url).encode(),
            headers={"content-type": "image/png"},
        )

    urls = [f"http://x/{i}.png" for i in range(20)]
    async with _client(handler) as client:
        fetcher = ImageFetcher(client, _img_config(max_concurrent_downloads=4))
        out = await fetcher.fetch_many(urls)
    assert len(out) == 20
