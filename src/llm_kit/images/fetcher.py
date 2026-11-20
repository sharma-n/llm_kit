"""Concurrent image fetcher with retry, in-memory LRU cache, and base64 encoding.

For each URL we return a ``data:<mime>;base64,<payload>`` string ready to drop
straight into the OpenAI vision content array. Big payloads are encoded on a
thread to keep the event loop responsive; small ones encode inline.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
from collections import OrderedDict

import httpx

from llm_kit.config.image import ImageConfig, ImageFailurePolicy
from llm_kit.errors import ImageFetchError, LLMError
from llm_kit.http.retry import retry_request
from llm_kit.messages.types import ImagePart

logger = logging.getLogger(__name__)


# 1x1 transparent PNG, used when ImageFailurePolicy.PLACEHOLDER is configured.
_PLACEHOLDER_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


_DATA_URL_RE = re.compile(r"^data:([^;,]+)((?:;[^;,]*)*);base64,(.*)$", re.DOTALL)


class ImageFetcher:
    """Fetches a batch of image URLs concurrently, returns ``url → data_url``.

    Caching is per-process and per-URL. The cache is populated from any
    successful fetch and consulted before issuing a new request, so duplicate
    URLs across a batch resolve once.
    """

    def __init__(self, client: httpx.AsyncClient, config: ImageConfig) -> None:
        self._client = client
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_downloads)
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_lock = asyncio.Lock()

    async def resolve(self, parts: list[ImagePart]) -> dict[ImagePart, str]:
        """Resolve image parts to data URLs, keyed by the part itself.

        - bytes parts: base64-encoded once (off-loop when large); never fetched
          or cached;
        - ``data:`` URL parts: validated and passed through; no network, not cached;
        - other URLs: fetched via ``fetch_many`` (LRU-cached).

        Failures (fetch errors *and* malformed data URLs) follow
        ``ImageConfig.on_failure``: omitted (SKIP), placeholder, or raised.
        """
        out: dict[ImagePart, str] = {}
        fetch_urls: list[str] = []
        for part in dict.fromkeys(parts):
            if part.data is not None:
                assert part.mime_type is not None
                out[part] = await self._encode(part.data, part.mime_type)
            elif part.url is not None and part.url.startswith("data:"):
                try:
                    out[part] = await self._check_data_url(part.url)
                except ImageFetchError as exc:
                    out.update(
                        {part: v for v in self._handle_failure(part.url, exc).values()}
                    )
            elif part.url is not None:
                fetch_urls.append(part.url)
        if fetch_urls:
            fetched = await self.fetch_many(fetch_urls)
            for part in parts:
                if part.url in fetched:
                    out[part] = fetched[part.url]
        return out

    async def _check_data_url(self, url: str) -> str:
        m = _DATA_URL_RE.match(url)
        if m is None:
            raise ImageFetchError(
                "malformed image data URL (expected data:<mime>;base64,<payload>)",
                url=_truncate(url),
            )
        payload = m.group(3)
        try:
            if len(payload) >= self._config.threadpool_encode_threshold_bytes:
                await asyncio.to_thread(base64.b64decode, payload, validate=True)
            else:
                base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageFetchError(
                f"image data URL is not valid base64: {exc}",
                url=_truncate(url),
                cause=exc,
            ) from exc
        return url

    async def fetch_many(self, urls: list[str]) -> dict[str, str]:
        """Resolve a batch of URLs in parallel.

        Returns a dict from URL → data URL. Failed URLs are omitted (for SKIP),
        substituted (for PLACEHOLDER), or cause this call to raise (for RAISE).
        """
        if not urls:
            return {}
        unique = list(dict.fromkeys(urls))  # preserve order, drop dupes

        results: dict[str, str] = {}
        tasks = [asyncio.create_task(self._fetch_one(u)) for u in unique]
        try:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            for t in tasks:
                t.cancel()
            raise

        for url, outcome in zip(unique, outcomes, strict=True):
            if isinstance(outcome, str):
                results[url] = outcome
                continue
            if isinstance(outcome, BaseException):
                results.update(self._handle_failure(url, outcome))
        return results

    def _handle_failure(self, url: str, exc: BaseException) -> dict[str, str]:
        policy = self._config.on_failure
        wrapped: ImageFetchError
        if isinstance(exc, ImageFetchError):
            wrapped = exc
        elif isinstance(exc, LLMError):
            wrapped = ImageFetchError(f"image fetch failed: {exc}", url=url, cause=exc)
        else:
            wrapped = ImageFetchError(
                f"image fetch failed: {type(exc).__name__}: {exc}",
                url=url,
                cause=exc,
            )

        if policy is ImageFailurePolicy.RAISE:
            raise wrapped
        if policy is ImageFailurePolicy.PLACEHOLDER:
            logger.warning("image %s failed, substituting placeholder: %s", url, wrapped)
            return {url: _PLACEHOLDER_DATA_URL}
        logger.warning("image %s failed, skipping: %s", url, wrapped)
        return {}

    async def _fetch_one(self, url: str) -> str:
        cached = await self._cache_get(url)
        if cached is not None:
            return cached

        async with self._semaphore:
            cached = await self._cache_get(url)  # double-check after wait
            if cached is not None:
                return cached

            response = await retry_request(
                lambda: self._client.get(url, timeout=self._config.fetch_timeout_seconds),
                retry=self._config.retry,
                operation=f"image_fetch[{_truncate(url)}]",
            )

            payload = response.content
            if len(payload) > self._config.max_bytes:
                raise ImageFetchError(
                    f"image exceeds max_bytes ({len(payload)} > {self._config.max_bytes})",
                    url=url,
                )

            mime = response.headers.get("content-type", self._config.default_mime_type)
            mime = mime.split(";", 1)[0].strip() or self._config.default_mime_type

            data_url = await self._encode(payload, mime)
            await self._cache_put(url, data_url)
            return data_url

    async def _encode(self, payload: bytes, mime: str) -> str:
        if len(payload) >= self._config.threadpool_encode_threshold_bytes:
            encoded = await asyncio.to_thread(_b64_encode, payload)
        else:
            encoded = _b64_encode(payload)
        return f"data:{mime};base64,{encoded}"

    async def _cache_get(self, url: str) -> str | None:
        if self._config.cache_size <= 0:
            return None
        async with self._cache_lock:
            value = self._cache.get(url)
            if value is not None:
                self._cache.move_to_end(url)
            return value

    async def _cache_put(self, url: str, data_url: str) -> None:
        if self._config.cache_size <= 0:
            return
        async with self._cache_lock:
            self._cache[url] = data_url
            self._cache.move_to_end(url)
            while len(self._cache) > self._config.cache_size:
                self._cache.popitem(last=False)


def _b64_encode(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _truncate(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
