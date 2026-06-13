"""Factory for the shared ``httpx.AsyncClient``."""

from __future__ import annotations

import httpx

from llm_kit.config.http import HTTPConfig


def build_async_client(config: HTTPConfig) -> httpx.AsyncClient:
    """Construct an AsyncClient with HTTP/2 + a sized connection pool.

    One client is shared across the LLM client and the image fetcher; httpx
    multiplexes by host so both layers benefit from a warm pool.
    """
    timeout = httpx.Timeout(
        connect=config.connect_timeout_seconds,
        read=config.read_timeout_seconds,
        write=config.write_timeout_seconds,
        pool=config.pool_timeout_seconds,
    )
    limits = httpx.Limits(
        max_connections=config.max_connections,
        max_keepalive_connections=config.max_keepalive_connections,
        keepalive_expiry=config.keepalive_expiry_seconds,
    )
    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        http2=config.http2,
        follow_redirects=True,
    )
