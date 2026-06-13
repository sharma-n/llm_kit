"""OpenAI-compatible embeddings client.

Targets any endpoint speaking ``POST /v1/embeddings`` with the OpenAI
request/response shape. Sits at the same layer as ``OpenAICompatibleLLM`` and
reuses the same shared ``httpx.AsyncClient``, retry wrapper, rate limiter, and
error hierarchy.

The defining difference from the chat client: ``/v1/embeddings`` takes an
*array* of inputs per request. The hot path is therefore micro-batched — see
``embed/driver.py``. A single request maps to *one* RPM debit and a TPM debit
sized to the sum of all inputs in the request.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import struct
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Self

import httpx

from llm_kit.config.app import AppConfig
from llm_kit.config.embed import EmbeddingConfig
from llm_kit.embed.driver import run_embed_stream
from llm_kit.embed.response import EmbeddingResponse, EmbeddingResult, EmbedItem, split_usage
from llm_kit.errors import ProviderError
from llm_kit.http.retry import retry_request
from llm_kit.http.session import build_async_client
from llm_kit.llm.response import TokenUsage
from llm_kit.rate_limit.limiter import RateLimiter

logger = logging.getLogger(__name__)

TextEstimator = Callable[[list[str]], int]


class OpenAICompatibleEmbedder:
    """HTTP client for any OpenAI-shaped embeddings endpoint."""

    def __init__(
        self,
        config: AppConfig,
        *,
        client: httpx.AsyncClient | None = None,
        token_estimator: TextEstimator | None = None,
        owns_client: bool | None = None,
    ) -> None:
        self._config = config
        self._client = client or build_async_client(config.http)
        # Same ownership rule as the LLM client: if we built it, we close it.
        self._owns_client = client is None if owns_client is None else owns_client

        self._limiter = RateLimiter(config.rate_limit)
        self._estimator = token_estimator or self._default_estimator
        self._api_key = self._resolve_api_key(config.embed)

    # ------------------------------------------------------------------ public

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        return cls(AppConfig.from_yaml(path))

    async def embed_one(self, text: str) -> EmbeddingResponse:
        """Embed a single string in its own request."""
        vectors, usage, model = await self.embed_request([text])
        return EmbeddingResponse(vector=vectors[0], index=0, usage=usage, model=model)

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResponse]:
        """Embed a caller-sized list in a *single* request.

        The caller is responsible for keeping ``texts`` within the provider's
        per-request limits (``max_inputs_per_request`` / ``max_tokens_per_request``).
        For arbitrarily large inputs use ``embed_batch``, which packs and streams.
        """
        if not texts:
            return []
        vectors, usage, model = await self.embed_request(texts)
        per = split_usage(usage, [self._estimator([t]) for t in texts])
        return [
            EmbeddingResponse(vector=v, index=i, usage=per[i], model=model)
            for i, v in enumerate(vectors)
        ]

    def embed_batch(
        self,
        items: Any,
        *,
        completed_ids: set[str] | None = None,
    ) -> AsyncIterator[EmbeddingResult]:
        """Stream ``EmbeddingResult``s as the packed requests complete.

        ``items`` is an iterable (or async iterable) of ``EmbedItem``. Inputs are
        micro-batched into request-sized chunks under the configured caps.
        """
        return run_embed_stream(
            embedder=self,
            items=items,
            batch_config=self._config.batch,
            embed_config=self._config.embed,
            estimator=self._estimator,
            completed_ids=completed_ids,
        )

    async def embed_batch_to_jsonl(
        self,
        items: Any,
        output_path: str | Path,
        *,
        resume: bool = True,
    ) -> int:
        """Stream embedding results to ``output_path`` (jsonl). Returns count.

        Like the chat side, the output file IS the checkpoint: with ``resume``,
        IDs already present are skipped.
        """
        from llm_kit.batch.jsonl import load_completed_ids  # reuse: scans `id`
        from llm_kit.embed.jsonl import write_embed_results_to_jsonl

        path = Path(output_path)
        completed = load_completed_ids(path) if resume and path.exists() else set()
        if completed:
            logger.info("resuming: %d items already in %s", len(completed), path)
        return await write_embed_results_to_jsonl(
            path,
            self.embed_batch(items, completed_ids=completed),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ---------------------------------------------------------- request engine

    async def embed_request(
        self, texts: list[str]
    ) -> tuple[list[list[float]], TokenUsage, str | None]:
        """Issue one embeddings HTTP request for ``texts`` (already chunk-sized).

        Returns the per-input vectors (in input order), the request-level token
        usage, and the model string. Used directly by the driver per chunk.
        """
        body = self._build_body(texts)
        estimated = self._estimator(texts)

        async with self._limiter.acquire(estimated_tokens=estimated):
            response = await retry_request(
                lambda: self._client.post(
                    self._endpoint(), headers=self._headers(), json=body
                ),
                retry=self._config.http.llm_retry,
                operation="embedding",
            )

        data = self._parse_json(response)
        vectors = self._extract_vectors(data, len(texts))
        usage = self._extract_usage(data)
        # Only true up when the provider actually reported a count, else the
        # delta would refund the full estimate and let the bucket drift.
        if usage.total_tokens > 0:
            self._limiter.true_up_tokens(estimated=estimated, actual=usage.total_tokens)
        return vectors, usage, data.get("model")

    # ---------------------------------------------------------------- internal

    def _resolve_api_key(self, config: EmbeddingConfig) -> str | None:
        if config.api_key:
            return config.api_key
        if config.api_key_env:
            return os.environ.get(config.api_key_env)
        return None

    def _endpoint(self) -> str:
        base = self._config.embed.base_url.rstrip("/")
        path = self._config.embed.embeddings_path
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        headers.update(self._config.embed.extra_headers)
        return headers

    def _build_body(self, texts: list[str]) -> dict[str, Any]:
        cfg = self._config.embed
        body: dict[str, Any] = {
            "model": cfg.model,
            "input": texts,
            "encoding_format": cfg.encoding_format,
        }
        if cfg.dimensions is not None:
            body["dimensions"] = cfg.dimensions
        return body

    def _parse_json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"non-JSON response from provider: {response.text[:200]!r}",
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(f"unexpected response shape: {type(data).__name__}")
        return data

    def _extract_vectors(self, data: dict[str, Any], expected: int) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list):
            raise ProviderError(f"missing 'data' array in embeddings response: {data!r}")
        if len(items) != expected:
            raise ProviderError(
                f"embeddings count mismatch: requested {expected}, got {len(items)}"
            )
        # Providers may return out of order; index field is authoritative.
        ordered = sorted(items, key=lambda d: d.get("index", 0) if isinstance(d, dict) else 0)
        return [self._decode_embedding(d.get("embedding")) for d in ordered]

    def _decode_embedding(self, raw: Any) -> list[float]:
        if isinstance(raw, list):
            return [float(x) for x in raw]
        if isinstance(raw, str):
            # base64-packed little-endian float32 blob
            buf = base64.b64decode(raw)
            return list(struct.unpack(f"<{len(buf) // 4}f", buf))
        raise ProviderError(f"unrecognized embedding payload type: {type(raw).__name__}")

    def _extract_usage(self, data: dict[str, Any]) -> TokenUsage:
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt) or prompt)
        return TokenUsage(prompt_tokens=prompt, completion_tokens=0, total_tokens=total)

    def _default_estimator(self, texts: list[str]) -> int:
        chars = sum(len(t) for t in texts)
        return max(1, int(chars / self._config.rate_limit.default_chars_per_token))


__all__ = ["EmbedItem", "EmbeddingResponse", "EmbeddingResult", "OpenAICompatibleEmbedder"]
