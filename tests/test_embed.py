"""Embeddings client tests: wire shape, ordering, base64, usage, errors."""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import replace

import httpx
import pytest
from conftest import embedding_response, make_config, mock_client

from llm_kit.config.embed import EmbeddingConfig
from llm_kit.embed.client import OpenAICompatibleEmbedder
from llm_kit.errors import AuthError, ProviderError


def _embedder(handler, cfg=None):
    cfg = cfg or make_config()
    return OpenAICompatibleEmbedder(cfg, client=mock_client(handler))


async def test_embed_one_returns_vector_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "test-embed-model"
        assert body["input"] == ["hello"]
        return httpx.Response(200, json=embedding_response(vectors=[[0.1, 0.2, 0.3]], prompt_tokens=4))

    async with _embedder(handler) as emb:
        resp = await emb.embed_one("hello")

    assert resp.vector == [0.1, 0.2, 0.3]
    assert resp.usage.total_tokens == 4
    assert resp.model == "test-embed-model"


async def test_embed_texts_preserves_input_order_when_provider_reorders():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=embedding_response(
                vectors=[[1.0], [2.0], [3.0]], out_of_order=True, prompt_tokens=9
            ),
        )

    async with _embedder(handler) as emb:
        responses = await emb.embed_texts(["a", "b", "c"])

    # Despite the provider returning data reversed, index-based re-sort restores
    # alignment: input i → vector i.
    assert [r.vector for r in responses] == [[1.0], [2.0], [3.0]]


async def test_embed_texts_usage_splits_to_request_total():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=embedding_response(vectors=[[1.0], [2.0]], prompt_tokens=30)
        )

    async with _embedder(handler) as emb:
        responses = await emb.embed_texts(["short", "muchlongertext"])

    # Per-input usage is approximate but sums exactly to the request total.
    assert sum(r.usage.total_tokens for r in responses) == 30


async def test_base64_encoding_decodes_to_same_floats():
    floats = [0.5, -1.25, 3.0]
    blob = base64.b64encode(struct.pack(f"<{len(floats)}f", *floats)).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["encoding_format"] == "base64"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test-embed-model",
                "data": [{"object": "embedding", "index": 0, "embedding": blob}],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )

    cfg = make_config()
    cfg.embed = replace(cfg.embed, encoding_format="base64")
    async with _embedder(handler, cfg) as emb:
        resp = await emb.embed_one("x")

    assert resp.vector == pytest.approx(floats)


async def test_dimensions_is_forwarded_when_set():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=embedding_response(vectors=[[0.0, 0.0]]))

    cfg = make_config()
    cfg.embed = replace(cfg.embed, dimensions=2)
    async with _embedder(handler, cfg) as emb:
        await emb.embed_one("x")

    assert seen["dimensions"] == 2


async def test_count_mismatch_raises_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        # Asked for two, provider returns one.
        return httpx.Response(200, json=embedding_response(vectors=[[1.0]]))

    async with _embedder(handler) as emb:
        with pytest.raises(ProviderError, match="count mismatch"):
            await emb.embed_texts(["a", "b"])


async def test_auth_error_propagates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    async with _embedder(handler) as emb:
        with pytest.raises(AuthError):
            await emb.embed_one("x")


async def test_injected_client_not_closed_on_aclose():
    client = mock_client(lambda r: httpx.Response(200, json=embedding_response(vectors=[[1.0]])))
    emb = OpenAICompatibleEmbedder(make_config(), client=client)
    await emb.aclose()
    assert not client.is_closed
    await client.aclose()


async def test_api_key_sent_as_bearer():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=embedding_response(vectors=[[1.0]]))

    async with _embedder(handler) as emb:
        await emb.embed_one("x")

    assert seen["auth"] == "Bearer test-key"
