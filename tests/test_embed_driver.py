"""Embed driver tests: packing, fan-out, failure isolation, resume, jsonl."""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from conftest import embedding_response, make_config, mock_client

from llm_kit.config.batch import BatchConfig, BatchFailurePolicy
from llm_kit.config.embed import EmbeddingConfig
from llm_kit.embed.client import OpenAICompatibleEmbedder
from llm_kit.embed.driver import run_embed_stream
from llm_kit.embed.response import EmbedItem
from llm_kit.errors import LLMError, ProviderError
from llm_kit.llm.response import TokenUsage


class FakeEmbedder:
    """Backend stub implementing the driver's ``embed_request`` contract.

    Records the texts of every request so tests can assert how the packer
    chunked them. Returns ``[float(text)]`` per input so fan-out alignment is
    checkable (item text is the stringified number it should map back to).
    """

    def __init__(self, *, fail_texts: set[str] | None = None) -> None:
        self.requests: list[list[str]] = []
        self.fail_texts = fail_texts or set()

    async def embed_request(self, texts):
        self.requests.append(list(texts))
        if self.fail_texts & set(texts):
            raise ProviderError("simulated chunk failure", status_code=400)
        vectors = [[float(t)] for t in texts]
        return vectors, TokenUsage(prompt_tokens=len(texts), total_tokens=len(texts)), "fake-model"


def _items(n: int) -> list[EmbedItem]:
    return [EmbedItem(id=f"item-{i:03d}", text=str(i)) for i in range(n)]


def _stream(embedder, items, *, embed_cfg=None, batch_cfg=None, estimator=None, completed_ids=None):
    return run_embed_stream(
        embedder=embedder,
        items=items,
        batch_config=batch_cfg or BatchConfig(worker_concurrency=2, input_buffer_size=8, progress_log_every=0),
        embed_config=embed_cfg or EmbeddingConfig(),
        estimator=estimator or (lambda texts: sum(len(t) for t in texts)),
        completed_ids=completed_ids,
    )


async def _collect(stream):
    return [r async for r in stream]


async def test_packs_by_input_count_cap():
    emb = FakeEmbedder()
    cfg = replace(EmbeddingConfig(), max_inputs_per_request=3)
    results = await _collect(_stream(emb, _items(7), embed_cfg=cfg))

    assert len(results) == 7
    # 7 items, cap 3 → chunks of 3, 3, 1.
    assert sorted(len(c) for c in emb.requests) == [1, 3, 3]


async def test_packs_by_token_cap():
    emb = FakeEmbedder()
    # Each single-digit text estimates to 1 token; cap 2 → at most 2 per chunk.
    cfg = replace(EmbeddingConfig(), max_inputs_per_request=1000, max_tokens_per_request=2)
    results = await _collect(_stream(emb, _items(5), embed_cfg=cfg))

    assert len(results) == 5
    assert all(len(c) <= 2 for c in emb.requests)
    assert sorted(len(c) for c in emb.requests) == [1, 2, 2]


async def test_oversized_single_input_goes_out_alone():
    emb = FakeEmbedder()
    cfg = replace(EmbeddingConfig(), max_tokens_per_request=1)
    items = [EmbedItem(id="big", text="1234567890")]  # 10 tokens > cap of 1
    results = await _collect(_stream(emb, items, embed_cfg=cfg))

    assert len(results) == 1
    assert results[0].ok
    assert emb.requests == [["1234567890"]]


async def test_fan_out_aligns_vector_to_id():
    emb = FakeEmbedder()
    cfg = replace(EmbeddingConfig(), max_inputs_per_request=4)
    results = await _collect(_stream(emb, _items(10), embed_cfg=cfg))

    by_id = {r.id: r for r in results}
    for i in range(10):
        r = by_id[f"item-{i:03d}"]
        assert r.ok
        # item text is str(i); fake returns [float(text)] → vector must be [i].
        assert r.response.vector == [float(i)]


async def test_per_input_usage_sums_to_chunk_total():
    emb = FakeEmbedder()
    cfg = replace(EmbeddingConfig(), max_inputs_per_request=5)
    results = await _collect(_stream(emb, _items(5), embed_cfg=cfg))
    # One chunk of 5 → request total 5; per-input usage must sum back to it.
    assert sum(r.response.usage.total_tokens for r in results) == 5


async def test_chunk_failure_records_error_for_every_input_in_chunk():
    emb = FakeEmbedder(fail_texts={"2"})  # the chunk containing "2" fails wholesale
    cfg = replace(EmbeddingConfig(), max_inputs_per_request=3)
    results = await _collect(_stream(emb, _items(7), embed_cfg=cfg))

    by_id = {r.id: r for r in results}
    # items 0,1,2 share the first chunk → all three error together.
    assert not by_id["item-000"].ok
    assert not by_id["item-001"].ok
    assert not by_id["item-002"].ok
    assert isinstance(by_id["item-002"].error, ProviderError)
    # later chunks unaffected
    assert by_id["item-006"].ok


async def test_raise_policy_aborts_stream():
    emb = FakeEmbedder(fail_texts={"0"})
    batch_cfg = BatchConfig(
        worker_concurrency=1, input_buffer_size=4, progress_log_every=0,
        on_failure=BatchFailurePolicy.RAISE,
    )
    with pytest.raises(LLMError):
        await _collect(_stream(emb, _items(3), batch_cfg=batch_cfg))


async def test_completed_ids_are_skipped():
    emb = FakeEmbedder()
    completed = {"item-000", "item-001"}
    results = await _collect(_stream(emb, _items(5), completed_ids=completed))

    ids = {r.id for r in results}
    assert ids == {"item-002", "item-003", "item-004"}
    # skipped items never reach a request
    assert all("0" not in c and "1" not in c for c in emb.requests)


# -------------------------------------------------------------- integration


async def test_embed_batch_to_jsonl_writes_and_resumes(tmp_path):
    out = tmp_path / "vectors.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = len(body["input"])
        return httpx.Response(200, json=embedding_response(vectors=[[float(i)] for i in range(n)]))

    cfg = make_config()
    cfg.embed = replace(cfg.embed, max_inputs_per_request=4)
    async with OpenAICompatibleEmbedder(cfg, client=mock_client(handler)) as emb:
        written = await emb.embed_batch_to_jsonl(_items(6), out)
    assert written == 6

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["id"] for r in records} == {f"item-{i:03d}" for i in range(6)}
    assert all("vector" in r["response"] for r in records)

    # Resume: nothing new to do, file already complete.
    async with OpenAICompatibleEmbedder(cfg, client=mock_client(handler)) as emb:
        written_again = await emb.embed_batch_to_jsonl(_items(6), out, resume=True)
    assert written_again == 0
