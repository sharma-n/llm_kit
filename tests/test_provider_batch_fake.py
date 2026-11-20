"""FakeBatchBackend, backend injection/registry, and the raw submit/fetch path."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from llm_kit import (
    BatchItem,
    BatchJob,
    BatchRequest,
    BatchState,
    LLMClient,
    Message,
    RawResult,
    register_batch_backend,
)
from llm_kit.config.llm import LLMConfig, MessageFormat
from llm_kit.errors import LLMError
from llm_kit.llm.parsing import parse_response
from llm_kit.messages.formatter import AnthropicFormatter, GeminiFormatter, OpenAIFormatter
from llm_kit.provider_batch import registry
from llm_kit.provider_batch.backend import BatchBackend
from llm_kit.provider_batch.types import SubmitResult
from llm_kit.testing import FakeBatchBackend

from conftest import make_config


def _no_network() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected network request: {request.method} {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _client(fake: BatchBackend, cfg=None) -> LLMClient:
    return LLMClient(cfg or make_config(), client=_no_network(), batch_backend=fake)


def _items(n: int) -> list[BatchItem]:
    return [
        BatchItem(id=f"i{k}", messages=[Message.user(f"q{k}")], metadata={"k": k})
        for k in range(n)
    ]


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------- AC 1-4


async def test_run_batch_api_end_to_end_no_network(tmp_path: Path) -> None:
    fake = FakeBatchBackend(polls_until_done=3)
    llm = _client(fake)
    out = tmp_path / "out.jsonl"
    n = await llm.run_batch_api(_items(4), out, poll_interval=0.0)
    assert n == 4
    rows = _read(out)
    assert sorted(r["id"] for r in rows) == ["i0", "i1", "i2", "i3"]
    assert all(r.get("error") is None for r in rows)
    (batch_id,) = fake.submitted
    assert fake.poll_counts[batch_id] >= 3
    assert len(fake.submitted[batch_id]) == 4


async def test_raw_body_reaches_backend_byte_for_byte() -> None:
    fake = FakeBatchBackend()
    llm = _client(fake)
    body = {"model": "x", "weird": [1, {"b": None, "a": "é"}], "n": 1.5, "nested": {"z": 1, "a": 2}}
    req = BatchRequest(custom_id="r1", body=body)
    job = await llm.submit_requests([req], metadata={"r1": {"tag": "t"}})
    (recorded,) = fake.submitted[job.batch_id]
    assert recorded.body is body
    assert recorded.body == body
    assert json.dumps(recorded.body) == json.dumps(body)
    assert job.metadata_for("r1") == {"tag": "t"}


async def test_one_failed_item_yields_exactly_one_error() -> None:
    fake = FakeBatchBackend(fail_items={"i2": "boom"})
    llm = _client(fake)
    job = await llm.submit_batch(_items(5))
    results = [r async for r in llm.fetch_batch_results(job)]
    assert len(results) == 5
    errs = [r for r in results if r.error is not None]
    assert [r.id for r in errs] == ["i2"]
    assert "boom" in str(errs[0].error)
    assert all(r.response is not None for r in results if r.id != "i2")


async def test_saved_job_resumed_by_new_client(tmp_path: Path) -> None:
    fake = FakeBatchBackend(polls_until_done=2)
    first = _client(fake)
    job = await first.submit_batch(_items(3))
    path = tmp_path / "job.json"
    job.save(path)

    second = _client(fake)
    loaded = BatchJob.load(path)
    out = tmp_path / "out.jsonl"
    n = await second.run_batch_api([], out, resume_job=loaded, poll_interval=0.0)
    assert n == 3
    results = [r async for r in second.fetch_batch_results(loaded)]
    assert sorted(r.id for r in results) == ["i0", "i1", "i2"]
    assert results[0].metadata["k"] in (0, 1, 2)


# ---------------------------------------------------------------- registry


def _restore_registry(snapshot: dict) -> None:
    registry._REGISTRY.clear()
    registry._REGISTRY.update(snapshot)


def test_builtin_backends_registered() -> None:
    assert {"openai", "anthropic", "gemini"} <= set(registry.registered_batch_backends())


def test_custom_backend_selected_via_config() -> None:
    snap = dict(registry._REGISTRY)
    try:
        fake = FakeBatchBackend()
        seen = {}

        def factory(config, client, api_key):
            seen["args"] = (config, client, api_key)
            return fake

        register_batch_backend("mine", factory)
        cfg = make_config()
        cfg.provider_batch.backend = "mine"
        llm = LLMClient(cfg, client=_no_network())
        assert llm._batch_backend is fake
        assert seen["args"][0] is cfg and seen["args"][2] == "test-key"
    finally:
        _restore_registry(snap)


def test_duplicate_registration_needs_replace() -> None:
    snap = dict(registry._REGISTRY)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_batch_backend("openai", lambda c, h, k: FakeBatchBackend())
        sentinel = FakeBatchBackend()
        register_batch_backend("openai", lambda c, h, k: sentinel, replace=True)
        assert LLMClient(make_config(), client=_no_network())._batch_backend is sentinel
    finally:
        _restore_registry(snap)


def test_unknown_backend_lists_registered() -> None:
    cfg = make_config()
    cfg.provider_batch.backend = "nope"
    with pytest.raises(ValueError, match="nope") as ei:
        LLMClient(cfg, client=_no_network())
    assert "openai" in str(ei.value) and "gemini" in str(ei.value)


def test_default_backend_follows_message_format() -> None:
    from llm_kit.provider_batch.anthropic import AnthropicBatchBackend

    cfg = make_config()
    cfg.llm.message_format = MessageFormat.ANTHROPIC
    assert isinstance(LLMClient(cfg, client=_no_network())._batch_backend, AnthropicBatchBackend)


def test_injected_backend_beats_config_name() -> None:
    cfg = make_config()
    cfg.provider_batch.backend = "nope"  # would raise if the registry were consulted
    fake = FakeBatchBackend()
    assert LLMClient(cfg, client=_no_network(), batch_backend=fake)._batch_backend is fake


# ---------------------------------------------------------------- BatchJob


def test_old_job_json_loads(tmp_path: Path) -> None:
    p = tmp_path / "old.json"
    p.write_text(
        json.dumps({"provider": "gemini", "batch_id": "b1", "input_file_id": None, "manifest": {}})
    )
    job = BatchJob.load(p)
    assert job.backend == "gemini"
    assert job.extra == {}


async def test_extra_and_backend_round_trip(tmp_path: Path) -> None:
    fake = FakeBatchBackend()
    job = await _client(fake).submit_batch(_items(2))
    assert job.backend == "fake"
    assert job.extra == {"fake": True, "n_requests": 2}
    p = tmp_path / "j.json"
    job.save(p)
    loaded = BatchJob.load(p)
    assert loaded.backend == "fake" and loaded.extra == job.extra


async def test_bare_string_submit_still_works() -> None:
    class Plain(FakeBatchBackend):
        async def submit(self, requests):
            r = await super().submit(requests)
            assert isinstance(r, SubmitResult)
            return r.batch_id

    job = await _client(Plain()).submit_batch(_items(1))
    assert job.extra == {}


async def test_backend_mismatch_warns_not_fails(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeBatchBackend()
    llm = _client(fake)
    job = await llm.submit_batch(_items(1))
    job.backend = "something-else"
    with caplog.at_level(logging.WARNING, logger="llm_kit.llm.client"):
        results = [r async for r in llm.fetch_batch_results(job)]
    assert len(results) == 1
    assert "something-else" in caplog.text


# ---------------------------------------------------------------- fake itself


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakeBatchBackend(), BatchBackend)


@pytest.mark.parametrize(
    ("fmt", "formatter"),
    [
        (MessageFormat.OPENAI, OpenAIFormatter(structured_output_style=LLMConfig().structured_output_style)),
        (MessageFormat.ANTHROPIC, AnthropicFormatter()),
        (MessageFormat.GEMINI, GeminiFormatter()),
    ],
)
def test_text_responder_parses_for_each_format(fmt, formatter) -> None:
    respond = FakeBatchBackend.text_responder(fmt, lambda r: f"hi {r.custom_id}")
    resp = parse_response(respond(BatchRequest("c9", {})), None, formatter)
    assert resp.text == "hi c9"


@pytest.mark.parametrize("fmt", list(MessageFormat))
async def test_fake_end_to_end_per_format(fmt: MessageFormat) -> None:
    cfg = make_config()
    cfg.llm.message_format = fmt
    fake = FakeBatchBackend(
        responder=FakeBatchBackend.text_responder(fmt, lambda r: f"A-{r.custom_id}")
    )
    llm = _client(fake, cfg)
    job = await llm.submit_batch(_items(2))
    got = {r.id: r.response.text async for r in llm.fetch_batch_results(job)}
    assert got == {"i0": "A-i0", "i1": "A-i1"}


async def test_shuffle_preserves_matching() -> None:
    fake = FakeBatchBackend(shuffle=True)
    llm = _client(fake)
    job = await llm.submit_batch(_items(5))
    results = [r async for r in llm.fetch_batch_results(job)]
    assert [r.id for r in results] == ["i4", "i3", "i2", "i1", "i0"]
    for r in results:
        assert r.response.text == f"fake:{r.id}"
        assert r.metadata == {"k": int(r.id[1:])}


async def test_mapping_responder_and_raw_fetch() -> None:
    fake = FakeBatchBackend(responder={"a": {"x": 1}})
    llm = _client(fake)
    job = await llm.submit_requests([BatchRequest("a", {})])
    raw = [r async for r in llm.fetch_raw_results(job)]
    assert raw == [RawResult("a", inner_body={"x": 1})]


async def test_fail_job_and_cancel() -> None:
    fake = FakeBatchBackend(fail_job=BatchState.EXPIRED, raw_state="timed_out")
    llm = _client(fake)
    job = await llm.submit_batch(_items(1))
    st = await llm.poll_batch(job)
    assert st.state is BatchState.EXPIRED and st.raw_state == "timed_out"

    fake2 = FakeBatchBackend(polls_until_done=5)
    job2 = await _client(fake2).submit_batch(_items(1))
    await fake2.cancel(job2.batch_id)
    assert job2.batch_id in fake2.cancelled
    assert (await fake2.poll(job2.batch_id)).state is BatchState.CANCELLED


async def test_unknown_batch_id_raises() -> None:
    with pytest.raises(LLMError):
        await FakeBatchBackend().poll("nope")


# ---------------------------------------------------------------- submit_requests guards


async def test_submit_requests_rejects_duplicates_and_oversize() -> None:
    cfg = make_config()
    cfg.provider_batch.max_requests_per_batch = 2
    llm = _client(FakeBatchBackend(), cfg)
    with pytest.raises(ValueError, match="duplicate"):
        await llm.submit_requests([BatchRequest("a", {}), BatchRequest("a", {})])
    with pytest.raises(ValueError, match="max_requests_per_batch"):
        await llm.submit_requests([BatchRequest(c, {}) for c in "abc"])
