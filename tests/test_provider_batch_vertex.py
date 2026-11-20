"""Vertex AI batch backend: duck-typed fake genai + in-memory GCS (no network, no SDK)."""

from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from llm_kit import (
    BatchItem,
    BatchJob,
    BatchJobFailedError,
    BatchRequest,
    BatchState,
    LLMClient,
    Message,
)
from llm_kit.config.llm import MessageFormat
from llm_kit.messages.types import ImagePart
from llm_kit.provider_batch import registry
from llm_kit.provider_batch.types import BatchJobStatus
from llm_kit.provider_batch.vertex import (
    LABEL_KEY,
    VertexBatchBackend,
    canonical_hash,
    decode_label,
    encode_label,
)

from conftest import make_config

PREFIX = "gs://bkt/work"


# ------------------------------------------------------------------- fakes


class FakeBlob:
    def __init__(self, store: dict[str, bytes], name: str) -> None:
        self._store, self.name = store, name

    def open(self, mode: str = "rb", **kw: Any) -> Any:
        if "w" in mode:
            blob = self

            class W(io.BytesIO):
                def close(self_inner) -> None:  # noqa: N805
                    blob._store[blob.name] = self_inner.getvalue()
                    super().close()

            return W()
        return io.BytesIO(self._store[self.name])

    def upload_from_string(self, data: bytes, content_type: str | None = None) -> None:
        self._store[self.name] = data

    def delete(self) -> None:
        del self._store[self.name]


class FakeBucket:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._store, name)


class FakeStorage:
    """One bucket; keys are object names (``bkt`` implied)."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def bucket(self, name: str) -> FakeBucket:
        assert name == "bkt"
        return FakeBucket(self.objects)

    def list_blobs(self, bucket: str, prefix: str = "") -> list[FakeBlob]:
        return [FakeBlob(self.objects, n) for n in sorted(self.objects) if n.startswith(prefix)]

    def lines(self, suffix: str) -> list[dict[str, Any]]:
        (name,) = [n for n in self.objects if n.endswith(suffix)]
        return [json.loads(x) for x in self.objects[name].decode().splitlines()]


def text_response(text: str) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
    }


class FakeGenai:
    """Mimics ``genai.Client(vertexai=True)``: ``client.aio.batches.{create,get,cancel}``.

    On the final poll it "runs" the job: reads input.jsonl from storage and writes
    predictions.jsonl under the job's dest.
    """

    def __init__(
        self,
        storage: FakeStorage,
        *,
        states: list[str] | None = None,
        shuffle: bool = True,
        echo_labels: bool = True,
        row_errors: set[str] | None = None,
        drop_ids: set[str] | None = None,
        final_state: str = "JOB_STATE_SUCCEEDED",
        write_output: bool = True,
        error: Any = None,
    ) -> None:
        self.storage = storage
        self.states = list(states or [])
        self.shuffle, self.echo_labels = shuffle, echo_labels
        self.row_errors, self.drop_ids = row_errors or set(), drop_ids or set()
        self.final_state, self.write_output, self.error = final_state, write_output, error
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.aio = SimpleNamespace(
            batches=SimpleNamespace(create=self._create, get=self._get, cancel=self._cancel)
        )

    async def _create(self, *, model: str, src: str, config: dict[str, Any]) -> Any:
        self.created.append({"model": model, "src": src, "config": config})
        name = f"projects/p/locations/global/batchPredictionJobs/{len(self.created)}"
        self.jobs[name] = {"src": src, "dest": config["dest"], "polls": 0, "written": False}
        return SimpleNamespace(name=name)

    async def _get(self, *, name: str) -> Any:
        j = self.jobs[name]
        j["polls"] += 1
        state = self.states.pop(0) if self.states else self.final_state
        if state == "JOB_STATE_SUCCEEDED" and self.write_output and not j["written"]:
            self._run(j)
            j["written"] = True
        return SimpleNamespace(
            name=name,
            state=state,
            error=self.error,
            src=SimpleNamespace(gcs_uri=[j["src"]]),
            dest=SimpleNamespace(gcs_uri=j["dest"]),
            completion_stats=SimpleNamespace(successful_count=2, failed_count=0, incomplete_count=0),
        )

    async def _cancel(self, *, name: str) -> None:
        self.cancelled.append(name)

    def _run(self, job: dict[str, Any]) -> None:
        in_name = job["src"].removeprefix("gs://bkt/")
        rows = [json.loads(x) for x in self.storage.objects[in_name].decode().splitlines()]
        keys = {}
        run = in_name.rsplit("/", 1)[0] + "/"
        for line in self.storage.objects[run + "keys.jsonl"].decode().splitlines():
            k = json.loads(line)
            keys[k["hash"]] = k["id"]
        out = []
        for row in rows:
            req = json.loads(json.dumps(row["request"]))
            cid = keys[canonical_hash(req)]
            if cid in self.drop_ids:
                continue
            if not self.echo_labels:
                req.pop("labels", None)
            if cid in self.row_errors:
                out.append({"request": req, "status": "INVALID_ARGUMENT: bad", "response": {}})
            else:
                out.append({"request": req, "status": "", "response": text_response(f"ans:{cid}")})
        if self.shuffle:
            out.reverse()
        dest = job["dest"].removeprefix("gs://bkt/")
        self.storage.objects[dest + "prediction-model-2026/predictions.jsonl"] = (
            "\n".join(json.dumps(r) for r in out) + "\n"
        ).encode()


def vconfig(**vertex: Any):
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.GEMINI
    cfg.llm.model = "gemini-test"
    cfg.provider_batch.backend = "vertex"
    cfg.provider_batch.vertex.gcs_prefix = PREFIX
    for k, v in vertex.items():
        setattr(cfg.provider_batch.vertex, k, v)
    return cfg


def setup(*, states=None, vertex=None, **fake_kw):
    storage = FakeStorage()
    genai = FakeGenai(storage, states=states, **fake_kw)
    cfg = vconfig(**(vertex or {}))
    backend = VertexBatchBackend(cfg, genai_client=genai, storage_client=storage)
    return cfg, storage, genai, backend


def llm_for(cfg, backend) -> LLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected network request")

    return LLMClient(
        cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), batch_backend=backend
    )


def reqs(*ids: str) -> list[BatchRequest]:
    return [
        BatchRequest(i, {"contents": [{"role": "user", "parts": [{"text": f"q-{i}"}]}]}) for i in ids
    ]


# --------------------------------------------------------------------- tests


async def test_submit_writes_expected_jsonl_and_job_body() -> None:
    cfg, storage, genai, backend = setup()
    res = await backend.submit(reqs("a", "b"))
    (created,) = genai.created
    run = res.extra["run_prefix"]
    assert run.startswith("gs://bkt/work/llm-kit-") and run.endswith("/")
    assert created["model"] == "gemini-test"
    assert created["src"] == run + "input.jsonl"
    assert created["config"]["dest"] == run + "output/"
    assert created["config"]["display_name"].startswith("llm-kit-")
    assert res.batch_id.endswith("/batchPredictionJobs/1")
    assert res.extra["input_uri"] == created["src"] and res.extra["output_uri"] == created["config"]["dest"]
    rows = storage.lines("input.jsonl")
    assert rows == [
        {
            "request": {
                "contents": [{"role": "user", "parts": [{"text": f"q-{i}"}]}],
                "labels": {LABEL_KEY: encode_label(i)},
            }
        }
        for i in ("a", "b")
    ]
    keys = storage.lines("keys.jsonl")
    assert [k["id"] for k in keys] == ["a", "b"]
    assert keys[0]["hash"] == canonical_hash(rows[0]["request"])  # label ignored in hash
    json.dumps(res.extra)  # JSON-serialisable


def test_label_roundtrip_and_limits() -> None:
    for cid in ("a", "item-1", "ünï/cödé"):
        enc = encode_label(cid)
        assert enc and len(enc) <= 63 and enc == enc.lower()
        assert decode_label(enc) == cid
    assert encode_label("z" * 100) is None
    assert decode_label("garbage!") is None and decode_label(None) is None


async def test_submit_requires_gcs_prefix() -> None:
    cfg, storage, genai, backend = setup(vertex={"gcs_prefix": None})
    with pytest.raises(ValueError, match="gcs_prefix"):
        await backend.submit(reqs("a"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("JOB_STATE_UNSPECIFIED", BatchState.PENDING),
        ("JOB_STATE_QUEUED", BatchState.PENDING),
        ("JOB_STATE_PENDING", BatchState.PENDING),
        ("JOB_STATE_RUNNING", BatchState.IN_PROGRESS),
        ("JOB_STATE_CANCELLING", BatchState.IN_PROGRESS),
        ("JOB_STATE_UPDATING", BatchState.IN_PROGRESS),
        ("JOB_STATE_SUCCEEDED", BatchState.COMPLETED),
        ("JOB_STATE_PARTIALLY_SUCCEEDED", BatchState.COMPLETED),
        ("JOB_STATE_FAILED", BatchState.FAILED),
        ("JOB_STATE_CANCELLED", BatchState.CANCELLED),
        ("JOB_STATE_EXPIRED", BatchState.EXPIRED),
        ("JOB_STATE_PAUSED", BatchState.PAUSED),
        ("JOB_STATE_SOMETHING_NEW", BatchState.IN_PROGRESS),
    ],
)
async def test_poll_maps_every_state(raw: str, expected: BatchState) -> None:
    cfg, storage, genai, backend = setup(states=[raw], write_output=False)
    res = await backend.submit(reqs("a"))
    status = await backend.poll(res.batch_id)
    assert status.state is expected and status.raw_state == raw
    assert status.output_ref == res.extra["output_uri"]
    assert status.counts == {"succeeded": 2, "failed": 0, "incomplete": 0}


async def test_poll_accepts_enum_like_state_and_error_detail() -> None:
    err = SimpleNamespace(message="quota exceeded", details=["d1"], code=8)
    cfg, storage, genai, backend = setup(
        states=[SimpleNamespace(value="JOB_STATE_FAILED")], error=err, write_output=False
    )
    res = await backend.submit(reqs("a"))
    st = await backend.poll(res.batch_id)
    assert st.state is BatchState.FAILED and st.error_detail == "quota exceeded; d1"


async def _fetch(backend: VertexBatchBackend, batch_id: str) -> dict[str, Any]:
    status = await backend.poll(batch_id)
    assert status.state is BatchState.COMPLETED
    return {r.custom_id: r async for r in backend.fetch_results(status)}


@pytest.mark.parametrize("echo_labels", [True, False])
async def test_results_matched_when_shuffled(echo_labels: bool) -> None:
    # echo_labels=False exercises the canonical-hash fallback.
    cfg, storage, genai, backend = setup(shuffle=True, echo_labels=echo_labels)
    ids = ["a", "b", "c", "d"]
    res = await backend.submit(reqs(*ids))
    got = await _fetch(backend, res.batch_id)
    assert set(got) == set(ids)
    for i in ids:
        assert got[i].error is None and got[i].inner_body == text_response(f"ans:{i}")


async def test_hash_fallback_handles_duplicate_requests() -> None:
    cfg, storage, genai, backend = setup(echo_labels=False)
    body = {"contents": [{"role": "user", "parts": [{"text": "same"}]}]}
    res = await backend.submit([BatchRequest("x", body), BatchRequest("y", dict(body))])
    got = await _fetch(backend, res.batch_id)
    assert set(got) == {"x", "y"}


async def test_row_error_missing_row_and_unmatched(caplog: pytest.LogCaptureFixture) -> None:
    cfg, storage, genai, backend = setup(row_errors={"b"}, drop_ids={"c"})
    res = await backend.submit(reqs("a", "b", "c"))
    # inject a row that matches nothing
    status = await backend.poll(res.batch_id)
    name = next(n for n in storage.objects if n.endswith("predictions.jsonl"))
    storage.objects[name] += json.dumps({"request": {"contents": []}, "response": {}}).encode() + b"\n"
    with caplog.at_level(logging.WARNING):
        got = {r.custom_id: r async for r in backend.fetch_results(status)}
    assert got["a"].error is None
    assert "bad" in str(got["b"].error)
    assert "no output row" in str(got["c"].error)
    assert any("could not be matched" in r.message for r in caplog.records)


async def test_failed_job_is_terminal_via_run_batch_api(tmp_path) -> None:
    cfg, storage, genai, backend = setup(
        final_state="JOB_STATE_FAILED",
        write_output=False,
        error=SimpleNamespace(message="boom", details=None),
    )
    llm = llm_for(cfg, backend)
    out = tmp_path / "out.jsonl"
    with pytest.raises(BatchJobFailedError) as ei:
        await llm.run_batch_api(
            [BatchItem(id="a", messages=[Message.user("hi")])], out, poll_interval=0
        )
    assert ei.value.state is BatchState.FAILED and ei.value.detail == "boom"
    assert not out.exists()


async def test_paused_is_waited_through(tmp_path) -> None:
    cfg, storage, genai, backend = setup(
        states=["JOB_STATE_PENDING", "JOB_STATE_PAUSED", "JOB_STATE_PAUSED", "JOB_STATE_RUNNING"]
    )
    llm = llm_for(cfg, backend)
    n = await llm.run_batch_api(
        [BatchItem(id="a", messages=[Message.user("hi")])], tmp_path / "o.jsonl", poll_interval=0
    )
    assert n == 1
    (job,) = genai.jobs.values()
    assert job["polls"] >= 5


async def test_upload_images_rewrites_inline_data_and_dedupes() -> None:
    cfg, storage, genai, backend = setup(vertex={"upload_images": True})
    png = b"\x89PNG-fake-bytes"
    b64 = base64.b64encode(png).decode()
    other = base64.b64encode(b"other").decode()

    def body(*datas: str) -> dict[str, Any]:
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "see"}]
                    + [{"inlineData": {"mimeType": "image/png", "data": d}} for d in datas],
                }
            ]
        }

    res = await backend.submit([BatchRequest("a", body(b64, other)), BatchRequest("b", body(b64))])
    rows = storage.lines("input.jsonl")
    for row in rows:
        assert "inlineData" not in json.dumps(row)
    uri = rows[0]["request"]["contents"][0]["parts"][1]["fileData"]["fileUri"]
    assert rows[0]["request"]["contents"][0]["parts"][1]["fileData"]["mimeType"] == "image/png"
    assert uri == rows[1]["request"]["contents"][0]["parts"][1]["fileData"]["fileUri"]  # deduped
    assert uri.startswith(res.extra["run_prefix"] + "images/") and uri.endswith(".png")
    assert storage.objects[uri.removeprefix("gs://bkt/")] == png
    imgs = [n for n in storage.objects if "/images/" in n]
    assert len(imgs) == 2 and len(res.extra["image_uris"]) == 2
    # hash is of the body as written, so matching still works
    got = await _fetch(backend, res.batch_id)
    assert set(got) == {"a", "b"}


async def test_inline_images_untouched_by_default_via_formatter() -> None:
    cfg, storage, genai, backend = setup()
    llm = llm_for(cfg, backend)
    item = BatchItem(
        id="a", messages=[Message.user("look", ImagePart.from_bytes(b"\x01\x02", "image/png"))]
    )
    await llm.submit_batch([item])
    part = storage.lines("input.jsonl")[0]["request"]["contents"][0]["parts"][1]
    assert part == {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(b"\x01\x02").decode()}}


async def test_upload_images_through_gemini_formatter_bodies() -> None:
    cfg, storage, genai, backend = setup(vertex={"upload_images": True})
    llm = llm_for(cfg, backend)
    item = BatchItem(
        id="a", messages=[Message.user("look", ImagePart.from_bytes(b"\x01\x02", "image/jpeg"))]
    )
    job = await llm.submit_batch([item])
    part = storage.lines("input.jsonl")[0]["request"]["contents"][0]["parts"][1]
    assert part["fileData"]["fileUri"].endswith(".jpg")
    assert job.extra["image_uris"] == [part["fileData"]["fileUri"]]


async def test_cleanup_deletes_expected_blobs() -> None:
    cfg, storage, genai, backend = setup(vertex={"upload_images": True})
    body = {"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": "AAAA"}}]}]}
    res = await backend.submit([BatchRequest("a", body)])
    await _fetch(backend, res.batch_id)
    storage.objects["work/unrelated.txt"] = b"keep"
    job = BatchJob(provider="gemini", batch_id=res.batch_id, backend="vertex", extra=res.extra)

    assert await backend.cleanup(job) == []  # config flags default False
    deleted = await backend.cleanup(job, output=True)
    assert len(deleted) == 1 and deleted[0].endswith("predictions.jsonl")
    assert any(n.endswith("input.jsonl") for n in storage.objects)
    deleted = await backend.cleanup(job, input=True)
    assert sorted(d.rsplit("/", 1)[-1] for d in deleted)[:2] == sorted(
        d.rsplit("/", 1)[-1] for d in deleted
    )[:2]
    assert list(storage.objects) == ["work/unrelated.txt"]


async def test_cleanup_uses_config_flags_and_batch_id_only() -> None:
    cfg, storage, genai, backend = setup(vertex={"cleanup_input": True, "cleanup_output": True})
    res = await backend.submit(reqs("a"))
    await _fetch(backend, res.batch_id)
    deleted = await backend.cleanup(res.batch_id)  # str: run dir derived from the job resource
    assert len(deleted) == 3  # input, keys, predictions
    assert storage.objects == {}


async def test_cancel_calls_sdk() -> None:
    cfg, storage, genai, backend = setup()
    await backend.cancel("projects/p/locations/global/batchPredictionJobs/9")
    assert genai.cancelled == ["projects/p/locations/global/batchPredictionJobs/9"]


async def test_reconnect_fetch_from_job_name_alone() -> None:
    cfg, storage, genai, backend = setup()
    res = await backend.submit(reqs("a", "b"))
    fresh = VertexBatchBackend(vconfig(), genai_client=genai, storage_client=storage)
    got = await _fetch(fresh, res.batch_id)
    assert set(got) == {"a", "b"} and got["a"].error is None


async def test_end_to_end_via_registry_with_injected_sdk_clients(tmp_path) -> None:
    storage = FakeStorage()
    genai = FakeGenai(storage, states=["JOB_STATE_RUNNING"])
    cfg = vconfig()
    assert registry.backend_name(cfg) == "vertex"
    original = registry._REGISTRY["vertex"]
    registry.register_batch_backend(
        "vertex",
        lambda config, client, key: VertexBatchBackend(
            config, genai_client=genai, storage_client=storage
        ),
        replace=True,
    )
    try:
        llm = LLMClient(
            cfg,
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(AssertionError(r)))
            ),
        )
    finally:
        registry.register_batch_backend("vertex", original, replace=True)
    assert isinstance(llm._batch_backend, VertexBatchBackend)
    items = [
        BatchItem(id=f"i{k}", messages=[Message.user(f"q{k}")], metadata={"k": k}) for k in range(3)
    ]
    out = tmp_path / "out.jsonl"
    assert await llm.run_batch_api(items, out, poll_interval=0) == 3
    rows = {json.loads(x)["id"]: json.loads(x) for x in out.read_text().splitlines()}
    assert set(rows) == {"i0", "i1", "i2"}
    assert all(not r.get("error") for r in rows.values())
    assert rows["i1"]["response"]["text"] == "ans:i1"
    assert [r["metadata"]["k"] for _, r in sorted(rows.items())] == [0, 1, 2]
    created = genai.created[0]
    assert created["model"] == "gemini-test"

    deleted = await llm.cleanup_batch(
        BatchJob("gemini", next(iter(genai.jobs)), backend="vertex"), input=True, output=True
    )
    assert len(deleted) == 3 and storage.objects == {}


async def test_cleanup_batch_unsupported_backend() -> None:
    from llm_kit.testing import FakeBatchBackend

    cfg = make_config()
    llm = llm_for(cfg, FakeBatchBackend())
    with pytest.raises(NotImplementedError):
        await llm.cleanup_batch(BatchJob("openai", "b"))


def test_factory_requires_gemini_format() -> None:
    cfg = vconfig()
    cfg.llm.message_format = MessageFormat.OPENAI
    with pytest.raises(ValueError, match="message_format: gemini"):
        registry.build_backend(cfg, None, None)  # type: ignore[arg-type]


def test_config_from_yaml_dict() -> None:
    from llm_kit import AppConfig

    cfg = AppConfig.from_dict(
        {
            "provider_batch": {
                "backend": "vertex",
                "vertex": {"project": "p", "gcs_prefix": "gs://b/x", "upload_images": True},
            }
        }
    )
    v = cfg.provider_batch.vertex
    assert (v.project, v.location, v.upload_images, v.cleanup_input) == ("p", "global", True, False)


# ---------------------------------------------------- base install without extra


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True
    )


_BLOCK = """
import sys
for m in ("google", "google.genai", "google.cloud", "google.cloud.storage", "google.auth"):
    sys.modules[m] = None
"""


def test_base_package_imports_without_google() -> None:
    r = _run(
        _BLOCK
        + """
import llm_kit, llm_kit.provider_batch, llm_kit.provider_batch.registry
import llm_kit.provider_batch.vertex  # module itself is SDK-free at import
assert not [m for m, v in sys.modules.items() if m.startswith("google") and v is not None]
print("ok")
"""
    )
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_selecting_vertex_without_extra_raises_hint() -> None:
    r = _run(
        _BLOCK
        + """
from llm_kit import AppConfig, LLMClient
cfg = AppConfig.from_dict({"llm": {"message_format": "gemini", "model": "gemini-x", "api_key": "k"},
                           "provider_batch": {"backend": "vertex"}})
try:
    LLMClient(cfg)
except ImportError as e:
    print("HINT", e)
else:
    raise SystemExit("no error")
"""
    )
    assert r.returncode == 0, r.stderr
    assert 'pip install "llm_kit[vertex]"' in r.stdout and "uv sync --extra vertex" in r.stdout
