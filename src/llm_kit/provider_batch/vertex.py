"""Vertex AI batch-prediction backend for Gemini (GCS transport).

Request/response *bodies* are the Gemini formatter's, unchanged (CLAUDE.md
invariant #10). This module is only transport + envelope:

  - submit:  rows ``{"request": <Gemini body>}`` are written as JSONL to
             ``{gcs_prefix}/llm-kit-<uuid>/input.jsonl``; a sidecar ``keys.jsonl``
             records ``custom_id`` <-> canonical hash of each body as written; a
             ``batchPredictionJobs`` job is created via ``google-genai``
             (``client.aio.batches.create``) with ``dest`` = ``<run>/output/``.
  - poll:    ``client.aio.batches.get``; ``JobState`` -> ``BatchState``.
  - fetch:   list ``<run>/output/`` and stream every ``predictions*.jsonl`` line by
             line. Vertex rows carry no custom id, so each row is matched to its
             request by (1) the ``llm_kit_id`` request *label* if the service echoed
             it, else (2) the canonical hash of the echoed ``request``.
  - cancel / cleanup: SDK cancel; ``cleanup`` deletes uploaded input / images / output.

SDK exception to invariant #1: this backend does **not** use the shared
``httpx.AsyncClient`` and never builds auth headers. ``google-genai`` and
``google-cloud-storage`` own their transport and resolve Application Default
Credentials themselves. Both are an optional extra (``llm_kit[vertex]``) and are
imported lazily, only when a client is not injected, so this module imports
without them. Both SDK clients are injectable (``genai_client``,
``storage_client``) for tests.

Image upload (``upload_images``) rewrites ``inlineData`` parts into ``fileData``
URIs. That is envelope/transport work (where the bytes live), not wire-format
logic: the formatter still decides what an image part *is*.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import json
import logging
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Iterator
from typing import Any

from llm_kit.config.app import AppConfig
from llm_kit.errors import ProviderError
from llm_kit.provider_batch.job import BatchJob
from llm_kit.provider_batch.types import (
    BatchJobStatus,
    BatchRequest,
    BatchState,
    RawResult,
    SubmitResult,
)

logger = logging.getLogger(__name__)

LABEL_KEY = "llm_kit_id"
_INPUT_NAME = "input.jsonl"
_KEYS_NAME = "keys.jsonl"
_OUTPUT_DIR = "output/"
_IMAGES_DIR = "images/"
_LINES_PER_HOP = 64  # lines read per thread hop when streaming a blob

_STATE_MAP: dict[str, BatchState] = {
    "JOB_STATE_UNSPECIFIED": BatchState.PENDING,
    "JOB_STATE_QUEUED": BatchState.PENDING,
    "JOB_STATE_PENDING": BatchState.PENDING,
    "JOB_STATE_RUNNING": BatchState.IN_PROGRESS,
    "JOB_STATE_CANCELLING": BatchState.IN_PROGRESS,
    "JOB_STATE_UPDATING": BatchState.IN_PROGRESS,
    "JOB_STATE_SUCCEEDED": BatchState.COMPLETED,
    "JOB_STATE_PARTIALLY_SUCCEEDED": BatchState.COMPLETED,
    "JOB_STATE_FAILED": BatchState.FAILED,
    "JOB_STATE_CANCELLED": BatchState.CANCELLED,
    "JOB_STATE_EXPIRED": BatchState.EXPIRED,
    "JOB_STATE_PAUSED": BatchState.PAUSED,
}

_MIME_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/heic": "heic",
    "image/heif": "heif",
    "image/bmp": "bmp",
    "application/pdf": "pdf",
}


# ------------------------------------------------------------------ helpers


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """``gs://bucket/some/path`` -> ``("bucket", "some/path")``."""
    if not uri.startswith("gs://"):
        raise ValueError(f"expected a gs:// URI, got {uri!r}")
    rest = uri[len("gs://") :]
    bucket, _, path = rest.partition("/")
    if not bucket:
        raise ValueError(f"gs:// URI has no bucket: {uri!r}")
    return bucket, path


def encode_label(custom_id: str) -> str | None:
    """Label-safe encoding of ``custom_id`` (lowercase ``[a-z0-9_-]``, <= 63 chars).

    ``"x"`` + lowercase unpadded base32 of the UTF-8 id; ``None`` if too long (such
    rows rely on the hash fallback).
    """
    enc = "x" + base64.b32encode(custom_id.encode()).decode().rstrip("=").lower()
    return enc if len(enc) <= 63 else None


def decode_label(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("x"):
        return None
    raw = value[1:].upper()
    try:
        return base64.b32decode(raw + "=" * (-len(raw) % 8)).decode()
    except Exception:  # noqa: BLE001 - any malformed label simply doesn't match
        return None


def canonical_hash(body: dict[str, Any]) -> str:
    """sha256 of the request body as canonical JSON, ignoring our own label."""
    b = dict(body)
    labels = b.get("labels")
    if isinstance(labels, dict):
        labels = {k: v for k, v in labels.items() if k != LABEL_KEY}
        if labels:
            b["labels"] = labels
        else:
            b.pop("labels")
    text = json.dumps(b, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _state_str(state: Any) -> str:
    return str(getattr(state, "value", state) or "").upper()


def _error_detail(error: Any) -> str | None:
    if not error:
        return None
    if isinstance(error, dict):
        msg, details = error.get("message"), error.get("details")
    else:
        msg, details = getattr(error, "message", None), getattr(error, "details", None)
    parts = [str(msg)] if msg else []
    if details:
        parts.extend(str(d) for d in details)
    return "; ".join(parts) or str(error)


def _row_error(row: dict[str, Any]) -> str | None:
    """Per-row failure text, or None if the row carries a usable response."""
    status = row.get("status")
    if status:
        return f"batch item error: {status!r}"
    if not isinstance(row.get("response"), dict):
        return f"batch item missing response: keys={sorted(row)!r}"
    return None


class VertexBatchBackend:
    """``BatchBackend`` for Vertex AI Gemini batch prediction over GCS."""

    name = "vertex"

    def __init__(
        self,
        config: AppConfig,
        *,
        genai_client: Any | None = None,
        storage_client: Any | None = None,
    ) -> None:
        self._config = config
        self._vertex = config.provider_batch.vertex
        self._genai = genai_client
        self._storage = storage_client

    # ---------------------------------------------------------- lazy SDK clients

    def _genai_client(self) -> Any:
        if self._genai is None:
            from google import genai  # lazy: optional extra

            self._genai = genai.Client(
                vertexai=True, project=self._vertex.project, location=self._vertex.location
            )
        return self._genai

    def _storage_client(self) -> Any:
        if self._storage is None:
            from google.cloud import storage  # lazy: optional extra

            self._storage = storage.Client(project=self._vertex.project)
        return self._storage

    # ------------------------------------------------------------------ submit

    async def submit(self, requests: list[BatchRequest]) -> SubmitResult:
        cfg = self._vertex
        if not cfg.gcs_prefix:
            raise ValueError(
                "provider_batch.vertex.gcs_prefix (gs://bucket/path) is required to submit"
            )
        bucket, base = parse_gs_uri(cfg.gcs_prefix)
        base = base.strip("/")
        run_id = uuid.uuid4().hex
        run_path = f"{base}/llm-kit-{run_id}/" if base else f"llm-kit-{run_id}/"
        run_prefix = f"gs://{bucket}/{run_path}"

        bodies = [r.body for r in requests]
        image_uris: list[str] = []
        if cfg.upload_images:
            bodies, image_uris = await self._upload_images(bucket, run_path, bodies)

        def input_rows() -> Iterator[str]:
            for r, body in zip(requests, bodies, strict=True):
                row_body = body
                label = encode_label(r.custom_id)
                if label is not None:
                    row_body = {**body, "labels": {**(body.get("labels") or {}), LABEL_KEY: label}}
                yield json.dumps({"request": row_body}, ensure_ascii=False)

        def key_rows() -> Iterator[str]:
            for r, body in zip(requests, bodies, strict=True):
                yield json.dumps(
                    {"id": r.custom_id, "hash": canonical_hash(body)}, ensure_ascii=False
                )

        await self._write_lines(bucket, run_path + _INPUT_NAME, input_rows())
        await self._write_lines(bucket, run_path + _KEYS_NAME, key_rows())

        input_uri = run_prefix + _INPUT_NAME
        output_uri = run_prefix + _OUTPUT_DIR
        job = await self._genai_client().aio.batches.create(
            model=self._model(),
            src=input_uri,
            config={
                "display_name": f"{cfg.display_name_prefix}-{run_id}",
                "dest": output_uri,
            },
        )
        extra: dict[str, Any] = {
            "run_prefix": run_prefix,
            "input_uri": input_uri,
            "output_uri": output_uri,
            "keys_uri": run_prefix + _KEYS_NAME,
            "image_uris": image_uris,
        }
        return SubmitResult(batch_id=str(job.name), extra=extra)

    def _model(self) -> str:
        # Bare "gemini-..." or full "publishers/google/models/..." — the SDK accepts both.
        return self._config.llm.model

    async def _upload_images(
        self, bucket: str, run_path: str, bodies: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        uploaded: dict[str, str] = {}  # sha256 -> gs uri

        async def rewrite(part: dict[str, Any]) -> dict[str, Any]:
            inline = part["inlineData"]
            mime = str(inline.get("mimeType") or "application/octet-stream")
            data = base64.b64decode(inline["data"])
            digest = hashlib.sha256(data).hexdigest()
            if digest not in uploaded:
                ext = _MIME_EXT.get(mime, "bin")
                path = f"{run_path}{_IMAGES_DIR}{digest}.{ext}"
                await self._write_bytes(bucket, path, data, mime)
                uploaded[digest] = f"gs://{bucket}/{path}"
            new = {k: v for k, v in part.items() if k != "inlineData"}
            new["fileData"] = {"mimeType": mime, "fileUri": uploaded[digest]}
            return new

        out: list[dict[str, Any]] = []
        for body in bodies:
            contents = body.get("contents")
            if not isinstance(contents, list):
                out.append(body)
                continue
            new_contents = []
            for content in contents:
                parts = content.get("parts") if isinstance(content, dict) else None
                if not isinstance(parts, list):
                    new_contents.append(content)
                    continue
                new_parts = [
                    await rewrite(p) if isinstance(p, dict) and "inlineData" in p else p
                    for p in parts
                ]
                new_contents.append({**content, "parts": new_parts})
            out.append({**body, "contents": new_contents})
        return out, list(uploaded.values())

    # -------------------------------------------------------------------- poll

    async def poll(self, batch_id: str) -> BatchJobStatus:
        job = await self._genai_client().aio.batches.get(name=batch_id)
        raw = _state_str(getattr(job, "state", None))
        state = _STATE_MAP.get(raw)
        if state is None:
            logger.warning("unknown Vertex job state %r for %s; treating as in_progress", raw, batch_id)
            state = BatchState.IN_PROGRESS
        counts: dict[str, int] = {}
        stats = getattr(job, "completion_stats", None)
        if stats is not None:
            for key, attr in (
                ("succeeded", "successful_count"),
                ("failed", "failed_count"),
                ("incomplete", "incomplete_count"),
            ):
                val = getattr(stats, attr, None)
                if isinstance(val, int):
                    counts[key] = val
        return BatchJobStatus(
            state=state,
            raw_state=raw,
            counts=counts,
            output_ref=self._output_dir(job),
            error_detail=_error_detail(getattr(job, "error", None)),
        )

    @staticmethod
    def _output_dir(job: Any) -> str | None:
        """Our ``<run>/output/`` directory, derived from the job (not memory)."""
        dest = getattr(job, "dest", None)
        uri = getattr(dest, "gcs_uri", None)
        if isinstance(uri, str) and uri:
            return uri.rstrip("/") + "/"
        src = getattr(getattr(job, "src", None), "gcs_uri", None)
        if isinstance(src, (list, tuple)) and src:
            src = src[0]
        if isinstance(src, str) and src:
            return src.rsplit("/", 1)[0] + "/" + _OUTPUT_DIR
        return None

    @staticmethod
    def _run_prefix_of(output_dir: str) -> str:
        return output_dir.rstrip("/").rsplit("/", 1)[0] + "/"

    # ------------------------------------------------------------------- fetch

    async def fetch_results(self, status: BatchJobStatus) -> AsyncIterator[RawResult]:
        if not status.output_ref:
            return
        bucket, out_path = parse_gs_uri(status.output_ref)
        out_path = out_path.rstrip("/") + "/"
        run_path = self._run_prefix_of(status.output_ref)
        _, keys_dir = parse_gs_uri(run_path)

        id_hash: dict[str, str] = {}
        by_hash: dict[str, deque[str]] = defaultdict(deque)
        keys_blob = self._blob(bucket, keys_dir + _KEYS_NAME)
        async for line in self._iter_lines(keys_blob):
            if not line.strip():
                continue
            k = json.loads(line)
            id_hash[k["id"]] = k["hash"]
            by_hash[k["hash"]].append(k["id"])
        remaining = set(id_hash)

        blobs = await asyncio.to_thread(
            lambda: sorted(
                (
                    b
                    for b in self._storage_client().list_blobs(bucket, prefix=out_path)
                    if b.name.rsplit("/", 1)[-1].startswith("predictions")
                    and b.name.endswith(".jsonl")
                ),
                key=lambda b: b.name,
            )
        )
        unmatched = 0
        for blob in blobs:
            async for line in self._iter_lines(blob):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    unmatched += 1
                    logger.warning("unparseable row in %s; skipped", blob.name)
                    continue
                cid = self._match(row, id_hash, by_hash, remaining)
                if cid is None:
                    unmatched += 1
                    logger.warning("could not match an output row in %s to any request; skipped", blob.name)
                    continue
                remaining.discard(cid)
                err = _row_error(row)
                if err is not None:
                    yield RawResult(custom_id=cid, error=ProviderError(err))
                else:
                    yield RawResult(custom_id=cid, inner_body=row["response"])
        if unmatched:
            logger.warning("%d output row(s) could not be matched and were skipped", unmatched)
        # Requests with no (matched) output row each get an error result.
        for cid in sorted(remaining):
            yield RawResult(
                custom_id=cid,
                error=ProviderError(f"no output row found for request {cid!r}"),
            )

    @staticmethod
    def _match(
        row: dict[str, Any],
        id_hash: dict[str, str],
        by_hash: dict[str, deque[str]],
        remaining: set[str],
    ) -> str | None:
        request = row.get("request")
        if not isinstance(request, dict):
            return None
        labels = request.get("labels")
        cid = decode_label(labels.get(LABEL_KEY)) if isinstance(labels, dict) else None
        if cid is not None and cid in remaining:
            bucket = by_hash.get(id_hash[cid])
            if bucket is not None and cid in bucket:
                bucket.remove(cid)
            return cid
        bucket = by_hash.get(canonical_hash(request))
        while bucket:
            cand = bucket.popleft()
            if cand in remaining:
                return cand
        return None

    # ------------------------------------------------------------ cancel/cleanup

    async def cancel(self, batch_id: str) -> None:
        await self._genai_client().aio.batches.cancel(name=batch_id)

    async def cleanup(
        self,
        job: BatchJob | str,
        *,
        input: bool | None = None,  # noqa: A002 - mirrors the config flag name
        output: bool | None = None,
    ) -> list[str]:
        """Delete this run's GCS objects; returns the deleted ``gs://`` URIs.

        ``input`` covers ``input.jsonl``, the ``keys.jsonl`` sidecar and uploaded
        images; ``output`` covers ``output/``. ``None`` falls back to the
        ``cleanup_input`` / ``cleanup_output`` config flags. Works from the job
        resource alone (the run directory is derived from the job's src/dest).
        """
        do_input = self._vertex.cleanup_input if input is None else input
        do_output = self._vertex.cleanup_output if output is None else output
        if not (do_input or do_output):
            return []
        run_prefix: str | None = None
        if isinstance(job, BatchJob):
            run_prefix = job.extra.get("run_prefix")
            batch_id = job.batch_id
        else:
            batch_id = job
        if not run_prefix:
            out = self._output_dir(await self._genai_client().aio.batches.get(name=batch_id))
            if not out:
                raise ValueError(f"cannot determine the GCS run directory of job {batch_id!r}")
            run_prefix = self._run_prefix_of(out)
        bucket, run_path = parse_gs_uri(run_prefix)

        def doomed(name: str) -> bool:
            rel = name[len(run_path) :]
            if rel.startswith(_OUTPUT_DIR):
                return do_output
            return do_input and (
                rel in (_INPUT_NAME, _KEYS_NAME) or rel.startswith(_IMAGES_DIR)
            )

        def run() -> list[str]:
            deleted: list[str] = []
            for blob in list(self._storage_client().list_blobs(bucket, prefix=run_path)):
                if doomed(blob.name):
                    blob.delete()
                    deleted.append(f"gs://{bucket}/{blob.name}")
            return deleted

        return await asyncio.to_thread(run)

    # ----------------------------------------------------------- storage helpers

    def _blob(self, bucket: str, path: str) -> Any:
        return self._storage_client().bucket(bucket).blob(path)

    async def _write_lines(self, bucket: str, path: str, lines: Iterator[str]) -> None:
        def run() -> None:
            blob = self._blob(bucket, path)
            with blob.open("wb", content_type="application/jsonl") as fh:
                for line in lines:
                    fh.write(line.encode() + b"\n")

        await asyncio.to_thread(run)

    async def _write_bytes(self, bucket: str, path: str, data: bytes, mime: str) -> None:
        await asyncio.to_thread(
            lambda: self._blob(bucket, path).upload_from_string(data, content_type=mime)
        )

    async def _iter_lines(self, blob: Any) -> AsyncIterator[bytes]:
        """Stream a blob line by line (bounded memory), reading off the event loop."""
        reader = await asyncio.to_thread(blob.open, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(
                    lambda: list(itertools.islice(reader, _LINES_PER_HOP))
                )
                if not chunk:
                    return
                for line in chunk:
                    yield line
        finally:
            await asyncio.to_thread(reader.close)
