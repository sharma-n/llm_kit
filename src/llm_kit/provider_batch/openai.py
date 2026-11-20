"""OpenAI Batch API backend (Files + Batches), fully wired.

Transport: upload a jsonl request file (Files API, ``purpose=batch``), create a
batch referencing it (Batches API), poll the batch object, then download the
output (and error) files. Each request line wraps the item body produced by the
formatter:

    {"custom_id": ..., "method": "POST", "url": "/v1/chat/completions", "body": {...}}

and each result line unwraps to:

    {"custom_id": ..., "response": {"status_code": 200, "body": {...}}, "error": null}

All HTTP calls go through the shared ``retry_request`` with the LLM retry policy.
This path does not touch the rate limiter — the provider meters batches on its
own enqueued-token limits.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_kit.config.app import AppConfig
from llm_kit.errors import ProviderError
from llm_kit.http.retry import retry_request
from llm_kit.provider_batch.types import BatchJobStatus, BatchRequest, BatchState, RawResult

logger = logging.getLogger(__name__)

# OpenAI batch status string -> normalized lifecycle state.
_STATE_MAP: dict[str, BatchState] = {
    "validating": BatchState.PENDING,
    "in_progress": BatchState.IN_PROGRESS,
    "finalizing": BatchState.IN_PROGRESS,
    "cancelling": BatchState.IN_PROGRESS,
    "completed": BatchState.COMPLETED,
    "failed": BatchState.FAILED,
    "expired": BatchState.EXPIRED,
    "cancelled": BatchState.CANCELLED,
}


class OpenAIBatchBackend:
    """File-upload transport against the OpenAI Batch API."""

    def __init__(self, client: httpx.AsyncClient, config: AppConfig, api_key: str | None) -> None:
        self._client = client
        self._config = config
        self._api_key = api_key

    # ------------------------------------------------------------------ submit

    async def submit(self, requests: list[BatchRequest]) -> str:
        jsonl = self._build_jsonl(requests)
        input_file_id = await self._upload(jsonl)
        return await self._create_batch(input_file_id)

    def _build_jsonl(self, requests: list[BatchRequest]) -> bytes:
        url = self._chat_path()
        lines = [
            json.dumps(
                {"custom_id": r.custom_id, "method": "POST", "url": url, "body": r.body},
                ensure_ascii=False,
            )
            for r in requests
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")

    async def _upload(self, jsonl: bytes) -> str:
        response = await retry_request(
            lambda: self._client.post(
                self._url(self._config.provider_batch.files_path),
                headers=self._auth_headers(),
                files={"file": ("batch.jsonl", jsonl, "application/jsonl")},
                data={"purpose": "batch"},
            ),
            retry=self._config.http.llm_retry,
            operation="batch_file_upload",
        )
        return self._parse(response).get("id", "")

    async def _create_batch(self, input_file_id: str) -> str:
        cfg = self._config.provider_batch
        body = {
            "input_file_id": input_file_id,
            "endpoint": self._chat_path(),
            "completion_window": cfg.completion_window,
        }
        response = await retry_request(
            lambda: self._client.post(
                self._url(cfg.batches_path),
                headers=self._json_headers(),
                json=body,
            ),
            retry=self._config.http.llm_retry,
            operation="batch_create",
        )
        return self._parse(response).get("id", "")

    # -------------------------------------------------------------------- poll

    async def poll(self, batch_id: str) -> BatchJobStatus:
        url = self._url(f"{self._config.provider_batch.batches_path}/{batch_id}")
        response = await retry_request(
            lambda: self._client.get(url, headers=self._auth_headers()),
            retry=self._config.http.llm_retry,
            operation="batch_poll",
        )
        data = self._parse(response)
        raw_state = str(data.get("status", ""))
        counts = data.get("request_counts") or {}
        return BatchJobStatus(
            state=_STATE_MAP.get(raw_state, BatchState.IN_PROGRESS),
            raw_state=raw_state,
            counts={k: int(v) for k, v in counts.items() if isinstance(v, int)},
            output_ref=data.get("output_file_id"),
            error_ref=data.get("error_file_id"),
            error_detail=_error_detail(data.get("errors")),
        )

    # ---------------------------------------------------------------- results

    async def fetch_results(self, status: BatchJobStatus) -> AsyncIterator[RawResult]:
        if status.output_ref:
            async for raw in self._download_file(status.output_ref):
                yield raw
        if status.error_ref:
            async for raw in self._download_file(status.error_ref):
                yield raw

    async def _download_file(self, file_id: str) -> AsyncIterator[RawResult]:
        url = self._url(f"{self._config.provider_batch.files_path}/{file_id}/content")
        response = await retry_request(
            lambda: self._client.get(url, headers=self._auth_headers()),
            retry=self._config.http.llm_retry,
            operation="batch_results_download",
        )
        for line in response.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed batch result line")
                continue
            yield _result_line_to_raw(record)

    # ------------------------------------------------------------------ cancel

    async def cancel(self, batch_id: str) -> None:
        url = self._url(f"{self._config.provider_batch.batches_path}/{batch_id}/cancel")
        await retry_request(
            lambda: self._client.post(url, headers=self._auth_headers()),
            retry=self._config.http.llm_retry,
            operation="batch_cancel",
        )

    # ---------------------------------------------------------------- internal

    def _chat_path(self) -> str:
        path = self._config.llm.chat_completions_path
        return path if path.startswith("/") else "/" + path

    def _url(self, path: str) -> str:
        base = self._config.llm.base_url.rstrip("/")
        return base + (path if path.startswith("/") else "/" + path)

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        headers.update(self._config.llm.extra_headers)
        return headers

    def _json_headers(self) -> dict[str, str]:
        return {"content-type": "application/json", **self._auth_headers()}

    def _parse(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"non-JSON response from batch endpoint: {response.text[:200]!r}",
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(f"unexpected batch response shape: {type(data).__name__}")
        return data


def _error_detail(errors: Any) -> str | None:
    """Join the batch object's ``errors.data[*].message`` (job-level, not per-item)."""
    if not isinstance(errors, dict):
        return None
    msgs = [
        str(e["message"])
        for e in errors.get("data") or []
        if isinstance(e, dict) and e.get("message")
    ]
    return "; ".join(msgs) or None


def _result_line_to_raw(record: dict[str, Any]) -> RawResult:
    custom_id = str(record.get("custom_id", ""))
    # A line-level error (e.g. from the error file) takes precedence.
    err = record.get("error")
    if err:
        return RawResult(custom_id=custom_id, error=_to_error(err))
    response = record.get("response") or {}
    status_code = response.get("status_code")
    body = response.get("body")
    if isinstance(status_code, int) and status_code >= 400:
        return RawResult(
            custom_id=custom_id,
            error=ProviderError(f"batch item HTTP {status_code}: {body!r}", status_code=status_code),
        )
    if not isinstance(body, dict):
        return RawResult(
            custom_id=custom_id,
            error=ProviderError(f"batch item missing response body: {record!r}"),
        )
    return RawResult(custom_id=custom_id, inner_body=body)


def _to_error(err: Any) -> ProviderError:
    if isinstance(err, dict):
        message = err.get("message") or json.dumps(err)
        code = err.get("code")
        status = code if isinstance(code, int) else None
        return ProviderError(f"batch item error: {message}", status_code=status)
    return ProviderError(f"batch item error: {err!r}")
