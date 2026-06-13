"""Anthropic Message Batches backend (inline transport).

Implements the documented Anthropic Message Batches API
(https://platform.claude.com/docs/en/build-with-claude/batch-processing):

  - submit:  ``POST /v1/messages/batches`` with ``{"requests": [{custom_id, params}]}``
  - poll:    ``GET /v1/messages/batches/{id}`` → ``processing_status`` (``in_progress`` /
             ``canceling`` / ``ended``), ``request_counts``, ``results_url``
  - results: stream the jsonl at ``results_url``; each line is
             ``{custom_id, result: {type: succeeded|errored|canceled|expired, message|error}}``

The per-item body (``params``) is the formatter's Messages-API body, and a
succeeded result's ``message`` is parsed by the same ``parse_response``. Verified
against the published wire format; exercised in-repo by mock-transport tests only
(no live-key integration test).
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

_BATCHES_PATH = "/v1/messages/batches"

# Anthropic processing_status / result types -> normalized state.
_STATE_MAP: dict[str, BatchState] = {
    "in_progress": BatchState.IN_PROGRESS,
    "canceling": BatchState.IN_PROGRESS,
    "ended": BatchState.COMPLETED,
}


class AnthropicBatchBackend:
    """Inline transport against the Anthropic Message Batches API."""

    def __init__(self, client: httpx.AsyncClient, config: AppConfig, api_key: str | None) -> None:
        self._client = client
        self._config = config
        self._api_key = api_key

    async def submit(self, requests: list[BatchRequest]) -> str:
        body = {
            "requests": [
                {"custom_id": r.custom_id, "params": r.body} for r in requests
            ]
        }
        response = await retry_request(
            lambda: self._client.post(
                self._url(_BATCHES_PATH), headers=self._json_headers(), json=body
            ),
            retry=self._config.http.llm_retry,
            operation="batch_create",
        )
        return self._parse(response).get("id", "")

    async def poll(self, batch_id: str) -> BatchJobStatus:
        url = self._url(f"{_BATCHES_PATH}/{batch_id}")
        response = await retry_request(
            lambda: self._client.get(url, headers=self._auth_headers()),
            retry=self._config.http.llm_retry,
            operation="batch_poll",
        )
        data = self._parse(response)
        raw_state = str(data.get("processing_status", ""))
        counts = data.get("request_counts") or {}
        return BatchJobStatus(
            state=_STATE_MAP.get(raw_state, BatchState.IN_PROGRESS),
            raw_state=raw_state,
            counts={k: int(v) for k, v in counts.items() if isinstance(v, int)},
            output_ref=data.get("results_url"),
        )

    async def fetch_results(self, status: BatchJobStatus) -> AsyncIterator[RawResult]:
        if not status.output_ref:
            return
        response = await retry_request(
            lambda: self._client.get(status.output_ref, headers=self._auth_headers()),
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

    async def cancel(self, batch_id: str) -> None:
        url = self._url(f"{_BATCHES_PATH}/{batch_id}/cancel")
        await retry_request(
            lambda: self._client.post(url, headers=self._auth_headers()),
            retry=self._config.http.llm_retry,
            operation="batch_cancel",
        )

    # ---------------------------------------------------------------- internal

    def _url(self, path: str) -> str:
        base = self._config.llm.base_url.rstrip("/")
        return base + (path if path.startswith("/") else "/" + path)

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
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


def _result_line_to_raw(record: dict[str, Any]) -> RawResult:
    custom_id = str(record.get("custom_id", ""))
    result = record.get("result") or {}
    rtype = result.get("type")
    if rtype == "succeeded":
        message = result.get("message")
        if isinstance(message, dict):
            return RawResult(custom_id=custom_id, inner_body=message)
        return RawResult(
            custom_id=custom_id,
            error=ProviderError(f"batch item missing message: {record!r}"),
        )
    # errored / canceled / expired
    detail = result.get("error") or rtype
    return RawResult(custom_id=custom_id, error=ProviderError(f"batch item {rtype}: {detail!r}"))
