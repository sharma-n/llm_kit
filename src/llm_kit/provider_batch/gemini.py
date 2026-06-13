"""Google Gemini batch backend (inline transport). SCAFFOLDED.

.. warning::
   This backend is **not verified against the live Gemini API** and Gemini's
   batch surface is the least settled of the three. It exists to keep the
   ``BatchBackend`` abstraction honest across providers. Field names, the
   long-running-operation polling shape, and the results envelope below are
   plausible but unconfirmed — verify against the live API before use. Tests
   cover envelope wrap/unwrap against canned mock responses only.

Transport (inline): submit an inline ``requests`` array to a batch endpoint,
poll the returned operation, then read per-request responses keyed back to each
item's ``custom_id`` via the request ``metadata.key`` field.
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


class GeminiBatchBackend:
    """Inline transport against the Gemini batch API. SCAFFOLDED."""

    def __init__(self, client: httpx.AsyncClient, config: AppConfig, api_key: str | None) -> None:
        self._client = client
        self._config = config
        self._api_key = api_key

    async def submit(self, requests: list[BatchRequest]) -> str:
        body = {
            "batch": {
                "inputConfig": {
                    "requests": {
                        "requests": [
                            {"request": r.body, "metadata": {"key": r.custom_id}}
                            for r in requests
                        ]
                    }
                }
            }
        }
        # Model is carried in chat_completions_path for Gemini (…/models/{model}:…).
        response = await retry_request(
            lambda: self._client.post(
                self._batch_url(), headers=self._json_headers(), json=body
            ),
            retry=self._config.http.llm_retry,
            operation="batch_create",
        )
        return self._parse(response).get("name", "")

    async def poll(self, batch_id: str) -> BatchJobStatus:
        url = self._url(f"/v1beta/{batch_id}")
        response = await retry_request(
            lambda: self._client.get(url, headers=self._auth_headers()),
            retry=self._config.http.llm_retry,
            operation="batch_poll",
        )
        data = self._parse(response)
        done = bool(data.get("done"))
        state = BatchState.COMPLETED if done else BatchState.IN_PROGRESS
        if done and data.get("error"):
            state = BatchState.FAILED
        return BatchJobStatus(
            state=state,
            raw_state="done" if done else "running",
            output_ref=batch_id if done else None,
        )

    async def fetch_results(self, status: BatchJobStatus) -> AsyncIterator[RawResult]:
        if not status.output_ref:
            return
        url = self._url(f"/v1beta/{status.output_ref}")
        response = await retry_request(
            lambda: self._client.get(url, headers=self._auth_headers()),
            retry=self._config.http.llm_retry,
            operation="batch_results_download",
        )
        data = self._parse(response)
        inlined = (data.get("response") or {}).get("inlinedResponses") or {}
        for entry in inlined.get("inlinedResponses") or []:
            yield _result_entry_to_raw(entry)

    async def cancel(self, batch_id: str) -> None:
        url = self._url(f"/v1beta/{batch_id}:cancel")
        await retry_request(
            lambda: self._client.post(url, headers=self._auth_headers()),
            retry=self._config.http.llm_retry,
            operation="batch_cancel",
        )

    # ---------------------------------------------------------------- internal

    def _batch_url(self) -> str:
        # chat_completions_path is …/models/{model}:generateContent; swap the verb.
        path = self._config.llm.chat_completions_path
        path = path.replace(":generateContent", ":batchGenerateContent")
        return self._url(path)

    def _url(self, path: str) -> str:
        base = self._config.llm.base_url.rstrip("/")
        return base + (path if path.startswith("/") else "/" + path)

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key
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


def _result_entry_to_raw(entry: dict[str, Any]) -> RawResult:
    custom_id = str((entry.get("metadata") or {}).get("key", ""))
    if entry.get("error"):
        return RawResult(
            custom_id=custom_id,
            error=ProviderError(f"batch item error: {entry['error']!r}"),
        )
    body = entry.get("response")
    if isinstance(body, dict):
        return RawResult(custom_id=custom_id, inner_body=body)
    return RawResult(
        custom_id=custom_id,
        error=ProviderError(f"batch item missing response: {entry!r}"),
    )
