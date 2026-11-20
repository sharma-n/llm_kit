"""Google Gemini Batch Mode backend (inline transport).

Implements the documented Gemini Batch API
(https://ai.google.dev/gemini-api/docs/batch-api) for inline requests:

  - submit:  ``POST /v1beta/models/{model}:batchGenerateContent`` with
             ``{"batch": {"inputConfig": {"requests": {"requests": [{request, metadata:{key}}]}}}}``;
             returns a long-running operation whose ``name`` (e.g. ``batches/123``) is the job id.
  - poll:    ``GET /v1beta/{name}`` → completion via ``metadata.state`` (``JOB_STATE_PENDING`` /
             ``RUNNING`` / ``SUCCEEDED`` / ``FAILED`` / ``CANCELLED`` / ``EXPIRED``), with the
             ``done`` boolean as a fallback signal.
  - results: once succeeded, ``response.inlinedResponses`` is a direct array; each element is
             ``{response: <GenerateContentResponse>, metadata: {key}}`` or ``{error: <status>}``.
             Results are correlated back to each item by the ``metadata.key`` we set on submit
             (the item's custom_id), falling back to array order.

Field casing follows protojson (camelCase and snake_case are both accepted); camelCase is used
here to match ``GeminiFormatter``. The inner ``request`` body and the succeeded ``response`` are
produced/parsed by the formatter and ``parse_response``. Verified against the published wire
format; exercised in-repo by mock-transport tests only (no live-key integration test).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_kit.config.app import AppConfig
from llm_kit.errors import ProviderError
from llm_kit.http.retry import retry_request
from llm_kit.provider_batch.types import BatchJobStatus, BatchRequest, BatchState, RawResult

logger = logging.getLogger(__name__)

# Gemini long-running-operation job state -> normalized lifecycle state.
_STATE_MAP: dict[str, BatchState] = {
    "JOB_STATE_PENDING": BatchState.PENDING,
    "JOB_STATE_RUNNING": BatchState.IN_PROGRESS,
    "JOB_STATE_SUCCEEDED": BatchState.COMPLETED,
    "JOB_STATE_FAILED": BatchState.FAILED,
    "JOB_STATE_CANCELLED": BatchState.CANCELLED,
    "JOB_STATE_EXPIRED": BatchState.EXPIRED,
    "JOB_STATE_PAUSED": BatchState.PAUSED,
}
# The Gemini Batch API's own enum spells the same states BATCH_STATE_*.
_STATE_MAP.update(
    {k.replace("JOB_STATE_", "BATCH_STATE_"): v for k, v in list(_STATE_MAP.items())}
)


class GeminiBatchBackend:
    """Inline transport against the Gemini Batch API."""

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
        # The create call returns the long-running operation; its name is the job id.
        return self._parse(response).get("name", "")

    async def poll(self, batch_id: str) -> BatchJobStatus:
        url = self._url(f"/v1beta/{batch_id}")
        response = await retry_request(
            lambda: self._client.get(url, headers=self._auth_headers()),
            retry=self._config.http.llm_retry,
            operation="batch_poll",
        )
        data = self._parse(response)
        raw_state = str((data.get("metadata") or {}).get("state", ""))
        state = _STATE_MAP.get(raw_state)
        if state is None:
            # No (or unknown) state field: fall back to the operation `done` flag.
            state = BatchState.COMPLETED if data.get("done") else BatchState.IN_PROGRESS
            if data.get("done") and data.get("error"):
                state = BatchState.FAILED
        error = data.get("error") or (data.get("metadata") or {}).get("error")
        detail = None
        if isinstance(error, dict):
            detail = str(error.get("message") or error) or None
        elif error:
            detail = str(error)
        return BatchJobStatus(
            state=state,
            raw_state=raw_state or ("done" if data.get("done") else "running"),
            error_detail=detail,
            # Results live on the operation itself; re-fetch it to read them.
            output_ref=batch_id if state is BatchState.COMPLETED else None,
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
        inlined = (data.get("response") or {}).get("inlinedResponses") or []
        for index, entry in enumerate(inlined):
            yield _result_entry_to_raw(entry, index)

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
        except ValueError as exc:
            raise ProviderError(
                f"non-JSON response from batch endpoint: {response.text[:200]!r}",
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(f"unexpected batch response shape: {type(data).__name__}")
        return data


def _result_entry_to_raw(entry: dict[str, Any], index: int) -> RawResult:
    # We set metadata.key = custom_id on submit; the API echoes it for correlation.
    # Fall back to array order (inline responses preserve request order) if absent.
    custom_id = str((entry.get("metadata") or {}).get("key", "") or index)
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
