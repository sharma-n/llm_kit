"""Test helpers for code built on llm_kit (no network, no provider)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from typing import Any

from llm_kit.config.llm import MessageFormat
from llm_kit.errors import LLMError
from llm_kit.provider_batch.types import (
    BatchJobStatus,
    BatchRequest,
    BatchState,
    RawResult,
    SubmitResult,
)

Responder = Callable[[BatchRequest], dict[str, Any]]


class FakeBatchBackend:
    """An in-memory ``BatchBackend`` for tests; pass it as ``LLMClient(batch_backend=...)``.

    Batch state lives on the instance, so a second ``LLMClient`` sharing the same
    fake can resume a job saved by the first. ``submit`` returns a
    ``SubmitResult`` whose ``extra`` is ``{"fake": True, "n_requests": N}`` so the
    extra-state plumbing is exercised end to end.

    Args:
        polls_until_done: the poll number (per batch) at which the job reaches
            its final state; earlier polls report ``IN_PROGRESS``. ``0``/``1``
            means done on the first poll.
        responder: a callable ``BatchRequest -> inner response body`` or a mapping
            ``custom_id -> body``. Defaults to an OpenAI-shaped echo of the id.
        fail_items: custom_ids that yield a per-item error (an iterable uses a
            default message; a mapping gives the message per id).
        fail_job: if set, the job ends in this (terminal, non-completed) state
            instead of ``COMPLETED``, with ``raw_state`` as the provider string and
            ``error_detail`` as the provider's failure message.
        partial_output: with ``fail_job``, expose the (otherwise complete) results as
            partial output so ``allow_partial=True`` fetches yield them.
        states: scripted states for successive polls (e.g. ``[PAUSED, PAUSED]``);
            once exhausted, polling falls through to the normal final state. A
            terminal entry ends the job in that state. Overrides ``polls_until_done``.
        shuffle: yield results in reverse submission order.
    """

    name = "fake"

    def __init__(
        self,
        *,
        polls_until_done: int = 1,
        responder: Responder | Mapping[str, dict[str, Any]] | None = None,
        fail_items: Mapping[str, str] | Iterable[str] = (),
        fail_job: BatchState | None = None,
        raw_state: str = "",
        error_detail: str | None = None,
        partial_output: bool = False,
        states: Sequence[BatchState] | None = None,
        shuffle: bool = False,
    ) -> None:
        self.polls_until_done = polls_until_done
        self.fail_job = fail_job
        self.raw_state = raw_state
        self.error_detail = error_detail
        self.partial_output = partial_output
        self.states = list(states) if states else []
        self.shuffle = shuffle
        if responder is None:
            responder = self.text_responder(
                MessageFormat.OPENAI, lambda r: f"fake:{r.custom_id}"
            )
        self._responder = responder
        if isinstance(fail_items, Mapping):
            self.fail_items: dict[str, str] = dict(fail_items)
        else:
            self.fail_items = {cid: "fake item failure" for cid in fail_items}

        self.submitted: dict[str, list[BatchRequest]] = {}
        self.cancelled: set[str] = set()
        self.poll_counts: dict[str, int] = {}
        self._next_id = 0

    # ------------------------------------------------------------ BatchBackend

    async def submit(self, requests: list[BatchRequest]) -> SubmitResult:
        self._next_id += 1
        batch_id = f"fake-batch-{self._next_id}"
        self.submitted[batch_id] = list(requests)
        self.poll_counts[batch_id] = 0
        return SubmitResult(batch_id, {"fake": True, "n_requests": len(requests)})

    async def poll(self, batch_id: str) -> BatchJobStatus:
        self._require(batch_id)
        self.poll_counts[batch_id] += 1
        n = len(self.submitted[batch_id])
        if batch_id in self.cancelled:
            return BatchJobStatus(BatchState.CANCELLED, "cancelled", output_ref=batch_id)
        poll_n = self.poll_counts[batch_id]
        if self.states:
            if poll_n <= len(self.states):
                scripted = self.states[poll_n - 1]
                if scripted.is_terminal:
                    return self._final_status(batch_id, scripted)
                return BatchJobStatus(scripted, scripted.value, {"total": n})
        elif poll_n < self.polls_until_done:
            return BatchJobStatus(BatchState.IN_PROGRESS, "in_progress", {"total": n})
        return self._final_status(batch_id, self.fail_job or BatchState.COMPLETED)

    def _final_status(self, batch_id: str, state: BatchState) -> BatchJobStatus:
        n = len(self.submitted[batch_id])
        if state is not BatchState.COMPLETED:
            return BatchJobStatus(
                state,
                self.raw_state or state.value,
                {"total": n},
                output_ref=batch_id if self.partial_output else None,
                error_detail=self.error_detail,
            )
        failed = sum(1 for r in self.submitted[batch_id] if r.custom_id in self.fail_items)
        return BatchJobStatus(
            BatchState.COMPLETED,
            "completed",
            {"total": n, "failed": failed, "succeeded": n - failed},
            output_ref=batch_id,
        )

    async def fetch_results(self, status: BatchJobStatus) -> AsyncIterator[RawResult]:
        if status.output_ref is None:
            return
        requests = self._require(status.output_ref)
        ordered = list(reversed(requests)) if self.shuffle else list(requests)
        for req in ordered:
            if req.custom_id in self.fail_items:
                yield RawResult(req.custom_id, error=LLMError(self.fail_items[req.custom_id]))
            elif callable(self._responder):
                yield RawResult(req.custom_id, inner_body=self._responder(req))
            else:
                yield RawResult(req.custom_id, inner_body=self._responder[req.custom_id])

    async def cancel(self, batch_id: str) -> None:
        self._require(batch_id)
        self.cancelled.add(batch_id)

    # ----------------------------------------------------------------- helpers

    def _require(self, batch_id: str) -> list[BatchRequest]:
        try:
            return self.submitted[batch_id]
        except KeyError:
            raise LLMError(f"unknown fake batch id {batch_id!r}") from None

    @staticmethod
    def text_responder(
        message_format: MessageFormat, text_fn: Callable[[BatchRequest], str]
    ) -> Responder:
        """Build a responder whose bodies ``parse_response`` reads for ``message_format``."""

        def respond(req: BatchRequest) -> dict[str, Any]:
            text = text_fn(req)
            if message_format is MessageFormat.ANTHROPIC:
                return {
                    "id": f"msg_{req.custom_id}",
                    "type": "message",
                    "role": "assistant",
                    "model": "fake-model",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            if message_format is MessageFormat.GEMINI:
                return {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                }
            return {
                "id": f"chatcmpl-{req.custom_id}",
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        return respond
