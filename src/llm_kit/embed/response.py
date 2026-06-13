"""Embedding request/result dataclasses.

Mirrors the chat side's ``BatchItem`` / ``BatchResult`` shape so the jsonl
sink and ID-based resume work the same way — the output file is still the
checkpoint, keyed by ``id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_kit.errors import LLMError
from llm_kit.llm.response import TokenUsage


@dataclass(slots=True)
class EmbeddingResponse:
    """One input's vector.

    ``usage`` is *approximate* per input: the embeddings API reports tokens per
    request, not per input, so the request total is split across its inputs in
    proportion to their estimated size (see ``split_usage``). The split is exact
    in aggregate — summing ``usage.total_tokens`` over a request's inputs equals
    the request's reported total.
    """

    vector: list[float]
    index: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None


@dataclass(slots=True)
class EmbedItem:
    """One unit of work for the embed driver."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingResult:
    """One completed embed unit. Exactly one of ``response``/``error`` is set.

    Note: embeddings are micro-batched, so an HTTP-level failure fails the whole
    chunk — every input in that request shares the same ``error``.
    """

    id: str
    response: EmbeddingResponse | None = None
    error: LLMError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def split_usage(total: TokenUsage, weights: list[int]) -> list[TokenUsage]:
    """Split a per-request ``TokenUsage`` across N inputs by ``weights``.

    The last input absorbs the rounding remainder so the parts sum exactly to
    ``total.total_tokens``. Embeddings have no completion tokens, so prompt and
    total are equal per input.
    """
    n = len(weights)
    if n == 0:
        return []
    total_tokens = total.total_tokens
    weight_sum = sum(weights)
    out: list[TokenUsage] = []
    allocated = 0
    for i, w in enumerate(weights):
        if i == n - 1:
            tokens = total_tokens - allocated
        elif weight_sum > 0:
            tokens = round(total_tokens * w / weight_sum)
            allocated += tokens
        else:  # all weights zero — distribute evenly
            tokens = total_tokens // n
            allocated += tokens
        tokens = max(tokens, 0)
        out.append(TokenUsage(prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens))
    return out
