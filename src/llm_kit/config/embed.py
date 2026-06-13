"""Embeddings endpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EmbeddingConfig:
    """Connection + request shaping for an OpenAI-compatible embeddings endpoint.

    The two ``max_*_per_request`` knobs control *micro-batching*: unlike chat
    completions (one prompt per request), ``/v1/embeddings`` accepts an array of
    inputs per request. The embed driver packs as many inputs as fit under both
    caps into each HTTP call, which is the dominant throughput lever for
    embeddings. Defaults mirror OpenAI's published limits.
    """

    base_url: str = "https://api.openai.com"
    embeddings_path: str = "/v1/embeddings"
    model: str = "text-embedding-3-small"

    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None  # resolved at client init; never persisted to YAML

    # Matryoshka truncation — only honored by models that support it (e.g.
    # text-embedding-3-*). Left None means "provider default dimensionality".
    dimensions: int | None = None

    # "float" returns a JSON array of floats; "base64" returns a packed
    # little-endian float32 blob (smaller on the wire). Both decode to the same
    # vector locally.
    encoding_format: str = "float"

    extra_headers: dict[str, str] = field(default_factory=dict)

    # Micro-batch caps. A single request is flushed when adding the next input
    # would exceed *either* of these.
    max_inputs_per_request: int = 2048
    max_tokens_per_request: int = 300_000
