"""Registry of batch backends, keyed by name.

Replaces a closed if-chain on ``MessageFormat``. A *factory* builds a backend from
``(config, shared httpx client, api_key)``; it may import optional dependencies
lazily inside its body, so merely registering a backend never requires them.

Selection: ``ProviderBatchConfig.backend`` if set, else the ``message_format``
value (``"openai"`` / ``"anthropic"`` / ``"gemini"``). The caller can also bypass
the registry entirely with ``LLMClient(batch_backend=...)``.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from llm_kit.config.app import AppConfig
from llm_kit.provider_batch.backend import BatchBackend

BackendFactory = Callable[[AppConfig, httpx.AsyncClient, str | None], BatchBackend]

_REGISTRY: dict[str, BackendFactory] = {}


def register_batch_backend(
    name: str, factory: BackendFactory, *, replace: bool = False
) -> None:
    """Register ``factory`` under ``name``; a duplicate needs ``replace=True``."""
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"batch backend {name!r} is already registered (pass replace=True to override)"
        )
    _REGISTRY[name] = factory


def registered_batch_backends() -> list[str]:
    return sorted(_REGISTRY)


def backend_name(config: AppConfig) -> str:
    """The backend name selected by ``config`` (explicit override or message_format)."""
    return config.provider_batch.backend or str(config.llm.message_format)


def build_backend(
    config: AppConfig, client: httpx.AsyncClient, api_key: str | None
) -> BatchBackend:
    name = backend_name(config)
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown batch backend {name!r}; registered: {', '.join(registered_batch_backends())}"
        ) from None
    return factory(config, client, api_key)


def _openai(config: AppConfig, client: httpx.AsyncClient, api_key: str | None) -> BatchBackend:
    from llm_kit.provider_batch.openai import OpenAIBatchBackend

    return OpenAIBatchBackend(client, config, api_key)


def _anthropic(config: AppConfig, client: httpx.AsyncClient, api_key: str | None) -> BatchBackend:
    from llm_kit.provider_batch.anthropic import AnthropicBatchBackend

    return AnthropicBatchBackend(client, config, api_key)


def _gemini(config: AppConfig, client: httpx.AsyncClient, api_key: str | None) -> BatchBackend:
    from llm_kit.provider_batch.gemini import GeminiBatchBackend

    return GeminiBatchBackend(client, config, api_key)


register_batch_backend("openai", _openai)
register_batch_backend("anthropic", _anthropic)
register_batch_backend("gemini", _gemini)
