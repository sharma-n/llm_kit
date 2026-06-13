"""llm_kit: provider-agnostic async toolkit for multimodal LLM apps."""

from llm_kit.config import AppConfig
from llm_kit.embed.client import OpenAICompatibleEmbedder
from llm_kit.embed.response import EmbeddingResponse, EmbeddingResult, EmbedItem
from llm_kit.errors import (
    AuthError,
    ImageFetchError,
    LLMError,
    NetworkError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)
from llm_kit.llm.base import BaseLLM
from llm_kit.llm.openai_compat import OpenAICompatibleLLM
from llm_kit.llm.response import BatchItem, BatchResult, LLMResponse
from llm_kit.messages.types import Message

__all__ = [
    "AppConfig",
    "AuthError",
    "BaseLLM",
    "BatchItem",
    "BatchResult",
    "EmbedItem",
    "EmbeddingResponse",
    "EmbeddingResult",
    "ImageFetchError",
    "LLMError",
    "LLMResponse",
    "Message",
    "NetworkError",
    "OpenAICompatibleEmbedder",
    "OpenAICompatibleLLM",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "ValidationError",
]
