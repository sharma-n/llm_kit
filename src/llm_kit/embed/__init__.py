"""Embeddings: OpenAI-compatible client + micro-batching driver."""

from llm_kit.embed.client import OpenAICompatibleEmbedder
from llm_kit.embed.response import EmbeddingResponse, EmbeddingResult, EmbedItem

__all__ = [
    "EmbedItem",
    "EmbeddingResponse",
    "EmbeddingResult",
    "OpenAICompatibleEmbedder",
]
