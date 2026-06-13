"""Translate internal ``Message`` objects into the OpenAI chat-completions shape.

Two responsibilities:
1. Build the ``messages`` array, replacing ``image_urls`` with base64 data URLs.
2. Inject ``response_format`` (or equivalent) when a pydantic schema is supplied.

This is the only file in the package that knows the OpenAI wire shape. Adding
a new provider style is a new formatter, not changes to the LLM client.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.messages.types import Message


class OpenAIFormatter:
    """Format messages + response schema into the OpenAI chat-completions body."""

    def __init__(self, *, structured_output_style: StructuredOutputStyle) -> None:
        self._style = structured_output_style

    def format_messages(
        self,
        messages: list[Message],
        encoded_images: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Build the ``messages`` array.

        ``encoded_images`` maps the original URL → ``"data:image/...;base64,..."``.
        URLs missing from the map are skipped (the fetcher's failure policy
        already decided what to do; we honor whatever it produced).
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            content = self._build_content(msg, encoded_images)
            out.append({"role": str(msg.role), "content": content})
        return out

    def response_format(self, model: type[BaseModel] | None) -> dict[str, Any] | None:
        if model is None or self._style is StructuredOutputStyle.NONE:
            return None
        if self._style is StructuredOutputStyle.OPENAI_JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": model.__name__,
                    "schema": model.model_json_schema(),
                    "strict": True,
                },
            }
        if self._style is StructuredOutputStyle.TOOL_USE:
            # Some providers (Anthropic, others) prefer schemas-as-tools. We
            # surface a tool definition; the LLM client merges this into the
            # body as ``tools`` + ``tool_choice``.
            return {
                "_style": "tool_use",
                "tool": {
                    "type": "function",
                    "function": {
                        "name": model.__name__,
                        "parameters": model.model_json_schema(),
                    },
                },
            }
        raise ValueError(f"Unknown structured output style: {self._style}")

    def _build_content(
        self,
        msg: Message,
        encoded_images: dict[str, str],
    ) -> str | list[dict[str, Any]]:
        # Text-only messages use the simple string form — smaller payload, and
        # some providers reject content-arrays for system messages.
        if not msg.image_urls:
            return msg.text or ""
        parts: list[dict[str, Any]] = []
        if msg.text:
            parts.append({"type": "text", "text": msg.text})
        for url in msg.image_urls:
            data_url = encoded_images.get(url)
            if data_url is None:
                continue
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        return parts
