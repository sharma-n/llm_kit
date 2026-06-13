"""Translate internal ``Message`` objects into provider-specific wire shapes.

Two formatter classes live here:
  - ``OpenAIFormatter`` — OpenAI chat-completions shape (also vLLM, Ollama, Together, …)
  - ``AnthropicFormatter`` — Anthropic Messages API shape

Each handles both sides of the request/response cycle:
  - Request: ``format_messages``, ``system_prompt``, ``structured_output_fields``,
    ``default_headers``
  - Response: ``extract_text``, ``extract_usage``, ``finish_reason``

The LLM client is provider-agnostic and delegates all wire-format decisions to the
active formatter.  Adding a new provider is a new formatter class — nothing in the
LLM client changes.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.errors import ProviderError
from llm_kit.messages.types import Message, Role


class OpenAIFormatter:
    """OpenAI chat-completions wire format (request + response)."""

    def __init__(self, *, structured_output_style: StructuredOutputStyle) -> None:
        self._style = structured_output_style

    # ------------------------------------------------------------------ request

    def format_messages(
        self,
        messages: list[Message],
        encoded_images: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Build the ``messages`` array.

        ``encoded_images`` maps original URL → ``"data:image/...;base64,..."``.
        URLs absent from the map are skipped (the fetcher's failure policy already
        decided what to do; we honour whatever it produced).
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            content = self._build_content(msg, encoded_images)
            out.append({"role": str(msg.role), "content": content})
        return out

    def system_prompt(self, messages: list[Message]) -> str | None:
        # OpenAI embeds system as a regular message; nothing to lift out.
        return None

    def structured_output_fields(self, model: type[BaseModel] | None) -> dict[str, Any]:
        """Return body fields to merge for structured output (empty dict if none)."""
        if model is None or self._style is StructuredOutputStyle.NONE:
            return {}
        if self._style is StructuredOutputStyle.OPENAI_JSON_SCHEMA:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": model.__name__,
                        "schema": model.model_json_schema(),
                        "strict": True,
                    },
                }
            }
        if self._style is StructuredOutputStyle.TOOL_USE:
            return {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": model.__name__,
                            "parameters": model.model_json_schema(),
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": model.__name__},
                },
            }
        raise ValueError(f"Unknown structured output style: {self._style}")

    def default_headers(self, api_key: str | None) -> dict[str, str]:
        if api_key:
            return {"authorization": f"Bearer {api_key}"}
        return {}

    # ----------------------------------------------------------------- response

    def extract_text(self, data: dict[str, Any]) -> str:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"missing choices[0].message in response: {data!r}", cause=exc
            ) from exc

        if self._style is StructuredOutputStyle.TOOL_USE:
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                args = tool_calls[0].get("function", {}).get("arguments")
                if isinstance(args, str):
                    return args
                if isinstance(args, dict):
                    return json.dumps(args)

        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Some providers return content as an array of typed parts even for
            # text-only responses; concatenate the text parts.
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content if content is not None else ""

    def extract_usage(self, data: dict[str, Any]) -> tuple[int, int, int]:
        """Return ``(prompt_tokens, completion_tokens, total_tokens)``."""
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0)
        return prompt, completion, total

    def finish_reason(self, data: dict[str, Any]) -> str | None:
        try:
            return data["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            return None

    # ------------------------------------------------------------------ private

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


class AnthropicFormatter:
    """Anthropic Messages API wire format (request + response)."""

    # ------------------------------------------------------------------ request

    def format_messages(
        self,
        messages: list[Message],
        encoded_images: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Build the ``messages`` array, excluding system messages.

        System text is hoisted to the top-level ``system`` field via
        ``system_prompt()``; including it here would be an API error.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role is Role.SYSTEM:
                continue
            content = self._build_content(msg, encoded_images)
            out.append({"role": str(msg.role), "content": content})
        return out

    def system_prompt(self, messages: list[Message]) -> str | None:
        texts = [msg.text or "" for msg in messages if msg.role is Role.SYSTEM]
        return "\n\n".join(texts) if texts else None

    def structured_output_fields(self, model: type[BaseModel] | None) -> dict[str, Any]:
        """Return body fields for structured output using Anthropic's output_config.format."""
        if model is None:
            return {}
        return {
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": model.model_json_schema(),
                }
            }
        }

    def default_headers(self, api_key: str | None) -> dict[str, str]:
        headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    # ----------------------------------------------------------------- response

    def extract_text(self, data: dict[str, Any]) -> str:
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderError(
                f"missing or unexpected 'content' in Anthropic response: {data!r}"
            )
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    def extract_usage(self, data: dict[str, Any]) -> tuple[int, int, int]:
        """Return ``(prompt_tokens, completion_tokens, total_tokens)``."""
        usage = data.get("usage") or {}
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
        return prompt, completion, prompt + completion

    def finish_reason(self, data: dict[str, Any]) -> str | None:
        return data.get("stop_reason")

    # ------------------------------------------------------------------ private

    def _build_content(
        self,
        msg: Message,
        encoded_images: dict[str, str],
    ) -> str | list[dict[str, Any]]:
        if not msg.image_urls:
            return msg.text or ""
        parts: list[dict[str, Any]] = []
        if msg.text:
            parts.append({"type": "text", "text": msg.text})
        for url in msg.image_urls:
            data_url = encoded_images.get(url)
            if data_url is None:
                continue
            image_part = _data_url_to_anthropic_image(data_url)
            if image_part is not None:
                parts.append(image_part)
        return parts


def _data_url_to_anthropic_image(data_url: str) -> dict[str, Any] | None:
    """Convert a ``data:image/...;base64,...`` URL to an Anthropic image block."""
    try:
        header, b64_data = data_url.split(",", 1)
        media_type = header[5 : header.index(";")]  # strip "data:" prefix, stop before ";"
    except (ValueError, IndexError):
        return None
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64_data},
    }
