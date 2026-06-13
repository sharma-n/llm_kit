"""Translate internal ``Message`` objects into provider-specific wire shapes.

Three formatter classes live here:
  - ``OpenAIFormatter`` — OpenAI chat-completions shape (also vLLM, Ollama, Together, …)
  - ``AnthropicFormatter`` — Anthropic Messages API shape
  - ``GeminiFormatter`` — Google Gemini ``generateContent`` shape

Each handles both sides of the request/response cycle:
  - Request: ``format_messages``, ``system_prompt``, ``structured_output_fields``,
    ``default_headers``, ``build_body``
  - Response: ``extract_text``, ``extract_usage``, ``finish_reason``

The LLM client is provider-agnostic and delegates all wire-format decisions to the
active formatter.  Adding a new provider is a new formatter class — nothing in the
LLM client changes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.errors import ProviderError
from llm_kit.messages.streaming import (
    FinishFragment,
    MetaFragment,
    StreamFragment,
    TextFragment,
    ToolCallFragment,
    UsageFragment,
)
from llm_kit.messages.tools import ToolCall, ToolDefinition
from llm_kit.messages.types import Message, Role


def _parse_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Coerce a tool call's ``arguments`` (JSON string or dict) to a dict.

    Preserves the original text under ``_raw`` if it is not valid JSON, so a
    malformed argument blob is surfaced rather than silently dropped.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_raw": raw}


def _openai_tools_fields(
    tools: list[ToolDefinition] | None,
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any]:
    """Render tool defs + choice into OpenAI top-level body fields (empty if none)."""
    if not tools:
        return {}
    fields: dict[str, Any] = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
    }
    if tool_choice is not None:
        fields["tool_choice"] = tool_choice
    return fields


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
            if msg.role is Role.TOOL:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": msg.text or "",
                    }
                )
                continue
            entry: dict[str, Any] = {
                "role": str(msg.role),
                "content": self._build_content(msg, encoded_images),
            }
            if msg.tool_calls:
                # An assistant turn that requested tool calls. OpenAI wants the
                # arguments back as a JSON *string*.
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            out.append(entry)
        return out

    def system_prompt(self, messages: list[Message]) -> str | None:
        # OpenAI embeds system as a regular message; nothing to lift out.
        return None

    def structured_output_fields(self, model: type[BaseModel] | None) -> dict[str, Any]:
        """Return top-level body fields to merge for structured output (empty dict if none)."""
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

    def build_body(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        extra_sampling: dict[str, Any],
        response_model: type[BaseModel] | None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages}
        # system is always None for OpenAI (stays in the messages array)
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body.update(extra_sampling)
        body.update(self.structured_output_fields(response_model))
        # Explicit tools override any tool_use structured-output fields above.
        body.update(_openai_tools_fields(tools, tool_choice))
        if stream:
            body["stream"] = True
            # Ask for usage in the terminal chunk so the limiter can be trued up.
            body["stream_options"] = {"include_usage": True}
        return body

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

    def extract_tool_calls(self, data: dict[str, Any]) -> list[ToolCall]:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return []
        calls = message.get("tool_calls") or []
        out: list[ToolCall] = []
        for i, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            out.append(
                ToolCall(
                    id=call.get("id") or f"call_{i}",
                    name=fn.get("name") or "",
                    arguments=_parse_arguments(fn.get("arguments")),
                )
            )
        return out

    # -------------------------------------------------------------- stream parse

    async def stream_events(
        self, lines: AsyncIterator[str]
    ) -> AsyncIterator[StreamFragment]:
        """Decode an OpenAI SSE stream into normalized fragments.

        OpenAI emits ``data: {chunk}`` lines (and a final ``data: [DONE]``). Each
        chunk's ``choices[0].delta`` carries text and/or tool-call fragments; the
        terminal chunk (requested via ``stream_options.include_usage``) carries
        ``usage`` with an empty ``choices`` array.
        """
        model_seen = False
        async for line in lines:
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if not model_seen and chunk.get("model"):
                model_seen = True
                yield MetaFragment(model=chunk["model"])

            usage = chunk.get("usage")
            if usage:
                yield UsageFragment(
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                )

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            text = delta.get("content")
            if text:
                yield TextFragment(text=text)

            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                yield ToolCallFragment(
                    index=int(tc.get("index", 0) or 0),
                    id=tc.get("id"),
                    name=fn.get("name"),
                    args_delta=fn.get("arguments") or "",
                )

            if choice.get("finish_reason") is not None:
                yield FinishFragment(reason=choice["finish_reason"])

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
        """Return top-level body fields for structured output using output_config.format."""
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

    def build_body(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        extra_sampling: dict[str, Any],
        response_model: type[BaseModel] | None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        if tools or stream:
            raise NotImplementedError(
                "tool calling and streaming are implemented for the OpenAI "
                "formatter only; Anthropic support is a fast-follow"
            )
        body: dict[str, Any] = {"model": model, "messages": messages}
        if system is not None:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body.update(extra_sampling)
        body.update(self.structured_output_fields(response_model))
        return body

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

    def extract_tool_calls(self, data: dict[str, Any]) -> list[ToolCall]:
        # Anthropic tool-call decoding is a fast-follow; not yet wired.
        return []

    async def stream_events(
        self, lines: AsyncIterator[str]
    ) -> AsyncIterator[StreamFragment]:
        raise NotImplementedError(
            "streaming is implemented for the OpenAI formatter only"
        )
        yield  # pragma: no cover - makes this an async generator

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


class GeminiFormatter:
    """Google Gemini generateContent wire format (request + response).

    Endpoint: POST /v1beta/models/{model}:generateContent
    Configure via LLMConfig:
      base_url = "https://generativelanguage.googleapis.com"
      chat_completions_path = "/v1beta/models/gemini-2.0-flash:generateContent"
      message_format = "gemini"
    """

    # ------------------------------------------------------------------ request

    def format_messages(
        self,
        messages: list[Message],
        encoded_images: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Build the ``contents`` array, excluding system messages.

        System text is hoisted to ``systemInstruction`` via ``system_prompt()``.
        Gemini uses ``"model"`` where we use ``"assistant"``.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role is Role.SYSTEM:
                continue
            role = "model" if msg.role is Role.ASSISTANT else str(msg.role)
            parts = self._build_parts(msg, encoded_images)
            out.append({"role": role, "parts": parts})
        return out

    def system_prompt(self, messages: list[Message]) -> str | None:
        texts = [msg.text or "" for msg in messages if msg.role is Role.SYSTEM]
        return "\n\n".join(texts) if texts else None

    def structured_output_fields(self, model: type[BaseModel] | None) -> dict[str, Any]:
        """Return ``generationConfig`` fields for structured output (merged by build_body)."""
        if model is None:
            return {}
        return {
            "responseMimeType": "application/json",
            "responseSchema": model.model_json_schema(),
        }

    def default_headers(self, api_key: str | None) -> dict[str, str]:
        if api_key:
            return {"x-goog-api-key": api_key}
        return {}

    def build_body(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        extra_sampling: dict[str, Any],
        response_model: type[BaseModel] | None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        if tools or stream:
            raise NotImplementedError(
                "tool calling and streaming are implemented for the OpenAI "
                "formatter only; Gemini support is a fast-follow"
            )
        # Model goes in the URL for Gemini, not the body.
        body: dict[str, Any] = {"contents": messages}
        if system is not None:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        gen_config: dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["topP"] = top_p
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        gen_config.update(extra_sampling)
        gen_config.update(self.structured_output_fields(response_model))
        if gen_config:
            body["generationConfig"] = gen_config
        return body

    # ----------------------------------------------------------------- response

    def extract_text(self, data: dict[str, Any]) -> str:
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"unexpected Gemini response shape: {data!r}", cause=exc
            ) from exc
        return "".join(
            p.get("text", "") for p in parts if isinstance(p, dict)
        )

    def extract_usage(self, data: dict[str, Any]) -> tuple[int, int, int]:
        """Return ``(prompt_tokens, completion_tokens, total_tokens)``."""
        um = data.get("usageMetadata") or {}
        prompt = int(um.get("promptTokenCount", 0) or 0)
        completion = int(um.get("candidatesTokenCount", 0) or 0)
        return prompt, completion, prompt + completion

    def finish_reason(self, data: dict[str, Any]) -> str | None:
        try:
            return data["candidates"][0].get("finishReason")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_tool_calls(self, data: dict[str, Any]) -> list[ToolCall]:
        # Gemini tool-call decoding is a fast-follow; not yet wired.
        return []

    async def stream_events(
        self, lines: AsyncIterator[str]
    ) -> AsyncIterator[StreamFragment]:
        raise NotImplementedError(
            "streaming is implemented for the OpenAI formatter only"
        )
        yield  # pragma: no cover - makes this an async generator

    # ------------------------------------------------------------------ private

    def _build_parts(
        self,
        msg: Message,
        encoded_images: dict[str, str],
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if msg.text:
            parts.append({"text": msg.text})
        for url in msg.image_urls:
            data_url = encoded_images.get(url)
            if data_url is None:
                continue
            image_part = _data_url_to_gemini_image(data_url)
            if image_part is not None:
                parts.append(image_part)
        # Gemini requires at least one part per turn.
        if not parts:
            parts = [{"text": ""}]
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


def _data_url_to_gemini_image(data_url: str) -> dict[str, Any] | None:
    """Convert a ``data:image/...;base64,...`` URL to a Gemini inlineData part."""
    try:
        header, b64_data = data_url.split(",", 1)
        media_type = header[5 : header.index(";")]
    except (ValueError, IndexError):
        return None
    return {"inlineData": {"mimeType": media_type, "data": b64_data}}
