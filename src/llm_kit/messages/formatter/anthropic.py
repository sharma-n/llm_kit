"""Anthropic Messages API wire format (request + response)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from llm_kit.errors import ProviderError
from llm_kit.messages.formatter._shared import parse_arguments
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


def _anthropic_tool_choice(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Map a provider-agnostic ``tool_choice`` onto Anthropic's shape.

    Dicts pass through (the caller supplied an Anthropic-shaped choice). The
    control words map to Anthropic's vocabulary; any other string is treated as
    a specific tool name.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, dict):
        return tool_choice
    control = {
        "auto": {"type": "auto"},
        "required": {"type": "any"},
        "any": {"type": "any"},
        "none": {"type": "none"},
    }
    return control.get(tool_choice, {"type": "tool", "name": tool_choice})


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


def _enforce_no_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add ``additionalProperties: false`` to every object in a JSON schema.

    Anthropic's structured-output API rejects schemas where any object type
    omits this field; Pydantic's ``model_json_schema()`` does not set it.
    """
    schema = dict(schema)
    if schema.get("type") == "object":
        ap = schema.get("additionalProperties")
        if ap is None or isinstance(ap, dict):
            schema["additionalProperties"] = False
    for key in ("properties", "$defs", "definitions"):
        if isinstance(schema.get(key), dict):
            schema[key] = {k: _enforce_no_additional_properties(v) for k, v in schema[key].items()}
    if isinstance(schema.get("items"), dict):
        schema["items"] = _enforce_no_additional_properties(schema["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            schema[key] = [_enforce_no_additional_properties(s) for s in schema[key]]
    return schema


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

        Two tool-specific shapes differ from OpenAI:
          - an assistant turn that called tools renders its ``tool_calls`` as
            ``tool_use`` content blocks (``input`` is a dict, not a JSON string);
          - a tool-result (``Role.TOOL``) turn becomes a ``tool_result`` block
            inside a **user** message. Adjacent tool results are coalesced into a
            single user turn, since Anthropic expects one user message to carry
            all results for the preceding assistant's tool calls.
        """
        out: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            if pending_results:
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for msg in messages:
            if msg.role is Role.SYSTEM:
                continue
            if msg.role is Role.TOOL:
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.text or "",
                    }
                )
                continue
            flush_results()
            content = self._build_content(msg, encoded_images)
            if msg.tool_calls:
                blocks: list[dict[str, Any]] = (
                    list(content)
                    if isinstance(content, list)
                    else ([{"type": "text", "text": content}] if content else [])
                )
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                    for tc in msg.tool_calls
                )
                content = blocks
            out.append({"role": str(msg.role), "content": content})
        flush_results()
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
                    "schema": _enforce_no_additional_properties(model.model_json_schema()),
                }
            }
        }

    def default_headers(self, api_key: str | None) -> dict[str, str]:
        headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def endpoint_path(self, path: str, *, stream: bool) -> str:
        # Anthropic streams from the same URL (``stream: true`` lives in the body).
        return path

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
        if tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
            choice = _anthropic_tool_choice(tool_choice)
            if choice is not None:
                body["tool_choice"] = choice
        if stream:
            body["stream"] = True
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
        content = data.get("content")
        if not isinstance(content, list):
            return []
        out: list[ToolCall] = []
        for i, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            out.append(
                ToolCall(
                    id=block.get("id") or f"call_{i}",
                    name=block.get("name") or "",
                    # Anthropic returns ``input`` already parsed as a dict.
                    arguments=parse_arguments(block.get("input")),
                )
            )
        return out

    async def stream_events(
        self, lines: AsyncIterator[str]
    ) -> AsyncIterator[StreamFragment]:
        """Decode an Anthropic Messages SSE stream into normalized fragments.

        Anthropic emits typed events (``event:`` + ``data:`` line pairs); we
        dispatch on the ``type`` field inside each ``data:`` JSON, so the
        ``event:`` lines can be ignored. The shapes that matter:

          - ``message_start``        — carries ``model`` and ``usage.input_tokens``
          - ``content_block_start``  — a ``tool_use`` block announces its id + name
          - ``content_block_delta``  — ``text_delta`` (text) or ``input_json_delta``
            (a piece of the tool's argument JSON, grouped by the block ``index``)
          - ``message_delta``        — ``stop_reason`` + final ``usage.output_tokens``
          - ``message_stop``         — end of stream
        """
        input_tokens = 0
        model_seen = False
        async for line in lines:
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")

            if etype == "message_start":
                message = event.get("message") or {}
                model = message.get("model")
                if model and not model_seen:
                    model_seen = True
                    yield MetaFragment(model=model)
                usage = message.get("usage") or {}
                input_tokens = int(usage.get("input_tokens", 0) or 0)

            elif etype == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    yield ToolCallFragment(
                        index=int(event.get("index", 0) or 0),
                        id=block.get("id"),
                        name=block.get("name"),
                        args_delta="",
                    )

            elif etype == "content_block_delta":
                delta = event.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    text = delta.get("text")
                    if text:
                        yield TextFragment(text=text)
                elif dtype == "input_json_delta":
                    yield ToolCallFragment(
                        index=int(event.get("index", 0) or 0),
                        args_delta=delta.get("partial_json") or "",
                    )

            elif etype == "message_delta":
                usage = event.get("usage") or {}
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                yield UsageFragment(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )
                reason = (event.get("delta") or {}).get("stop_reason")
                if reason is not None:
                    yield FinishFragment(reason=reason)

            elif etype == "message_stop":
                break

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
