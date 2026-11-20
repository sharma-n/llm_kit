"""OpenAI chat-completions wire format (also vLLM, Ollama, Together, …)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from llm_kit.config.llm import StructuredOutputStyle
from llm_kit.errors import ProviderError
from llm_kit.messages.formatter._shared import message_text, parse_arguments, resolve_effort
from llm_kit.messages.schema import ResponseSchema, json_schema, schema_name
from llm_kit.messages.settings import Reasoning
from llm_kit.messages.streaming import (
    FinishFragment,
    MetaFragment,
    StreamFragment,
    TextFragment,
    ToolCallFragment,
    UsageFragment,
)
from llm_kit.messages.tools import ToolCall, ToolDefinition
from llm_kit.messages.types import ImagePart, Message, Role, TextPart


def _strictify_schema(node: Any) -> Any:
    """Make a Pydantic JSON schema satisfy OpenAI strict structured output.

    OpenAI ``strict: true`` requires every ``object`` to set
    ``additionalProperties: false`` *and* list every property key in
    ``required`` — Pydantic's ``model_json_schema()`` does neither. We walk the
    schema (including ``$defs``, array ``items``, and ``anyOf``/``allOf``/
    ``oneOf`` combinators) and patch each object node in place. Returns ``node``
    so callers can wrap it inline.
    """
    if isinstance(node, dict):
        if node.get("type") == "object":
            if "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            elif isinstance(node.get("additionalProperties"), dict):
                node["additionalProperties"] = False
        for key in ("properties", "$defs", "definitions"):
            section = node.get(key)
            if isinstance(section, dict):
                for sub in section.values():
                    _strictify_schema(sub)
        for key in ("items", "additionalProperties", "anyOf", "allOf", "oneOf", "prefixItems"):
            _strictify_schema(node.get(key))
    elif isinstance(node, list):
        for item in node:
            _strictify_schema(item)
    return node


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


def _openai_usage(usage: Any) -> tuple[int, int, int, int]:
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens", 0) or 0)
    return prompt, completion, total, reasoning


class OpenAIFormatter:
    """OpenAI chat-completions wire format (request + response)."""

    def __init__(self, *, structured_output_style: StructuredOutputStyle) -> None:
        self._style = structured_output_style

    # ------------------------------------------------------------------ request

    def format_messages(
        self,
        messages: list[Message],
        encoded_images: dict[ImagePart, str],
    ) -> list[dict[str, Any]]:
        """Build the ``messages`` array.

        ``encoded_images`` maps each ``ImagePart`` → ``"data:image/...;base64,..."``.
        Parts absent from the map are skipped (the image failure policy already
        decided what to do; we honour whatever it produced). Parts are emitted
        in the order given, one block per part.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role is Role.TOOL:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": message_text(msg),
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

    def structured_output_fields(self, model: ResponseSchema | None) -> dict[str, Any]:
        """Return top-level body fields to merge for structured output (empty dict if none)."""
        if model is None or self._style is StructuredOutputStyle.NONE:
            return {}
        if self._style is StructuredOutputStyle.OPENAI_JSON_SCHEMA:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name(model),
                        "schema": _strictify_schema(json_schema(model)),
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
                            "name": schema_name(model),
                            "parameters": json_schema(model),
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": schema_name(model)},
                },
            }
        raise ValueError(f"Unknown structured output style: {self._style}")

    def default_headers(self, api_key: str | None) -> dict[str, str]:
        if api_key:
            return {"authorization": f"Bearer {api_key}"}
        return {}

    def endpoint_path(self, path: str, *, stream: bool) -> str:
        # OpenAI streams from the same URL (``stream: true`` lives in the body).
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
        response_model: ResponseSchema | None,
        reasoning: Reasoning | None = None,
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
        effort = resolve_effort(reasoning)
        if effort is not None:
            body["reasoning_effort"] = effort.value
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

    def extract_usage(self, data: dict[str, Any]) -> tuple[int, int, int, int]:
        """Return ``(prompt, completion, total, reasoning)`` token counts.

        OpenAI's ``completion_tokens`` and ``total_tokens`` already include the
        reasoning tokens reported under ``completion_tokens_details``.
        """
        return _openai_usage(data.get("usage"))

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
                    arguments=parse_arguments(fn.get("arguments")),
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
                prompt, completion, total, reasoning = _openai_usage(usage)
                yield UsageFragment(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=total,
                    reasoning_tokens=reasoning,
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
        encoded_images: dict[ImagePart, str],
    ) -> str | list[dict[str, Any]]:
        # A lone text part (or none) uses the simple string form — smaller
        # payload, and some providers reject content-arrays for system messages.
        if all(isinstance(p, TextPart) for p in msg.parts) and len(msg.parts) <= 1:
            return msg.parts[0].text if msg.parts else ""  # type: ignore[union-attr]
        blocks: list[dict[str, Any]] = []
        for part in msg.parts:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": part.text})
                continue
            data_url = encoded_images.get(part)
            if data_url is None:
                continue
            blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        return blocks
