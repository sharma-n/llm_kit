"""Google Gemini generateContent wire format (request + response)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from llm_kit.errors import ProviderError
from llm_kit.messages.formatter._shared import message_text, parse_arguments, resolve_effort
from llm_kit.messages.schema import ResponseSchema, json_schema
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


def _gemini_tool_config(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Map a provider-agnostic ``tool_choice`` onto Gemini's ``toolConfig`` shape.

    Dicts pass through (caller supplied a Gemini-shaped ``toolConfig``). The
    control words map onto ``functionCallingConfig.mode``; any other string is a
    specific tool name, expressed as ``ANY`` mode restricted to that function.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, dict):
        return tool_choice
    mode_map = {"auto": "AUTO", "required": "ANY", "any": "ANY", "none": "NONE"}
    mode = mode_map.get(tool_choice)
    if mode is not None:
        return {"functionCallingConfig": {"mode": mode}}
    return {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": [tool_choice],
        }
    }


def _gemini_tool_response(content: str) -> dict[str, Any]:
    """Wrap a tool-result string in the object Gemini's ``functionResponse`` wants.

    Gemini requires ``response`` to be an object. If the observation is itself a
    JSON object, use it directly; otherwise wrap it under ``result``.
    """
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"result": content}
    return value if isinstance(value, dict) else {"result": value}


def _data_url_to_gemini_image(data_url: str) -> dict[str, Any] | None:
    """Convert a ``data:image/...;base64,...`` URL to a Gemini inlineData part."""
    try:
        header, b64_data = data_url.split(",", 1)
        media_type = header[5 : header.index(";")]
    except (ValueError, IndexError):
        return None
    return {"inlineData": {"mimeType": media_type, "data": b64_data}}


def _gemini_usage(um: Any) -> tuple[int, int, int, int]:
    um = um or {}
    prompt = int(um.get("promptTokenCount", 0) or 0)
    completion = int(um.get("candidatesTokenCount", 0) or 0)
    reasoning = int(um.get("thoughtsTokenCount", 0) or 0)
    total = int(um.get("totalTokenCount", 0) or 0) or prompt + completion + reasoning
    return prompt, completion, total, reasoning


class GeminiFormatter:
    """Google Gemini generateContent wire format (request + response).

    Endpoint: POST /v1beta/models/{model}:generateContent
    Streaming swaps the verb and SSE flag: :streamGenerateContent?alt=sse
    (see ``endpoint_path``).
    Configure via LLMConfig:
      base_url = "https://generativelanguage.googleapis.com"
      chat_completions_path = "/v1beta/models/gemini-2.0-flash:generateContent"
      message_format = "gemini"
    """

    # ------------------------------------------------------------------ request

    def format_messages(
        self,
        messages: list[Message],
        encoded_images: dict[ImagePart, str],
    ) -> list[dict[str, Any]]:
        """Build the ``contents`` array, excluding system messages.

        System text is hoisted to ``systemInstruction`` via ``system_prompt()``.
        Gemini uses ``"model"`` where we use ``"assistant"``.

        Tool handling differs from OpenAI in two ways:
          - an assistant turn that called tools renders its ``tool_calls`` as
            ``functionCall`` parts (``args`` is a dict, like Anthropic's ``input``);
          - Gemini has no tool/function *role* — only ``user``/``model`` — so a
            tool-result (``Role.TOOL``) turn becomes a ``functionResponse`` part
            inside a **user** message, and adjacent results coalesce into one user
            turn. Gemini matches a result to its call by function **name**, not id,
            so we recover the name from the preceding assistant turn's calls.
        """
        out: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []
        names_by_id: dict[str, str] = {}

        def flush_results() -> None:
            if pending_results:
                out.append({"role": "user", "parts": list(pending_results)})
                pending_results.clear()

        for msg in messages:
            if msg.role is Role.SYSTEM:
                continue
            if msg.role is Role.TOOL:
                call_id = msg.tool_call_id or ""
                pending_results.append(
                    {
                        "functionResponse": {
                            "name": names_by_id.get(call_id, call_id),
                            "response": _gemini_tool_response(message_text(msg)),
                        }
                    }
                )
                continue
            flush_results()
            role = "model" if msg.role is Role.ASSISTANT else str(msg.role)
            parts = self._build_parts(msg, encoded_images)
            if msg.tool_calls:
                # Drop the lone empty-text placeholder when the turn is calls-only.
                if parts == [{"text": ""}]:
                    parts = []
                for tc in msg.tool_calls:
                    names_by_id[tc.id] = tc.name
                    parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
            out.append({"role": role, "parts": parts})
        flush_results()
        return out

    def system_prompt(self, messages: list[Message]) -> str | None:
        texts = [message_text(msg) for msg in messages if msg.role is Role.SYSTEM]
        return "\n\n".join(texts) if texts else None

    def structured_output_fields(self, model: ResponseSchema | None) -> dict[str, Any]:
        """Return ``generationConfig`` fields for structured output (merged by build_body).

        Both Pydantic classes and dict schemas go in ``responseJsonSchema`` (not
        the OpenAPI-subset ``responseSchema``) so ``$ref`` / ``$defs`` work.
        """
        if model is None:
            return {}
        return {
            "responseMimeType": "application/json",
            "responseJsonSchema": json_schema(model),
        }

    def default_headers(self, api_key: str | None) -> dict[str, str]:
        if api_key:
            return {"x-goog-api-key": api_key}
        return {}

    def endpoint_path(self, path: str, *, stream: bool) -> str:
        """Adjust the configured path for streaming.

        Unlike OpenAI/Anthropic, Gemini streams from a different action verb
        (``:streamGenerateContent`` instead of ``:generateContent``) and requires
        ``?alt=sse`` to receive SSE frames rather than a single JSON array. The
        URL is part of Gemini's wire contract, so this lives in the formatter
        (invariant #2), not in the client.
        """
        if not stream:
            return path
        base, _, query = path.partition("?")
        base = base.replace(":generateContent", ":streamGenerateContent")
        params = f"{query}&alt=sse" if query else "alt=sse"
        return f"{base}?{params}"

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
        # ``stream`` does not affect the Gemini body — it is expressed entirely in
        # the URL (``:streamGenerateContent?alt=sse``), handled by endpoint_path.
        # Model also goes in the URL for Gemini, not the body.
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
        effort = resolve_effort(reasoning)
        if effort is not None:
            # Gemini 3.x+ takes ``thinkingLevel`` (``thinkingBudget`` is gone);
            # merge so an ``includeThoughts`` from extra_sampling survives.
            thinking = dict(gen_config.get("thinkingConfig") or {})
            thinking["thinkingLevel"] = effort.value
            gen_config["thinkingConfig"] = thinking
        if gen_config:
            body["generationConfig"] = gen_config
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                }
            ]
            tool_config = _gemini_tool_config(tool_choice)
            if tool_config is not None:
                body["toolConfig"] = tool_config
        return body

    # ----------------------------------------------------------------- response

    def extract_text(self, data: dict[str, Any]) -> str:
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"unexpected Gemini response shape: {data!r}", cause=exc
            ) from exc
        # Skip thought-summary parts (``"thought": true``): they are not the answer.
        return "".join(
            p.get("text", "")
            for p in parts
            if isinstance(p, dict) and not p.get("thought")
        )

    def extract_usage(self, data: dict[str, Any]) -> tuple[int, int, int, int]:
        """Return ``(prompt, completion, total, reasoning)``.

        ``candidatesTokenCount`` excludes thinking; ``thoughtsTokenCount`` is
        reported separately. ``total`` prefers the provider's
        ``totalTokenCount``, else prompt + candidates + thoughts.
        """
        return _gemini_usage(data.get("usageMetadata"))

    def finish_reason(self, data: dict[str, Any]) -> str | None:
        try:
            return data["candidates"][0].get("finishReason")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_tool_calls(self, data: dict[str, Any]) -> list[ToolCall]:
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            return []
        out: list[ToolCall] = []
        for i, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            fc = part.get("functionCall")
            if not isinstance(fc, dict):
                continue
            out.append(
                ToolCall(
                    # Gemini function calls carry no id in the baseline wire; an
                    # optional ``id`` appears only with parallel calling. Synthesize
                    # a stable one when absent.
                    id=fc.get("id") or f"call_{i}",
                    name=fc.get("name") or "",
                    # ``args`` is already a parsed dict.
                    arguments=parse_arguments(fc.get("args")),
                )
            )
        return out

    async def stream_events(
        self, lines: AsyncIterator[str]
    ) -> AsyncIterator[StreamFragment]:
        """Decode a Gemini ``streamGenerateContent`` SSE stream into fragments.

        With ``?alt=sse`` each event is a ``data: {GenerateContentResponse}`` line.
        Each chunk's ``candidates[0].content.parts`` carries text and/or
        ``functionCall`` parts; ``usageMetadata`` is cumulative (the last chunk
        holds the final totals), and ``finishReason`` lands on the terminal chunk.

        Unlike OpenAI/Anthropic, a streamed ``functionCall`` arrives **whole** in
        one chunk (we do not set ``streamFunctionCallArguments``), so each becomes
        a single ``ToolCallFragment`` carrying its complete (dict) args as JSON.
        """
        model_seen = False
        tool_index = 0
        async for line in lines:
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":  # not emitted by Gemini, tolerated anyway
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            model = chunk.get("modelVersion")
            if model and not model_seen:
                model_seen = True
                yield MetaFragment(model=model)

            candidates = chunk.get("candidates") or []
            if candidates:
                candidate = candidates[0]
                parts = (candidate.get("content") or {}).get("parts") or []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if text and not part.get("thought"):
                        yield TextFragment(text=text)
                    fc = part.get("functionCall")
                    if isinstance(fc, dict):
                        yield ToolCallFragment(
                            index=tool_index,
                            id=fc.get("id"),
                            name=fc.get("name"),
                            args_delta=json.dumps(fc.get("args") or {}),
                        )
                        tool_index += 1
                reason = candidate.get("finishReason")
                if reason is not None:
                    yield FinishFragment(reason=reason)

            usage = chunk.get("usageMetadata")
            if usage:
                prompt, completion, total, reasoning = _gemini_usage(usage)
                yield UsageFragment(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=total,
                    reasoning_tokens=reasoning,
                )

    # ------------------------------------------------------------------ private

    def _build_parts(
        self,
        msg: Message,
        encoded_images: dict[ImagePart, str],
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for part in msg.parts:
            if isinstance(part, TextPart):
                if part.text:  # Gemini rejects empty text parts
                    parts.append({"text": part.text})
                continue
            data_url = encoded_images.get(part)
            if data_url is None:
                continue
            image_part = _data_url_to_gemini_image(data_url)
            if image_part is not None:
                parts.append(image_part)
        # Gemini requires at least one part per turn.
        if not parts:
            parts = [{"text": ""}]
        return parts
