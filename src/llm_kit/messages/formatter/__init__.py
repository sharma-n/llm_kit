"""Translate internal ``Message`` objects into provider-specific wire shapes.

One formatter class per provider, each in its own module:
  - ``OpenAIFormatter``    ([openai.py]) — OpenAI chat-completions shape
    (also vLLM, Ollama, Together, …)
  - ``AnthropicFormatter`` ([anthropic.py]) — Anthropic Messages API shape
  - ``GeminiFormatter``    ([gemini.py]) — Google Gemini ``generateContent`` shape

Each owns **both** sides of the wire (invariant #2 in CLAUDE.md):
  - Request: ``format_messages``, ``system_prompt``, ``structured_output_fields``,
    ``default_headers``, ``endpoint_path`` (URL shape, incl. Gemini's streaming
    verb-swap), ``build_body`` (renders ``tools``/``tool_choice``/``stream``)
  - Response: ``extract_text``, ``extract_usage``, ``finish_reason``,
    ``extract_tool_calls``
  - Stream: ``stream_events`` (provider SSE → normalized fragments)

The only cross-provider helper is ``parse_arguments`` in ``_shared.py``; each
provider's ``tool_choice`` mapper, ``functionResponse`` wrapper, and image
converter live next to the class that uses them.

The LLM client is provider-agnostic and delegates all wire-format decisions to
the active formatter. Adding a new provider is a new module + class here —
nothing in the LLM client changes.
"""

from __future__ import annotations

from llm_kit.messages.formatter.anthropic import AnthropicFormatter
from llm_kit.messages.formatter.gemini import GeminiFormatter
from llm_kit.messages.formatter.openai import OpenAIFormatter

__all__ = ["AnthropicFormatter", "GeminiFormatter", "OpenAIFormatter"]
