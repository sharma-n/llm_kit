# Configuration

[← back to README](../README.md)

Everything is driven by a YAML file loaded into [`AppConfig`](../src/llm_kit/config/app.py). Environment-variable interpolation with `${VAR}` and `${VAR:-default}` works inside string values.

The full schema is in [examples/config.yaml](../examples/config.yaml). The sections:

| Section | Knobs |
|---|---|
| `llm` | `base_url`, `model`, `api_key_env`, sampling defaults, `message_format`, `structured_output_style` |
| `embed` | `base_url`, `model`, `api_key_env`, `dimensions`, `encoding_format`, `max_inputs_per_request`, `max_tokens_per_request` |
| `http` | connect/read/pool timeouts, pool sizes, HTTP/2, retry policy for the LLM call |
| `rate_limit` | `requests_per_minute`, `tokens_per_minute`, `max_concurrent_requests` |
| `image` | `max_concurrent_downloads`, `cache_size`, `on_failure`, retry policy for image fetches |
| `batch` | `worker_concurrency`, `input_buffer_size`, `on_failure`, progress logging |
| `provider_batch` | `completion_window`, `files_path`, `batches_path`, `poll_interval_seconds`, `max_requests_per_batch` (the offline 24h Batch API driver) |

A minimal config (only override what you need):

```yaml
llm:
  base_url: https://api.openai.com
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
rate_limit:
  requests_per_minute: 1000
  tokens_per_minute: 1500000
  max_concurrent_requests: 64
```

All other fields take their dataclass defaults.

## Provider portability

`message_format` selects the wire format. Three values are supported:

**OpenAI and OpenAI-compatible endpoints** (vLLM, Ollama, Together, …) — the default:

```yaml
llm:
  base_url: https://api.together.xyz       # or http://localhost:11434 for Ollama
  model: meta-llama/Llama-3.3-70B-Instruct
  api_key_env: TOGETHER_API_KEY
  message_format: openai
  structured_output_style: openai_json_schema  # or tool_use, or none
```

`structured_output_style` fine-tunes how structured outputs are expressed for OpenAI-compatible endpoints; it has no effect on Anthropic or Gemini.

**Anthropic:**

```yaml
llm:
  base_url: https://api.anthropic.com
  chat_completions_path: /v1/messages
  model: claude-opus-4-8
  api_key_env: ANTHROPIC_API_KEY
  message_format: anthropic
```

**Google Gemini:**

```yaml
llm:
  base_url: https://generativelanguage.googleapis.com
  chat_completions_path: /v1beta/models/gemini-2.0-flash:generateContent
  api_key_env: GEMINI_API_KEY
  message_format: gemini
```

For Gemini the model name lives in `chat_completions_path` (not in `model`), because Gemini embeds it in the URL.

> Token streaming and tool-call parsing are implemented for **all three** wire formats — OpenAI, Anthropic, and Gemini. (Gemini streams from a different URL — `:streamGenerateContent?alt=sse` — which the formatter derives automatically; no extra config needed.) All three also support non-streaming `invoke`, structured outputs, and the batch paths.
