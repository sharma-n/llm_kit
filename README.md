# llm_kit

A provider-agnostic, async, high-throughput toolkit for multimodal LLM applications.

Bring your own endpoint — OpenAI, Anthropic, Google Gemini, or anything else that speaks one of their wire formats — your own API key, and a YAML config. `llm_kit` handles rate limiting, image fetching + base64 encoding, retries, structured outputs, token streaming, tool-call parsing, 1M-scale concurrent batching, and provider-native 24h batch jobs (≈50% cheaper).

## Why this exists

Most LLM client libraries assume one provider, one model, and synchronous calls. When you're building agentic systems or running large multimodal batches, you usually need:

- **Provider freedom** — swap OpenAI for vLLM / Together / Anthropic / a local Ollama instance with a config change, not a code change.
- **Multimodal by default** — every message is an ordered list of text and image parts (URLs, `data:` URLs, or raw bytes); images are fetched/encoded and inlined transparently, preserving order.
- **Real throughput** — async, persistent HTTPS connections, RPM and TPM rate limits respected, hundreds of concurrent requests.
- **Big batches without OOM** — run a million prompts through and stream results to disk as they complete.

That's what this is.

## Install

This is a `uv`-managed project on Python 3.13+. Supported (and tested in CI): Python 3.13 and 3.14.

```bash
uv sync               # production deps only
uv sync --extra dev   # adds pytest + pytest-asyncio
```

The package is `llm_kit`.

## Quick start

### Single request

```python
import asyncio
from llm_kit import ImagePart, Message, LLMClient

async def main():
    async with LLMClient.from_yaml("examples/config.yaml") as llm:
        response = await llm.invoke([
            Message.system("You are a terse assistant."),
            Message.user("In one sentence, what is the capital of France?"),
        ])
    print(response.text)
    print(response.usage)        # TokenUsage(prompt=..., completion=..., total=...)

asyncio.run(main())
```

Set `OPENAI_API_KEY` (or whichever env var your config points at) and run with `uv run python examples/single_invoke.py`.

### Single request with images

Messages are ordered lists of text and image parts (`Message.user` turns `str` arguments into `TextPart`s). An `ImagePart` is a URL (fetched and base64-encoded before sending), a `data:` URL (passed through), or raw bytes (`ImagePart.from_bytes(data, "image/png")`, encoded once, never fetched). Order is preserved on the wire.

```python
response = await llm.invoke([
    Message.system("Describe the image in one sentence."),
    Message.user(
        "Caption this:",
        ImagePart.from_url("https://example.com/cat.jpg"),
        "and compare it with:",
        ImagePart.from_url("https://example.com/dog.png"),
    ),
])
```

### Structured outputs (one model does double duty)

Pass a pydantic model — it's both injected as the provider's native structured-output schema in the request AND used to validate the response. The validated object lands at `response.parsed`.

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    confidence: float

response = await llm.invoke(
    [Message.user("how do you feel?")],
    response_model=Sentiment,
)
print(response.parsed.label, response.parsed.confidence)
```

If the model rejects the response, you get a typed `ValidationError` (subclass of `LLMError`) with the raw content attached for debugging.

**JSON schema dicts.** Anywhere a `response_model` is accepted (`invoke`, `invoke_stream`, `BatchItem`) you can pass a plain JSON schema `dict` instead. The schema goes to the provider the same way, but there is no Pydantic validation: `response.parsed` is just the decoded JSON (`dict`/`list`/...). Text that isn't valid JSON raises `ValidationError` (a `BatchResult.error` in batches). The schema's `"title"` (default `"response"`) names it where a provider needs a name; your dict is never mutated. Gemini receives both kinds as `generationConfig.responseJsonSchema`, so `$ref`/`$defs` work.

```python
schema = {
    "title": "Verdict",
    "type": "object",
    "properties": {"label": {"type": "string"}, "score": {"type": "number"}},
    "required": ["label"],
}
response = await llm.invoke([Message.user("verdict?")], response_model=schema)
print(response.parsed["label"])
```

### Per-request settings and reasoning

`max_tokens`, `temperature` and `top_p` default to `LLMConfig`, but any call or batch item can override them with `RequestSettings`; `None` fields inherit. `reasoning` asks the model to think, and each formatter renders it natively:

```python
from llm_kit import BatchItem, Reasoning, ReasoningEffort, RequestSettings

settings = RequestSettings(
    temperature=0.0,
    max_tokens=2000,
    reasoning=Reasoning(effort=ReasoningEffort.HIGH),   # or Reasoning(budget_tokens=8000)
)
await llm.invoke(messages, settings=settings)
BatchItem(id="a", messages=messages, settings=settings)   # per item, in one batch
```

| Provider | `effort` | `budget_tokens` |
|---|---|---|
| OpenAI | `reasoning_effort` | mapped to effort (<=1024 minimal, <=4096 low, <=16384 medium, else high) |
| Anthropic | mapped to a budget (minimal 1024, low 2048, medium 8192, high 24576) | `thinking: {type: enabled, budget_tokens}` |
| Gemini | `generationConfig.thinkingConfig.thinkingLevel` (always a level, never `thinkingBudget`) | mapped to a level as for OpenAI |

If both are given, OpenAI/Gemini use `effort` and Anthropic uses `budget_tokens`. llm_kit does not strip `temperature` or adjust `max_tokens` for Anthropic thinking; the provider validates. Thinking content is never part of `response.text` (Gemini `thought` parts and Anthropic `thinking` blocks are dropped). `response.usage.reasoning_tokens` reports thinking tokens where the provider does (OpenAI, Gemini; Anthropic reports 0), and `total_tokens` always includes them. OpenAI includes them in `completion_tokens`; Gemini reports them separately from `completion_tokens`. `estimated_cost` prices output as `total_tokens - prompt_tokens` per item (`TokenUsage.output_tokens`), so thinking is billed at the completion rate exactly once for every provider.

Precedence is `LLMConfig` < `settings` < `overrides`. `overrides` is deep-merged into the request body (dicts merge recursively; lists and scalars replace), so `overrides={"generationConfig": {"temperature": 0.9}}` leaves the response schema in place.

### Streaming

`invoke_stream` streams a single call as a sequence of **typed events** rather than bare tokens — so the same stream can carry text *and* tool calls. Forward `TextChunk`s to your user as they arrive (time-to-first-token); the terminal `StreamEnd` carries the fully-assembled `LLMResponse` (text, usage, finish reason, validated `response_model`).

```python
from llm_kit import TextChunk, StreamEnd

async for event in llm.invoke_stream([Message.user("Write a haiku about latency.")]):
    if isinstance(event, TextChunk):
        print(event.text, end="", flush=True)
    elif isinstance(event, StreamEnd):
        print(f"\n{event.response.usage}")
```

Streaming is **single-call only** — the throughput-oriented batch path never streams. Retries cover establishing the stream; once bytes start flowing, a mid-stream failure propagates rather than silently replaying.

### Tool calling — `llm_kit` parses, you execute

Pass `tools` to `invoke` or `invoke_stream` and `llm_kit` renders them into the provider's native tool slot, then **decodes** the model's response into provider-agnostic `ToolCall`s (id, name, parsed `arguments`). It does *not* run them — executing tools, the multi-step loop, and feeding results back are your application's job. The dividing line is **decode vs. execute**.

```python
from llm_kit import ToolDefinition, Message

weather = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)

resp = await llm.invoke([Message.user("Weather in Paris?")], tools=[weather])
for call in resp.tool_calls:
    print(call.name, call.arguments)        # get_weather {'city': 'Paris'}
    result = run_my_tool(call.name, call.arguments)
    # feed the observation back on the next turn:
    follow_up = [
        Message.user("Weather in Paris?"),
        Message.assistant_tool_calls(resp.tool_calls),
        Message.tool_result(call.id, result),
    ]
```

The same `tools=` argument works with `invoke_stream`; a `ToolCallStarted` event fires as soon as a call's name is known (a UI hint), and the assembled calls land on `StreamEnd.response.tool_calls`. See [examples/stream_invoke.py](examples/stream_invoke.py).

> Streaming and tool parsing are implemented for **all three** wire formats — OpenAI, Anthropic, and Gemini. For a stateful agent built *on* this surface — memory, multi-turn, an MCP tool loop — see the `agent_kit` sibling project.

### Batch and embeddings

For many prompts, build a `list[BatchItem]` and stream results as they complete — the same pipeline drives an async iterator, a resumable JSONL sink, or a per-result callback, in O(buffer) memory (not O(N)):

```python
from llm_kit import BatchItem

items = [BatchItem(id=f"item-{i}", messages=[Message.user(q)]) for i, q in enumerate(prompts)]
written = await llm.batch_to_jsonl(items, "out.jsonl", resume=True)
```

There's also a provider-native 24h Batch API (≈50% cheaper) and an embeddings client with input micro-batching. See **[docs/batch.md](docs/batch.md)** and **[docs/embeddings.md](docs/embeddings.md)**.

Gemini on **Vertex AI** (GCS-based batch prediction, Application Default Credentials) is available as the `vertex` batch backend behind an optional extra: `pip install "llm_kit[vertex]"` (`uv sync --extra vertex`), then `provider_batch.backend: vertex` with `llm.message_format: gemini`. See [docs/batch.md](docs/batch.md#vertex-ai-gemini-batch-backend); a manual real-project smoke test is in [docs/vertex_smoke_test.md](docs/vertex_smoke_test.md).

### Testing offline batches without a network

`llm_kit.testing.FakeBatchBackend` is an in-memory provider batch backend. Inject it and the whole `run_batch_api` / `submit_batch` / `fetch_batch_results` flow runs with no HTTP:

```python
from llm_kit import LLMClient
from llm_kit.testing import FakeBatchBackend

fake = FakeBatchBackend(polls_until_done=2, fail_items={"item-3": "boom"}, shuffle=True)
llm = LLMClient(config, batch_backend=fake)
await llm.run_batch_api(items, "out.jsonl", poll_interval=0)
fake.submitted            # {batch_id: [BatchRequest, ...]} as the backend received them
```

Use `responder=` (callable or `{custom_id: body}`) or `FakeBatchBackend.text_responder(fmt, fn)` to shape responses, `fail_job=BatchState.EXPIRED` (with `raw_state=`, `error_detail=`, `partial_output=`) for whole-job failure, and `states=[BatchState.PAUSED, ...]` to script poll states. Fetching from a running job raises `BatchJobNotFinishedError` (or pass `wait=True`); a failed/expired/cancelled one raises `BatchJobFailedError` (or pass `allow_partial=True`). To bypass `BatchItem`, use `llm.submit_requests([BatchRequest(id, body)])` and `llm.fetch_raw_results(job)`. Details in [docs/batch.md](docs/batch.md).

## Documentation

| Topic | |
|---|---|
| **[Configuration](docs/configuration.md)** | YAML schema, env-var interpolation, provider portability (OpenAI / Anthropic / Gemini) |
| **[Batching](docs/batch.md)** | The three live sinks, crash resume, and the provider-native 24h Batch API |
| **[Embeddings](docs/embeddings.md)** | The embeddings client and input micro-batching |
| **[Rate limiting, images & benchmarks](docs/performance.md)** | The three rate-limit gates, image-fetcher tuning, and load-test results |
| **[Errors](docs/errors.md)** | The typed `LLMError` hierarchy and retry classification |

## Examples

- [examples/single_invoke.py](examples/single_invoke.py) — one chat call
- [examples/stream_invoke.py](examples/stream_invoke.py) — token streaming + tool-call parsing
- [examples/batch_structured.py](examples/batch_structured.py) — small batch with per-item pydantic schemas and image inputs
- [examples/embed_batch.py](examples/embed_batch.py) — single + micro-batched embeddings
- [examples/provider_batch.py](examples/provider_batch.py) — provider-native 24h batch API (submit / poll / fetch)

## Tests

```bash
uv run pytest                                       # all tests, ~1.5s
uv run pytest tests/test_batch.py                   # one module
uv run pytest tests/test_batch.py::test_per_item_failure_is_recorded_default
```

The suite uses `httpx.MockTransport` for wire-level coverage and a `FakeLLM` (in [tests/conftest.py](tests/conftest.py)) for batch-driver coverage. No network access needed.

## Project layout

```
src/llm_kit/
  config/        # dataclasses + YAML loader (env-var interpolation)
  http/          # shared httpx.AsyncClient + retry wrapper (incl. streaming retry)
  rate_limit/    # token bucket + composed limiter (sem + RPM + TPM)
  images/        # async fetch + LRU cache + base64 encode
  messages/      # Message + tool types + OpenAI / Anthropic / Gemini wire-format adapters (incl. stream decode)
  llm/           # BaseLLM + LLMClient (invoke / invoke_stream) + LLMResponse + shared response/stream parsing
  embed/         # OpenAICompatibleEmbedder + micro-batching driver + jsonl
  batch/         # bounded live pipeline + jsonl writer + ID-based resume
  provider_batch/# offline 24h Batch API driver (OpenAI / Anthropic / Gemini)
  errors.py      # typed exception hierarchy
examples/        # runnable examples
docs/            # reference docs (configuration, batch, embeddings, performance, errors)
tests/           # pytest suite (190 tests)
benchmarks/      # load tests
```

## License

MIT. See [LICENSE](LICENSE).
