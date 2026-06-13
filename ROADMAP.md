# Roadmap — extension ideas

## What this kit is

`llm_kit` is a **stateless, high-throughput batch engine** for chat completions
and embeddings. Its load-bearing strength is the bounded batch pipeline
(O(buffer), not O(N); resumable from the output jsonl) plus strict bottom-up
layering (`config → http → rate_limit/images/messages → llm/embed → batch`).

The guiding principle for everything below is one question:

> **Does it make a large batch job cheaper, faster, more reliable, or more
> resumable?**

That is the bar. Capabilities that only make the kit more *general-purpose*
(swappable everything, broad integrations, the version churn and leaky
abstractions that come with them) are a different product and aren't tracked
here.

One boundary is worth stating: `llm_kit` **parses** tool calls —
pure wire decoding, which is squarely the formatter's job (invariant #2) — but
it does **not execute** them. The execution loop, tool-result feedback, and
bounded multi-step orchestration are stateful; they live in the `agent_kit`
sibling project (SPEC.md), built *on* this engine and never folded into the
batch path. Streaming is likewise single-call only — it never touches the
batch pipeline.

---

## ✅ Done

- **Embeddings + micro-batching driver.** `OpenAICompatibleEmbedder` with a
  dedicated chunk-as-unit packer (CLAUDE.md invariant #9), jsonl resume sink, and
  `benchmarks/embed_throughput.py`. RAG storage/retrieval was scoped out.

- **Provider-native batch API.** [`provider_batch/`](src/llm_kit/provider_batch/):
  submit a `list[BatchItem]` to a provider's 24h async endpoint, poll, download,
  and reconcile into the same `BatchResult`s the live pipeline yields. Request/
  response *bodies* reuse the formatter; *transport* is a `BatchBackend` selected
  by `message_format`. All three backends implemented (OpenAI file-upload;
  Anthropic + Gemini inline), with a persistent `BatchJob` handle for
  cross-process resume. Mock-transport tests only — no live-key tests in-repo.

- **Cost & usage accounting.** [`batch/ledger.py`](src/llm_kit/batch/ledger.py):
  `UsageLedger` accumulates token usage across a run; `batch_to_jsonl` returns a
  `BatchSummary` (counts + token totals + estimated cost). Pricing is opt-in via
  two `LLMConfig` fields set to the provider's published per-token rates.

- **Streaming + tool-call parsing.** `LLMClient.invoke_stream` streams one call as
  a typed event sequence (`TextChunk` → optional `ToolCallStarted` → terminal
  `StreamEnd` with the assembled `LLMResponse`); `invoke` and `invoke_stream` both
  accept `tools`/`tool_choice` and surface provider-agnostic `ToolCall`s. SSE
  decoding and tool-call reassembly live in the formatter; `StreamAssembler` is
  the provider-agnostic analog of `parse_response`.

- **Provider formatters — all three.** OpenAI, Anthropic, and Gemini wire formats
  in the [`messages/formatter/`](src/llm_kit/messages/formatter/) package (one
  module per provider), selected by `LLMConfig.message_format`. Each owns both
  sides of the wire — body assembly, structured output via the provider's native
  mechanism, response parsing, tools, and stream decoding (incl. Gemini's
  streaming URL swap via `endpoint_path`). Adding a fourth provider is a new
  formatter class; nothing in `LLMClient` changes.

---

## Moat-deepening — make the batch engine better

Each of these passes the test cleanly and reuses `http/session`, `http/retry`,
`rate_limit`, and the batch pipeline almost verbatim. Recommended near-term
order.

1. **Adaptive rate limiting.** The limiter is a fixed token bucket today; real
   provider limits are often undocumented and time-varying. Feed `429` responses
   and `Retry-After` headers back into the limiter to auto-tune RPM/TPM toward the
   provider's *actual* ceiling — additive-increase on sustained success,
   multiplicative-decrease on throttling. The signal is already observed in
   `http/retry.py`; this just routes it into `rate_limit/`. Pure
   throughput + reliability, and it removes the guesswork in hand-tuning caps.

2. **Failure re-drive pass.** After a run, re-process *only* the items that came
   back with a `BatchResult.error`. The output jsonl already records which ids
   failed (invariant #6), so this is a thin filter over the existing resume
   path: read the output, select error rows, re-feed their inputs. Turns a
   partial batch into a complete one without re-paying for the successes — a
   resumability + reliability win that's almost free given what's already there.

3. **Cache-aware cost accounting.** Providers report cached-prompt tokens
   separately (the cache-read/cache-write split from prompt caching) and price
   them differently. The ledger currently folds everything into prompt/completion.
   Surfacing cached-token counts and a third price field makes the `BatchSummary`
   cost estimate *honest* when prompt caching is in play — which, for a batch with
   a shared system prompt, is exactly when the savings are largest. Small, and it
   ties cost accounting to the caching providers already do for us.

4. **Real-tokenizer estimator (optional extra).** The default TPM estimator is
   char/4; the constructor already takes `token_estimator=fn`. Shipping a
   `tiktoken`-backed estimator behind a `[tokenizer]` extra tightens TPM
   accounting, which means fewer surprise `429`s and a higher usable throughput
   ceiling. Opt-in, never a core dep.

---

## Out of scope — a different project on purpose

These are real capabilities, but they add general-purpose statefulness that
pushes *against* the throughput core. Pursuing one is a deliberate decision to
become a different project, not "next on the list."

- **Agent execution loop.** Multi-step tool execution, observation feedback,
  timeouts, orchestration. The *parsing* half is shipped (see Done); everything
  *stateful* lives in `agent_kit` (SPEC.md), built on the engine.

- **RAG (storage + retrieval).** Vector-store adapters put you in the
  integration-maintenance business this kit exists to avoid. If ever pursued, the
  only genuinely batch-adjacent piece is a token-aware **chunker**; a `VectorStore`
  Protocol + in-memory impl + pgvector/FAISS/Qdrant adapters would all be optional
  extras behind a `[rag]` flag, never core.

---

## Cross-cutting infrastructure

Multiplies the value of everything above; slots in whenever it starts paying for
itself.

- **Observability.** OpenTelemetry spans + metrics on the request path (latency,
  retries, token counts, queue depths). Most valuable while the engine is the
  product — it makes throughput legible, and it's the natural source of the
  signals item #1 (adaptive limiting) consumes.
- **Middleware / interceptor hooks.** Pre/post-request redaction, templating,
  content-based retry. Useful, but breadth rather than depth.
