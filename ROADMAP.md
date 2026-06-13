# Roadmap — extension ideas

## What this kit is (and the one test for every item below)

`llm_kit` is a **stateless, high-throughput batch engine** for chat completions
and embeddings. Its load-bearing strength is the bounded batch pipeline
(O(buffer), not O(N), resumable from the output jsonl) plus strict bottom-up
layering (`config → http → rate_limit/images/messages → llm/embed → batch`).

That is a deliberately *narrow* identity. It is not a general-purpose LLM
framework, and the difference matters: a framework optimizes for "can it talk to
anything" (breadth, integrations, swappable components — and the leaky
abstractions and version churn that come with them). This kit optimizes for "can
it process a million things without falling over." Those are different products.

So every item below gets one test:

> **Does it make a large batch job cheaper, faster, more reliable, or more
> resumable? Or does it just make `llm_kit` more *general*?**

The first deepens the moat. The second dissolves it — and the second is where
"am I just rebuilding LangChain?" comes from, correctly. Items are grouped by
that test, not by raw capability gap.

---

## ✅ Done

- **Embeddings + micro-batching driver.** `OpenAICompatibleEmbedder`, a dedicated
  chunking driver (chunk-as-unit, see CLAUDE.md invariant #9), jsonl resume sink,
  and `benchmarks/embed_throughput.py`. RAG storage/retrieval was explicitly
  scoped **out** — see the "Identity fork" section to take it further.

---

## Moat-deepening — make the batch engine better

These pass the test cleanly. Each makes a large job cheaper, faster, more
reliable, or more resumable, and each reuses `http/session`, `http/retry`,
`rate_limit`, the error hierarchy, and the batch pipeline almost verbatim. This
is the recommended near-term order.

1. **Provider-native batch API (the "fourth sink").** The most on-brand item in
   the whole roadmap. Same `BatchItem` list (alongside stream/jsonl/callback):
   submit to OpenAI/Anthropic 24h async batch endpoints (~50% cheaper), poll,
   reconcile. Purely a batch concern, complementary to the existing client-side
   concurrency, and a capability general frameworks don't really own.
2. **Cost & usage accounting.** Aggregate the per-response `TokenUsage` (already
   collected and trued-up) into a priced ledger per batch, priced from config.
   Turns "throughput" into "throughput you can budget." Small, additive, and a
   batch-engine feature rather than framework furniture.
3. **Response caching / dedup.** Mirror the image-fetcher LRU at the response
   layer, keyed by a hash of messages + params. In a 1M batch with repeated
   prompts this is real money saved; the pattern already exists in `images/`.
4. **Anthropic-native formatter.** Cheap and low-risk: CLAUDE.md invariant #2
   already reserves the seam, so this is a wire-format addition, not a new layer
   of statefulness. Unlocks Claude's native content/tool-use shape without
   touching the LLM client. Target the latest Claude models (Opus 4.x / Fable 5)
   as the default.

---

## Identity fork — valuable, but a different project

These are real capabilities, but they **fail the test**: they add general-purpose
statefulness that pushes *against* the throughput core rather than deepening it.
Doing them is a deliberate decision to become a different kind of project — make
it on purpose, not as "next on the list."

- **Tool-calling / agent loop.** The biggest capability gap vs. the README's
  "agentic systems" framing — and the item that fights the moat hardest. Each
  item becomes a *multi-step* unit, so the O(buffer) memory and per-item-failure
  isolation invariants need a deliberate extension, not a bolt-on. This is also
  precisely the LangChain/LangGraph surface area, where the work is already
  commoditized. Note `tool_use` currently exists only as a *structured-output
  style*, not an execution loop. If pursued, treat it as a sibling project built
  *on* the engine, not folded into the batch path.
- **RAG (storage + retrieval).** "The interface is the product" is the LangChain
  thesis verbatim, and shipping vector-store adapters puts you in the
  integration-maintenance business that is LangChain's whole burden. If pursued:
  - **Chunker** — token-aware splitter with overlap. Small, owned, the most
    batch-adjacent piece.
  - **`VectorStore` Protocol** — `add(ids, vectors, metadata)` / `search(query, k)`.
  - **One built-in impl** — in-memory numpy cosine index, honest to ~10⁵–10⁶
    vectors, clearly labeled as not a production vector DB (behind a `[rag]` extra).
  - **Adapters as optional extras** — pgvector / FAISS / Qdrant behind the
    Protocol, *never* core deps.
  - **`Retriever`** — query-embed + store-search, returns context for the chat
    client.
- **Streaming `invoke()`.** Token streaming for interactive/single-call use.
  Genuinely useful but most at odds with the batch design — if done, scope it
  strictly to single calls, never the batch path.

---

## Cross-cutting infrastructure

Multiplies the value of everything above; infrastructure, not features. Slots in
whenever it starts paying for itself.

- **Observability.** OpenTelemetry spans + metrics around the request path
  (latency, retries, token counts, queue depths). Most valuable while the engine
  is the product — it makes throughput legible.
- **Middleware / interceptor hooks.** Pre/post-request: redaction, prompt
  templating, content-based retry. Generic framework furniture — useful but
  breadth, not depth.
- **Prompt management / templating.** Versioned prompts, variable injection. Same
  caveat — general-purpose, not throughput-specific.

---

## Suggested order

Provider-native batch API → cost accounting → response caching → Anthropic
formatter. These four keep the kit firmly in its lane and each one deepens the
batch engine. Only *after* that, if the project's identity is meant to widen,
take up the agent loop or RAG as a conscious fork. Cross-cutting infra slots in
whenever it starts paying for itself.
