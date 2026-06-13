# Embeddings

[← back to README](../README.md)

Same async, rate-limited, resumable machinery, pointed at `/v1/embeddings`. The
key difference from chat: the embeddings endpoint takes an **array** of inputs
per request, so the batch driver **micro-batches** — it packs many inputs into
each HTTP call (bounded by `max_inputs_per_request` and `max_tokens_per_request`)
instead of one-request-per-input. Each input still comes back as its own
`EmbeddingResult`, keyed by `id`.

```python
import asyncio
from llm_kit import EmbedItem, OpenAICompatibleEmbedder

async def main():
    async with OpenAICompatibleEmbedder.from_yaml("examples/config.yaml") as emb:
        # single string
        r = await emb.embed_one("The quick brown fox.")
        print(len(r.vector), r.usage)

        # caller-sized list, one request
        rs = await emb.embed_texts(["a", "b", "c"])

        # streamed batch — packed into as few requests as the caps allow
        items = [EmbedItem(id=f"doc-{i}", text=f"document {i}") for i in range(100_000)]
        async for result in emb.embed_batch(items):
            if result.ok:
                use(result.response.vector)

        # resumable jsonl sink (the file IS the checkpoint, same as chat)
        await emb.embed_batch_to_jsonl(items, "vectors.jsonl", resume=True)

asyncio.run(main())
```

Rate limiting maps cleanly onto the array shape: **one request = one RPM debit**,
and the **TPM debit is the sum of the request's inputs**, trued up from the
response's reported usage. Per-input `usage` is the request total split across
inputs by estimated size (exact in aggregate). Because failures are per-request,
an HTTP error fails the whole chunk — every input in it shares the error (or, under
`batch.on_failure: raise`, aborts the stream).

The `embed` config section mirrors `llm`:

```yaml
embed:
  base_url: https://api.openai.com   # or Together / a local server
  model: text-embedding-3-small
  api_key_env: OPENAI_API_KEY
  # dimensions: 512                   # Matryoshka truncation, if the model supports it
  max_inputs_per_request: 2048
  max_tokens_per_request: 300000
```

See [docs/performance.md](performance.md) for the throughput benchmark — `max_inputs_per_request` is the dominant lever.
