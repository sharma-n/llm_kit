"""Embeddings example: one-shot, then a streamed micro-batched batch.

Inputs are packed into request-sized chunks automatically (see EmbeddingConfig
max_inputs_per_request / max_tokens_per_request). Each input still yields one
EmbeddingResult, keyed by its id.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm_kit import EmbedItem, OpenAICompatibleEmbedder


async def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    async with OpenAICompatibleEmbedder.from_yaml(str(config_path)) as emb:
        # Single string.
        one = await emb.embed_one("The quick brown fox.")
        print(f"dims: {len(one.vector)}  usage: {one.usage}")

        # Streamed batch — packed into as few requests as the caps allow.
        items = [EmbedItem(id=f"doc-{i:04d}", text=f"document number {i}") for i in range(50)]
        ok = 0
        async for result in emb.embed_batch(items):
            if result.ok:
                ok += 1
            else:
                print(f"{result.id} failed: {result.error}")
        print(f"embedded {ok}/{len(items)} documents")

        # Resumable sink: the jsonl file IS the checkpoint.
        # written = await emb.embed_batch_to_jsonl(items, "vectors.jsonl", resume=True)


if __name__ == "__main__":
    asyncio.run(main())
