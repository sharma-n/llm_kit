"""Minimal example: one chat completion, no images, no schema."""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm_kit import Message, OpenAICompatibleLLM


async def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    async with OpenAICompatibleLLM.from_yaml(str(config_path)) as llm:
        response = await llm.invoke(
            [
                Message.system("You are a terse assistant."),
                Message.user("In one sentence, what is the capital of France?"),
            ]
        )
    print(response.text)
    print(f"usage: {response.usage}")
    print(f"latency: {response.latency_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
