"""Streaming + tool calling: the agent-fork enablers in one place.

Run against a real OpenAI-compatible endpoint (uses examples/config.yaml).
First streams a plain answer token-by-token; then offers a tool and prints the
assembled tool call the model wants executed (llm_kit parses it — *you* run it).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm_kit import (
    LLMClient,
    Message,
    StreamEnd,
    TextChunk,
    ToolCallStarted,
    ToolDefinition,
)


async def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    async with LLMClient.from_yaml(str(config_path)) as llm:
        # 1. Plain streaming — forward TextChunks as they arrive.
        print("streaming answer: ", end="", flush=True)
        async for event in llm.invoke_stream(
            [
                Message.system("You are a terse assistant."),
                Message.user("Name three primary colors."),
            ]
        ):
            if isinstance(event, TextChunk):
                print(event.text, end="", flush=True)
            elif isinstance(event, StreamEnd):
                print(f"\nusage: {event.response.usage}")

        # 2. Tool calling — offer a tool, print the parsed call.
        weather = ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        print("\ntool turn:")
        async for event in llm.invoke_stream(
            [Message.user("What's the weather in Paris?")],
            tools=[weather],
            tool_choice="auto",
        ):
            if isinstance(event, ToolCallStarted):
                print(f"  model wants: {event.name}(...)")
            elif isinstance(event, StreamEnd):
                for call in event.response.tool_calls:
                    print(f"  parsed call -> {call.name}({call.arguments})")


if __name__ == "__main__":
    asyncio.run(main())
