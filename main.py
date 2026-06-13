import asyncio
from llm_kit import Message, OpenAICompatibleLLM

async def main():
    async with OpenAICompatibleLLM.from_yaml("examples/config.yaml") as llm:
        response = await llm.invoke([
            Message.system("You are a terse assistant."),
            Message.user("In one sentence, what is the capital of France?"),
        ])
    print(response.text)
    print(response.usage)


if __name__ == "__main__":
    asyncio.run(main())
