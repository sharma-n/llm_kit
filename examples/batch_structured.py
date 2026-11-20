"""Batch example: each item validates against its own pydantic schema.

Demonstrates:
- per-item ``response_model`` for structured outputs
- multimodal input (image URL → base64 transparently)
- streaming results to a jsonl file with ID-based resume
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from llm_kit import BatchItem, ImagePart, Message, LLMClient


class ImageCaption(BaseModel):
    caption: str = Field(description="A one-sentence caption.")
    contains_text: bool


def make_items() -> list[BatchItem]:
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/120px-Cat03.jpg",
    ]
    return [
        BatchItem(
            id=f"cap-{i:04d}",
            messages=[
                Message.system("Caption the image and report whether it contains text."),
                Message.user("Caption this:", ImagePart.from_url(url)),
            ],
            response_model=ImageCaption,
        )
        for i, url in enumerate(image_urls)
    ]


async def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    output_path = Path(__file__).with_name("batch_output.jsonl")

    async with LLMClient.from_yaml(str(config_path)) as llm:
        written = await llm.batch_to_jsonl(make_items(), output_path, resume=True)

    print(f"wrote {written} results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
