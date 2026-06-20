"""Provider-native (offline) Batch API example.

Unlike ``batch_structured.py`` (the *live* pipeline, which fires concurrent
requests now), this submits the whole batch to the provider's 24h async batch
endpoint — roughly half the price, at the cost of latency. The job outlives your
process, so the realistic flow is three separate commands:

    python examples/provider_batch.py submit   # upload + create job, save the handle
    python examples/provider_batch.py status    # poll; repeat until it says completed
    python examples/provider_batch.py fetch      # download + reconcile to jsonl

The ``run`` command does all three in one process (handy for a tiny demo, but it
blocks until the provider finishes — minutes to hours for real jobs).

The handle (``batch_job.json``) is the checkpoint: it stores the provider batch
id and a custom_id→metadata manifest so a *fresh* process can reconnect. Note the
re-supply of ``make_items()`` at fetch time — ``response_model`` is a Python type
and is not persisted, so validation needs the original items back (matched by id).
The backend is selected by ``message_format`` in config.yaml — ``openai`` (Files +
Batches), ``anthropic`` (Message Batches), or ``gemini`` (Batch Mode) — all using
the same commands below.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from llm_kit import BatchItem, BatchJob, LLMClient, Message
from llm_kit.batch.jsonl import write_results_to_jsonl

CONFIG_PATH = Path(__file__).with_name("config.yaml")
HANDLE_PATH = Path(__file__).with_name("batch_job.json")
OUTPUT_PATH = Path(__file__).with_name("provider_batch_output.jsonl")


class Sentiment(BaseModel):
    label: str = Field(description="one of: positive, negative, neutral")
    confidence: float


def make_items() -> list[BatchItem]:
    reviews = [
        "Absolutely loved it, would buy again.",
        "Broke after two days. Total waste of money.",
        "It's fine. Does the job, nothing special.",
    ]
    return [
        BatchItem(
            id=f"review-{i:04d}",
            messages=[
                Message.system("Classify the sentiment of the review."),
                Message.user(text),
            ],
            response_model=Sentiment,
            metadata={"source": "demo"},
        )
        for i, text in enumerate(reviews)
    ]


async def submit() -> None:
    """Phase 1: serialize + upload + create the job, then persist the handle."""
    async with LLMClient.from_yaml(str(CONFIG_PATH)) as llm:
        job = await llm.submit_batch(make_items())
    job.save(HANDLE_PATH)
    print(f"submitted batch {job.batch_id}; handle saved to {HANDLE_PATH.name}")
    print("run 'status' to poll, then 'fetch' once it is completed")


async def status() -> None:
    """Phase 2: reconnect to the in-flight job and print its state."""
    job = BatchJob.load(HANDLE_PATH)
    async with LLMClient.from_yaml(str(CONFIG_PATH)) as llm:
        snap = await llm.poll_batch(job)
    print(f"batch {job.batch_id}: state={snap.state} (raw={snap.raw_state!r}) counts={snap.counts}")
    if snap.state.is_terminal:
        print("terminal — run 'fetch' to download results")


async def fetch() -> None:
    """Phase 3: download + reconcile into jsonl (re-supplying items for validation)."""
    job = BatchJob.load(HANDLE_PATH)
    items = make_items()
    ok = 0

    async def _print_and_yield(results):
        nonlocal ok
        async for result in results:
            if result.ok:
                ok += 1
                print(f"{result.id}: {result.response.parsed}")
            else:
                print(f"{result.id} failed: {result.error}")
            yield result

    async with LLMClient.from_yaml(str(CONFIG_PATH)) as llm:
        OUTPUT_PATH.unlink(missing_ok=True)  # truncate for the demo
        raw = llm.fetch_batch_results(job, items=items)  # type: ignore[attr-defined]
        written = await write_results_to_jsonl(OUTPUT_PATH, _print_and_yield(raw))
    print(f"reconciled {ok}/{written} results")


async def run() -> None:
    """All three phases in one process (blocks until the provider finishes)."""
    items = make_items()
    async with LLMClient.from_yaml(str(CONFIG_PATH)) as llm:
        # poll_interval is a courtesy floor; tighten it in config for small jobs.
        written = await llm.run_batch_api(items, OUTPUT_PATH, poll_interval=10.0)
    print(f"wrote {written} results to {OUTPUT_PATH.name}")


COMMANDS = {"submit": submit, "status": status, "fetch": fetch, "run": run}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd not in COMMANDS:
        print(f"usage: {Path(__file__).name} [{'|'.join(COMMANDS)}]")
        raise SystemExit(2)
    asyncio.run(COMMANDS[cmd]())
