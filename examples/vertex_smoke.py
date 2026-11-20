"""Manual smoke test for the Vertex AI batch backend (needs a real GCP project).

Prerequisites: ``uv sync --extra vertex``, ``gcloud auth application-default login``,
a bucket you can read/write, and Vertex AI enabled. See docs/vertex_smoke_test.md.

    uv run --extra vertex python examples/vertex_smoke.py \
        --project my-proj --bucket my-bucket [--location global] \
        [--model gemini-2.5-flash] [--upload-images] [--cleanup] [--image-kb 200]

Submits two text items plus one inline-image item, waits, prints state transitions,
timings, per-item results, and whether request labels were echoed in the output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
import zlib

from llm_kit import AppConfig, BatchItem, LLMClient, Message
from llm_kit.config.llm import MessageFormat
from llm_kit.messages.types import ImagePart
from llm_kit.provider_batch.vertex import LABEL_KEY, decode_label


def make_png(size_kb: int) -> bytes:
    """A valid PNG whose compressed size is roughly ``size_kb`` (random-ish pixels)."""
    import os

    side = 64
    while True:
        raw = b"".join(b"\x00" + os.urandom(side * 3) for _ in range(side))
        def chunk(tag: bytes, data: bytes) -> bytes:
            body = tag + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 0))
            + chunk(b"IEND", b"")
        )
        if len(png) >= size_kb * 1024 or side > 4096:
            return png
        side = int(side * 1.5)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="llm-kit-smoke")
    ap.add_argument("--location", default="global")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--upload-images", action="store_true")
    ap.add_argument("--image-kb", type=int, default=50, help="approx size of the test PNG")
    ap.add_argument("--poll-interval", type=float, default=15.0)
    ap.add_argument("--cleanup", action="store_true", help="delete input/output afterwards")
    args = ap.parse_args()

    cfg = AppConfig()
    cfg.llm.message_format = MessageFormat.GEMINI
    cfg.llm.model = args.model
    cfg.llm.api_key = "unused-adc"  # ADC is used; this is never sent
    cfg.provider_batch.backend = "vertex"
    v = cfg.provider_batch.vertex
    v.project, v.location = args.project, args.location
    v.gcs_prefix = f"gs://{args.bucket}/{args.prefix}"
    v.upload_images = args.upload_images

    png = make_png(args.image_kb)
    items = [
        BatchItem(id="text-1", messages=[Message.user("Reply with the single word: alpha")]),
        BatchItem(id="text-2", messages=[Message.user("Reply with the single word: beta")]),
        BatchItem(
            id="image-1",
            messages=[Message.user("Describe this image in five words.", ImagePart.from_bytes(png, "image/png"))],
        ),
    ]
    print(f"image bytes: {len(png)} ({len(png) / 1024:.0f} KiB); upload_images={args.upload_images}")

    llm = LLMClient(cfg)
    t0 = time.monotonic()
    job = await llm.submit_batch(items)
    print(f"submitted in {time.monotonic() - t0:.1f}s: {job.batch_id}")
    print("extra:", json.dumps(job.extra, indent=2))

    last = None
    while True:
        status = await llm.poll_batch(job)
        if status.raw_state != last:
            print(f"[{time.monotonic() - t0:7.1f}s] {status.raw_state} -> {status.state} counts={status.counts}")
            last = status.raw_state
        if status.state.is_terminal:
            break
        await asyncio.sleep(args.poll_interval)
    if status.error_detail:
        print("job error:", status.error_detail)

    # Findings probe: are labels echoed in the raw output rows?
    backend = llm._batch_backend
    echoed = None
    try:
        from itertools import islice

        run_prefix = job.extra["run_prefix"]
        bucket_name, path = run_prefix[5:].split("/", 1)
        for blob in backend._storage_client().list_blobs(bucket_name, prefix=path + "output/"):
            if blob.name.endswith(".jsonl"):
                with blob.open("rb") as fh:
                    for line in islice(fh, 5):
                        row = json.loads(line)
                        labels = (row.get("request") or {}).get("labels") or {}
                        echoed = decode_label(labels.get(LABEL_KEY)) is not None
                        print(f"first row keys={sorted(row)} labels_echoed={echoed} line_bytes={len(line)}")
                        break
                break
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print("label probe failed:", exc)

    if status.state.is_terminal:
        try:
            async for r in llm.fetch_batch_results(job, items=items, allow_partial=True):
                text = r.response.text if r.response else None
                print(f"{r.id}: {'ERROR ' + str(r.error) if r.error else repr(text)}")
        except Exception as exc:  # noqa: BLE001
            print("fetch failed:", exc)
    print(f"total {time.monotonic() - t0:.1f}s")

    if args.cleanup:
        print("deleted:", await llm.cleanup_batch(job, input=True, output=True))
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
