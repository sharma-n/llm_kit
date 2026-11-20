# Vertex AI batch: manual smoke test

The Vertex backend is covered in-repo only by duck-typed fakes (no GCP credentials were available when it was written). This script runs one tiny real job and records what the unit tests cannot know.

## Run it

```bash
uv sync --extra vertex
gcloud auth application-default login          # ADC; llm_kit never handles tokens itself
uv run --extra vertex python examples/vertex_smoke.py \
    --project <project> --bucket <bucket> --location global --cleanup
```

Needs Vertex AI enabled on the project, `roles/aiplatform.user`, and object read/write/list/delete on the bucket. Useful flags: `--upload-images` (send `fileData` gs:// URIs instead of inline base64), `--image-kb N` (approximate PNG size, to probe the inline size limit; try 50, 1000, 5000, 20000), `--model`, `--prefix`, `--poll-interval`. Without `--cleanup` the run directory `gs://<bucket>/<prefix>/llm-kit-<uuid>/` is kept for inspection (`gsutil cat .../output/*/predictions.jsonl | head`).

It submits two text items and one inline-image item, prints each state transition with elapsed time, the per-item results, and, for every `.jsonl` under `output/`, its relative path, row count, first-row keys, whether our `llm_kit_id` label was echoed (and its decoded value) and the max line size in bytes.

## What to check

- Job reaches `JOB_STATE_SUCCEEDED`; all three ids come back with text and no per-item error.
- If labels are not echoed, matching must still work through the canonical-hash fallback (the script prints results either way; a row-matching failure shows up as `no output row found`).

## Run log: 2026-11-20 (gemini-3.5-flash, asia-southeast1, `upload_images=False`)

2 text items + 1 inline 61 KiB PNG. The job succeeded, the inline image was accepted, and 3/3 results matched correctly. Output layout was `output/prediction-model-2026-11-20T11:39:38.622633Z/predictions.jsonl` plus `.../incremental_predictions/predictions-chunked-2026-11-20T11:45:50.004844Z.jsonl`. Timeline: PENDING 4.7s, QUEUED 309s, RUNNING 385s, SUCCEEDED 613s (615s total). `--cleanup` removed input, keys and output. The old label probe printed nothing (it opened only the first `.jsonl` by name, the incremental file, and stopped), so whether labels are echoed, the max inline image size and the max line size are still to be determined: rerun with the improved probe and a larger `--image-kb`.

## Findings (fill in after running)

| Question | Finding | Date / model / location |
|---|---|---|
| Does Vertex accept `labels` on a batch `request` (job created without error)? | Job created and succeeded; all 3 results matched correctly | 2026-11-20, gemini-3.5-flash, asia-southeast1 |
| Is `request.labels.llm_kit_id` echoed in output rows? | not yet run | |
| Is the echoed `request` canonical-hash-equal to what was written (key order / field normalisation / inlineData)? | not yet run | |
| Output row shape (`request`, `response`, `status`, `processed_time`?) and per-row error format | not yet run | |
| Output ordering vs input | not yet run | |
| Max inline image size accepted (`--image-kb` sweep, `upload_images` off) | 61 KiB inline PNG accepted; maximum still to be determined (rerun with larger `--image-kb`) | 2026-11-20, gemini-3.5-flash, asia-southeast1 |
| Max JSONL line size accepted | not yet run | |
| Behaviour with `--upload-images` (fileData URIs accepted, permissions needed for the Vertex service agent) | not yet run | |
| Output directory layout / file names | `output/prediction-model-<ts>/predictions.jsonl` (final) plus `output/prediction-model-<ts>/incremental_predictions/predictions-chunked-<ts>.jsonl` (progress chunk repeating rows). The backend reads only the final file; incremental is a fallback when no final exists | 2026-11-20, gemini-3.5-flash, asia-southeast1 |
| Typical time to completion for 3 requests | PENDING 4.7s, QUEUED 309s, RUNNING 385s, SUCCEEDED 613s; 615s total. Counts reported `{'incomplete': 3}` then `{'succeeded': 3}` | 2026-11-20, gemini-3.5-flash, asia-southeast1 |
| Anything else surprising | not yet run | |

If labels turn out to be rejected, remove the label in `VertexBatchBackend.submit` (the hash fallback does not depend on it). If the echoed request is normalised so hashes differ, the fallback in `_match` needs a looser canonical form.
