"""Vertex AI batch-prediction configuration (``ProviderBatchConfig.vertex``).

Used only by the ``vertex`` provider-batch backend, which pairs with
``llm.message_format: gemini`` (the Gemini formatter still builds every request
body). Auth is Application Default Credentials, resolved by Google's SDKs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VertexBatchConfig:
    """Knobs for the Vertex AI (``batchPredictionJobs``) batch backend."""

    # GCP project id. None => whatever the SDK/ADC resolves (GOOGLE_CLOUD_PROJECT, ...).
    project: str | None = None
    # Vertex location; "global" is supported for Gemini models.
    location: str = "global"
    # gs://bucket[/path] under which each run gets its own ``llm-kit-<uuid>/``
    # directory (input.jsonl, keys.jsonl, output/, images/). Required at submit.
    gcs_prefix: str | None = None
    # Upload image bytes to GCS and reference them as fileData gs:// URIs instead of
    # inlining base64 (for requests too large to inline).
    upload_images: bool = False
    # Defaults for ``cleanup()``: delete uploaded input (+ sidecar + images) / output.
    cleanup_input: bool = False
    cleanup_output: bool = False
    # Job display name is ``{display_name_prefix}-{run id}``.
    display_name_prefix: str = "llm-kit"
