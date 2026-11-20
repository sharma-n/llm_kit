"""Provider-native (offline) Batch API driver.

A peer of ``batch/`` (the live concurrent pipeline). Same input/output contracts
(``list[BatchItem]`` → ``BatchResult``), entirely different middle: requests are
serialized and handed to a provider's 24h async batch endpoint, then polled and
reconciled. See ``backend.BatchBackend`` for the transport abstraction and
``job.BatchJob`` for the persisted resume handle.
"""

from llm_kit.provider_batch.backend import BatchBackend
from llm_kit.provider_batch.job import BatchJob
from llm_kit.provider_batch.registry import BackendFactory, register_batch_backend
from llm_kit.provider_batch.types import (
    BatchJobStatus,
    BatchRequest,
    BatchState,
    RawResult,
    SubmitResult,
)

__all__ = [
    "BackendFactory",
    "BatchBackend",
    "BatchJob",
    "BatchJobStatus",
    "BatchRequest",
    "BatchState",
    "RawResult",
    "SubmitResult",
    "register_batch_backend",
]
