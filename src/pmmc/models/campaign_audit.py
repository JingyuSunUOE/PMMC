"""Task-local campaign lineage and durable successful-call recovery.

Neither attempt IDs nor journal locations enter provider prompts or request hashes.
The journal is a recovery record, not a provider exactly-once guarantee.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pmmc.schemas.model import EmbeddingResult, GenerationResult
from pmmc.schemas.trace import ModelCallTrace

if TYPE_CHECKING:
    from pmmc.experiments.step8_artifacts import CampaignArtifactStore


@dataclass(frozen=True)
class CampaignModelAudit:
    store: CampaignArtifactStore
    campaign_id: str
    attempt_id: str
    job_id: str
    source_fingerprint: str
    request_semaphore: asyncio.Semaphore | None = None

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "step8_campaign_id": self.campaign_id,
            "step8_attempt_id": self.attempt_id,
            "step8_component_job_id": self.job_id,
        }

    def identity(self, call_id: str, request: dict[str, object], model_hash: str) -> str:
        from pmmc.compiler.ids import canonical_content_hash

        return canonical_content_hash(
            ["step8-audited-response-v1", call_id, request, model_hash, self.source_fingerprint]
        )

    def restore_result(self, call_id: str, key: str) -> dict[str, Any] | None:
        return self.store.restored_response(request_id=call_id, request_hash=key)

    def save_result(
        self,
        call_id: str,
        key: str,
        result: GenerationResult | EmbeddingResult,
        runtime: dict[str, str | None] | None = None,
    ) -> None:
        self.store.journal_response(
            request_id=call_id,
            request_hash=key,
            response={
                "result": result.model_dump(mode="json"),
                "lineage": self.metadata,
                **({"runtime": runtime} if runtime is not None else {}),
            },
            attempt_id=self.attempt_id,
        )

    def restore_trace(self, key: str) -> ModelCallTrace | None:
        path = self.store.root / "_store/model-traces" / (key + ".json")
        if not path.is_file():
            return None
        import json

        receipt = json.loads(path.read_bytes())
        if receipt.get("request_hash") != key:
            raise ValueError("model_trace_journal_identity_mismatch")
        trace = ModelCallTrace.model_validate(self.store.load_json(receipt["trace_sha256"]))
        response = self.restore_result(trace.model_call_id, key)
        if response is not None:
            runtime_id, service_id = response_runtime(response)
            if service_id is not None and trace.runtime_id != runtime_id:
                # Older response-only recovery used the replaying runtime with
                # the producer's service. Preserve that journal as evidence.
                if trace.runtime_service_id != service_id or any(
                    trace.metadata.get(k) != v for k, v in response["lineage"].items()
                ):
                    raise ValueError("response_trace_producer_mismatch")
                trace = trace.model_copy(
                    update={
                        "runtime_id": runtime_id,
                        "metadata": {
                            **trace.metadata,
                            "response_runtime_recovery": {
                                "version": "response-producer-runtime-v1",
                                "original_trace_sha256": receipt["trace_sha256"],
                                "original_runtime_id": trace.runtime_id,
                                "producer_runtime_id": runtime_id,
                                "producer_runtime_service_id": service_id,
                            },
                        },
                    }
                )
        return trace

    def save_trace(self, key: str, trace: ModelCallTrace) -> None:
        from pmmc.experiments.step8_artifacts import write_json

        digest = self.store.put_json(trace.model_dump(mode="json"))
        write_json(
            self.store.root / "_store/model-traces" / (key + ".json"),
            {"request_hash": key, "trace_sha256": digest},
            immutable=True,
        )


def response_runtime(response: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read producer identity, never the runtime replaying a durable response."""
    metadata = response["result"].get("metadata", {})
    service_id = metadata.get("runtime_service_id")
    recorded = response.get("runtime")
    if recorded is not None:
        runtime_id = recorded["runtime_id"]
        if service_id is not None and service_id != recorded["runtime_service_id"]:
            raise ValueError("response_runtime_service_mismatch")
        service_id = recorded["runtime_service_id"]
    elif service_id is not None:
        # ManagedRuntimeRunner has always named services runtime:stage:service.
        runtime_id, separator, rest = service_id.partition(":")
        if not separator or ":" not in rest or not runtime_id or not rest.split(":", 1)[1]:
            raise ValueError("legacy_response_runtime_identity_missing")
    else:
        runtime_id = None
    if service_id is not None and (
        runtime_id is None or not service_id.startswith(runtime_id + ":")
    ):
        raise ValueError("response_runtime_identity_mismatch")
    return runtime_id, service_id


_CURRENT: ContextVar[CampaignModelAudit | None] = ContextVar("campaign_model_audit", default=None)


def current_campaign_audit() -> CampaignModelAudit | None:
    return _CURRENT.get()


@contextmanager
def campaign_model_scope(audit: CampaignModelAudit) -> Iterator[None]:
    token = _CURRENT.set(audit)
    try:
        yield
    finally:
        _CURRENT.reset(token)
