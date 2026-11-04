"""Local audited execution fixture; no provider, database, or remote object store."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.models.campaign_audit import CampaignModelAudit
from pmmc.models.execution import ModelExecutionService
from pmmc.models.fake import DeterministicFakeGateway
from pmmc.models.registry import ModelRegistry, ModelRegistryEntry
from pmmc.schemas import DataScope


def fixture(tmp_path: Path):
    store = CampaignArtifactStore(tmp_path)
    audit = CampaignModelAudit(store, "campaign", "attempt-test", "job-native", "a" * 64)
    registry = ModelRegistry(
        registry_version="journal-test",
        models=[
            ModelRegistryEntry(
                alias="deterministic-fake",
                endpoint_kind="fake",
                provider_model="deterministic-fake",
                expected_actual_model_id="deterministic-fake@v1",
                model_revision="v1",
                tokenizer_name="fake",
                tokenizer_revision="v1",
                capabilities=["chat_text"],
                context_window=32768,
            )
        ],
    )

    async def put_bytes(*, artifact_id, data, **_):
        return SimpleNamespace(artifact_id=artifact_id, sha256=store.put(data))

    artifacts = SimpleNamespace(scope=DataScope.RUNTIME, put_bytes=put_bytes)
    writer = SimpleNamespace(add=AsyncMock(side_effect=lambda trace: trace))
    gateway = AsyncMock(spec=DeterministicFakeGateway)
    service = ModelExecutionService(
        registry=registry, artifact_service=artifacts, model_calls=writer
    )
    return service, audit, gateway, store, artifacts, writer, service.trace_sink
