"""Step 8 stage adapters over the existing compiler and baseline implementations."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, write_json
from pmmc.experiments.step8_jobs import BenchmarkStageJob, ComponentReceipt
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_engine, create_session_factory
from pmmc.infrastructure.settings import AppMode, Settings
from pmmc.models.registry import load_model_registry
from pmmc.schemas.base import DatasetName, JsonDict
from pmmc.schemas.step8 import BenchmarkCampaignManifest, Step8BenchmarkConfig


def _compiler_namespace(manifest: BenchmarkCampaignManifest) -> str | None:
    if manifest.diagnostic_only:
        return manifest.campaign_id
    return "exploration-dev-all" if manifest.phase.value == "exploration" else None


def compiler_subprotocol_hash(manifest: BenchmarkCampaignManifest, scope: dict[str, Any]) -> str:
    from pmmc.experiments.step8_model_identity import compiler_model_entry_hashes

    treatment = scope.get("treatment", "main")
    if treatment == "no_reranker":
        treatment = "main"
    return canonical_content_hash(
        {
            **({"diagnostic_namespace": namespace} if (namespace := _compiler_namespace(manifest))
               else {}),
            "snapshot": scope["snapshot_id"],
            "conversation": scope["conversation_id"],
            "treatment": treatment,
            "seed": manifest.protocol.seed,
            "questioner": manifest.protocol.config_hashes["questioner_config"],
            "planner": manifest.protocol.config_hashes["planner_config"],
            "doubter": manifest.protocol.config_hashes["doubter_config"],
            "operators": manifest.protocol.config_hashes["foundation_config"],
            "model_entries": compiler_model_entry_hashes(manifest.protocol),
            "compiler_sources": {
                key: manifest.protocol.source_fingerprints[key]
                for key in (
                    "unit_prepare",
                    "questioner",
                    "planner0",
                    "text_index",
                    "multimodal_index",
                    "dry_run0",
                    "r0_gate",
                    "doubter",
                    "projection",
                )
            },
        }
    )


def compiler_instance_id(manifest: BenchmarkCampaignManifest, scope: dict[str, Any]) -> str:
    return "step8-compiler-" + compiler_subprotocol_hash(manifest, scope)[:24]


class ProductionComponentBackend:
    def __init__(
        self,
        *,
        settings: Settings,
        config: Step8BenchmarkConfig,
        campaign: BenchmarkCampaignManifest,
        campaign_dir: Path,
        store: CampaignArtifactStore,
    ) -> None:
        from pmmc.experiments.execution import load_execution_config
        from pmmc.experiments.step8_config_contract import verify_frozen_execution_inputs

        verify_frozen_execution_inputs(settings, config, campaign.protocol)
        self.execution_profile = load_execution_config(
            settings.resolve_path(config.execution_profile)
        )
        if self.execution_profile.unit_preparation.workers != config.execution.cpu_workers:
            raise ValueError("step8_execution_profile_cpu_workers_mismatch")
        self.execution_profile_hash = self.execution_profile.config_hash
        self.config, self.campaign, self.campaign_dir, self.store = (
            config,
            campaign,
            campaign_dir,
            store,
        )
        self.settings = settings.model_copy(
            update={
                "mode": AppMode.PAPER if campaign.paper_result else AppMode.DEV,
                "runs_dir": campaign_dir / "private/runs",
                "outputs_dir": campaign_dir / "private/outputs",
                "artifacts_dir": settings.project_root / config.output_root / "_store/artifacts",
                "model_registry_path": settings.resolve_path(config.model_registry),
                "runtime_model_lock_path": settings.resolve_path(config.model_lock),
                "runtime_profile_path": settings.resolve_path(config.runtime_profile),
                "compiler_config_path": settings.resolve_path(config.foundation_config),
                "questioner_config_path": settings.resolve_path(config.questioner_config),
                "database_pool_size": min(config.execution.database_concurrency, 24),
                "database_max_overflow": 0,
                "model_max_concurrency": config.execution.model_concurrency,
                "compiler_cpu_workers": config.execution.cpu_workers,
                "qbank_text_query_batch_size": self.execution_profile.qbank.text_query_batch_size,
                "qbank_multimodal_query_batch_size": (
                    self.execution_profile.qbank.multimodal_query_batch_size
                ),
                "qbank_embedding_concurrency": self.execution_profile.qbank.embedding_concurrency,
                "qbank_score_concurrency": self.execution_profile.qbank.score_concurrency,
                "token_counter_backend": self.execution_profile.unit_preparation.counter_backend,
                "token_counter_cache_dir": (
                    settings.token_counter_cache_dir
                    if settings.local_scratch_dir is not None
                    else settings.resolve_path(config.output_root) / "_cache/token-counters"
                ),
                "local_asset_cache_enabled": (
                    self.execution_profile.unit_preparation.local_asset_cache
                ),
            }
        )
        # Runtime lock validation binds the original registry. Execution consumers
        # need resolved commits, but must not feed that derivative back into the lock gate.
        self.runtime_settings = self.settings
        from pmmc.models.locking import (
            load_model_lock,
            resolve_registry_from_lock,
            write_resolved_registry,
        )

        registry = resolve_registry_from_lock(
            load_model_registry(self.settings.resolved_model_registry_path),
            load_model_lock(self.settings.resolve_path(config.model_lock)),
        )
        resolved = campaign_dir / "private/configs/resolved-models.yaml"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        write_resolved_registry(registry, resolved)
        self.settings = self.settings.model_copy(update={"model_registry_path": resolved})

    async def preflight(self, job: BenchmarkStageJob) -> None:
        from pmmc.experiments.step8_config_contract import verify_frozen_execution_inputs
        from pmmc.experiments.step8_jobs import scientific_source_fingerprint

        verify_frozen_execution_inputs(self.runtime_settings, self.config, self.campaign.protocol)
        if job.campaign_id != self.campaign.campaign_id:
            raise ValueError("component_campaign_identity_mismatch")
        if job.source_fingerprint != self.campaign.protocol.source_fingerprints.get(job.stage):
            raise ValueError("component_scientific_source_changed")
        if (
            scientific_source_fingerprint(self.settings.project_root, job.stage)
            != job.source_fingerprint
            and self.execution_maintenance(job) is None
        ):
            raise ValueError("component_scientific_source_changed")
        if not self.settings.database_url:
            raise ValueError("campaign_database_preflight_missing_credentials")
        if job.operation in {"snapshot", "corpus"} and not self.settings.database_admin_url:
            raise ValueError("campaign_import_requires_cpu_controller_credentials")
        if not self.settings.object_store_endpoint:
            raise ValueError("campaign_object_store_preflight_missing_endpoint")

    def execution_maintenance(self, job: BenchmarkStageJob) -> dict[str, Any] | None:
        from pmmc.experiments.step8_jobs import scientific_source_fingerprint
        from pmmc.experiments.step8_maintenance import APPROVED_TRANSITION, execution_maintenance

        if job.campaign_id != self.campaign.campaign_id:
            raise ValueError("component_campaign_identity_mismatch")
        from pmmc.experiments.step8_compile_job_maintenance import CAMPAIGN_ID

        if job.campaign_id not in {APPROVED_TRANSITION["campaign_id"], CAMPAIGN_ID}:
            return None
        try:
            return execution_maintenance(
                job=job, protocol=self.campaign.protocol, directory=self.campaign_dir,
                project_root=self.settings.project_root,
                actual_source_fingerprint=scientific_source_fingerprint(
                    self.settings.project_root, job.stage
                ),
            )
        except ValueError as error:
            raise ValueError("component_scientific_source_changed") from error

    def _upstream(self, job: BenchmarkStageJob) -> list[ComponentReceipt]:
        return [
            ComponentReceipt.model_validate_json(
                (self.campaign_dir / "checkpoints/components" / f"{key}.json").read_text()
            )
            for key in job.dependency_ids
        ]

    def _parent_value(self, job: BenchmarkStageJob, key: str) -> Any:
        seen = set()
        queue = list(job.dependency_ids)
        while queue:
            identity = queue.pop(0)
            if identity in seen:
                continue
            seen.add(identity)
            receipt = ComponentReceipt.model_validate_json(
                (self.campaign_dir / "checkpoints/components" / f"{identity}.json").read_text()
            )
            if key in receipt.output:
                return receipt.output[key]
            queue.extend(receipt.upstream_receipt_hashes)
        raise ValueError("required_component_parent_output_missing_" + key)

    def _compiler_configs(self, scope: dict[str, Any]):
        from pmmc.compiler.doubter_config import load_doubter_config
        from pmmc.compiler.planner_config import load_planner_treatment_config
        from pmmc.compiler.questioner_config import load_questioner_config

        q = load_questioner_config(self.settings.resolve_path(self.config.questioner_config))
        d = load_doubter_config(self.settings.resolve_path(self.config.doubter_config))
        treatment = scope.get("treatment", "main")
        p = load_planner_treatment_config(
            self.settings.resolve_path(self.config.planner_config),
            treatment=treatment if treatment != "no_reranker" else "main",
        )
        if treatment == "no_raw_image":
            q = type(q).model_validate({**q.model_dump(), "memory_raw_image_policy": "forbid"})
        return q, p, d

    async def execute(
        self, job: BenchmarkStageJob, upstream: dict[str, str]
    ) -> tuple[JsonDict, list[str], dict[str, int]]:
        if set(upstream) != set(job.dependency_ids):
            raise ValueError("component_dependency_receipt_set_mismatch")
        scope = dict(job.parameters)
        handler = getattr(self, "_" + job.operation, None)
        if handler is None or job.operation.startswith("_"):
            raise ValueError("unregistered_production_component_operation")
        from pmmc.runtime.grounding import step8_prompt_scope

        with step8_prompt_scope(self.settings.project_root):
            result = await handler(job, scope)
        if is_dataclass(result) and not isinstance(result, type):
            result = asdict(result)
        if not isinstance(result, dict):
            raise TypeError("production component must return an audit record")
        # Canonical roundtrip rejects accidental non-serializable/private service objects.
        payload = json.loads(json.dumps(result))
        digest = self.store.put_json(payload)
        artifact_hashes = [digest]
        if "questioner_import_proof_sha256" in payload:
            artifact_hashes.append(payload["questioner_import_proof_sha256"])
        if "questioner_adoption" in payload:
            artifact_hashes.append(payload["questioner_adoption"]["source_proof_sha256"])
        if "r0_prefix_import_proof_sha256" in payload:
            artifact_hashes.append(payload["r0_prefix_import_proof_sha256"])
        if "r0_prefix_adoption" in payload:
            artifact_hashes.append(payload["r0_prefix_adoption"]["source_proof_sha256"])
        if "partial_prefix_adoption" in payload:
            artifact_hashes.append(payload["partial_prefix_adoption"]["source_proof_sha256"])
        return payload, list(dict.fromkeys(artifact_hashes)), {"component_outputs": 1}

    async def _snapshot(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        from pmmc.experiments.step8_snapshot import prepare_campaign_snapshot, snapshot_event
        from pmmc.infrastructure.database.importer import CanonicalSnapshotImporter
        from pmmc.infrastructure.database.storage_updates import (
            persist_asset_sync,
            persist_snapshot_archive,
        )
        from pmmc.infrastructure.object_store import (
            create_evaluator_object_stores,
            create_runtime_object_store,
        )
        from pmmc.infrastructure.storage_sync import archive_snapshot, sync_assets

        dataset = DatasetName(scope["dataset"])
        snapshot_event("role_credentials", "start", dataset=dataset.value)
        runtime_store = create_runtime_object_store(self.settings)
        _, evaluator_store = create_evaluator_object_stores(self.settings)
        snapshot_event("role_credentials", "done", minio_admin_required=False,
                       scope="configuration_only")
        path, snapshot, validation = await asyncio.to_thread(
            prepare_campaign_snapshot, self.settings, self.config, self.campaign,
            dataset, scope["snapshot_id"],
        )
        engine = create_engine(self.settings, DatabaseRole.ADMIN)
        try:
            snapshot_event("database_import", "start", dataset=dataset.value)
            await CanonicalSnapshotImporter(engine).import_snapshot(path)
            snapshot_event("database_import", "done", dataset=dataset.value)
            snapshot_event("archive", "start", dataset=dataset.value)
            archived = await archive_snapshot(
                path, runtime_store=runtime_store, evaluation_store=evaluator_store
            )
            await persist_snapshot_archive(engine, archived)
            snapshot_event("archive", "done", dataset=dataset.value)
            snapshot_event("asset_sync", "start", dataset=dataset.value)
            assets = await sync_assets(
                path, project_root=self.settings.benchmark_asset_path, runtime_store=runtime_store,
                expected_asset_sha256=validation["asset_sha256"],
            )
            await persist_asset_sync(engine, assets)
            snapshot_event("asset_sync", "done", dataset=dataset.value)
        finally:
            await engine.dispose()
        return {
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_path": str(path),
            "canonical_content_hash": snapshot.canonical_content_hash,
            "raw_validation": validation,
        }

    async def _corpus(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        from pmmc.baselines.corpus import build_corpus
        from pmmc.infrastructure.database.baseline_repositories import BaselineCorpusImporter
        from pmmc.models import TransformersTokenCounter

        registry = load_model_registry(self.settings.resolved_model_registry_path)
        entry = registry.get("qwen3-embedding-8b")
        counter = TransformersTokenCounter(
            entry.tokenizer_name,
            entry.tokenizer_revision,
            local_files_only=True,
            cache_dir=self.settings.model_cache_path,
        )
        path, manifest = await asyncio.to_thread(
            build_corpus,
            Path(self._parent_value(job, "snapshot_path")),
            output_root=self.settings.baseline_corpora_path,
            counter=counter,
        )
        engine = create_engine(self.settings, DatabaseRole.ADMIN)
        try:
            await BaselineCorpusImporter(engine).import_corpus(path)
        finally:
            await engine.dispose()
        return {"corpus_id": manifest.corpus_id, "corpus_path": str(path)}

    async def _units(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        from pmmc.experiments.step8_partial_import import partial_stage

        imported = await partial_stage(self, job, scope)
        if imported is not None:
            return imported
        from pmmc.compiler.preparation import CompilationUnitPreparationService
        from pmmc.infrastructure.database.repositories import (
            PostgresMemoryRepository,
            PostgresSessionRepository,
        )

        q, _, _ = self._compiler_configs(scope)
        engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
        try:
            selection = {str(scope["conversation_id"])}
            memories = await PostgresMemoryRepository(sessions).list_snapshot(
                str(scope["snapshot_id"]), conversation_ids=selection
            )
            source_sessions = await PostgresSessionRepository(sessions).list_snapshot(
                str(scope["snapshot_id"]), conversation_ids=selection
            )
        finally:
            await engine.dispose()
        preparation = CompilationUnitPreparationService(
            settings=self.settings,
            registry=load_model_registry(self.settings.resolved_model_registry_path),
            config=q,
            workers=self.config.execution.cpu_workers,
            max_tasks_per_child=self.execution_profile.unit_preparation.max_tasks_per_child,
            cache_policy=self.execution_profile.unit_preparation.cache_policy,
        )
        manifest, _, path = await asyncio.to_thread(
            preparation.prepare,
            run_id=compiler_instance_id(self.campaign, scope),
            snapshot_id=str(scope["snapshot_id"]),
            sessions=source_sessions,
            memories=memories,
        )
        return {
            "unit_preparation": str(path),
            "preparation_id": manifest.preparation_id,
            "compiler_run_id": compiler_instance_id(self.campaign, scope),
        }

    async def _compiler(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        from pmmc.experiments.step8_partial_import import partial_stage

        imported = await partial_stage(self, job, scope)
        if imported is not None:
            return imported
        from pmmc.compiler.program_workflow import ProgramWorkflowPhase, run_program_workflow
        from pmmc.experiments.step8_upstream_import import (
            seal_r0_prefix,
            try_r0_prefix_import,
            verified_r0_prefix_stage,
        )

        if job.stage == "questioner":
            imported = await try_r0_prefix_import(self, scope)
            if imported is not None:
                return imported
            imported = await self._try_questioner_import(scope)
            if imported is not None:
                return imported
        else:
            imported = await verified_r0_prefix_stage(self, scope, job.stage)
            if imported is not None:
                return imported

        phases = {
            "questioner": ["questioner"],
            "planner0": ["planner"],
            "text_index": ["qbank_index_text", "memory_index_text", "program_query_text"],
            "multimodal_index": [
                "qbank_index_multimodal",
                "memory_index_multimodal",
                "program_query_multimodal",
            ],
            "dry_run0": ["dry_run"],
            "r0_gate": ["r0_input"],
            "doubter": ["doubter"],
            "projection": ["decision", "finalize"],
        }
        q, p, d = self._compiler_configs(scope)
        result = None
        for phase in phases[job.stage]:
            if phase in {"qbank_index_text", "qbank_index_multimodal"}:
                adopted = await self._index_adopted_questioner(scope, phase)
                if adopted:
                    continue
            result = await run_program_workflow(
                settings=self.settings,
                registry=load_model_registry(self.settings.resolved_model_registry_path),
                questioner=q,
                planner=p,
                doubter=d,
                snapshot_id=str(scope["snapshot_id"]),
                questioner_freeze_or_audit_hash=compiler_subprotocol_hash(self.campaign, scope),
                questioner_identity_kind="step8_method_protocol",
                phase=ProgramWorkflowPhase(phase),
                run_id=compiler_instance_id(self.campaign, scope),
                conversation_ids={str(scope["conversation_id"])},
                seed=self.campaign.protocol.seed,
                split="paper_test" if self.campaign.paper_result else "development",
                unit_preparation=Path(self._parent_value(job, "unit_preparation"))
                if phase == "questioner"
                else None,
            )
        assert result is not None
        output: dict[str, Any] = {
            "compiler_run_id": result.run_id,
            "phase": result.phase.value,
            "program_count": result.program_count,
            "run_status": result.run_status,
            "eligible_candidate_count": result.eligible_candidate_count,
        }
        if job.stage == "questioner":
            output.update(await self._seal_questioner(scope))
        if job.stage == "r0_gate" and not self.config.metadata.get("partial_import_policy"):
            output.update(await seal_r0_prefix(self, scope))
        elif job.stage == "r0_gate":
            output["r0_import_proof"] = "not_sealed_for_partial_migration"
        if job.stage == "projection":
            from pmmc.experiments.step8_instances import build_compiled_qbank_instance_manifest

            engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
            try:
                instance = await build_compiled_qbank_instance_manifest(
                    sessions,
                    protocol=self.campaign.protocol,
                    compiler_run_id=result.run_id,
                    compiler_subprotocol_hash=compiler_subprotocol_hash(self.campaign, scope),
                    scope=scope,
                    diagnostic_namespace=_compiler_namespace(self.campaign),
                )
            finally:
                await engine.dispose()
            output["compiled_instance"] = instance.model_dump(mode="json")
            output["compiled_instance_hash"] = instance.content_hash
        return output

    async def _index_adopted_questioner(self, scope: dict[str, Any], phase: str) -> bool:
        """Reuse the normal index builder after proving adopted Q ownership.

        The ordinary Questioner workflow reconstructs generation metadata and
        units for equality checking. Imported facts intentionally preserve their
        producer lineage, so they must use a verified, persisted-unit index path.
        """
        from pmmc.compiler.workflow import QuestionerWorkflowPhase, _run_index_phase
        from pmmc.experiments.step8_upstream_import import questioner_compatibility
        from pmmc.infrastructure.database.compiler_repositories import (
            PostgresCompilerUnitRepository,
        )
        from pmmc.infrastructure.database.step8_upstream_import import (
            PostgresQuestionerImportRepository,
        )
        from pmmc.infrastructure.object_store import create_runtime_object_store

        run_id = compiler_instance_id(self.campaign, scope)
        engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
        try:
            repository = PostgresQuestionerImportRepository(
                sessions, create_runtime_object_store(self.settings)
            )
            if not await repository.verify_adopted(
                run_id, questioner_compatibility(self.campaign, scope)
            ):
                return False
            async with sessions() as session:
                units = await PostgresCompilerUnitRepository(session).list_for_run(run_id)
            questioner, _, _ = self._compiler_configs(scope)
            await _run_index_phase(
                settings=self.settings,
                registry=load_model_registry(self.settings.resolved_model_registry_path),
                config=questioner,
                sessions=sessions,
                run_id=run_id,
                snapshot_id=str(scope["snapshot_id"]),
                conversation_count=1,
                unit_values=units,
                phase=(
                    QuestionerWorkflowPhase.INDEX_TEXT
                    if phase == "qbank_index_text"
                    else QuestionerWorkflowPhase.INDEX_MULTIMODAL
                ),
                complete_run=False,
            )
            print(
                f"STEP8_QUESTIONER_ADOPTION_INDEX state=verified phase={phase} "
                f"unit_count={len(units)} questioner_provider_calls=0",
                flush=True,
            )
            return True
        finally:
            await engine.dispose()

    async def _seal_questioner(self, scope: dict[str, Any]) -> dict[str, Any]:
        from pmmc.experiments.step8_upstream_import import publish_proof, questioner_compatibility
        from pmmc.infrastructure.database.step8_upstream_import import (
            PostgresQuestionerImportRepository,
        )
        from pmmc.infrastructure.object_store import create_runtime_object_store

        engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
        try:
            proof = await PostgresQuestionerImportRepository(
                sessions, create_runtime_object_store(self.settings)
            ).seal(
                run_id=compiler_instance_id(self.campaign, scope),
                campaign_id=self.campaign.campaign_id,
                protocol_hash=self.campaign.protocol.content_hash,
                compatibility=questioner_compatibility(self.campaign, scope),
            )
        finally:
            await engine.dispose()
        return {"questioner_import_proof_sha256": publish_proof(self.store, proof)}

    async def _adopt_questioner(
        self, job: BenchmarkStageJob, scope: dict[str, Any]
    ) -> dict[str, Any]:
        if scope.get("treatment") != "fixed_planner":
            raise ValueError("questioner_controlled_adoption_requires_fixed_planner")
        from pmmc.experiments.step8_partial_import import partial_stage

        imported = await partial_stage(self, job, scope)
        if imported is not None:
            return imported
        proof_hash = str(self._parent_value(job, "questioner_import_proof_sha256"))
        result = await self._try_questioner_import(scope, required_proof_hash=proof_hash)
        if result is None:
            raise ValueError("fixed_planner_cannot_generate_independent_questioner")
        return result

    async def _try_questioner_import(
        self,
        scope: dict[str, Any],
        *,
        required_proof_hash: str | None = None,
        prefix_proof: Any | None = None,
    ) -> dict[str, Any] | None:
        from pmmc.compiler.doubter_config import doubter_config_hash
        from pmmc.compiler.doubter_prompt import (
            doubter_prompt_sha256,
            doubter_response_schema_sha256,
        )
        from pmmc.compiler.planner_config import planner_config_hash
        from pmmc.compiler.planner_prompt import (
            planner_prompt_sha256,
            planner_response_schema_sha256,
        )
        from pmmc.compiler.questioner_prompt import questioner_prompt_sha256
        from pmmc.experiments.step8_upstream_import import (
            QuestionerImportProof,
            compatible_proofs,
            questioner_compatibility,
        )
        from pmmc.infrastructure.database.step8_upstream_import import (
            PostgresQuestionerImportRepository,
        )
        from pmmc.infrastructure.object_store import create_runtime_object_store

        target = compiler_instance_id(self.campaign, scope)
        compatibility = questioner_compatibility(self.campaign, scope)
        q, p, d = self._compiler_configs(scope)
        engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
        try:
            repository = PostgresQuestionerImportRepository(
                sessions, create_runtime_object_store(self.settings)
            )
            registered, lineage = await repository.target_registration(target)
            if registered and lineage is None:
                if required_proof_hash is not None:
                    raise ValueError("fixed_planner_questioner_already_generated_independently")
                return None
            if required_proof_hash is not None:
                proof = QuestionerImportProof.model_validate(
                    self.store.load_json(required_proof_hash)
                )
            else:
                proofs = compatible_proofs(self.store, compatibility, exclude_run_id=target)
                if lineage is not None:
                    proofs = [
                        value for value in proofs if value.content_hash == lineage["proof_hash"]
                    ]
                    if not proofs:
                        raise ValueError("questioner_import_resume_proof_missing")
                if not proofs:
                    return None
                proof = proofs[0]
            registry = load_model_registry(self.settings.resolved_model_registry_path)
            metadata = {
                "compiler_subprotocol_hash": compiler_subprotocol_hash(self.campaign, scope),
                "questioner_freeze_or_audit_hash": compiler_subprotocol_hash(self.campaign, scope),
                "questioner_identity_kind": "step8_method_protocol",
                "planner_config_hash": planner_config_hash(p),
                "planner_response_schema_hash": planner_response_schema_sha256(p),
                "doubter_response_schema_hash": doubter_response_schema_sha256(d),
                "active_round": d.active_round,
                "max_round": d.max_round,
            }
            imported = await repository.adopt(
                proof=proof,
                compatibility=compatibility,
                target_run_id=target,
                compiler_updates={
                    "questioner_freeze_manifest_hash": compiler_subprotocol_hash(
                        self.campaign, scope
                    ),
                    "planner_config_hash": planner_config_hash(p),
                    "planner_model_alias": p.planner_model_alias,
                    "planner_model_revision": registry.get(p.planner_model_alias).model_revision,
                    "planner_prompt_version": p.prompt_version,
                    "planner_prompt_hash": planner_prompt_sha256(p),
                    "planner_renderer_version": p.renderer_version,
                    "planner_parser_version": p.parser_version,
                    "planner_schema_version": p.schema_version,
                    "program_policy_version": p.policy_version,
                    "dry_run_policy_version": p.dry_run_policy,
                    "doubter_config_hash": doubter_config_hash(d),
                    "doubter_model_alias": d.doubter_model_alias,
                    "doubter_model_revision": registry.get(d.doubter_model_alias).model_revision,
                    "doubter_prompt_version": d.prompt_version,
                    "doubter_prompt_hash": doubter_prompt_sha256(d),
                    "doubter_renderer_version": d.renderer_version,
                    "doubter_parser_version": d.parser_version,
                    "doubter_schema_version": d.schema_version,
                    "r0_input_policy_version": d.r0_input_policy_version,
                    "routing_projection_version": d.routing_projection_version,
                    "active_round": d.active_round,
                    "max_round": d.max_round,
                },
                manifest_metadata=metadata,
                prompt_hashes={
                    q.prompt_version: questioner_prompt_sha256(q.prompt_version),
                    q.verifier_prompt_version: questioner_prompt_sha256(q.verifier_prompt_version),
                    d.prompt_version: doubter_prompt_sha256(d),
                },
            )
            imported["source_proof_sha256"] = self.store.put_json(proof.model_dump(mode="json"))
            prefix_imported = None
            if prefix_proof is not None:
                from pmmc.experiments.step8_upstream_import import r0_prefix_compatibility
                from pmmc.infrastructure.database.step8_upstream_import import (
                    PostgresR0PrefixImportRepository,
                )

                prefix_imported = await PostgresR0PrefixImportRepository(
                    sessions, create_runtime_object_store(self.settings)
                ).adopt_prefix(
                    proof=prefix_proof,
                    compatibility=r0_prefix_compatibility(self.campaign, scope),
                    target_run_id=target,
                )
                self.store.put_json(prefix_proof.model_dump(mode="json"))
        finally:
            await engine.dispose()
        print(
            "STEP8_QUESTIONER_ADOPTION "
            + json.dumps(
                {
                    "state": "verified",
                    "treatment": scope.get("treatment", "main"),
                    "eligible_count": proof.eligible_count,
                    "unit_count": proof.unit_count,
                    "provider_calls": 0,
                    "reused": imported["reused"],
                    "source_proof_hash": proof.content_hash,
                    "downstream_rebuilt": []
                    if prefix_imported
                    else ["planner0", "indexes", "dry_run0", "r0_gate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return {
            "compiler_run_id": target,
            "phase": "questioner",
            "program_count": 0,
            "eligible_candidate_count": proof.eligible_count,
            "run_status": "running",
            "questioner_adoption": imported,
            **({"r0_prefix_adoption": prefix_imported} if prefix_imported else {}),
            **await self._seal_questioner(scope),
        }

    async def _baseline_index(
        self, job: BenchmarkStageJob, scope: dict[str, Any]
    ) -> dict[str, Any]:
        from pmmc.baselines.rag_config import load_rag_config
        from pmmc.baselines.rag_index import build_rag_index
        from pmmc.schemas import BaselineMethod

        result = await build_rag_index(
            settings=self.settings,
            registry=load_model_registry(self.settings.resolved_model_registry_path),
            config=load_rag_config(
                self._answer_settings(
                    self.campaign.protocol.answer_backbones[0],
                    caption_only=scope.get("treatment") == "no_raw_image",
                ).resolved_rag_config_path
            ),
            corpus_id=str(self._parent_value(job, "corpus_id")),
            method=BaselineMethod(scope["method"]),
        )
        return {
            "collection": result.collection.model_dump(mode="json"),
            "run_id": result.run_id,
            "document_count": result.document_count,
            "reused": result.reused,
        }

    def _answer_settings(self, alias: str, *, caption_only: bool = False) -> Settings:
        files = {}
        for kind, source in (
            ("full", self.settings.resolved_full_context_config_path),
            ("rag", self.settings.resolved_rag_config_path),
        ):
            value = yaml.safe_load(source.read_text())
            value.update(
                answer_model_alias=alias,
                answer_contract_version="typed-answer-contract-v3",
                reserved_output_tokens=512,
                max_output_tokens=512,
                max_input_tokens=32256,
                total_context_tokens=32768,
            )
            if kind == "full":
                value.update(
                    prompt_version="pmmc-answer-contract-v3",
                    prompt_path="prompts/runtime/pmmc_answer_contract_v3.txt",
                )
            else:
                value.update(
                    answer_prompt_version="pmmc-answer-contract-v3",
                    answer_prompt_path="prompts/runtime/pmmc_answer_contract_v3.txt",
                )
                value["memory_raw_image_policy"] = "forbid" if caption_only else "allow"
            suffix = "-caption-only" if caption_only else ""
            path = self.campaign_dir / "private/configs" / f"{kind}-{alias}{suffix}.json"
            write_json(path, value, immutable=True)
            files[kind] = path
        return self.settings.model_copy(
            update={"full_context_config_path": files["full"], "rag_config_path": files["rag"]}
        )

    async def _suite(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        from pmmc.baselines.suite import create_baseline_suite
        from pmmc.schemas import BaselineMethod, BenchmarkSelection

        caption_only = scope.get("treatment") == "no_raw_image"
        settings = self._answer_settings(str(scope["answer_model"]), caption_only=caption_only)
        manifest, path = await create_baseline_suite(
            settings=settings,
            snapshot_id=str(scope["snapshot_id"]),
            selection=BenchmarkSelection(
                datasets=[DatasetName(scope["dataset"])], query_ids=[str(scope["query_id"])]
            ),
            corpus_id=str(self._parent_value(job, "corpus_id")),
            answer_model_alias=str(scope["answer_model"]),
            methods=[BaselineMethod.NATIVE_MULTIMODAL_RAG] if caption_only else None,
            metadata={
                "partition_id": self.campaign.partition_id,
                "partition_role": "paper_test" if self.campaign.paper_result else "development",
                "paper_result": self.campaign.paper_result,
                "method_protocol_hash": self.campaign.protocol.content_hash,
                "answer_contract_version": "typed-answer-contract-v3",
                "answer_prompt_version": "pmmc-answer-contract-v3",
                "allow_memory_raw_images": not caption_only,
                **({"internal_pmmc_fallback": True} if caption_only else {}),
            },
        )
        return {"suite_id": manifest.suite_id, "suite_path": str(path)}

    async def _baseline_retrieve(
        self, job: BenchmarkStageJob, scope: dict[str, Any]
    ) -> dict[str, Any]:
        from pmmc.baselines.rag_workflow import RagRunPhase
        from pmmc.baselines.suite import run_baseline_suite
        from pmmc.schemas import BaselineMethod

        suite_id = str(self._parent_value(job, "suite_id"))
        await run_baseline_suite(
            settings=self._answer_settings(
                str(scope["answer_model"]), caption_only=scope.get("treatment") == "no_raw_image"
            ),
            suite_id=suite_id,
            phase=RagRunPhase.RETRIEVE,
            methods=[BaselineMethod(scope["method"])],
        )
        return {"suite_id": suite_id, "retrieved_method": scope["method"]}

    async def _baseline_answer(
        self, job: BenchmarkStageJob, scope: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._baseline_phase(job, scope, "generate")

    async def _baseline_grounding(
        self, job: BenchmarkStageJob, scope: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._baseline_phase(job, scope, "ground")

    async def _native_fallback_generate(self, job, scope):
        return await self._baseline_phase(job, scope, "generate", native_only=True)

    async def _native_fallback_ground(self, job, scope):
        return await self._baseline_phase(job, scope, "ground", native_only=True)

    async def _baseline_phase(
        self, job: BenchmarkStageJob, scope: dict[str, Any], phase: str, *,
        native_only: bool = False,
    ) -> dict[str, Any]:
        from pmmc.baselines.rag_workflow import RagRunPhase
        from pmmc.baselines.suite import run_baseline_suite
        from pmmc.schemas import BaselineMethod

        suite_id = str(self._parent_value(job, "suite_id"))
        await run_baseline_suite(
            settings=self._answer_settings(
                str(scope["answer_model"]), caption_only=scope.get("treatment") == "no_raw_image"
            ),
            suite_id=suite_id,
            phase=RagRunPhase(phase),
            methods=[BaselineMethod.NATIVE_MULTIMODAL_RAG] if native_only else None,
        )
        return {"suite_id": suite_id, "answer_phase": phase}

    async def _query_embedding(
        self, job: BenchmarkStageJob, scope: dict[str, Any]
    ) -> dict[str, Any]:
        from pmmc.runtime.query_executor_config import load_query_executor_config
        from pmmc.runtime.query_workflow import prepare_query_executor_runtime_embeddings
        from pmmc.schemas.query_executor import QBankRuntimeChannel

        result = await prepare_query_executor_runtime_embeddings(
            settings=self.settings,
            registry=load_model_registry(self.settings.resolved_model_registry_path),
            source_compiler_run_id=compiler_instance_id(self.campaign, scope),
            snapshot_id=str(scope["snapshot_id"]),
            query_ids=[str(scope["query_id"])],
            channel=QBankRuntimeChannel("text" if job.stage == "text_index" else "multimodal"),
            joint_evidence=(job.stage == "multimodal_index" and scope["treatment"] != "no_raw_image"
                            and load_query_executor_config(self.settings.resolve_path(
                                self.config.query_executor_config
                            )).joint_relevance_enabled),
        )
        return dict(result)

    async def _query_match(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        return await self._query_phase(job, scope, "match")

    async def _query_execute(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        # The match phase checkpoints actual executions, not a synthetic success counter.
        run_id = str(self._parent_value(job, "query_executor_run_id"))
        from sqlalchemy import select

        from pmmc.infrastructure.database.query_executor_models import QueryRuntimePhaseStateModel

        engine, sessions = create_session_factory(self.settings, DatabaseRole.ONLINE)
        try:
            async with sessions() as session:
                row = await session.scalar(
                    select(QueryRuntimePhaseStateModel).where(
                        QueryRuntimePhaseStateModel.query_executor_run_id == run_id,
                        QueryRuntimePhaseStateModel.query_id == str(scope["query_id"]),
                        QueryRuntimePhaseStateModel.phase == "match",
                    )
                )
                if row is None:
                    raise ValueError("runtime_match_execution_checkpoint_missing")
                return {"query_executor_run_id": run_id, "phase_hash": row.content_hash}
        finally:
            await engine.dispose()

    async def _query_answer(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        return await self._query_phase(job, scope, "answer")

    async def _query_grounding(
        self, job: BenchmarkStageJob, scope: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._query_phase(job, scope, "select")

    async def _query_phase(
        self, job: BenchmarkStageJob, scope: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        from pmmc.baselines.suite import baseline_suite_child_run_id
        from pmmc.experiments.step8_instances import verify_compiled_qbank_instance_manifest
        from pmmc.runtime.query_executor_config import load_query_executor_config
        from pmmc.runtime.query_workflow import run_query_executor_workflow
        from pmmc.schemas import BaselineMethod
        from pmmc.schemas.step8 import CompiledQBankInstanceManifest, MethodProtocolFreeze

        instance = CompiledQBankInstanceManifest.model_validate(
            self._parent_value(job, "compiled_instance")
        )
        freeze_path = self.campaign_dir / "manifests/method-freeze.json"
        freeze = (
            MethodProtocolFreeze.model_validate_json(freeze_path.read_text())
            if freeze_path.is_file()
            else None
        )
        engine, sessions = create_session_factory(self.settings, DatabaseRole.RUNTIME)
        try:
            await verify_compiled_qbank_instance_manifest(
                sessions,
                protocol=self.campaign.protocol,
                compiler_run_id=compiler_instance_id(self.campaign, scope),
                compiler_subprotocol_hash=compiler_subprotocol_hash(self.campaign, scope),
                scope=scope,
                manifest=instance,
                expected_manifest_hash=str(self._parent_value(job, "compiled_instance_hash")),
                freeze=freeze,
                paper=self.campaign.paper_result,
                diagnostic_namespace=_compiler_namespace(self.campaign),
            )
        finally:
            await engine.dispose()

        config = load_query_executor_config(
            self.settings.resolve_path(self.config.query_executor_config)
        )
        # Treatment-specific config validation is mandatory; never mutate a frozen program online.
        protocol = self.campaign.protocol
        config = type(config).model_validate(
            {
                **config.model_dump(),
                "treatment": scope["treatment"],
                "experiment_seed": protocol.seed,
                "active_match_threshold": protocol.match_threshold,
                "active_gap_threshold": protocol.gap_threshold,
                "no_reranker_match_threshold": protocol.no_reranker_match_threshold,
                "no_reranker_gap_threshold": protocol.no_reranker_gap_threshold,
                "query_evidence_relevance_threshold": protocol.text_relevance_threshold,
                "multimodal_evidence_relevance_threshold": protocol.multimodal_relevance_threshold,
            }
        )
        suite_id = str(self._parent_value(job, "suite_id"))
        fallback_id = baseline_suite_child_run_id(suite_id, BaselineMethod.NATIVE_MULTIMODAL_RAG)
        run_id = (
            "step8-query-"
            + canonical_content_hash(
                {
                    "protocol": self.campaign.protocol.content_hash,
                    "scope": scope,
                }
            )[:24]
        )
        result = await run_query_executor_workflow(
            settings=self._answer_settings(
                str(scope["answer_model"]), caption_only=scope.get("treatment") == "no_raw_image"
            ),
            registry=load_model_registry(self.settings.resolved_model_registry_path),
            config=config,
            foundation_config_path=self.settings.resolve_path(self.config.foundation_config),
            source_compiler_run_id=compiler_instance_id(self.campaign, scope),
            coverage_run_id=None,
            fallback_run_id=fallback_id,
            query_ids=[str(scope["query_id"])],
            pilot_projection=not self.campaign.paper_result,
            run_id=run_id,
            answer_model_alias=str(scope["answer_model"]),
            phase=phase,
            primary_only=scope.get("readout") == "primary",
            source_match_run_id=(
                "step8-query-"
                + canonical_content_hash(
                    {
                        "protocol": self.campaign.protocol.content_hash,
                        "scope": {
                            **scope,
                            "answer_model": self.campaign.protocol.answer_backbones[0],
                        },
                    }
                )[:24]
                if scope["answer_model"] != self.campaign.protocol.answer_backbones[0]
                else None
            ),
        )
        return {
            "query_executor_run_id": run_id,
            "suite_id": suite_id,
            "phase": phase,
            "workflow": {
                "manifest": result.manifest.model_dump(mode="json"),
                "batch": asdict(result.batch),
                "answer_prompt_hash": result.answer_prompt_hash,
                "imported_text_batch_count": result.imported_text_batch_count,
                "imported_multimodal_batch_count": result.imported_multimodal_batch_count,
            },
        }

    async def _evaluate(self, job: BenchmarkStageJob, scope: dict[str, Any]) -> dict[str, Any]:
        from pmmc.evaluation.step8_workflow import evaluate_step8_query
        from pmmc.schemas.step8 import CampaignPhase

        if self.campaign.phase == CampaignPhase.EXPLORATION:
            return await self._scoped_evaluate(job, scope, "paired_main")
        return await evaluate_step8_query(
            settings=self.settings,
            campaign_id=self.campaign.campaign_id,
            query_executor_run_id=str(self._parent_value(job, "query_executor_run_id")),
            suite_id=str(self._parent_value(job, "suite_id")),
            query_id=str(scope["query_id"]),
        )

    async def _primary_evaluate(self, job, scope):
        return await self._scoped_evaluate(job, scope, "pmmc_main")

    async def _analysis_evaluate(self, job, scope):
        return await self._scoped_evaluate(job, scope, "pmmc")

    async def _scoped_evaluate(self, job, scope, method_scope):
        from pmmc.evaluation.step8_workflow import evaluate_step8_query

        return await evaluate_step8_query(
            self.settings, self.campaign.campaign_id,
            str(self._parent_value(job, "query_executor_run_id")),
            str(self._parent_value(job, "suite_id")), str(scope["query_id"]),
            method_scope=method_scope,
        )
