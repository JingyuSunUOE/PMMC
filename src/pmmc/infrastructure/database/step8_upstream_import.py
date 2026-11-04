"""Transactional, FK-preserving adoption of a sealed Questioner fact graph."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any, cast

from sqlalchemy import Table, insert, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import canonical_content_hash, question_content_hash
from pmmc.experiments.manifest import run_config_hash
from pmmc.experiments.step8_artifacts import json_bytes
from pmmc.experiments.step8_upstream_import import (
    IMPORT_VERSION,
    R0_ADOPTION_VERSION,
    R0_PREFIX_STAGES,
    QuestionerImportProof,
    R0PrefixImportProof,
    assert_compatible,
    assert_r0_prefix_compatible,
    questioner_compatibility,
    r0_prefix_compatibility,
    remap_questioner_rows,
)
from pmmc.infrastructure.database import compiler_models as c
from pmmc.infrastructure.database import questioner_models as q
from pmmc.infrastructure.database.compiler_mappers import question_from_rows
from pmmc.infrastructure.database.doubter_models import R0InputManifestModel
from pmmc.infrastructure.database.experiment_models import (
    EmbeddingCollectionModel,
    EmbeddingModel,
    ExperimentJobModel,
    JobAttemptModel,
    RuntimeArtifactModel,
    RuntimeModelCallModel,
)
from pmmc.infrastructure.database.experiment_repositories import PostgresRunRepository
from pmmc.infrastructure.database.models import RunManifestModel
from pmmc.infrastructure.database.questioner_repositories import questioner_v3_unit_issues
from pmmc.infrastructure.database.step8_models import (
    BenchmarkCampaignModel,
    BenchmarkComponentReceiptModel,
    BenchmarkStageJobModel,
)
from pmmc.infrastructure.object_store import ObjectStore
from pmmc.schemas import RunManifest, RunStatus
from pmmc.schemas.base import utc_now
from pmmc.schemas.step8 import STEP8_ALEMBIC_REVISION, BenchmarkCampaignManifest


class QuestionerImportConflict(ValueError):
    """Expose a fixed diagnostic code without rendering private fact values."""

    def __init__(self, code: str):
        self.error_code = code
        super().__init__(code)


def _index_diagnostic(row: Mapping[str, Any]) -> bool:
    """Recognize non-gating text-index links, including R0 ownership projections."""
    method, score = row.get("method"), row.get("score")
    metadata = row.get("metadata")
    if not (
        isinstance(method, str)
        and method.endswith(":semantic-nearest")
        and row.get("hard_duplicate") is False
        and isinstance(metadata, dict)
        and metadata.get("diagnostic_only") is True
        and isinstance(score, dict)
        and score.get("score_type") == "cosine"
        and isinstance(score.get("metadata"), dict)
        and isinstance(score["metadata"].get("collection_id"), str)
        and score["metadata"]["collection_id"]
    ):
        return False
    return True


def _questioner_duplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The same table contains Q-owned dedupe and later index-owned diagnostics.
    # Do not remove rows from the database or omit them from full R0 proofs.
    result = [row for row in rows if not _index_diagnostic(row)]
    if len(result) != len(rows):
        print(
            f"STEP8_QUESTIONER_FACT_SCOPE retained_dedupe={len(result)} "
            f"index_diagnostics={len(rows) - len(result)} database_rows_deleted=0",
            flush=True,
        )
    return result


# An explicit allowlist is essential: a new compiler/evaluation table must not
# silently become importable merely because it has a run_id column.
QUESTIONER_TABLES: tuple[Table, ...] = tuple(
    cast(Table, model.__table__)
    for model in (
        c.CompilerUnitModel,
        c.CompilerUnitMemoryModel,
        q.CompilerUnitLinkModel,
        q.QuestionOpportunitySetModel,
        q.QuestionOpportunityModel,
        q.QuestionOpportunityEvidenceRefModel,
        RuntimeArtifactModel,
        ExperimentJobModel,
        JobAttemptModel,
        RuntimeModelCallModel,
        q.QuestionerGenerationModel,
        c.QBankQuestionModel,
        c.QBankParaphraseModel,
        c.QBankEvidenceRefModel,
        q.QuestionerAnswerClaimModel,
        q.QuestionerVerificationBatchModel,
        q.QuestionerCandidateVerificationModel,
        q.QuestionerCandidateSupersessionModel,
        q.QuestionerCandidateGateResultModel,
        q.QBankDuplicateLinkModel,
        q.QBankCandidateFamilyModel,
        q.QBankCandidateFamilyMemberModel,
        q.RuntimeStructuredOutputItemDiagnosticModel,
        c.CompilerUnitStageResultModel,
    )
)

# Round revisions, Doubter feedback, projections, online execution and evaluator
# tables are intentionally absent. Selection below is by real FK ownership, not
# by a stage receipt's declared counts.
R0_PREFIX_TABLES: tuple[Table, ...] = tuple(
    cast(Table, model.__table__)
    for model in (
        c.CompilerStageJobModel,
        c.PlannerGenerationBatchModel,
        c.PlannerRequirementContractModel,
        c.PlannerCandidateDraftModel,
        c.MemoryProgramModel,
        c.ProgramStepModel,
        c.ProgramExpectedEvidenceRefModel,
        c.PlannerBatchProgramModel,
        c.PlannerCandidateOutcomeModel,
        c.ProgramValidationResultModel,
        c.ProgramValidationIssueModel,
        EmbeddingCollectionModel,
        EmbeddingModel,
        q.QBankIndexManifestModel,
        c.MemoryOperatorIndexManifestModel,
        c.ProgramQueryEmbeddingBatchModel,
        c.ProgramQueryEmbeddingModel,
        c.ProgramExecutionModel,
        c.ProgramExecutionStepModel,
        c.ProgramExecutionEvidenceRefModel,
        c.ProgramDryRunAssessmentModel,
        c.ProgramDryRunEvidenceMatchModel,
        R0InputManifestModel,
    )
)


def r0_prefix_table_contract_hash() -> str:
    return canonical_content_hash(
        {
            table.name: {
                "columns": [
                    (column.name, str(column.type), column.nullable, column.computed is not None)
                    for column in table.columns
                ],
                "primary_key": [column.name for column in table.primary_key],
                "foreign_keys": sorted(
                    (fk.parent.name, fk.target_fullname) for fk in table.foreign_keys
                ),
            }
            for table in (*QUESTIONER_TABLES, *R0_PREFIX_TABLES)
        }
    )


def table_contract_hash() -> str:
    return canonical_content_hash(
        {
            table.name: [
                (column.name, str(column.type), column.nullable, column.computed is not None)
                for column in table.columns
            ]
            for table in QUESTIONER_TABLES
        }
    )


def _sort_rows(table: Table, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [column.name for column in table.primary_key.columns]
    rows.sort(key=lambda row: tuple(str(row[key]) for key in keys))
    if table.name == "qbank_questions":
        rows.sort(key=lambda row: int(row["candidate_version"]))
    if table.name == "model_calls":
        remaining = {row["model_call_id"]: row for row in rows}
        rows = []
        while remaining:
            ready = [
                row
                for row in remaining.values()
                if row["parent_model_call_id"] is None
                or row["parent_model_call_id"] in {item["model_call_id"] for item in rows}
            ]
            if not ready:
                raise ValueError("questioner_import_model_call_parent_not_closed")
            for row in ready:
                rows.append(row)
                del remaining[row["model_call_id"]]
    return rows


def _rehash_questions(rows: dict[str, list[dict[str, Any]]]) -> None:
    # Candidate hashes bind run/unit/owned IDs. Producer bytes remain immutable,
    # but the receiving candidate must satisfy the normal repository hash rule.
    candidate_hashes = {}
    for row in rows["qbank_questions"]:
        question = question_from_rows(
            row,
            [p for p in rows["qbank_paraphrases"] if p["qbank_id"] == row["qbank_id"]],
            [e for e in rows["qbank_evidence_refs"] if e["qbank_id"] == row["qbank_id"]],
        )
        row["metadata"]["upstream_import"]["source_question_hash"] = row["question_hash"]
        row["question_hash"] = question_content_hash(question)
        if row["candidate_content_hash"] is not None:
            row["candidate_content_hash"] = row["question_hash"]
        candidate_hashes[row["qbank_id"]] = row["candidate_content_hash"]
    # Gate hashes reference the receiving candidate, not the producer's ID scope.
    # remap_questioner_rows has already preserved each source row hash in metadata.
    for gate in rows["questioner_candidate_gate_results"]:
        content_hash = candidate_hashes.get(gate["qbank_id"])
        if content_hash is None:
            raise ValueError("questioner_import_gate_candidate_hash_missing")
        gate["content_hash"] = content_hash


class PostgresQuestionerImportRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], store: ObjectStore):
        self.sessions, self.store = sessions, store

    async def target_registration(self, run_id: str) -> tuple[bool, dict[str, Any] | None]:
        async with self.sessions() as session:
            compiler = await session.get(c.CompilerRunModel, run_id)
            return (
                compiler is not None,
                compiler.metadata_json.get("questioner_upstream_import") if compiler else None,
            )

    async def verify_adopted(self, run_id: str, compatibility: dict[str, Any]) -> bool:
        """Validate the persisted adoption graph before an index-only consumer."""
        async with self.sessions() as session:
            compiler = await session.get(c.CompilerRunModel, run_id)
            lineage = compiler.metadata_json.get("questioner_upstream_import") if compiler else None
            if lineage is None:
                return False
            if (
                lineage.get("version") != IMPORT_VERSION
                or lineage.get("compatibility_hash") != canonical_content_hash(compatibility)
                or lineage.get("imported_stages") != ["questioner"]
            ):
                raise QuestionerImportConflict("questioner_import_consumer_identity_mismatch")
            headers, rows = await self._read(session, run_id)
            self._scope(headers, compatibility)
            if canonical_content_hash(rows) != lineage["target_rows_hash"]:
                raise QuestionerImportConflict("questioner_import_consumer_fact_conflict")
            source_headers, source_rows = await self._read(session, lineage["source_run_id"])
            if (
                canonical_content_hash({"headers": source_headers, "rows": source_rows})
                != lineage["source_facts_hash"]
            ):
                raise QuestionerImportConflict("questioner_import_consumer_source_conflict")
        await self._artifacts(rows["run_artifacts"])
        return True

    async def _read(
        self,
        session: AsyncSession,
        run_id: str,
        *,
        validate: bool = True,
        include_index_diagnostics: bool = False,
    ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
        revisions = list(
            await session.scalars(text("SELECT version_num FROM public.alembic_version"))
        )
        if revisions != [STEP8_ALEMBIC_REVISION]:
            raise ValueError("questioner_import_database_revision_mismatch")
        compiler_row = (
            (
                await session.execute(
                    select(c.CompilerRunModel.__table__).where(c.CompilerRunModel.run_id == run_id)
                )
            )
            .mappings()
            .one_or_none()
        )
        manifest_row = await session.get(RunManifestModel, run_id)
        if compiler_row is None or manifest_row is None:
            raise ValueError("questioner_import_source_facts_missing")
        compiler = dict(compiler_row)
        if compiler["compiler_schema_version"] != "0.9.0":
            raise ValueError("questioner_import_legacy_compiler_forbidden")
        rows: dict[str, list[dict[str, Any]]] = {}
        calls = list(
            (
                await session.execute(
                    select(RuntimeModelCallModel.__table__).where(
                        RuntimeModelCallModel.run_id == run_id,
                        RuntimeModelCallModel.stage == "questioner",
                    )
                )
            ).mappings()
        )
        job_ids = {row["job_id"] for row in calls}
        artifact_ids = {
            row[key]
            for row in calls
            for key in ("input_asset_id", "output_asset_id")
            if row[key] is not None
        }
        jobs = list(
            (
                await session.execute(
                    select(ExperimentJobModel.__table__).where(
                        ExperimentJobModel.run_id == run_id,
                        or_(
                            ExperimentJobModel.job_id.in_(job_ids),
                            ExperimentJobModel.stage == "compile",
                        ),
                    )
                )
            ).mappings()
        )
        job_ids = {row["job_id"] for row in jobs}
        attempts = list(
            (
                await session.execute(
                    select(JobAttemptModel.__table__).where(JobAttemptModel.job_id.in_(job_ids))
                )
            ).mappings()
        )
        artifact_ids.update(
            row["error_artifact_id"]
            for row in [*jobs, *attempts]
            if row["error_artifact_id"] is not None
        )
        if any(row["run_id"] != run_id or row["status"] != "succeeded" for row in jobs):
            raise ValueError("questioner_import_nonterminal_job")
        special = {"model_calls": calls, "experiment_jobs": jobs, "job_attempts": attempts}
        for table in QUESTIONER_TABLES:
            if table.name in special:
                values = special[table.name]
            else:
                statement = select(*[column for column in table.columns if column.computed is None])
                statement = statement.where(table.c.run_id == run_id)
                if table.name == "run_artifacts":
                    statement = statement.where(table.c.artifact_id.in_(artifact_ids))
                elif table.name == "compiler_unit_stage_results":
                    statement = statement.where(table.c.stage == "questioner")
                elif table.name == "structured_output_item_diagnostics":
                    statement = statement.where(
                        table.c.stage.in_(
                            ["questioner", "questioner_verifier", "questioner_revision"]
                        )
                    )
                values = list((await session.execute(statement)).mappings())
            rows[table.name] = _sort_rows(table, [dict(value) for value in values])
            if table.name == "qbank_duplicate_links" and not include_index_diagnostics:
                rows[table.name] = _questioner_duplicate_rows(rows[table.name])
        if set(row["artifact_id"] for row in rows["run_artifacts"]) != artifact_ids:
            raise ValueError("questioner_import_artifact_denominator_missing")
        if validate:
            units = rows["compiler_units"]
            completed = {
                row["compiler_unit_id"]
                for row in rows["compiler_unit_stage_results"]
                if row["status"] == "success"
            }
            if (
                not units
                or len(units) != compiler["unit_count"]
                or completed != {row["compiler_unit_id"] for row in units}
            ):
                raise ValueError("questioner_import_incomplete_unit_denominator")
            for unit in units:
                if await questioner_v3_unit_issues(
                    session, run_id=run_id, compiler_unit_id=unit["compiler_unit_id"]
                ):
                    raise ValueError("questioner_import_source_protocol_integrity_failed")
            for row in rows["qbank_questions"]:
                candidate = question_from_rows(
                    row,
                    [p for p in rows["qbank_paraphrases"] if p["qbank_id"] == row["qbank_id"]],
                    [e for e in rows["qbank_evidence_refs"] if e["qbank_id"] == row["qbank_id"]],
                )
                digest = question_content_hash(candidate)
                if candidate.question_hash != digest or candidate.candidate_content_hash not in {
                    None,
                    digest,
                }:
                    raise ValueError("questioner_import_candidate_hash_mismatch")
        return {"compiler": compiler, "manifest": manifest_row.manifest_json}, rows

    async def _artifacts(self, rows: list[dict[str, Any]]) -> dict[str, str]:
        hashes = {}
        for row in rows:
            if row["state"] != "ready" or row["size_bytes"] is None:
                raise ValueError("questioner_import_artifact_not_ready")
            data = await self.store.get_bytes(row["object_key"])
            if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("questioner_import_artifact_checksum_mismatch")
            hashes[row["artifact_id"]] = row["sha256"]
        return hashes

    @staticmethod
    def _scope(headers: dict[str, Any], expected: dict[str, Any]) -> None:
        compiler, manifest = headers["compiler"], headers["manifest"]
        if (
            compiler["snapshot_id"] != expected["snapshot_id"]
            or manifest["dataset"] != expected["dataset"]
            or compiler["metadata"].get("conversation_ids") != [expected["conversation_id"]]
            or manifest.get("seed") != expected["seed"]
            or manifest.get("split") != expected["partition_role"]
            or manifest["metadata"]["questioner_config"].get("memory_raw_image_policy", "allow")
            != expected["memory_raw_image_policy"]
        ):
            raise ValueError("questioner_import_source_scope_mismatch")

    async def seal(
        self, *, run_id: str, campaign_id: str, protocol_hash: str, compatibility: dict[str, Any]
    ) -> QuestionerImportProof:
        async with self.sessions() as session:
            headers, rows = await self._read(session, run_id)
        self._scope(headers, compatibility)
        # Slow object I/O happens after closing the read transaction.
        artifacts = await self._artifacts(rows["run_artifacts"])
        return QuestionerImportProof(
            source_campaign_id=campaign_id,
            source_protocol_hash=protocol_hash,
            source_run_id=run_id,
            compatibility=compatibility,
            compatibility_hash=canonical_content_hash(compatibility),
            schema_revision=STEP8_ALEMBIC_REVISION,
            table_contract_hash=table_contract_hash(),
            facts_hash=canonical_content_hash({"headers": headers, "rows": rows}),
            row_counts={name: len(values) for name, values in rows.items()},
            artifact_hashes=artifacts,
            unit_count=len(rows["compiler_units"]),
            eligible_count=sum(
                row["status"] == "eligible_for_planning"
                for row in rows["questioner_candidate_gate_results"]
            ),
        )

    async def adopt(
        self,
        *,
        proof: QuestionerImportProof,
        compatibility: dict[str, Any],
        target_run_id: str,
        compiler_updates: Mapping[str, Any],
        manifest_metadata: Mapping[str, Any],
        prompt_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        assert_compatible(proof, compatibility)
        if (
            proof.source_run_id == target_run_id
            or proof.table_contract_hash != table_contract_hash()
        ):
            raise ValueError("questioner_import_target_or_table_contract_invalid")
        async with self.sessions() as session:
            source_campaign = await session.get(BenchmarkCampaignModel, proof.source_campaign_id)
            if (
                source_campaign is None
                or source_campaign.protocol_hash != proof.source_protocol_hash
            ):
                raise ValueError("questioner_import_source_campaign_missing_or_changed")
            source_manifest = BenchmarkCampaignManifest.model_validate(source_campaign.payload)
            source_scope = {
                **{
                    key: compatibility[key] for key in ("snapshot_id", "conversation_id", "dataset")
                },
                "treatment": "no_raw_image"
                if compatibility["memory_raw_image_policy"] == "forbid"
                else "main",
            }
            if questioner_compatibility(source_manifest, source_scope) != compatibility:
                raise ValueError("questioner_import_source_protocol_not_compatible")
            proof_sha256 = hashlib.sha256(json_bytes(proof.model_dump(mode="json"))).hexdigest()
            receipt = await session.scalar(
                select(BenchmarkComponentReceiptModel.receipt_hash)
                .join(
                    BenchmarkStageJobModel,
                    BenchmarkStageJobModel.job_id == BenchmarkComponentReceiptModel.job_id,
                )
                .where(
                    BenchmarkStageJobModel.campaign_id == proof.source_campaign_id,
                    BenchmarkStageJobModel.stage == "questioner",
                    BenchmarkStageJobModel.status == "completed",
                    BenchmarkStageJobModel.receipt_hash
                    == BenchmarkComponentReceiptModel.receipt_hash,
                    BenchmarkComponentReceiptModel.payload["output"]["compiler_run_id"].astext
                    == proof.source_run_id,
                    BenchmarkComponentReceiptModel.payload["output"][
                        "questioner_import_proof_sha256"
                    ].astext
                    == proof_sha256,
                )
                .limit(1)
            )
            if receipt is None:
                raise ValueError("questioner_import_source_component_not_committed")
            headers, rows = await self._read(session, proof.source_run_id)
        self._scope(headers, compatibility)
        if (
            canonical_content_hash({"headers": headers, "rows": rows}) != proof.facts_hash
            or {name: len(values) for name, values in rows.items()} != proof.row_counts
            or len(rows["compiler_units"]) != proof.unit_count
            or sum(
                r["status"] == "eligible_for_planning"
                for r in rows["questioner_candidate_gate_results"]
            )
            != proof.eligible_count
        ):
            raise ValueError("questioner_import_source_fact_hash_mismatch")
        if await self._artifacts(rows["run_artifacts"]) != proof.artifact_hashes:
            raise ValueError("questioner_import_artifact_proof_mismatch")
        rebound, identities = remap_questioner_rows(
            rows,
            primary_keys={
                table.name: [key.name for key in table.primary_key] for table in QUESTIONER_TABLES
            },
            source_run_id=proof.source_run_id,
            target_run_id=target_run_id,
            proof_hash=proof.content_hash,
        )
        _rehash_questions(rebound)
        rebound = {
            table.name: _sort_rows(table, rebound[table.name]) for table in QUESTIONER_TABLES
        }
        target_rows_hash = canonical_content_hash(rebound)
        lineage = {
            "version": IMPORT_VERSION,
            "source_run_id": proof.source_run_id,
            "source_campaign_id": proof.source_campaign_id,
            "source_protocol_hash": proof.source_protocol_hash,
            "proof_hash": proof.content_hash,
            "source_facts_hash": proof.facts_hash,
            "target_rows_hash": target_rows_hash,
            "compatibility_hash": proof.compatibility_hash,
            "identity_map_hash": canonical_content_hash(identities),
            "imported_stages": ["questioner"],
            "provider_calls": 0,
        }
        async with self.sessions() as session, session.begin():
            # One atomic graph append, without holding a transaction over model or object I/O.
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:identity, 0))"),
                {"identity": target_run_id},
            )
            current_headers, current_rows = await self._read(
                session, proof.source_run_id, validate=False
            )
            if (
                canonical_content_hash({"headers": current_headers, "rows": current_rows})
                != proof.facts_hash
            ):
                raise ValueError("questioner_import_source_changed_during_verification")
            existing = await session.get(c.CompilerRunModel, target_run_id)
            if existing is not None:
                if existing.metadata_json.get("questioner_upstream_import") != lineage:
                    raise ValueError("questioner_import_target_already_owned")
                _, stored_rows = await self._read(session, target_run_id)
                if canonical_content_hash(stored_rows) != target_rows_hash:
                    raise ValueError("questioner_import_target_fact_conflict")
                return {**lineage, "reused": True, "eligible_count": proof.eligible_count}
            if await session.get(RunManifestModel, target_run_id) is not None:
                raise ValueError("questioner_import_target_partial_registration")
            metadata = {
                **headers["manifest"]["metadata"],
                **manifest_metadata,
                "questioner_upstream_import": lineage,
            }
            selection_hash = canonical_content_hash(
                {
                    "snapshot_id": compatibility["snapshot_id"],
                    "conversation_ids": [compatibility["conversation_id"]],
                    "unit_ids": [row["compiler_unit_id"] for row in rebound["compiler_units"]],
                    "seed": compatibility["seed"],
                    "shard_count": 1,
                    "shard_index": 0,
                    "split": compatibility["partition_role"],
                    "diagnostic_force_revision": False,
                }
            )
            manifest = RunManifest.model_validate(
                {
                    **headers["manifest"],
                    "run_id": target_run_id,
                    "metadata": metadata,
                    "prompt_hashes": dict(prompt_hashes),
                    "selection_hash": selection_hash,
                    "created_at": utc_now(),
                    "config_hash": None,
                }
            )
            manifest = manifest.model_copy(update={"config_hash": run_config_hash(manifest)})
            await PostgresRunRepository(session).register(manifest, [])
            compiler = {
                **headers["compiler"],
                **compiler_updates,
                "run_id": target_run_id,
                "created_at": manifest.created_at,
                "metadata": {
                    **headers["compiler"]["metadata"],
                    **manifest_metadata,
                    "selection_hash": selection_hash,
                    "questioner_upstream_import": lineage,
                },
            }
            await session.execute(
                insert(cast(Table, c.CompilerRunModel.__table__)).values(compiler)
            )
            for table in QUESTIONER_TABLES:
                # Parents precede children, including model calls and Q revisions.
                for row in rebound[table.name]:
                    await session.execute(insert(table).values(row))
            await PostgresRunRepository(session).transition(target_run_id, RunStatus.RUNNING)
            _, inserted_rows = await self._read(session, target_run_id)
            if canonical_content_hash(inserted_rows) != target_rows_hash:
                raise ValueError("questioner_import_post_insert_proof_failed")
        return {**lineage, "reused": False, "eligible_count": proof.eligible_count}


def validate_r0_prefix_rows(
    rows: dict[str, list[dict[str, Any]]], run_id: str, *, require_r0: bool = True
) -> None:
    """Check the complete pre-Doubter denominator, including physical vectors."""
    from pmmc.compiler.ids import memory_program_content_hash
    from pmmc.infrastructure.database.compiler_mappers import memory_program_from_rows

    if set(rows) != {t.name for t in (*QUESTIONER_TABLES, *R0_PREFIX_TABLES)}:
        raise ValueError("r0_prefix_import_table_denominator_mismatch")
    for table in (*QUESTIONER_TABLES, *R0_PREFIX_TABLES):
        values = rows[table.name]
        keys = [column.name for column in table.primary_key]
        if len({tuple(row[key] for key in keys) for row in values}) != len(values):
            raise ValueError("r0_prefix_import_duplicate_primary_key")
    units = {r["compiler_unit_id"] for r in rows["compiler_units"]}
    # Native v3 Q decisions live in gates; post-D decisions live in separate
    # tables. Do not silently import a legacy mutable Q acceptance projection.
    if any(
        row.get("accepted") is not None or row.get("rejection_reason") is not None
        for row in rows["qbank_questions"]
    ):
        raise ValueError("r0_prefix_import_legacy_question_decision_forbidden")
    eligible = {
        r["qbank_id"]
        for r in rows["questioner_candidate_gate_results"]
        if r["status"] == "eligible_for_planning"
    }
    programs = {r["plan_id"]: r for r in rows["memory_programs"]}
    executions = {r["execution_id"]: r for r in rows["program_executions"]}
    assessments = rows["program_dry_run_assessments"]
    if any(r["round_id"] != 0 or r["run_id"] != run_id for r in programs.values()):
        raise ValueError("r0_prefix_import_revision_forbidden")
    for stage in ("planner", "dry_run"):
        jobs = [r for r in rows["compiler_stage_jobs"] if r["stage"] == stage]
        if (
            {r["compiler_unit_id"] for r in jobs} != units
            or len(jobs) != len(units)
            or any(r["status"] != "succeeded" or r["round_id"] != 0 for r in jobs)
        ):
            raise ValueError("r0_prefix_import_stage_not_terminal")
    if (
        {r["qbank_id"] for r in rows["planner_candidate_outcomes"]} != eligible
        or len(rows["planner_candidate_outcomes"]) != len(eligible)
        or len(assessments) != len(programs)
        or {r["plan_id"] for r in assessments} != set(programs)
        or any(r["qbank_id"] not in eligible for r in programs.values())
    ):
        raise ValueError("r0_prefix_import_candidate_denominator_mismatch")
    for row in programs.values():
        program = memory_program_from_rows(
            row,
            [r for r in rows["program_steps"] if r["plan_id"] == row["plan_id"]],
            [r for r in rows["program_expected_evidence_refs"] if r["plan_id"] == row["plan_id"]],
        )
        if program.program_hash != memory_program_content_hash(program):
            raise ValueError("r0_prefix_import_program_hash_mismatch")
    for row in executions.values():
        if (
            row["purpose"] != "compiler_dry_run"
            or row["query_id"] is not None
            or row["source_compiler_run_id"] != run_id
            or row["run_id"] != run_id
            or row["plan_id"] not in programs
            or row["status"] not in {"success", "failed", "skipped"}
        ):
            raise ValueError("r0_prefix_import_non_r0_execution")
    for row in assessments:
        execution = executions.get(row["execution_id"])
        if execution is None or execution["plan_id"] != row["plan_id"]:
            raise ValueError("r0_prefix_import_execution_lineage_mismatch")
    manifests = rows["qbank_index_manifests"] + rows["memory_operator_index_manifests"]
    collections = {r["collection_id"]: r for r in rows["embedding_collections"]}
    if (
        len(rows["qbank_index_manifests"]) != 1
        or len(rows["memory_operator_index_manifests"]) != 1
        or any(r["status"] != "ready" for r in manifests)
    ):
        raise ValueError("r0_prefix_import_index_not_ready")
    referenced = {
        value for r in manifests for key, value in r.items() if key.endswith("collection_id")
    }
    vectors = {r["embedding_id"]: r for r in rows["embeddings"]}
    refs = rows["program_query_embeddings"]
    if len(refs) != 2 * len(eligible) or {(r["qbank_id"], r["channel"]) for r in refs} != {
        (qid, channel)
        for qid in eligible
        for channel in ("question_text_query", "question_joint_query")
    }:
        raise ValueError("r0_prefix_import_query_vector_denominator_mismatch")
    for ref in refs:
        vector = vectors.get(ref["embedding_id"])
        if vector is None or vector["qbank_id"] != ref["qbank_id"]:
            raise ValueError("r0_prefix_import_vector_lineage_mismatch")
        if vector["input_hash"] != ref["input_hash"]:
            raise ValueError("r0_prefix_import_vector_input_hash_mismatch")
        referenced.add(vector["collection_id"])
    if set(collections) != referenced:
        raise ValueError("r0_prefix_import_collection_denominator_mismatch")
    for collection in collections.values():
        if (
            collection["run_id"] != run_id
            or collection["status"] != "ready"
            or collection["dimension"] != 4096
            or not collection["model_revision"]
        ):
            raise ValueError("r0_prefix_import_collection_owner_or_identity_invalid")
    for vector in vectors.values():
        values = vector["embedding"]
        if (
            vector["collection_id"] not in collections
            or len(values) != vector["dimension"]
            or vector["dimension"] != 4096
            or not all(math.isfinite(value) for value in values)
        ):
            raise ValueError("r0_prefix_import_vector_bytes_invalid")
    qindex = rows["qbank_index_manifests"][0]
    paraphrases = {
        r["paraphrase_id"] for r in rows["qbank_paraphrases"] if r["qbank_id"] in eligible
    }
    if qindex["eligible_question_count"] != len(eligible) or qindex["paraphrase_count"] != len(
        paraphrases
    ):
        raise ValueError("r0_prefix_import_qbank_index_count_mismatch")
    for field in (
        "text_question_collection_id",
        "multimodal_question_collection_id",
        "text_paraphrase_collection_id",
        "multimodal_paraphrase_collection_id",
    ):
        records = [v for v in vectors.values() if v["collection_id"] == qindex[field]]
        key, expected = (
            ("paraphrase_id", paraphrases) if "paraphrase" in field else ("qbank_id", eligible)
        )
        if len(records) != len(expected) or {r[key] for r in records} != expected:
            raise ValueError("r0_prefix_import_qbank_vector_denominator_mismatch")
    r0 = rows["r0_input_manifests"]
    if require_r0 and (
        len(r0) != 1
        or not r0[0]["hard_gate_passed"]
        or r0[0]["candidate_count"] != len(eligible)
        or r0[0]["program_count"] != len(programs)
        or r0[0]["execution_count"] != len(assessments)
        or r0[0]["terminal_outcome_count"] != len(eligible)
    ):
        raise ValueError("r0_prefix_import_r0_denominator_mismatch")


def validate_r0_memory_vectors(
    rows: dict[str, list[dict[str, Any]]], memories: Any, *, memory_raw_image_policy: str
) -> None:
    from pmmc.compiler.memory_index import make_memory_operator_index, memory_index_inputs
    from pmmc.schemas import EmbeddingInput

    stored = rows["memory_operator_index_manifests"][0]
    expected, _, _ = make_memory_operator_index(
        **{
            key: stored[key]
            for key in (
                "run_id",
                "snapshot_id",
                "dataset",
                "selection_hash",
                "text_model_alias",
                "text_model_revision",
                "multimodal_model_alias",
                "multimodal_model_revision",
            )
        },
        memory_raw_image_policy=memory_raw_image_policy,
    )
    if any(
        stored[key] != getattr(expected, key)
        for key in (
            "content_hash",
            "memory_index_id",
            "text_collection_id",
            "visual_collection_id",
            "representation_version",
        )
    ):
        raise ValueError("r0_prefix_import_memory_index_identity_mismatch")
    inputs = memory_index_inputs(memories, memory_raw_image_policy=memory_raw_image_policy)
    for collection_key, values in zip(
        ("text_collection_id", "visual_collection_id"), inputs, strict=True
    ):
        expected_hashes = {
            item.source_id: canonical_content_hash(
                EmbeddingInput(
                    input_id=item.source_id,
                    text=item.text,
                    image_refs=[item.image] if item.image is not None else [],
                    metadata={"memory_id": item.memory_id},
                ).model_dump(mode="json")
            )
            for item in values
        }
        records = [r for r in rows["embeddings"] if r["collection_id"] == stored[collection_key]]
        if (
            len(records) != len(expected_hashes)
            or {r["source_id"]: r["input_hash"] for r in records} != expected_hashes
        ):
            raise ValueError("r0_prefix_import_memory_vector_denominator_or_input_mismatch")


def normalize_r0_prefix_rows(
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Exclude only the legacy post-D decision flag from the pre-D fact graph.

    Native R0 selects every round-zero program, without accepted filtering, and
    memory_program_content_hash excludes this flag. Planner0's native default is
    True; neither source nor target adoption may inherit a later D decision.
    """
    normalized = copy.deepcopy(rows)
    for row in normalized["memory_programs"]:
        row["accepted"] = True
    return normalized


def project_r0_prefix_rows(
    rows: dict[str, list[dict[str, Any]]],
    questioner_rows: dict[str, list[dict[str, Any]]],
    *,
    proof: R0PrefixImportProof,
    target_run_id: str,
    target_selection_hash: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Project measured facts, never synthesize a provider response or execution.

    Imported content_hash leaves use an explicit adoption hash domain over the
    complete target row and the original producer hash. Native program and memory
    index hashes keep their normal contracts. Request/input/output hashes remain
    producer measurements and are bound by immutable source-row provenance.
    R0 itself is deliberately omitted: the normal CPU gate reconstructs it.
    """
    from pmmc.compiler.ids import memory_program_content_hash
    from pmmc.compiler.memory_index import make_memory_operator_index
    from pmmc.infrastructure.database.compiler_mappers import memory_program_from_rows

    rows = normalize_r0_prefix_rows(rows)
    validate_r0_prefix_rows(rows, proof.source_run_id)
    source_index = rows["memory_operator_index_manifests"][0]
    memory_index, text_collection, visual_collection = make_memory_operator_index(
        **{
            key: source_index[key]
            for key in (
                "snapshot_id",
                "dataset",
                "text_model_alias",
                "text_model_revision",
                "multimodal_model_alias",
                "multimodal_model_revision",
            )
        },
        run_id=target_run_id,
        selection_hash=target_selection_hash,
        memory_raw_image_policy=proof.compatibility["questioner"]["memory_raw_image_policy"],
    )
    overrides = {
        source_index["memory_index_id"]: memory_index.memory_index_id,
        source_index["text_collection_id"]: text_collection.collection_id,
        source_index["visual_collection_id"]: visual_collection.collection_id,
    }
    tables = (*QUESTIONER_TABLES, *R0_PREFIX_TABLES)
    keys = {t.name: [column.name for column in t.primary_key] for t in tables}
    projected, identities = remap_questioner_rows(
        rows,
        primary_keys=keys,
        source_run_id=proof.source_run_id,
        target_run_id=target_run_id,
        proof_hash=proof.content_hash,
        identity_overrides=overrides,
    )
    qprojected, _ = remap_questioner_rows(
        questioner_rows,
        primary_keys=keys,
        source_run_id=proof.source_run_id,
        target_run_id=target_run_id,
        proof_hash=proof.questioner_proof.content_hash,
    )
    _rehash_questions(qprojected)
    for table in tables:
        name = table.name
        base = {
            tuple(row[k] for k in keys[name]): target
            for row, target in zip(
                questioner_rows.get(name, []), qprojected.get(name, []), strict=True
            )
        }
        for ordinal, (source, target) in enumerate(zip(rows[name], projected[name], strict=True)):
            pk = tuple(source[k] for k in keys[name])
            if pk in base:
                projected[name][ordinal] = base[pk]
                continue
            # These are producer bytes or canonical evidence, not ownership references.
            for field in ("args", "output", "message", "error", "embedding"):
                if field in source:
                    target[field] = copy.deepcopy(source[field])
            if "metadata" in target:
                target["metadata"].pop("upstream_import", None)
                target["metadata"]["r0_prefix_import"] = {
                    "version": R0_ADOPTION_VERSION,
                    "source_run_id": proof.source_run_id,
                    "source_primary_key": {key: source[key] for key in keys[name]},
                    "source_row_hash": canonical_content_hash(source),
                    "source_content_hash": source.get("content_hash"),
                    "proof_hash": proof.content_hash,
                }
            if name == "compiler_stage_jobs":
                target["claimed_by"] = target["lease_expires_at"] = None
            if "content_hash" in target:
                target["content_hash"] = canonical_content_hash(
                    {
                        "version": R0_ADOPTION_VERSION,
                        "table": name,
                        "source_content_hash": source["content_hash"],
                        "target": {
                            key: value for key, value in target.items() if key != "content_hash"
                        },
                    }
                )
    index_row = projected["memory_operator_index_manifests"][0]
    index_row.update(
        content_hash=memory_index.content_hash,
        selection_hash=target_selection_hash,
        memory_index_id=memory_index.memory_index_id,
    )
    for collection in (text_collection, visual_collection):
        row = next(
            r
            for r in projected["embedding_collections"]
            if r["collection_id"] == collection.collection_id
        )
        row["metadata"].update(collection.metadata)
    contracts = {r["qbank_id"]: r for r in projected["planner_requirement_contracts"]}
    for row in projected["memory_programs"]:
        if row["qbank_id"] in contracts and "requirement_contract_hash" in row["metadata"]:
            row["metadata"]["requirement_contract_hash"] = contracts[row["qbank_id"]][
                "content_hash"
            ]
        program = memory_program_from_rows(
            row,
            [s for s in projected["program_steps"] if s["plan_id"] == row["plan_id"]],
            [
                e
                for e in projected["program_expected_evidence_refs"]
                if e["plan_id"] == row["plan_id"]
            ],
        )
        row["metadata"]["r0_prefix_import"]["source_program_hash"] = row["program_hash"]
        row["program_hash"] = memory_program_content_hash(program)
    projected["r0_input_manifests"] = []
    return {table.name: _sort_rows(table, projected[table.name]) for table in tables}, identities


def r0_prefix_rows_hash(rows: dict[str, list[dict[str, Any]]]) -> str:
    """R0 is revalidated natively; this digest binds the exact imported inputs."""
    return canonical_content_hash({**normalize_r0_prefix_rows(rows), "r0_input_manifests": []})


def _prefix_call_dependencies(
    call_rows: Any,
) -> tuple[set[str], set[str], set[str]]:
    """Direct embedding attempts have no queue job, but remain measured calls."""
    job_ids: set[str] = set()
    direct_ids: set[str] = set()
    artifact_ids: set[str] = set()
    for row in call_rows:
        if (
            row["stage"] not in {"questioner", "planner", "embed"}
            or row.get("query_id") is not None
            or row.get("plan_id") is not None
        ):
            raise ValueError("r0_prefix_import_non_prefix_call")
        if row["job_id"] is None:
            if row["stage"] != "embed":
                raise ValueError("r0_prefix_import_unbound_non_embedding_call")
            direct_ids.add(row["model_call_id"])
        else:
            job_ids.add(row["job_id"])
        for key in ("input_asset_id", "output_asset_id"):
            if row[key] is not None:
                artifact_ids.add(row[key])
        error = row["metadata"].get("error_artifact_id")
        if error is not None:
            if not isinstance(error, str) or not error:
                raise ValueError("r0_prefix_import_invalid_error_artifact_id")
            artifact_ids.add(error)
    return job_ids, direct_ids, artifact_ids


class _PostgresR0PrefixReader(PostgresQuestionerImportRepository):
    """Read the closed pre-D fact graph using only the caller's DB snapshot."""

    async def _read_prefix(
        self, session: AsyncSession, run_id: str, *, require_r0: bool = True
    ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
        headers, rows = await self._read(session, run_id, include_index_diagnostics=True)

        async def read(table: Table, predicate: Any) -> list[dict[str, Any]]:
            values = (await session.execute(select(table).where(predicate))).mappings()
            result = [dict(value) for value in values]
            if table.name == "embeddings":
                for value in result:
                    # pgvector returns numpy arrays. Hash actual float values,
                    # never a truncated numpy display representation.
                    value["embedding"] = [float(v) for v in value["embedding"]]
            return _sort_rows(table, result)

        by_name = {t.name: t for t in R0_PREFIX_TABLES}
        programs = by_name["memory_programs"]
        plan_ids = select(programs.c.plan_id).where(
            programs.c.run_id == run_id, programs.c.round_id == 0
        )
        executions = by_name["program_executions"]
        execution_ids = select(executions.c.execution_id).where(
            executions.c.run_id == run_id,
            executions.c.plan_id.in_(plan_ids),
            executions.c.purpose == "compiler_dry_run",
        )
        validation = by_name["program_validation_results"]
        validation_ids = select(validation.c.validation_result_id).where(
            validation.c.run_id == run_id, validation.c.plan_id.in_(plan_ids)
        )
        assessments = by_name["program_dry_run_assessments"]
        assessment_ids = select(assessments.c.assessment_id).where(
            assessments.c.run_id == run_id, assessments.c.plan_id.in_(plan_ids)
        )
        for table in R0_PREFIX_TABLES:
            name = table.name
            if name in {"embeddings", "embedding_collections"}:
                continue
            owner = table.c.compiler_run_id if name == "qbank_index_manifests" else table.c.run_id
            predicate = owner == run_id
            if "round_id" in table.c:
                predicate &= table.c.round_id == 0
            if "plan_id" in table.c:
                predicate &= table.c.plan_id.in_(plan_ids)
            if "execution_id" in table.c:
                predicate &= table.c.execution_id.in_(execution_ids)
            if name == "program_validation_issues":
                predicate &= table.c.validation_result_id.in_(validation_ids)
            if name == "program_dry_run_evidence_matches":
                predicate &= table.c.assessment_id.in_(assessment_ids)
            rows[name] = await read(table, predicate)
        diagnostics = cast(Table, q.RuntimeStructuredOutputItemDiagnosticModel.__table__)
        rows[diagnostics.name] = await read(
            diagnostics,
            (diagnostics.c.run_id == run_id)
            & diagnostics.c.stage.in_(
                ["questioner", "questioner_verifier", "questioner_revision", "planner"]
            ),
        )
        collection_ids = {
            value
            for name in ("qbank_index_manifests", "memory_operator_index_manifests")
            for row in rows[name]
            for key, value in row.items()
            if key.endswith("collection_id")
        }
        vector_ids = {row["embedding_id"] for row in rows["program_query_embeddings"]}
        vectors = cast(Table, EmbeddingModel.__table__)
        query_vectors = await read(vectors, vectors.c.embedding_id.in_(vector_ids))
        collection_ids.update(row["collection_id"] for row in query_vectors)
        collections = cast(Table, EmbeddingCollectionModel.__table__)
        rows["embedding_collections"] = await read(
            collections, collections.c.collection_id.in_(collection_ids)
        )
        rows["embeddings"] = await read(vectors, vectors.c.collection_id.in_(collection_ids))
        call_ids = {
            row["model_call_id"]
            for name in (
                "planner_generation_batches",
                "program_query_embedding_batches",
                "embeddings",
            )
            for row in rows[name]
            if row.get("model_call_id") is not None
        }
        call_ids.update(row["model_call_id"] for row in rows["model_calls"])
        calls = cast(Table, RuntimeModelCallModel.__table__)
        # Include failed physical attempts too. Revision calls use the planner
        # stage as well, but bind a source plan; they are not Planner0 costs.
        call_ids.update(
            await session.scalars(
                select(calls.c.model_call_id).where(
                    calls.c.run_id == run_id,
                    calls.c.query_id.is_(None),
                    calls.c.plan_id.is_(None),
                    or_(
                        calls.c.stage == "embed",
                        calls.c.stage == "planner",
                    ),
                )
            )
        )
        # Follow request ancestry, but never smuggle a foreign producer or a D call.
        while True:
            call_rows = list(
                (
                    await session.execute(select(calls).where(calls.c.model_call_id.in_(call_ids)))
                ).mappings()
            )
            parents = {r["parent_model_call_id"] for r in call_rows if r["parent_model_call_id"]}
            if parents <= call_ids:
                break
            call_ids.update(parents)
        if {r["model_call_id"] for r in call_rows} != call_ids or any(
            r["run_id"] != run_id or r["stage"] in {"doubter", "planner_revision"}
            for r in call_rows
        ):
            raise ValueError("r0_prefix_import_call_lineage_not_closed")
        rows["model_calls"] = _sort_rows(calls, [dict(r) for r in call_rows])
        rows[diagnostics.name] = [
            row for row in rows[diagnostics.name] if row["model_call_id"] in call_ids
        ]
        jobs = cast(Table, ExperimentJobModel.__table__)
        job_ids, direct_ids, artifact_ids = _prefix_call_dependencies(call_rows)
        job_ids.update(r["job_id"] for r in rows["experiment_jobs"])
        headers["direct_model_call_ids"] = sorted(direct_ids)
        rows["experiment_jobs"] = await read(jobs, jobs.c.job_id.in_(job_ids))
        if {r["job_id"] for r in rows["experiment_jobs"]} != job_ids or any(
            r["run_id"] != run_id or r["status"] != "succeeded" for r in rows["experiment_jobs"]
        ):
            raise ValueError("r0_prefix_import_model_job_not_terminal")
        attempts = cast(Table, JobAttemptModel.__table__)
        rows["job_attempts"] = await read(attempts, attempts.c.job_id.in_(job_ids))
        artifact_ids.update(
            r["error_artifact_id"]
            for name in ("experiment_jobs", "job_attempts")
            for r in rows[name]
            if r["error_artifact_id"] is not None
        )
        artifacts = cast(Table, RuntimeArtifactModel.__table__)
        rows["run_artifacts"] = await read(artifacts, artifacts.c.artifact_id.in_(artifact_ids))
        if {r["artifact_id"] for r in rows["run_artifacts"]} != artifact_ids or any(
            r["run_id"] != run_id for r in rows["run_artifacts"]
        ):
            raise ValueError("r0_prefix_import_artifact_lineage_not_closed")
        rows = normalize_r0_prefix_rows(rows)
        validate_r0_prefix_rows(rows, run_id, require_r0=require_r0)
        from pmmc.infrastructure.database.repositories import PostgresMemoryRepository

        @asynccontextmanager
        async def same_snapshot():
            yield session

        memories = await PostgresMemoryRepository(cast(Any, same_snapshot)).list_snapshot(
            headers["compiler"]["snapshot_id"],
            conversation_ids=set(headers["compiler"]["metadata"]["conversation_ids"]),
        )
        policy = headers["manifest"]["metadata"]["questioner_config"].get(
            "memory_raw_image_policy", "allow"
        )
        validate_r0_memory_vectors(rows, memories, memory_raw_image_policy=policy)
        headers["canonical_memories"] = [m.model_dump(mode="json") for m in memories]
        return headers, rows


def _prefix_registration_hash(headers: dict[str, Any]) -> str:
    value = copy.deepcopy(headers)
    for owner in (value["compiler"], value["manifest"]):
        owner["metadata"].pop("r0_prefix_upstream_import", None)
    value["manifest"].pop("config_hash", None)
    return canonical_content_hash(value)


class PostgresR0PrefixImportRepository(_PostgresR0PrefixReader):
    """Atomic bounded append after Q adoption, followed by native CPU R0 validation."""

    async def prefix_registration(self, run_id: str) -> dict[str, Any] | None:
        async with self.sessions() as session:
            row = await session.get(c.CompilerRunModel, run_id)
            return row.metadata_json.get("r0_prefix_upstream_import") if row else None

    async def adopt_prefix(
        self, *, proof: R0PrefixImportProof, compatibility: dict[str, Any], target_run_id: str
    ) -> dict[str, Any]:
        if proof.source_run_id == target_run_id:
            raise ValueError("r0_prefix_import_same_owner")
        source_headers, source_rows = await self.verify_source(proof, compatibility)
        if not await self.verify_adopted(target_run_id, compatibility["questioner"]):
            raise ValueError("r0_prefix_import_requires_exact_questioner_adoption")
        async with self.sessions() as session:
            _, source_q = await self._read(session, proof.source_run_id)
            target_headers, target_q = await self._read(session, target_run_id)
        qlineage = target_headers["compiler"]["metadata"].get("questioner_upstream_import", {})
        if qlineage.get("proof_hash") != proof.questioner_proof.content_hash:
            raise ValueError("r0_prefix_import_questioner_producer_mismatch")
        # Doubter changes must not smuggle a different P/operator/R0 registration.
        allowed = {
            "run_id",
            "created_at",
            "metadata",
            "questioner_freeze_manifest_hash",
            "routing_projection_version",
            "active_round",
            "max_round",
        }
        for key, value in source_headers["compiler"].items():
            if key not in allowed and not key.startswith("doubter_"):
                if target_headers["compiler"].get(key) != value:
                    raise ValueError("r0_prefix_import_target_registration_incompatible_" + key)
        selection_hash = target_headers["manifest"]["selection_hash"]
        projected, identities = project_r0_prefix_rows(
            source_rows,
            source_q,
            proof=proof,
            target_run_id=target_run_id,
            target_selection_hash=selection_hash,
        )
        lineage = {
            "version": R0_ADOPTION_VERSION,
            "source_run_id": proof.source_run_id,
            "source_campaign_id": proof.source_campaign_id,
            "source_protocol_hash": proof.source_protocol_hash,
            "proof_json": proof.model_dump(mode="json"),
            "proof_hash": proof.content_hash,
            "source_proof_sha256": hashlib.sha256(
                json_bytes(proof.model_dump(mode="json"))
            ).hexdigest(),
            "source_facts_hash": proof.facts_hash,
            "target_rows_hash": r0_prefix_rows_hash(projected),
            "identity_map_hash": canonical_content_hash(identities),
            "target_selection_hash": selection_hash,
            "target_registration_hash": _prefix_registration_hash(target_headers),
            "compatibility_hash": proof.compatibility_hash,
            "imported_stages": list(R0_PREFIX_STAGES[1:]),
            "provider_calls": 0,
        }
        tables = (*QUESTIONER_TABLES, *R0_PREFIX_TABLES)
        qkeys = {
            table.name: {
                tuple(row[key.name] for key in table.primary_key) for row in target_q[table.name]
            }
            for table in QUESTIONER_TABLES
        }
        reused = False
        async with self.sessions() as session, session.begin():
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:identity, 0))"),
                {"identity": target_run_id},
            )
            current_headers, current_rows = await self._read_prefix(session, proof.source_run_id)
            if (
                canonical_content_hash({"headers": current_headers, "rows": current_rows})
                != proof.facts_hash
            ):
                raise ValueError("r0_prefix_import_source_changed_during_verification")
            compiler = await session.get(c.CompilerRunModel, target_run_id)
            manifest = await session.get(RunManifestModel, target_run_id)
            assert compiler is not None and manifest is not None
            current_target_headers, current_q = await self._read(session, target_run_id)
            if (
                current_q != target_q
                or _prefix_registration_hash(current_target_headers)
                != lineage["target_registration_hash"]
            ):
                raise ValueError("r0_prefix_import_target_changed_during_verification")
            existing = compiler.metadata_json.get("r0_prefix_upstream_import")
            if existing is not None:
                if (
                    existing != lineage
                    or manifest.manifest_json["metadata"].get("r0_prefix_upstream_import")
                    != lineage
                ):
                    raise ValueError("r0_prefix_import_target_lineage_conflict")
                reused = True
            else:
                for table in R0_PREFIX_TABLES:
                    if table.name == "embeddings":
                        continue  # Every embedding belongs to a checked collection.
                    owner = (
                        table.c.compiler_run_id
                        if table.name == "qbank_index_manifests"
                        else table.c.run_id
                    )
                    if (
                        await session.scalar(
                            select(next(iter(table.primary_key)))
                            .where(owner == target_run_id)
                            .limit(1)
                        )
                        is not None
                    ):
                        raise ValueError("r0_prefix_import_target_partial_prefix")
                for table in tables:
                    if table.name == "r0_input_manifests":
                        continue
                    for row in projected[table.name]:
                        pk = tuple(row[key.name] for key in table.primary_key)
                        if pk not in qkeys.get(table.name, set()):
                            await session.execute(insert(table).values(row))
                compiler.metadata_json = {
                    **compiler.metadata_json,
                    "r0_prefix_upstream_import": lineage,
                }
                payload = RunManifest.model_validate(
                    {
                        **manifest.manifest_json,
                        "metadata": {
                            **manifest.manifest_json["metadata"],
                            "r0_prefix_upstream_import": lineage,
                        },
                        "config_hash": None,
                    }
                )
                payload = payload.model_copy(update={"config_hash": run_config_hash(payload)})
                manifest.manifest_json = payload.model_dump(mode="json")
                manifest.config_hash = payload.config_hash
                await session.flush()
            _, actual = await self._read_prefix(session, target_run_id, require_r0=False)
            if r0_prefix_rows_hash(actual) != lineage["target_rows_hash"]:
                raise ValueError("r0_prefix_import_post_insert_fact_conflict")
        # A crash here is resumable, but no receipt is returned until the normal
        # gate reconstructs and verifies the target-owned R0 manifest.
        await self.verify_target(target_run_id, compatibility, materialize_r0=True)
        return {**lineage, "reused": reused}

    async def verify_target(
        self, run_id: str, compatibility: dict[str, Any], *, materialize_r0: bool = False
    ) -> dict[str, Any] | None:
        from pmmc.compiler.r0_gate import verify_r0_input

        lineage = await self.prefix_registration(run_id)
        if lineage is None:
            return None
        if (
            lineage.get("version") != R0_ADOPTION_VERSION
            or lineage.get("imported_stages") != list(R0_PREFIX_STAGES[1:])
            or lineage.get("provider_calls") != 0
        ):
            raise ValueError("r0_prefix_import_consumer_lineage_invalid")
        proof = R0PrefixImportProof.model_validate(lineage["proof_json"])
        _, source_rows = await self.verify_source(proof, compatibility)
        async with self.sessions() as session:
            qheaders, source_q = await self._read(session, proof.source_run_id)
            headers, rows = await self._read_prefix(session, run_id, require_r0=not materialize_r0)
            target_headers, _ = await self._read(session, run_id)
        if (
            proof.source_run_id != lineage.get("source_run_id")
            or proof.source_campaign_id != lineage.get("source_campaign_id")
            or proof.source_protocol_hash != lineage.get("source_protocol_hash")
            or target_headers["manifest"]["selection_hash"] != lineage["target_selection_hash"]
            or proof.content_hash != lineage["proof_hash"]
            or proof.facts_hash != lineage["source_facts_hash"]
            or proof.compatibility_hash != lineage["compatibility_hash"]
            or hashlib.sha256(json_bytes(proof.model_dump(mode="json"))).hexdigest()
            != lineage["source_proof_sha256"]
            or headers["manifest"]["metadata"].get("r0_prefix_upstream_import") != lineage
            or _prefix_registration_hash(target_headers) != lineage["target_registration_hash"]
            or canonical_content_hash({"headers": qheaders, "rows": source_q})
            != proof.questioner_proof.facts_hash
        ):
            raise ValueError("r0_prefix_import_consumer_proof_conflict")
        projected, identities = project_r0_prefix_rows(
            source_rows,
            source_q,
            proof=proof,
            target_run_id=run_id,
            target_selection_hash=lineage["target_selection_hash"],
        )
        if (
            r0_prefix_rows_hash(rows) != r0_prefix_rows_hash(projected)
            or r0_prefix_rows_hash(rows) != lineage["target_rows_hash"]
            or canonical_content_hash(identities) != lineage["identity_map_hash"]
        ):
            raise ValueError("r0_prefix_import_consumer_fact_conflict")
        await self._artifacts(rows["run_artifacts"])
        r0 = await verify_r0_input(self.sessions, run_id=run_id)
        source_r0 = source_rows["r0_input_manifests"][0]
        for key in (
            "candidate_count",
            "answerable_candidate_count",
            "control_candidate_count",
            "program_count",
            "execution_count",
            "terminal_outcome_count",
            "hard_gate_passed",
            "issue_codes",
            "protocol_integrity_issues",
            "candidate_scientific_rejections",
            "coverage_zero_units",
            "downstream_viability",
        ):
            if getattr(r0, key) != source_r0[key]:
                raise ValueError("r0_prefix_import_target_r0_semantics_changed")
        return lineage

    async def seal_prefix(
        self,
        *,
        run_id: str,
        campaign_id: str,
        protocol_hash: str,
        compatibility: dict[str, Any],
    ) -> R0PrefixImportProof:
        from pmmc.compiler.r0_gate import verify_r0_input

        # Refuse missing R0 before invoking the normal (build-or-verify) gate.
        async with self.sessions() as session:
            await session.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            )
            headers, rows = await self._read_prefix(session, run_id)
        r0 = await verify_r0_input(self.sessions, run_id=run_id)
        qproof = await self.seal(
            run_id=run_id,
            campaign_id=campaign_id,
            protocol_hash=protocol_hash,
            compatibility=compatibility["questioner"],
        )
        self._scope(headers, compatibility["questioner"])
        if rows["r0_input_manifests"][0]["content_hash"] != r0.content_hash:
            raise ValueError("r0_prefix_import_r0_changed_during_verification")
        artifacts = await self._artifacts(rows["run_artifacts"])
        return R0PrefixImportProof(
            source_campaign_id=campaign_id,
            source_protocol_hash=protocol_hash,
            source_run_id=run_id,
            compatibility=compatibility,
            compatibility_hash=canonical_content_hash(compatibility),
            schema_revision=STEP8_ALEMBIC_REVISION,
            table_contract_hash=r0_prefix_table_contract_hash(),
            facts_hash=canonical_content_hash({"headers": headers, "rows": rows}),
            row_counts={name: len(values) for name, values in rows.items()},
            artifact_hashes=artifacts,
            questioner_proof=qproof,
            r0_content_hash=r0.content_hash,
        )

    async def verify_source(
        self, proof: R0PrefixImportProof, compatibility: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
        from pmmc.compiler.r0_gate import verify_r0_input
        from pmmc.experiments.step8_jobs import ComponentReceipt

        assert_r0_prefix_compatible(proof, compatibility)
        if proof.table_contract_hash != r0_prefix_table_contract_hash():
            raise ValueError("r0_prefix_import_table_contract_mismatch")
        async with self.sessions() as session:
            await session.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            )
            source = await session.get(BenchmarkCampaignModel, proof.source_campaign_id)
            if source is None or source.protocol_hash != proof.source_protocol_hash:
                raise ValueError("r0_prefix_import_source_campaign_missing_or_changed")
            campaign = BenchmarkCampaignManifest.model_validate(source.payload)
            qscope = compatibility["questioner"]
            scope = {key: qscope[key] for key in ("snapshot_id", "dataset", "conversation_id")}
            scope["treatment"] = compatibility["treatment"]
            if (
                campaign.protocol.content_hash != proof.source_protocol_hash
                or r0_prefix_compatibility(campaign, scope) != compatibility
            ):
                raise ValueError("r0_prefix_import_source_protocol_not_compatible")
            receipts = list(
                (
                    await session.execute(
                        select(BenchmarkStageJobModel, BenchmarkComponentReceiptModel)
                        .join(
                            BenchmarkComponentReceiptModel,
                            BenchmarkStageJobModel.receipt_hash
                            == BenchmarkComponentReceiptModel.receipt_hash,
                        )
                        .where(
                            BenchmarkStageJobModel.campaign_id == proof.source_campaign_id,
                            BenchmarkStageJobModel.stage.in_(R0_PREFIX_STAGES),
                            BenchmarkStageJobModel.status == "completed",
                            BenchmarkComponentReceiptModel.payload["output"][
                                "compiler_run_id"
                            ].astext
                            == proof.source_run_id,
                        )
                    )
                ).all()
            )
            committed = {}
            for job, stored in receipts:
                from pmmc.experiments.step8_jobs import BenchmarkStageJob

                receipt = ComponentReceipt.model_validate(stored.payload)
                definition = BenchmarkStageJob.model_validate(job.payload)
                if (
                    receipt.content_hash != stored.receipt_hash
                    or receipt.job_id != job.job_id
                    or receipt.scientific_input_hash != job.scientific_input_hash
                    or receipt.source_fingerprint
                    != campaign.protocol.source_fingerprints[job.stage]
                    or receipt.source_fingerprint != job.source_fingerprint
                    or receipt.schema_revision != STEP8_ALEMBIC_REVISION
                    or definition.job_id != job.job_id
                    or definition.stage != job.stage
                    or definition.campaign_id != proof.source_campaign_id
                    or definition.source_fingerprint != job.source_fingerprint
                    or stored.job_id != job.job_id
                    or receipt.output.get("compiler_run_id") != proof.source_run_id
                    or definition.scientific_input_hash != job.scientific_input_hash
                    or set(receipt.upstream_receipt_hashes) != set(job.dependency_ids)
                    or definition.dependency_ids != job.dependency_ids
                    or stored.component_identity != receipt.component_identity
                    or stored.artifact_hashes != receipt.artifact_hashes
                    or job.stage in committed
                ):
                    raise ValueError("r0_prefix_import_source_receipt_identity_invalid")
                committed[job.stage] = receipt
                for dependency_id, receipt_hash in receipt.upstream_receipt_hashes.items():
                    dependency = await session.get(BenchmarkStageJobModel, dependency_id)
                    parent = await session.get(BenchmarkComponentReceiptModel, receipt_hash)
                    if (
                        dependency is None
                        or parent is None
                        or dependency.status != "completed"
                        or dependency.campaign_id != proof.source_campaign_id
                        or dependency.receipt_hash != receipt_hash
                        or parent.job_id != dependency_id
                        or ComponentReceipt.model_validate(parent.payload).content_hash
                        != receipt_hash
                    ):
                        raise ValueError("r0_prefix_import_upstream_receipt_lineage_mismatch")
            if set(committed) != set(R0_PREFIX_STAGES):
                raise ValueError("r0_prefix_import_source_components_not_committed")
            digest = hashlib.sha256(json_bytes(proof.model_dump(mode="json"))).hexdigest()
            r0_receipt = committed["r0_gate"]
            if (
                r0_receipt.output.get("r0_prefix_import_proof_sha256") != digest
                or digest not in r0_receipt.artifact_hashes
            ):
                raise ValueError("r0_prefix_import_source_proof_not_committed")
            questioner_digest = hashlib.sha256(
                json_bytes(proof.questioner_proof.model_dump(mode="json"))
            ).hexdigest()
            if (
                committed["questioner"].output.get("questioner_import_proof_sha256")
                != questioner_digest
                or questioner_digest not in committed["questioner"].artifact_hashes
            ):
                raise ValueError("r0_prefix_import_questioner_proof_not_committed")
            headers, rows = await self._read_prefix(session, proof.source_run_id)
        self._scope(headers, compatibility["questioner"])
        if (
            canonical_content_hash({"headers": headers, "rows": rows}) != proof.facts_hash
            or {name: len(values) for name, values in rows.items()} != proof.row_counts
        ):
            raise ValueError("r0_prefix_import_source_fact_hash_mismatch")
        if await self._artifacts(rows["run_artifacts"]) != proof.artifact_hashes:
            raise ValueError("r0_prefix_import_artifact_proof_mismatch")
        r0 = await verify_r0_input(self.sessions, run_id=proof.source_run_id)
        if r0.content_hash != proof.r0_content_hash:
            raise ValueError("r0_prefix_import_source_r0_mismatch")
        qproof = await self.seal(
            run_id=proof.source_run_id,
            campaign_id=proof.source_campaign_id,
            protocol_hash=proof.source_protocol_hash,
            compatibility=compatibility["questioner"],
        )
        if qproof != proof.questioner_proof:
            raise ValueError("r0_prefix_import_questioner_proof_mismatch")
        async with self.sessions() as session:
            await session.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            )
            current_headers, current_rows = await self._read_prefix(session, proof.source_run_id)
        if (
            canonical_content_hash({"headers": current_headers, "rows": current_rows})
            != proof.facts_hash
        ):
            raise ValueError("r0_prefix_import_source_changed_during_verification")
        return headers, rows


class PostgresR0PrefixProofRepository(PostgresR0PrefixImportRepository):
    """Compatibility name for callers using only the source proof/reader API."""
