"""Portable contracts for the round-0 compiler program quality pilot."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path, PurePosixPath

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    PROGRAM_PILOT_SCHEMA_VERSION,
    BaseSchema,
    DatasetName,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)
from pmmc.schemas.pilot import PilotDecisionValue, QuestionerPilotSample


class ProgramPilotStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    NEEDS_REMEDIATION = "needs_remediation"
    APPROVED = "approved"
    REJECTED = "rejected"


class ProgramPilotDatasetSource(BaseSchema):
    dataset: DatasetName
    runtime_export_dir: Path
    coverage_export_dir: Path
    snapshot_dir: Path

    @model_validator(mode="after")
    def _validate_dataset(self) -> ProgramPilotDatasetSource:
        if self.dataset is DatasetName.UNKNOWN:
            raise ValueError("program pilot source requires a concrete dataset")
        return self


class CompilerProgramPilotSourceManifest(BaseSchema):
    partition_id: NonEmptyStr
    samples: list[QuestionerPilotSample]
    datasets: list[ProgramPilotDatasetSource]
    benchmark_root: Path
    output_root: Path

    @model_validator(mode="after")
    def _validate_sources(self) -> CompilerProgramPilotSourceManifest:
        if len(self.samples) != len({item.dataset for item in self.samples}):
            raise ValueError("program pilot source samples repeat a dataset")
        if len(self.datasets) != len({item.dataset for item in self.datasets}):
            raise ValueError("program pilot source paths repeat a dataset")
        sample_datasets = {item.dataset for item in self.samples}
        if {item.dataset for item in self.datasets} != sample_datasets:
            raise ValueError("program pilot source samples and paths differ")
        if any(item.partition_id != self.partition_id for item in self.samples):
            raise ValueError("program pilot source samples use another partition")
        return self


class ProgramPilotArtifactRef(BaseSchema):
    role: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: NonEmptyStr
    size_bytes: NonNegativeInt
    private: bool = False

    @model_validator(mode="after")
    def _validate_path(self) -> ProgramPilotArtifactRef:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("program pilot artifact must stay below the bundle root")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("program pilot artifact requires a SHA256 digest")
        return self


class CompilerProgramPilotManifest(BaseSchema):
    pilot_id: NonEmptyStr
    schema_version: NonEmptyStr = PROGRAM_PILOT_SCHEMA_VERSION
    protocol_version: NonEmptyStr = "compiler-program-quality-pilot-v3"
    partition_id: NonEmptyStr
    samples: list[QuestionerPilotSample]
    status: ProgramPilotStatus = ProgramPilotStatus.PLANNED
    source_runtime_export_ids: dict[NonEmptyStr, NonEmptyStr]
    source_coverage_export_ids: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    automatic_integrity_passed: bool
    artifacts: list[ProgramPilotArtifactRef] = Field(default_factory=list)
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_manifest(self) -> CompilerProgramPilotManifest:
        if self.schema_version not in {
            "0.1.0",
            "0.2.0",
            PROGRAM_PILOT_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported compiler program pilot schema")
        expected_protocol = {
            "0.1.0": "compiler-program-quality-pilot-v1",
            "0.2.0": "compiler-program-quality-pilot-v2",
            PROGRAM_PILOT_SCHEMA_VERSION: "compiler-program-quality-pilot-v3",
        }[self.schema_version]
        if self.protocol_version != expected_protocol:
            raise ValueError("unsupported compiler program pilot protocol")
        if not self.samples:
            raise ValueError("program pilot requires samples")
        if len(self.samples) != len({item.dataset for item in self.samples}):
            raise ValueError("program pilot samples cannot repeat a dataset")
        if any(item.partition_id != self.partition_id for item in self.samples):
            raise ValueError("program pilot samples must use the frozen partition")
        datasets = {item.dataset.value for item in self.samples}
        if set(self.source_runtime_export_ids) != datasets:
            raise ValueError("program pilot runtime exports differ from sample datasets")
        if self.source_coverage_export_ids and set(self.source_coverage_export_ids) != datasets:
            raise ValueError("program pilot coverage exports differ from sample datasets")
        if self.status is ProgramPilotStatus.AWAITING_APPROVAL:
            if not self.automatic_integrity_passed:
                raise ValueError("awaiting-approval pilot must pass automatic integrity")
        if self.status is ProgramPilotStatus.NEEDS_REMEDIATION:
            if self.automatic_integrity_passed:
                raise ValueError("needs-remediation pilot must fail automatic integrity")
        ensure_unique_strings([item.relative_path for item in self.artifacts], "artifact paths")
        return self


class CompilerProgramPilotAuditManifest(BaseSchema):
    pilot_id: NonEmptyStr
    protocol_version: NonEmptyStr = "compiler-program-audit-v3"
    partition_id: NonEmptyStr
    candidate_program_audit_count: NonNegativeInt
    query_coverage_audit_count: NonNegativeInt
    candidate_program_audit_sha256: NonEmptyStr
    query_coverage_audit_sha256: NonEmptyStr
    all_candidate_program_facts_sha256: NonEmptyStr
    automatic_summary_sha256: NonEmptyStr
    source_runtime_manifest_hashes: dict[NonEmptyStr, NonEmptyStr]
    source_coverage_manifest_hashes: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    automatic_integrity_passed: bool
    integrity_issue_types: list[NonEmptyStr] = Field(default_factory=list)
    severe_terminal_error_count: NonNegativeInt = 0
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_audit(self) -> CompilerProgramPilotAuditManifest:
        if self.protocol_version not in {
            "compiler-program-audit-v1",
            "compiler-program-audit-v2",
            "compiler-program-audit-v3",
        }:
            raise ValueError("unsupported compiler program audit protocol")
        if self.candidate_program_audit_count != 160:
            raise ValueError("program pilot requires 160 candidate-program audit cards")
        if self.query_coverage_audit_count != 40:
            raise ValueError("program pilot requires 40 query coverage audit cards")
        if self.automatic_integrity_passed == bool(self.integrity_issue_types):
            raise ValueError("program pilot integrity status and issues disagree")
        if self.source_coverage_manifest_hashes and set(
            self.source_coverage_manifest_hashes
        ) != set(self.source_runtime_manifest_hashes):
            raise ValueError("program pilot source runtime/coverage datasets differ")
        for field_name in (
            "candidate_program_audit_sha256",
            "query_coverage_audit_sha256",
            "all_candidate_program_facts_sha256",
            "automatic_summary_sha256",
            "content_hash",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, field_name)) is None:
                raise ValueError(f"program pilot has invalid SHA256 field: {field_name}")
        return self


class CompilerProgramPilotDecision(BaseSchema):
    pilot_id: NonEmptyStr
    decision: PilotDecisionValue
    reviewer_id: NonEmptyStr
    rationale: NonEmptyStr
    audit_manifest_hash: NonEmptyStr
    decided_at: UtcDateTime = Field(default_factory=utc_now)
    content_hash: NonEmptyStr

    @model_validator(mode="after")
    def _validate_hashes(self) -> CompilerProgramPilotDecision:
        for field_name in ("audit_manifest_hash", "content_hash"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, field_name)) is None:
                raise ValueError(f"program decision has invalid SHA256 field: {field_name}")
        return self


class CompilerProgramFreezeManifest(BaseSchema):
    freeze_id: NonEmptyStr
    pilot_id: NonEmptyStr
    decision_hash: NonEmptyStr
    program_pilot_audit_hash: NonEmptyStr
    questioner_freeze_id: NonEmptyStr
    questioner_freeze_hash: NonEmptyStr
    partition_id: NonEmptyStr
    pilot_sample_ids: list[NonEmptyStr]
    planner_config_hash: NonEmptyStr
    planner_prompt_hash: NonEmptyStr
    planner_schema_hash: NonEmptyStr
    planner_renderer_version: NonEmptyStr
    planner_parser_version: NonEmptyStr
    program_policy_version: NonEmptyStr
    operator_set_version: NonEmptyStr
    operator_registry_hash: NonEmptyStr
    dry_run_policy_version: NonEmptyStr
    memory_index_representation_version: NonEmptyStr
    model_commits: dict[NonEmptyStr, NonEmptyStr]
    runtime_export_ids: dict[NonEmptyStr, NonEmptyStr]
    r_max: NonNegativeInt = 0
    evidence_lineage_version: NonEmptyStr | None = None
    qbank_semantic_schema_version: NonEmptyStr | None = None
    semantic_judge_config_hash: NonEmptyStr | None = None
    semantic_judge_prompt_hash: NonEmptyStr | None = None
    semantic_judge_schema_hash: NonEmptyStr | None = None
    semantic_export_ids: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=dict
    )
    approved_by: NonEmptyStr
    approved_at: UtcDateTime
    content_hash: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_freeze(self) -> CompilerProgramFreezeManifest:
        ensure_unique_strings(self.pilot_sample_ids, "pilot_sample_ids")
        if not self.model_commits:
            raise ValueError("program freeze requires pinned model commits")
        if self.operator_set_version == "pmmc-operators-v2":
            if self.r_max != 0:
                raise ValueError("Planner v1 freeze requires r_max=0")
            if self.program_policy_version != "planner-program-policy-v1":
                raise ValueError("Planner v1 freeze has the wrong program policy")
            if self.dry_run_policy_version != "evidence_driven_v2":
                raise ValueError("Planner v1 freeze has the wrong dry-run policy")
        elif self.operator_set_version == "pmmc-operators-v3":
            if self.r_max != 0:
                raise ValueError("Planner v1.1 freeze requires r_max=0")
            if self.program_policy_version != "planner-program-policy-v2":
                raise ValueError("Planner v1.1 freeze has the wrong program policy")
            if self.dry_run_policy_version != "evidence_driven_v3":
                raise ValueError("Planner v1.1 freeze has the wrong dry-run policy")
            semantic_values = (
                self.evidence_lineage_version,
                self.qbank_semantic_schema_version,
                self.semantic_judge_config_hash,
                self.semantic_judge_prompt_hash,
                self.semantic_judge_schema_hash,
            )
            if any(value is None for value in semantic_values):
                raise ValueError(
                    "Planner v1.1 freeze requires evidence and semantic identities"
                )
            if not self.semantic_export_ids:
                raise ValueError(
                    "Planner v1.1 freeze requires semantic export identities"
                )
        elif self.operator_set_version == "pmmc-operators-v4":
            if self.r_max != 2:
                raise ValueError("Step 6D compiler freeze requires active r_max=2")
            if self.program_policy_version != "planner-program-policy-v3":
                raise ValueError("Planner v1.2 freeze has the wrong program policy")
            if self.dry_run_policy_version != "evidence_driven_v4":
                raise ValueError("Planner v1.2 freeze has the wrong dry-run policy")
            semantic_values = (
                self.evidence_lineage_version,
                self.qbank_semantic_schema_version,
                self.semantic_judge_config_hash,
                self.semantic_judge_prompt_hash,
                self.semantic_judge_schema_hash,
            )
            if any(value is None for value in semantic_values):
                raise ValueError(
                    "Step 6D freeze requires evidence and semantic identities"
                )
            if not self.semantic_export_ids:
                raise ValueError("Step 6D freeze requires semantic export identities")
        elif self.operator_set_version == "pmmc-operators-v5":
            if self.r_max != 2:
                raise ValueError("Step 7 compiler freeze requires active r_max=2")
            if self.program_policy_version != "planner-program-policy-v4":
                raise ValueError("Planner v1.3 freeze has the wrong program policy")
            if self.dry_run_policy_version != "evidence_driven_v5":
                raise ValueError("Planner v1.3 freeze has the wrong dry-run policy")
            semantic_values = (
                self.evidence_lineage_version,
                self.qbank_semantic_schema_version,
                self.semantic_judge_config_hash,
                self.semantic_judge_prompt_hash,
                self.semantic_judge_schema_hash,
            )
            if any(value is None for value in semantic_values):
                raise ValueError("Step 7 freeze requires evidence and semantic identities")
            if not self.semantic_export_ids:
                raise ValueError("Step 7 freeze requires semantic export identities")
        elif self.operator_set_version == "pmmc-operators-v6":
            if self.r_max != 2:
                raise ValueError("Step 7B compiler freeze requires active r_max=2")
            if self.program_policy_version != "planner-program-policy-v5":
                raise ValueError("Planner v1.4 freeze has the wrong program policy")
            if self.dry_run_policy_version != "evidence_driven_v6":
                raise ValueError("Planner v1.4 freeze has the wrong dry-run policy")
            semantic_values = (
                self.evidence_lineage_version,
                self.qbank_semantic_schema_version,
                self.semantic_judge_config_hash,
                self.semantic_judge_prompt_hash,
                self.semantic_judge_schema_hash,
            )
            if any(value is None for value in semantic_values):
                raise ValueError("Step 7B freeze requires evidence and semantic identities")
            if not self.semantic_export_ids:
                raise ValueError("Step 7B freeze requires semantic export identities")
        else:
            raise ValueError("unsupported compiler freeze operator set")
        hashes = (
            "decision_hash",
            "program_pilot_audit_hash",
            "questioner_freeze_hash",
            "planner_config_hash",
            "planner_prompt_hash",
            "planner_schema_hash",
            "operator_registry_hash",
            "content_hash",
        )
        if any(re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None for name in hashes):
            raise ValueError("program freeze contains an invalid SHA256 identity")
        for field_name in (
            "semantic_judge_config_hash",
            "semantic_judge_prompt_hash",
            "semantic_judge_schema_hash",
        ):
            value = getattr(self, field_name)
            if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(
                    f"program freeze has invalid SHA256 field: {field_name}"
                )
        if any(
            re.fullmatch(r"[0-9a-f]{40}", revision) is None
            for revision in self.model_commits.values()
        ):
            raise ValueError("program freeze requires 40-character model commits")
        return self


__all__ = [
    "CompilerProgramFreezeManifest",
    "CompilerProgramPilotAuditManifest",
    "CompilerProgramPilotDecision",
    "CompilerProgramPilotManifest",
    "CompilerProgramPilotSourceManifest",
    "ProgramPilotDatasetSource",
    "ProgramPilotArtifactRef",
    "ProgramPilotStatus",
]
