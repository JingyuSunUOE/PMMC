"""Versioned scientific identities for incremental, paired benchmark campaigns.

Execution attempts and selection expansion deliberately do not change a method
or cell identity. Approval is a separate, explicit artifact, never an exit code.
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from pmmc.compiler.ids import canonical_content_hash
from pmmc.schemas.base import BaseSchema, DatasetName, JsonDict, NonEmptyStr, UtcDateTime

STEP8_SCHEMA_VERSION: Literal["0.1.0"] = "0.1.0"
STEP8_ALEMBIC_REVISION = "20261104_0038"
# Read-only historical reports remain interpretable; new workers require the current schema.
STEP8_READABLE_REVISIONS = frozenset({"20260927_0036", STEP8_ALEMBIC_REVISION})
ANSWER_BACKBONES = (
    "answer-qwen3.5-9b",
    "answer-qwen3.5-27b",
    "answer-gpt-6-luna",
    "answer-claude-haiku-5.5",
)
# Historical tuples are immutable: old results must still parse without relabelling.
LUNA_ANSWER_BACKBONES = (*ANSWER_BACKBONES[:3], "answer-claude-haiku-4.5")
LEGACY_ANSWER_BACKBONES = (*ANSWER_BACKBONES[:2], "answer-gpt-5-mini", LUNA_ANSWER_BACKBONES[3])
BASELINE_METHODS = (
    "full_context_captions",
    "full_context_raw_images",
    "caption_text_rag",
    "contrastive_multimodal_rag",
    "native_multimodal_rag",
)
MATCH_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
GAP_THRESHOLDS = (0.0, 0.05, 0.1)
RELEVANCE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


class CampaignPhase(StrEnum):
    DEVELOPMENT = "development"
    EXPLORATION = "exploration"
    PAPER = "paper"


class ResourceClass(StrEnum):
    CPU = "cpu"
    GPU_9B = "gpu_9b"
    GPU_27B = "gpu_27b"
    GPU_EMBEDDING = "gpu_embedding"
    API = "api"


class Treatment(StrEnum):
    MAIN = "main"
    FIXED_PLANNER = "fixed_planner"
    NO_RAW_IMAGE = "no_raw_image"
    NO_RERANKER = "no_reranker"


class OracleAt50Protocol(BaseSchema):
    """Opt-in research cost on one existing main PMMC cell, not every backbone."""

    answer_model: Literal["answer-qwen3.5-9b"] = "answer-qwen3.5-9b"
    question_budget: Literal[1, 2, 4, 8] = 8
    round_id: int = Field(default=2, ge=0, le=3)
    variant: Literal["active"] = "active"
    treatment: Literal["main"] = "main"
    candidate_limit: Literal[50] = 50
    policy: Literal["all_program_evidence_no_relevance_gate_no_rag"] = (
        "all_program_evidence_no_relevance_gate_no_rag"
    )


class ScientificProtocol(BaseSchema):
    """No paths, execution image, spend cap, or selection fraction in this identity."""

    schema_version: Literal["0.1.0", "0.2.0", "0.3.0"] = "0.3.0"
    questioner: str = "pmmc-questioner-v3.6"
    verifier: str = "pmmc-questioner-verifier-v7"
    planner: str = "pmmc-planner-v1.5"
    operator_set: str = "pmmc-operators-v7"
    doubter: str = "pmmc-doubter-v1.3"
    executor: str = "query-executor-v3"
    answer_contract: str = "typed-answer-contract-v3"
    compiler_alias: str = "compiler-qwen3.5-9b"
    reranker_alias: str = "reranker-qwen3.5-9b"
    grounding_alias: str = "grounding-qwen3.5-9b"
    judge_alias: str = "judge-qwen3.5-9b"
    answer_backbones: tuple[str, ...] = ANSWER_BACKBONES
    question_budget: Literal[8] = 8
    active_round: Literal[2] = 2
    max_round: Literal[3] = 3
    match_threshold: float = 0.6
    gap_threshold: float = 0.05
    text_relevance_threshold: float = 0.3
    multimodal_relevance_threshold: float = 0.3
    no_reranker_match_threshold: float = 0.6
    no_reranker_gap_threshold: float = 0.05
    seed: int = Field(default=0, ge=0)
    model_lock_hash: NonEmptyStr
    model_entry_hashes: dict[str, NonEmptyStr]
    partition_hash: NonEmptyStr
    exposure_manifest_hash: NonEmptyStr
    source_fingerprints: dict[str, NonEmptyStr]
    config_hashes: dict[str, NonEmptyStr]
    oracle_at_50: OracleAt50Protocol | None = None

    @model_serializer(mode="wrap")
    def _optional_oracle_identity(self, handler):
        payload = handler(self)
        if self.oracle_at_50 is None:
            payload.pop("oracle_at_50", None)
        return payload

    @model_validator(mode="after")
    def _frozen_choices(self) -> ScientificProtocol:
        expected = {
            "0.1.0": LEGACY_ANSWER_BACKBONES,
            "0.2.0": LUNA_ANSWER_BACKBONES,
            "0.3.0": ANSWER_BACKBONES,
        }[self.schema_version]
        if self.answer_backbones != expected:
            raise ValueError("Step 8 requires the four declared answer backbones in stable order")
        if self.match_threshold not in MATCH_THRESHOLDS:
            raise ValueError("match threshold is outside the declared grid")
        if self.gap_threshold not in GAP_THRESHOLDS:
            raise ValueError("gap threshold is outside the declared grid")
        if self.no_reranker_match_threshold not in MATCH_THRESHOLDS:
            raise ValueError("no-reranker threshold is outside its declared grid")
        if self.no_reranker_gap_threshold not in GAP_THRESHOLDS:
            raise ValueError("no-reranker gap is outside its declared grid")
        if any(
            value not in RELEVANCE_THRESHOLDS
            for value in (
                self.text_relevance_threshold,
                self.multimodal_relevance_threshold,
            )
        ):
            raise ValueError("relevance threshold is outside the declared grid")
        if not self.source_fingerprints or not self.config_hashes:
            raise ValueError("protocol must bind scientific source and configuration")
        if not self.model_entry_hashes or any(
            not alias.strip() or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            for alias, digest in self.model_entry_hashes.items()
        ):
            raise ValueError("model entry identities require nonempty keys and lowercase SHA256")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


class ThresholdApprovalEvidence(BaseSchema):
    """A policy selection linked to, but never rewriting, development facts."""

    schema_version: Literal["step8-threshold-approval-v1", "step8-threshold-approval-v2"] = (
        "step8-threshold-approval-v1"
    )
    source_campaign_id: NonEmptyStr
    source_campaign_manifest_hash: NonEmptyStr
    provisional_protocol: ScientificProtocol
    selected_policy: dict[str, float]
    cohort_selection_hashes: dict[str, NonEmptyStr]
    cohort_proof_sha256: dict[str, NonEmptyStr]
    cohort_receipt_hashes: dict[str, NonEmptyStr]
    audit_inventory_sha256: NonEmptyStr

    @model_validator(mode="after")
    def _selection_contract(self) -> ThresholdApprovalEvidence:
        fields = {
            "match_threshold",
            "gap_threshold",
            "no_reranker_match_threshold",
            "no_reranker_gap_threshold",
            "text_relevance_threshold",
            "multimodal_relevance_threshold",
        }
        if set(self.selected_policy) != fields:
            raise ValueError("threshold approval must bind the complete selected policy")
        ScientificProtocol.model_validate(
            {
                **self.provisional_protocol.model_dump(),
                **self.selected_policy,
            }
        )
        for name in ("text_relevance_threshold", "multimodal_relevance_threshold"):
            if self.schema_version == "step8-threshold-approval-v1" and self.selected_policy[
                name
            ] != getattr(self.provisional_protocol, name):
                raise ValueError("changed relevance requires separately generated context answers")
        for values in (
            self.cohort_selection_hashes,
            self.cohort_proof_sha256,
            self.cohort_receipt_hashes,
        ):
            if set(values) != {"regression", "confirmation"}:
                raise ValueError("threshold approval requires both development cohorts")
        for digest in (
            self.source_campaign_manifest_hash,
            self.audit_inventory_sha256,
            *self.cohort_selection_hashes.values(),
            *self.cohort_proof_sha256.values(),
            *self.cohort_receipt_hashes.values(),
        ):
            if re.fullmatch(r"[a-f0-9]{64}", digest) is None:
                raise ValueError("threshold approval evidence requires SHA256 identities")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


class MethodProtocolFreeze(BaseSchema):
    schema_version: Literal["0.1.0"] = STEP8_SCHEMA_VERSION
    protocol: ScientificProtocol
    decision: Literal["approved", "rejected"]
    reviewer_id: NonEmptyStr
    rationale_sha256: NonEmptyStr
    development_receipt_hash: NonEmptyStr
    confirmation_receipt_hash: NonEmptyStr
    approved_at: UtcDateTime
    protocol_integrity_passed: bool
    scientific_viability_passed: bool
    exposure_complete: bool
    threshold_approval: ThresholdApprovalEvidence | None = None

    @model_validator(mode="after")
    def _approval_gate(self) -> MethodProtocolFreeze:
        if self.decision == "approved" and not all(
            (
                self.protocol_integrity_passed,
                self.scientific_viability_passed,
                self.exposure_complete,
            )
        ):
            raise ValueError("cannot approve an incomplete or nonviable protocol")
        if self.decision == "approved":
            evidence = self.threshold_approval
            if evidence is None:
                raise ValueError("approved protocol requires saved threshold selection evidence")
            expected = ScientificProtocol.model_validate(
                {
                    **evidence.provisional_protocol.model_dump(),
                    **evidence.selected_policy,
                }
            )
            if self.protocol != expected:
                raise ValueError("frozen protocol differs from the approved threshold policy")
            if evidence.cohort_receipt_hashes != {
                "regression": self.development_receipt_hash,
                "confirmation": self.confirmation_receipt_hash,
            }:
                raise ValueError("freeze receipts differ from threshold approval evidence")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


class ExperimentCell(BaseSchema):
    schema_version: Literal["0.1.0"] = STEP8_SCHEMA_VERSION
    protocol_hash: NonEmptyStr
    dataset: DatasetName
    answer_model: NonEmptyStr
    method: NonEmptyStr
    treatment: Treatment = Treatment.MAIN
    question_budget: int = Field(default=8, ge=1, le=8)
    round_id: int = Field(default=2, ge=0, le=3)
    source_variant: Literal["active", "no_fallback", "forced_top1"] = "active"

    @model_validator(mode="after")
    def _cell_contract(self) -> ExperimentCell:
        if self.answer_model not in (*ANSWER_BACKBONES, *LEGACY_ANSWER_BACKBONES):
            raise ValueError("unknown answer backbone")
        if self.method not in ("pmmc", *BASELINE_METHODS):
            raise ValueError("unknown paired benchmark method")
        if self.question_budget not in (1, 2, 4, 8):
            raise ValueError("unsupported question budget")
        if self.method != "pmmc" and (
            self.treatment is not Treatment.MAIN
            or self.question_budget != 8
            or self.round_id != 2
            or self.source_variant != "active"
        ):
            raise ValueError("baseline cells do not inherit PMMC ablations")
        if self.answer_model != ANSWER_BACKBONES[0] and (
            self.treatment is not Treatment.MAIN or self.question_budget != 8 or self.round_id != 2
        ):
            raise ValueError("complete ablations are restricted to the 9B answer backbone")
        if self.treatment is not Treatment.MAIN and (
            self.question_budget != 8 or self.round_id != 2
        ):
            raise ValueError("auxiliary treatments are frozen at b8/r2")
        return self

    @property
    def cell_id(self) -> str:
        return "cell-" + canonical_content_hash(self.model_dump(mode="json"))[:24]

    @property
    def readable_name(self) -> str:
        model = re.sub(r"[^a-z0-9.-]", "-", self.answer_model.lower())
        return (
            f"{self.dataset.value}/{model}/{self.method}/{self.treatment.value}/"
            f"b{self.question_budget}-r{self.round_id}-{self.source_variant}"
        )


def build_experiment_matrix(protocol: ScientificProtocol, request: dict | None = None,
                            *, variant: str = "wr") -> list[ExperimentCell]:
    cells: dict[str, ExperimentCell] = {}
    for dataset in (DatasetName.MEMLENS, DatasetName.MEMGALLERY):
        for model in protocol.answer_backbones:
            for method in ("pmmc", *BASELINE_METHODS):
                variants = (
                    ("active", "no_fallback", "forced_top1") if method == "pmmc" else ("active",)
                )
                for source_variant in variants:
                    cell = ExperimentCell(
                        protocol_hash=protocol.content_hash,
                        dataset=dataset,
                        answer_model=model,
                        method=method,
                        source_variant=source_variant,
                    )
                    cells[cell.cell_id] = cell
        for budget in (1, 2, 4, 8):
            for round_id in range(4):
                for source_variant in ("active", "no_fallback", "forced_top1"):
                    cell = ExperimentCell(
                        protocol_hash=protocol.content_hash,
                        dataset=dataset,
                        answer_model=ANSWER_BACKBONES[0],
                        method="pmmc",
                        question_budget=budget,
                        round_id=round_id,
                        source_variant=source_variant,
                    )
                    cells[cell.cell_id] = cell
        for treatment in (Treatment.FIXED_PLANNER, Treatment.NO_RAW_IMAGE, Treatment.NO_RERANKER):
            for source_variant in ("active", "no_fallback", "forced_top1"):
                cell = ExperimentCell(
                    protocol_hash=protocol.content_hash,
                    dataset=dataset,
                    answer_model=ANSWER_BACKBONES[0],
                    method="pmmc",
                    treatment=treatment,
                    source_variant=source_variant,
                )
                cells[cell.cell_id] = cell
    selected = cells.values()
    if request is not None:
        stage = request["stage"]
        def included(cell: ExperimentCell) -> bool:
            if cell.dataset.value not in request["benchmarks"] or cell.answer_model not in request["models"]:
                return False
            if cell.method != "pmmc":
                return (variant == "wr" and stage in {"all", "baselines"}
                        and cell.method in request["methods"])
            if stage == "baselines":
                return False
            if variant == "wr" and stage in {"analysis", "all"} and "ablation" in request["analyses"]:
                return True
            return (cell.treatment == Treatment.MAIN and cell.source_variant == "active"
                    and (cell.question_budget, cell.round_id) == (8, 2))
        return sorted(filter(included, selected), key=lambda value: value.readable_name)
    return sorted(selected, key=lambda value: value.readable_name)


class CompiledQBankInstanceManifest(BaseSchema):
    protocol_hash: NonEmptyStr
    dataset: DatasetName
    snapshot_id: NonEmptyStr
    compiler_run_id: NonEmptyStr
    scope_hash: NonEmptyStr
    treatment: Treatment
    round_membership_hashes: dict[int, NonEmptyStr]
    collection_hashes: dict[str, NonEmptyStr]
    artifact_hashes: list[NonEmptyStr]
    r0_manifest_hash: NonEmptyStr

    @model_validator(mode="after")
    def _rounds(self) -> CompiledQBankInstanceManifest:
        if set(self.round_membership_hashes) != {0, 1, 2, 3}:
            raise ValueError("all four exact round memberships are required")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


class SelectionExpansionManifest(BaseSchema):
    campaign_id: NonEmptyStr
    phase: CampaignPhase
    selection: Literal["diagnostic", "regression", "confirmation", "dev-all", "5", "20", "100"]
    partition_hash: NonEmptyStr
    query_ids: dict[DatasetName, list[NonEmptyStr]]
    added_query_ids: dict[DatasetName, list[NonEmptyStr]]
    stratum_counts: dict[DatasetName, dict[str, int]]
    eligible_counts: dict[DatasetName, int]
    parent_selection_hash: str | None = None
    release_plan_hash: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")
    def _optional_request(self, handler):
        payload = handler(self)
        if self.release_plan_hash is None:
            payload.pop("release_plan_hash", None)
        return payload

    @model_validator(mode="after")
    def _selected(self) -> SelectionExpansionManifest:
        if self.release_plan_hash is not None:
            if self.phase != CampaignPhase.EXPLORATION or not self.query_ids or any(
                set(mapping) != set(self.query_ids)
                for mapping in (self.added_query_ids, self.eligible_counts, self.stratum_counts)
            ):
                raise ValueError("invalid release selection denominator")
        elif set(self.query_ids) != {DatasetName.MEMLENS, DatasetName.MEMGALLERY}:
            raise ValueError("both datasets must be represented")
        if (self.phase is CampaignPhase.PAPER) != (self.selection in {"5", "20", "100"}):
            raise ValueError("selection and campaign phase disagree")
        if (self.phase is CampaignPhase.EXPLORATION) != (self.selection == "dev-all"):
            raise ValueError("dev-all requires an independent exploration campaign")
        for dataset, values in self.query_ids.items():
            if not values or len(values) != len(set(values)):
                raise ValueError("selected queries must be nonempty and unique")
            if not set(self.added_query_ids.get(dataset, ())) <= set(values):
                raise ValueError("added queries must be in the selection")
            if self.phase is CampaignPhase.EXPLORATION and (
                self.parent_selection_hash is not None
                or self.eligible_counts.get(dataset) != len(values)
                or self.added_query_ids.get(dataset) != values
            ):
                raise ValueError("exploration must cover the complete development selection")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


class BenchmarkCampaignManifest(BaseSchema):
    schema_version: Literal["0.1.0"] = STEP8_SCHEMA_VERSION
    campaign_id: NonEmptyStr
    phase: CampaignPhase
    protocol: ScientificProtocol
    partition_id: NonEmptyStr
    snapshot_ids: dict[DatasetName, NonEmptyStr]
    matrix_hash: NonEmptyStr
    created_at: UtcDateTime
    protocol_freeze_hash: str | None = None
    paper_result: bool = False
    diagnostic_only: bool = False
    release_plan_hash: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")
    def _optional_request(self, handler):
        payload = handler(self)
        if self.release_plan_hash is None:
            payload.pop("release_plan_hash", None)
        return payload

    @model_validator(mode="after")
    def _paper_gate(self) -> BenchmarkCampaignManifest:
        if self.release_plan_hash is not None:
            if self.phase != CampaignPhase.EXPLORATION or not self.snapshot_ids:
                raise ValueError("invalid release campaign snapshots")
        elif set(self.snapshot_ids) != {DatasetName.MEMLENS, DatasetName.MEMGALLERY}:
            raise ValueError("campaign snapshots must cover both benchmarks")
        if self.phase is CampaignPhase.PAPER and not self.protocol_freeze_hash:
            raise ValueError("paper campaigns require an approved method protocol freeze")
        if self.paper_result != (self.phase is CampaignPhase.PAPER):
            raise ValueError("paper_result must follow the campaign phase")
        if self.diagnostic_only and self.paper_result:
            raise ValueError("diagnostic prefix cannot be paper output")
        if self.phase is CampaignPhase.EXPLORATION and (
            self.protocol_freeze_hash is not None or self.diagnostic_only
        ):
            raise ValueError("exploration cannot claim diagnostic or frozen status")
        return self


class AnalysisManifest(BaseSchema):
    protocol: Literal["paired-cluster-analysis-v1"] = "paired-cluster-analysis-v1"
    campaign_id: NonEmptyStr
    selection_hash: NonEmptyStr
    matrix_hash: NonEmptyStr
    seed: int = Field(default=0, ge=0)
    bootstrap_replicates: Literal[10000] = 10000
    confidence_level: float = Field(default=0.95, ge=0.95, le=0.95)
    cluster_by: dict[str, str] = Field(
        default_factory=lambda: {
            "memlens": "conversation_id",
            "memgallery": "scenario_id",
        }
    )
    primary_comparisons: tuple[str, ...] = BASELINE_METHODS
    multiple_testing: Literal["holm"] = "holm"
    input_hashes: dict[str, NonEmptyStr]
    analysis_source_hashes: dict[str, NonEmptyStr] = Field(default_factory=dict)


class Step8ExecutionConfig(BaseSchema):
    development_budget_limit_usd: Decimal = Field(default=Decimal("20"), gt=0, allow_inf_nan=False)
    cpu_workers: int = Field(default=32, ge=1, le=64)
    database_concurrency: int = Field(default=24, ge=1, le=64)
    model_concurrency: int = Field(default=64, ge=1, le=256)
    api_concurrency: int = Field(default=8, ge=1, le=64)
    gpu_idle_grace_seconds: int = Field(default=120, ge=0, le=600)
    heartbeat_seconds: int = Field(default=30, ge=1)
    lease_seconds: int = Field(default=900, ge=60)
    checkpoint_seconds: int = Field(default=1800, ge=60)
    checkpoint_jobs: int = Field(default=250, ge=1)
    periodic_checkpoint_retain: Literal[2] = 2
    transfer_part_bytes: int = Field(default=2 * 1024**3, ge=1024**2)


class Step8BenchmarkConfig(BaseSchema):
    schema_version: Literal["0.1.0"] = STEP8_SCHEMA_VERSION
    protocol_name: NonEmptyStr = "step8-scientific-contract-v3-gpt6-luna-haiku55"
    seed: int = Field(default=0, ge=0)
    partition_dir: Path
    exposure_manifest: Path
    regression_manifest: Path
    model_lock: Path
    model_registry: Path = Path("configs/models/step8.yaml")
    questioner_config: Path = Path("configs/compiler/questioner-v3.6.yaml")
    planner_config: Path = Path("configs/compiler/planner-v1.5.yaml")
    doubter_config: Path = Path("configs/compiler/doubter-v1.3.yaml")
    foundation_config: Path = Path("configs/compiler/pmmc-v7.yaml")
    query_executor_config: Path = Path("configs/runtime/query-executor-v3.yaml")
    runtime_profile: Path = Path("configs/runtime/h100-step8.yaml")
    execution_profile: Path = Path("configs/execution/h100-replicated-8gpu-step8.yaml")
    output_root: Path = Path("outputs/step8")
    execution: Step8ExecutionConfig = Field(default_factory=Step8ExecutionConfig)
    match_threshold: float = 0.6
    gap_threshold: float = 0.05
    text_relevance_threshold: float = 0.3
    multimodal_relevance_threshold: float = 0.3
    no_reranker_match_threshold: float = 0.6
    no_reranker_gap_threshold: float = 0.05
    oracle_at_50: OracleAt50Protocol | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @property
    def release_request(self) -> dict | None:
        if "release_request" not in self.metadata:
            return None
        from pmmc.release_plan import normalize_plan

        request = normalize_plan(self.metadata["release_request"])
        if request != self.metadata["release_request"]:
            raise ValueError("release request must be canonical")
        return request
