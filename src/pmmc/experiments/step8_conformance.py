"""Bounded, query-free production transport probes, separate from scientific runs.

Call ``run_step8_conformance`` with the aliases loaded by the current physical
worker. Native-only selections construct no GPU gateways. A returned receipt
certifies only its selected probes; it is never a benchmark result or denominator.
The caller supplies a resolved registry and existing API budget approval settings.
No approval, threshold calibration, model startup, or benchmark import happens here.
Compiler and judge aliases certify transport only, not compiler or evaluator
semantics. Judge-alias probes use synthetic grounding at runtime ANSWER stage;
they never create JUDGE facts or acquire evaluator credentials.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.baselines.runner import TransactionalModelCallWriter
from pmmc.compiler.ids import canonical_content_hash
from pmmc.experiments.manifest import run_config_hash
from pmmc.experiments.step8_artifacts import CampaignArtifactStore, sha256_file, write_json
from pmmc.experiments.step8_jobs import scientific_source_fingerprint
from pmmc.experiments.step8_runner import safe_failure_record
from pmmc.infrastructure.database.constants import DatabaseRole
from pmmc.infrastructure.database.engine import create_session_factory
from pmmc.infrastructure.database.experiment_mappers import run_state_from_row, run_state_to_row
from pmmc.infrastructure.database.experiment_models import RunStateModel
from pmmc.infrastructure.database.experiment_repositories import (
    ExperimentUnitOfWork,
    _insert_idempotent,
    _model_mapping,
)
from pmmc.infrastructure.database.mappers import run_manifest_to_row
from pmmc.infrastructure.database.models import RunManifestModel
from pmmc.infrastructure.database.step8_repositories import PostgresBenchmarkRepository
from pmmc.infrastructure.experiment_artifacts import ExperimentArtifactService
from pmmc.infrastructure.object_store import create_runtime_object_store
from pmmc.infrastructure.settings import Settings
from pmmc.models.campaign_audit import CampaignModelAudit, campaign_model_scope
from pmmc.models.execution import (
    ModelCallContext,
    ParsedModelOutput,
    _canonical_bytes,
    _generation_manifest,
)
from pmmc.models.factory import create_generation_gateway, create_model_execution_service
from pmmc.models.provider_config import prepare_provider_answer_request
from pmmc.models.provider_images import ImageBytes, validate_image_bytes
from pmmc.models.registry import ModelRegistry
from pmmc.models.step8_accounting import (
    SelectionApproval,
    load_settings_rate_card,
    require_selection_approval,
)
from pmmc.runtime.answering import (
    parse_typed_answer,
    prepare_typed_answer_request,
    typed_answer_public_output_tokens,
)
from pmmc.runtime.grounding import build_grounding_request, strict_json_object
from pmmc.runtime.qbank_reranker import (
    RERANKER_PROMPT,
    RERANKER_PROMPT_VERSION,
    effective_reranker_output,
    preflight_reranker_contract,
    reranker_contract_hash,
)
from pmmc.schemas import (
    AnswerGenerationInput,
    AppStage,
    AssetType,
    DataScope,
    GenerationRequest,
    GenerationResult,
    ImageRef,
    ModelCallTrace,
    ModelContentPart,
    ModelMessage,
    QBankRerankerOutput,
    QueryInput,
    RerankerMatchLevel,
    RunManifest,
    RunState,
    RunStatus,
    TokenUsage,
)
from pmmc.schemas.answer_output import (
    AnswerContractV3,
    AnswerEvidenceSource,
    AnswerSourceImage,
    GroundingAssessment,
    TypedAnswerContext,
)
from pmmc.schemas.base import BaseSchema, NonEmptyStr

CONFORMANCE_VERSION: Final = "step8-production-conformance-v1"
ProbeRole = Literal["reranker", "answer", "grounding", "compiler", "judge"]
ExecutionKind = Literal["live", "offline_transport"]
_ALIASES: dict[str, tuple[ProbeRole, str]] = {
    "compiler-qwen3.5-9b": ("compiler", "qwen_vllm"),
    "reranker-qwen3.5-9b": ("reranker", "qwen_vllm"),
    "answer-qwen3.5-9b": ("answer", "qwen_vllm"),
    "grounding-qwen3.5-9b": ("grounding", "qwen_vllm"),
    "judge-qwen3.5-9b": ("judge", "qwen_vllm"),
    "answer-qwen3.5-27b": ("answer", "qwen_vllm"),
    "answer-gpt-6-luna": ("answer", "openai"),
    "answer-claude-haiku-5.5": ("answer", "anthropic"),
}
_QUERY = "What color was Nora's raincoat during her walk by the lake?"
_POSITIVE = "Which color raincoat did Nora wear on the lakeside walk?"
_NEGATIVE = "What color was Nora's backpack during her walk by the lake?"
_IMAGE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsGqe"
    "ExAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA4LMf88OL0EKXAAAAAAElFTkSuQmCC"
)
_IMAGE_SHA = hashlib.sha256(_IMAGE).hexdigest()
_IMAGE_REF = ImageRef(
    image_id="conformance-source-image",
    sha256=_IMAGE_SHA,
    mime_type="image/png",
    width=64,
    height=64,
)


class ConformanceError(RuntimeError):
    """Content-free diagnostics; inspect private artifacts for provider details."""

    def __init__(self, code: str, *, probe_id: str | None = None) -> None:
        self.code, self.probe_id = code, probe_id
        super().__init__(code)


class ProbeReceipt(BaseSchema):
    model_config = ConfigDict(frozen=True)
    probe_id: NonEmptyStr
    model_alias: NonEmptyStr
    role: ProbeRole
    model_call_id: NonEmptyStr
    request_hash: NonEmptyStr
    trace_sha256: NonEmptyStr
    result_sha256: NonEmptyStr
    input_artifact_id: NonEmptyStr
    output_artifact_id: NonEmptyStr
    usage: TokenUsage
    api_cost_usd: Decimal | None = Field(ge=0)
    price_estimated: bool | None


class ConformanceReceipt(BaseSchema):
    model_config = ConfigDict(frozen=True)
    version: Literal["step8-production-conformance-v1"] = CONFORMANCE_VERSION
    run_id: NonEmptyStr
    plan_hash: NonEmptyStr
    campaign_id: NonEmptyStr
    protocol_hash: NonEmptyStr
    selection_hash: NonEmptyStr
    execution_kind: ExecutionKind
    probe_aliases: tuple[NonEmptyStr, ...]
    roles: tuple[ProbeRole, ...]
    probes: tuple[ProbeReceipt, ...]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    api_cost_usd: Decimal = Field(ge=0)
    unpriced_local_calls: int = Field(ge=0)
    scientific_assignments: Literal[0] = 0
    paper_result: Literal[False] = False
    passed: Literal[True] = True

    @property
    def content_hash(self) -> str:
        return canonical_content_hash(self.model_dump(mode="json"))


@dataclass(frozen=True)
class ConformanceProbe:
    probe_id: str
    role: ProbeRole
    request: GenerationRequest
    stage: AppStage
    context: TypedAnswerContext | None = None


@dataclass(frozen=True)
class ConformancePlan:
    run_id: str
    plan_hash: str
    protocol_hash: str
    campaign_id: str
    selection_hash: str
    execution_kind: ExecutionKind
    probe_aliases: tuple[str, ...]
    roles: tuple[ProbeRole, ...]
    probes: tuple[ConformanceProbe, ...]


class _FixtureImageResolver:
    async def resolve(self, image: ImageRef) -> ImageBytes:
        if image != _IMAGE_REF:
            raise ConformanceError("conformance_image_not_in_frozen_fixture")
        return validate_image_bytes(image, ImageBytes(_IMAGE, "image/png"))


class _ConformanceAudit(CampaignModelAudit):
    @property
    def metadata(self) -> dict[str, str]:
        return {
            **super().metadata,
            "step8_conformance_run_id": self.attempt_id,
            "step8_conformance_probe_id": self.job_id,
            "step8_cost_scope": "research_conformance",
        }


def _answer_input(alias: str) -> AnswerGenerationInput:
    query = QueryInput(
        query_id="conformance-only",
        dataset="unknown",
        conversation_id="conformance-only",
        raw_query="What color is the square in the image? Use one lowercase color word.",
    )
    source = AnswerEvidenceSource(
        evidence_alias="E01",
        evidence_id="conformance-source",
        text="The attached image is the source for this question.",
        images=[AnswerSourceImage(image_id=_IMAGE_REF.image_id, image=_IMAGE_REF)],
    )
    request = prepare_typed_answer_request(
        GenerationRequest(
            request_id="conformance-template",
            model_alias=alias,
            messages=[
                ModelMessage(role="user", parts=[ModelContentPart(kind="text", text="source")])
            ],
            max_output_tokens=512,
            temperature=0,
            top_p=1,
            seed=0,
            timeout_seconds=120,
            metadata={"answer_contract_version": "typed-answer-contract-v3"},
        ),
        query=query,
        evidence_sources=[source],
    )
    # These in-memory IDs support the production request builder only. Never persist
    # them as snapshot/query foreign keys or register them as scientific assignments.
    return AnswerGenerationInput(
        method_id=CONFORMANCE_VERSION,
        answer_variant="probe",
        run_id="conformance-template",
        snapshot_id="conformance-not-a-snapshot",
        query=query,
        request=request,
        evidence_ids_in_prompt=[source.evidence_id],
        image_ids_in_prompt=[_IMAGE_REF.image_id],
        source_lineage={"conformance_only": True},
        metadata=request.metadata,
    )


def _templates(aliases: tuple[str, ...], registry: ModelRegistry) -> tuple[ConformanceProbe, ...]:
    probes = []
    for alias in aliases:
        role, _kind = _ALIASES[alias]
        if role == "reranker":
            preflight_reranker_contract()
            for name, candidate in (("positive", _POSITIVE), ("negative", _NEGATIVE)):
                request = GenerationRequest(
                    request_id="conformance-template",
                    model_alias=alias,
                    messages=[
                        ModelMessage(
                            role="system",
                            parts=[ModelContentPart(kind="text", text=RERANKER_PROMPT)],
                        ),
                        ModelMessage(
                            role="user",
                            parts=[
                                ModelContentPart(
                                    kind="text",
                                    text=(
                                        "<QUERY>"
                                        + _QUERY
                                        + '</QUERY>\n<CANDIDATE alias="C01">'
                                        + candidate
                                        + "</CANDIDATE>\n<QUESTION_TYPE>factoid</QUESTION_TYPE>\n"
                                        "<REQUIRED_MODALITIES>text</REQUIRED_MODALITIES>"
                                    ),
                                )
                            ],
                        ),
                    ],
                    max_output_tokens=128,
                    provider_max_output_tokens=128,
                    temperature=0,
                    top_p=1,
                    seed=0,
                    thinking_mode="disabled",
                    timeout_seconds=120,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "pmmc_qbank_reranker_v1",
                            "strict": True,
                            "schema": QBankRerankerOutput.model_json_schema(),
                        },
                    },
                    metadata={
                        "prompt_version": RERANKER_PROMPT_VERSION,
                        "reranker_contract_hash": reranker_contract_hash(),
                        "single_pair_scope": True,
                        "private_compiler_facts_present": False,
                    },
                )
                probes.append(ConformanceProbe("reranker-" + name, role, request, AppStage.MATCH))
        elif role in {"answer", "compiler"}:
            prepared = _answer_input(alias)
            config = registry.get(alias).provider_config
            assert config is not None
            request = prepare_provider_answer_request(prepared.request, config)
            probes.append(
                ConformanceProbe(
                    alias + "-image",
                    role,
                    request,
                    AppStage.COMPILE if role == "compiler" else AppStage.ANSWER,
                    TypedAnswerContext.model_validate(prepared.metadata["typed_answer_context"]),
                )
            )
        else:
            request = build_grounding_request(
                _answer_input("answer-qwen3.5-9b"),
                AnswerContractV3(
                    answer_kind="text",
                    answer_text="blue",
                    selected_image_aliases=[],
                    supported=True,
                    used_evidence_aliases=["E01"],
                    abstain_reason=None,
                ),
                answer_model_call_id="conformance-proposed-answer",
                model_alias=alias,
            )
            request = request.model_copy(update={"provider_max_output_tokens": 512})
            probes.append(
                ConformanceProbe(
                    alias + "-image",
                    role,
                    request,
                    # This is source-only grounding transport, including for the
                    # judge alias. JUDGE is reserved for real evaluation parents.
                    AppStage.ANSWER,
                )
            )
    return tuple(probes)


def build_step8_conformance_plan(
    *,
    settings: Settings,
    registry: ModelRegistry,
    protocol_hash: str,
    campaign_id: str,
    selection_hash: str,
    probe_aliases: Sequence[str],
    roles: Sequence[ProbeRole] | None = None,
    execution_kind: ExecutionKind = "live",
) -> ConformancePlan:
    """Pure preflight; selected aliases determine the finite call bound (one or two each)."""
    if not all(re.fullmatch(r"[a-f0-9]{64}", value) for value in (protocol_hash, selection_hash)):
        raise ConformanceError("conformance_scope_hash_invalid")
    if (
        not campaign_id.strip()
        or not probe_aliases
        or len(set(probe_aliases)) != len(probe_aliases)
    ):
        raise ConformanceError("conformance_alias_selection_invalid")
    if any(alias not in _ALIASES for alias in probe_aliases):
        raise ConformanceError("conformance_alias_unsupported")
    aliases = tuple(alias for alias in _ALIASES if alias in probe_aliases)
    selected_roles = tuple(dict.fromkeys(_ALIASES[alias][0] for alias in aliases))
    if roles is not None and (not roles or set(roles) != set(selected_roles)):
        raise ConformanceError("conformance_roles_must_exactly_match_aliases")
    subset = ModelRegistry(
        registry_version=registry.registry_version,
        models=[registry.get(alias) for alias in aliases],
    )
    subset.assert_paper_ready()
    for entry in subset.models:
        if entry.provider_config is None or entry.provider_config.kind != _ALIASES[entry.alias][1]:
            raise ConformanceError("conformance_requires_production_provider")
        expected = "Qwen/Qwen3.5-27B" if entry.alias == "answer-qwen3.5-27b" else "Qwen/Qwen3.5-9B"
        if entry.provider_config.kind == "qwen_vllm" and entry.expected_actual_model_id != expected:
            raise ConformanceError("conformance_physical_model_mismatch")
    templates = _templates(aliases, registry)
    paid = any(_ALIASES[alias][1] in {"openai", "anthropic"} for alias in aliases)
    if paid and (not settings.step8_budget_approval_id or not settings.step8_rate_card_path):
        raise ConformanceError("conformance_budget_configuration_required")
    identity = {
        "version": CONFORMANCE_VERSION,
        "protocol_hash": protocol_hash,
        "campaign_id": campaign_id,
        "selection_hash": selection_hash,
        "registry_hash": registry.resolved_hash_for_aliases(aliases),
        "execution_kind": execution_kind,
        "runtime_id": settings.active_runtime_id,
        "runtime_services": {
            alias: settings.active_runtime_service_map.get(alias) for alias in aliases
        },
        "fixture_image_sha256": _IMAGE_SHA,
        "budget_approval_id": settings.step8_budget_approval_id if paid else None,
        "rate_card_hash": sha256_file(settings.resolve_path(settings.step8_rate_card_path))
        if paid and settings.step8_rate_card_path is not None
        else None,
        "module_sha256": sha256_file(Path(__file__)),
        "sources": {
            stage: scientific_source_fingerprint(settings.project_root, stage)
            for stage in {
                "matching"
                if role == "reranker"
                else "answers"
                if role in {"answer", "compiler"}
                else "grounding"
                for role in selected_roles
            }
        },
        "probes": [
            {"probe_id": probe.probe_id, "request": probe.request.model_dump(mode="json")}
            for probe in templates
        ],
    }
    plan_hash = canonical_content_hash(identity)
    run_id = "step8-conformance-" + plan_hash[:32]
    probes = tuple(
        ConformanceProbe(
            probe.probe_id,
            probe.role,
            probe.request.model_copy(
                update={
                    "request_id": "step8-conformance-request-"
                    + canonical_content_hash([run_id, probe.probe_id])[:32],
                    "metadata": {
                        **probe.request.metadata,
                        "step8_campaign_id": campaign_id,
                        "step8_selection_hash": selection_hash,
                        "step8_budget_approval_id": settings.step8_budget_approval_id
                        if _ALIASES[probe.request.model_alias][1] in {"openai", "anthropic"}
                        else None,
                        "conformance_run_id": run_id,
                        "conformance_role": probe.role,
                        "conformance_probe_scope": "synthetic_transport",
                        "scientific_assignments": 0,
                        "paper_result": False,
                    },
                }
            ),
            probe.stage,
            probe.context,
        )
        for probe in templates
    )
    return ConformancePlan(
        run_id,
        plan_hash,
        protocol_hash,
        campaign_id,
        selection_hash,
        execution_kind,
        aliases,
        selected_roles,
        probes,
    )


def _parse(
    probe: ConformanceProbe, result: GenerationResult, registry: ModelRegistry
) -> ParsedModelOutput:
    entry = registry.get(probe.request.model_alias)
    assert entry.provider_config is not None
    kind = entry.provider_config.kind
    if (
        result.request_id != probe.request.request_id
        or result.model_alias != entry.alias
        or result.actual_model_id != entry.expected_actual_model_id
        or result.finish_reason != "stop"
        or result.metadata.get("canonical_schema_validated") is not True
        or result.metadata.get("effective_reasoning_mode") != entry.provider_config.reasoning_mode
        or result.usage.metadata.get("usage_reported") is not True
        or re.search(r"</?think\b", result.text, re.IGNORECASE)
    ):
        raise ConformanceError("conformance_provider_contract_failed", probe_id=probe.probe_id)
    if kind in {"openai", "anthropic"}:
        accounting = result.metadata.get("step8_accounting", {})
        if (
            not isinstance(accounting, dict)
            or accounting.get("reservation_state") != "settled"
            or accounting.get("selection_hash") != probe.request.metadata["step8_selection_hash"]
            or accounting.get("approval_id") != probe.request.metadata["step8_budget_approval_id"]
            or result.usage.metadata.get("usage_reported") is not True
        ):
            raise ConformanceError("conformance_paid_call_not_settled", probe_id=probe.probe_id)
    output: QBankRerankerOutput | AnswerContractV3 | GroundingAssessment
    if probe.role == "reranker":
        output = QBankRerankerOutput.model_validate(strict_json_object(result.text))
        _, level = effective_reranker_output(output)
        correct = (
            level in {RerankerMatchLevel.EXACT, RerankerMatchLevel.STRONG}
            if probe.probe_id == "reranker-positive"
            else level is RerankerMatchLevel.NONE and output.same_information_need == "fail"
        )
    elif probe.role in {"answer", "compiler"}:
        assert probe.context is not None
        output = parse_typed_answer(
            result.text,
            context=probe.context,
            finish_reason=result.finish_reason,
            output_tokens=typed_answer_public_output_tokens(result),
        )
        correct = (
            output.supported
            and output.answer_kind == "text"
            and (output.answer_text or "").strip().casefold() == "blue"
            and output.used_evidence_aliases == ["E01"]
        )
    else:
        output = GroundingAssessment.model_validate(strict_json_object(result.text))
        correct = output.supported and output.query_closure == output.source_support == "pass"
    if not correct:
        raise ConformanceError("conformance_frozen_probe_failed", probe_id=probe.probe_id)
    return ParsedModelOutput(value=output.model_dump(mode="json"))


async def _register_run(sessions: async_sessionmaker[AsyncSession], manifest: RunManifest) -> None:
    # Conformance has no benchmark denominator, but runtime writes still require
    # BOTH parents. Commit them together; repair legacy manifest-only attempts.
    if (
        manifest.dataset_snapshot_ids
        or manifest.dataset.value != "unknown"
        or manifest.paper_mode
        or manifest.method != CONFORMANCE_VERSION
        or not manifest.run_id.startswith("step8-conformance-")
        or manifest.metadata.get("scientific_assignments") != 0
        or manifest.config_hash != run_config_hash(manifest)
    ):
        raise ConformanceError("conformance_run_scope_invalid")
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        if uow.session is None:
            raise ConformanceError("conformance_runtime_session_missing")
        await _insert_idempotent(
            uow.session, RunManifestModel, run_manifest_to_row(manifest), manifest.run_id
        )
        stored = await uow.session.get(RunStateModel, manifest.run_id)
        reused = stored is not None
        if stored is None:
            stored = await _insert_idempotent(
                uow.session,
                RunStateModel,
                run_state_to_row(RunState(run_id=manifest.run_id)),
                manifest.run_id,
            )
        state = run_state_from_row(_model_mapping(stored))
        if state.selected_count or state.succeeded_count or state.failed_count:
            raise ConformanceError("conformance_run_denominator_nonzero")
        if state.status not in {RunStatus.CREATED, RunStatus.RUNNING}:
            raise ConformanceError("conformance_run_terminal_immutable")
    print(json.dumps({
        "event": "STEP8_CONFORMANCE_RUN_REGISTERED",
        "state": "ready", "registration_reused": reused,
        "run_status": state.status.value, "scientific_assignments": 0,
    }, sort_keys=True), flush=True)


def verify_step8_conformance_receipt(
    receipt: ConformanceReceipt,
    *,
    plan: ConformancePlan,
    store: CampaignArtifactStore,
    registry: ModelRegistry,
) -> None:
    """Fail closed on offline, stale, partial, or modified selected-probe receipts."""
    if (
        receipt.execution_kind != "live"
        or plan.execution_kind != "live"
        or receipt.run_id != plan.run_id
        or receipt.plan_hash != plan.plan_hash
        or receipt.campaign_id != plan.campaign_id
        or receipt.protocol_hash != plan.protocol_hash
        or receipt.selection_hash != plan.selection_hash
        or receipt.probe_aliases != plan.probe_aliases
        or receipt.roles != plan.roles
        or tuple(item.probe_id for item in receipt.probes) != tuple(p.probe_id for p in plan.probes)
        or receipt.input_tokens != sum(item.usage.input_tokens for item in receipt.probes)
        or receipt.output_tokens != sum(item.usage.output_tokens for item in receipt.probes)
        or receipt.api_cost_usd
        != sum((item.api_cost_usd or Decimal(0) for item in receipt.probes), Decimal(0))
        or receipt.unpriced_local_calls != sum(item.api_cost_usd is None for item in receipt.probes)
    ):
        raise ConformanceError("conformance_receipt_not_production_ready")
    for proof, probe in zip(receipt.probes, plan.probes, strict=True):
        if (
            proof.model_alias != probe.request.model_alias
            or proof.role != probe.role
            or proof.request_hash != canonical_content_hash(probe.request.model_dump(mode="json"))
        ):
            raise ConformanceError("conformance_receipt_probe_mismatch")
        trace = ModelCallTrace.model_validate(store.load_json(proof.trace_sha256))
        result = GenerationResult.model_validate(store.load_json(proof.result_sha256))
        if (
            proof.model_call_id
            != "step8-conformance-call-"
            + canonical_content_hash([plan.run_id, probe.probe_id])[:32]
            or trace.model_call_id != proof.model_call_id
            or trace.run_id != plan.run_id
            or trace.query_id is not None
            or trace.snapshot_id is not None
            or trace.status.value != "success"
            or trace.data_scope is not DataScope.RUNTIME
            or trace.input_asset_id != proof.input_artifact_id
            or trace.output_asset_id != proof.output_artifact_id
            or trace.model_alias != proof.model_alias
            or trace.stage is not probe.stage
            or trace.actual_model_id != result.actual_model_id
            or trace.prompt_hash
            != hashlib.sha256(_canonical_bytes(_generation_manifest(probe.request))).hexdigest()
            or proof.usage != result.usage
            or trace.metadata.get("step8_conformance_run_id") != plan.run_id
        ):
            raise ConformanceError("conformance_receipt_trace_mismatch")
        if trace.parsed_output != _parse(probe, result, registry).value:
            raise ConformanceError("conformance_receipt_parsed_output_mismatch")


async def run_step8_conformance(
    *,
    settings: Settings,
    registry: ModelRegistry,
    protocol_hash: str,
    campaign_id: str,
    selection_hash: str,
    output_root: Path,
    probe_aliases: Sequence[str],
    roles: Sequence[ProbeRole] | None = None,
    provider_clients: Mapping[str, Any] | None = None,
) -> ConformanceReceipt:
    """Run only selected probes through real adapters, audited storage, and accounting.

    ``provider_clients`` is an offline SDK-transport testing hook. Its receipts are
    explicitly non-production, even when every probe passes. Live native probes
    require the existing settings budget approval/rate card and real credentials.
    Successful response journals recover without a second provider dispatch.
    ``roles``, when supplied, must exactly match the roles of ``probe_aliases``.
    Costs include all billed usage, including reasoning tokens; aggregate resumed
    receipts by model_call_id rather than adding the same receipt twice.
    """
    plan = build_step8_conformance_plan(
        settings=settings,
        registry=registry,
        protocol_hash=protocol_hash,
        campaign_id=campaign_id,
        selection_hash=selection_hash,
        probe_aliases=probe_aliases,
        roles=roles,
        execution_kind="offline_transport" if provider_clients is not None else "live",
    )
    shared_root = settings.resolve_path(output_root)
    root = shared_root / "conformance" / plan.run_id
    root.mkdir(parents=True, exist_ok=True)
    store = CampaignArtifactStore(shared_root)
    settings = settings.model_copy(
        update={
            "model_dev_max_attempts": 1,
            "model_paper_max_attempts": 1,
            "model_max_concurrency": 1,
            "step8_output_root": shared_root,
            "step8_attempt_id": plan.run_id,
        }
    )
    registry = ModelRegistry(
        registry_version=registry.registry_version,
        models=[registry.get(alias) for alias in plan.probe_aliases],
    )
    gateways = {}
    for alias in plan.probe_aliases:
        kind = _ALIASES[alias][1]
        if provider_clients is not None and kind not in provider_clients:
            raise ConformanceError("conformance_offline_transport_missing")
        gateways[alias] = create_generation_gateway(
            settings,
            registry,
            alias,
            image_bytes_resolver=_FixtureImageResolver(),
            provider_client=provider_clients[kind] if provider_clients is not None else None,
        )
    path = root / "manifest.json"
    if path.exists():
        manifest = RunManifest.model_validate_json(path.read_bytes())
        if (
            manifest.run_id != plan.run_id
            or manifest.metadata.get("conformance_plan_hash") != plan.plan_hash
            or manifest.method != CONFORMANCE_VERSION
            or manifest.experiment_name != CONFORMANCE_VERSION
            or manifest.selection_hash != plan.plan_hash
            or manifest.config_hash != run_config_hash(manifest)
            or manifest.metadata.get("scientific_assignments") != 0
            or manifest.metadata.get("execution_kind") != plan.execution_kind
        ):
            raise ConformanceError("conformance_manifest_identity_mismatch")
    else:
        manifest = RunManifest(
            run_id=plan.run_id,
            experiment_name=CONFORMANCE_VERSION,
            dataset="unknown",
            method=CONFORMANCE_VERSION,
            paper_mode=False,
            selection_hash=plan.plan_hash,
            metadata={
                "conformance_plan_hash": plan.plan_hash,
                "campaign_id": campaign_id,
                "protocol_hash": protocol_hash,
                "budget_selection_hash": selection_hash,
                "execution_kind": plan.execution_kind,
                "scientific_assignments": 0,
                "paper_result": False,
                "probe_ids": [p.probe_id for p in plan.probes],
            },
        )
        manifest = manifest.model_copy(update={"config_hash": run_config_hash(manifest)})
        write_json(path, manifest.model_dump(mode="json"), immutable=True)
    engine, sessions = create_session_factory(settings, DatabaseRole.RUNTIME)
    current_probe = None
    phase = "budget_approval"
    primary: BaseException | None = None

    def record_failure(error: BaseException) -> None:
        facts = safe_failure_record(error, stage="conformance", locator="probe")
        try:
            write_json(
                root / "failure.json",
                {
                    "run_id": plan.run_id,
                    "probe_id": current_probe,
                    "code": error.code
                    if isinstance(error, ConformanceError)
                    else "conformance_execution_failed",
                    "error_type": type(error).__name__,
                    "error_code": facts["error_code"],
                    "unknown_provider_outcome": facts["unknown_provider_outcome"],
                    "diagnostics": facts["diagnostics"],
                    "phase": phase,
                    "scientific_assignments": 0,
                },
            )
        except Exception as secondary:
            error.add_note("conformance failure persistence failed: " + type(secondary).__name__)

    try:
        repository = PostgresBenchmarkRepository(sessions)
        for alias in plan.probe_aliases:
            if _ALIASES[alias][1] in {"openai", "anthropic"}:
                rate_card = load_settings_rate_card(settings, registry.get(alias))
                await require_selection_approval(
                    repository,
                    SelectionApproval(
                        campaign_id,
                        selection_hash,
                        str(settings.step8_budget_approval_id),
                        rate_card.sha256,
                    ),
                )
        if any(_ALIASES[alias][1] in {"openai", "anthropic"} for alias in plan.probe_aliases):
            from pmmc.experiments.step8_conformance_replacement import guard_paid_plan

            phase = "paid_plan_identity"
            authorization = await guard_paid_plan(
                sessions=sessions, store=store, plan=plan, registry=registry,
                approval_id=str(settings.step8_budget_approval_id),
            )
            if authorization is not None:
                write_json(root / "replacement-authorization.json", authorization, immutable=True)
                print(json.dumps({
                    "event": "STEP8_CONFORMANCE_REPLACEMENT", "state": "authorized",
                    "authorization_hash": authorization["content_hash"],
                    "original_state": "unknown", "original_actual_usd": None,
                    "original_reserved_usd": authorization["original"]["reserved_usd"],
                    "budget_increased": False, "max_replacements": 1,
                }, sort_keys=True), flush=True)
        phase = "run_registration"
        await _register_run(sessions, manifest)
        artifacts = ExperimentArtifactService(
            sessions,
            create_runtime_object_store(settings),
            DataScope.RUNTIME,
            artifact_prefix="conformance",
        )
        execution = create_model_execution_service(
            settings,
            registry,
            artifact_service=artifacts,
            model_calls=TransactionalModelCallWriter(sessions),
        )
        if any(probe.role != "reranker" for probe in plan.probes):
            phase = "fixture_artifact"
            await artifacts.put_bytes(
                artifact_id=plan.run_id + ":fixture-image",
                run_id=plan.run_id,
                artifact_type=AssetType.RAW_IMAGE,
                data=_IMAGE,
                mime_type="image/png",
                extension="png",
            )
        receipts = []
        for probe in plan.probes:
            current_probe = probe.probe_id
            phase = "provider_probe"
            call_id = (
                "step8-conformance-call-"
                + canonical_content_hash([plan.run_id, probe.probe_id])[:32]
            )
            audit = _ConformanceAudit(
                store, campaign_id, plan.run_id, probe.probe_id, plan.plan_hash
            )
            key = audit.identity(
                call_id,
                _generation_manifest(probe.request),
                registry.resolved_hash_for_aliases([probe.request.model_alias]),
            )
            # Native API gateways already use the budget ledger's durable dispatch
            # claim. For local GPU calls, an unknown prior dispatch also fails closed.
            if (
                _ALIASES[probe.request.model_alias][1] == "qwen_vllm"
                and audit.restore_result(call_id, key) is None
            ):
                claim = root / "dispatch" / key
                claim.parent.mkdir(exist_ok=True)
                try:
                    claim.mkdir()
                except FileExistsError:
                    raise ConformanceError(
                        "conformance_prior_dispatch_unresolved", probe_id=probe.probe_id
                    ) from None
                write_json(
                    claim / "intent.json",
                    {"model_call_id": call_id, "request_hash": key},
                    immutable=True,
                )
            with campaign_model_scope(audit):
                audited = await execution.generate_audited(
                    gateways[probe.request.model_alias],
                    probe.request,
                    ModelCallContext(model_call_id=call_id, run_id=plan.run_id, stage=probe.stage),
                    output_parser=partial(_parse, probe, registry=registry),
                )
            trace = audited.trace
            if (
                trace.status.value != "success"
                or not trace.input_asset_id
                or not trace.output_asset_id
            ):
                raise ConformanceError("conformance_call_not_successful", probe_id=probe.probe_id)
            cost, estimated = None, None
            if _ALIASES[probe.request.model_alias][1] in {"openai", "anthropic"}:
                rate_card = load_settings_rate_card(
                    settings, registry.get(probe.request.model_alias)
                )
                cost = rate_card.actual(audited.result, registry.get(probe.request.model_alias))
                estimated = rate_card.card.estimated
                if cost is None:
                    raise ConformanceError("conformance_paid_cost_unknown", probe_id=probe.probe_id)
                cost = cost.quantize(Decimal("0.00000001"), rounding=ROUND_CEILING)
            receipts.append(
                ProbeReceipt(
                    probe_id=probe.probe_id,
                    model_alias=probe.request.model_alias,
                    role=probe.role,
                    model_call_id=call_id,
                    request_hash=canonical_content_hash(probe.request.model_dump(mode="json")),
                    trace_sha256=store.put_json(trace.model_dump(mode="json")),
                    result_sha256=store.put_json(audited.result.model_dump(mode="json")),
                    input_artifact_id=trace.input_asset_id,
                    output_artifact_id=trace.output_asset_id,
                    usage=audited.result.usage,
                    api_cost_usd=cost,
                    price_estimated=estimated,
                )
            )
        phase = "receipt"
        receipt = ConformanceReceipt(
            run_id=plan.run_id,
            plan_hash=plan.plan_hash,
            campaign_id=campaign_id,
            protocol_hash=protocol_hash,
            selection_hash=selection_hash,
            execution_kind=plan.execution_kind,
            probe_aliases=plan.probe_aliases,
            roles=plan.roles,
            probes=tuple(receipts),
            input_tokens=sum(item.usage.input_tokens for item in receipts),
            output_tokens=sum(item.usage.output_tokens for item in receipts),
            api_cost_usd=sum((item.api_cost_usd or Decimal(0) for item in receipts), Decimal(0)),
            unpriced_local_calls=sum(item.api_cost_usd is None for item in receipts),
        )
        write_json(root / "receipt.json", receipt.model_dump(mode="json"), immutable=True)
        return receipt
    except BaseException as exc:
        primary = exc
        record_failure(exc)
        raise
    finally:
        try:
            await engine.dispose()
        except BaseException as secondary:
            if primary is None:
                record_failure(secondary)
                raise
            primary.add_note("conformance engine disposal failed: " + type(secondary).__name__)
