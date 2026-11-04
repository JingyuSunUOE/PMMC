"""Typed PMMC operator registry, validation, and deterministic linear execution."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from pmmc.compiler.config import CompilerFoundationConfig
from pmmc.compiler.execution import program_evidence_quotas
from pmmc.compiler.ids import (
    canonical_content_hash,
    memory_program_content_hash,
    stable_compiler_id,
)
from pmmc.memory.evidence_roles import EVIDENCE_ROLE_POLICY, produced_evidence_roles
from pmmc.schemas import (
    BaseSchema,
    EvidenceType,
    MemoryProgram,
    Modality,
    OperatorControlAction,
    OperatorEvidence,
    OperatorExecutionContext,
    OperatorExecutionPurpose,
    OperatorOutput,
    OperatorSpec,
    ProgramExecutionEvidenceRef,
    ProgramExecutionStepTrace,
    ProgramExecutionTrace,
    ProgramValidationIssue,
    ProgramValidationResult,
    ProgramValidationSeverity,
    TraceStatus,
)
from pmmc.schemas.base import (
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    ensure_unique_strings,
    utc_now,
)


class KeywordSearchArgs(BaseSchema):
    query: NonEmptyStr | None = None
    top_k: PositiveInt = 10
    entity_mentions: list[NonEmptyStr] = Field(default_factory=list)
    term_source: Literal["canonical_question"] | None = None
    term_budget: PositiveInt | None = Field(default=None, le=8)
    server_derived: bool = False

    @model_validator(mode="after")
    def _validate_entities(self) -> KeywordSearchArgs:
        ensure_unique_strings(self.entity_mentions, "entity_mentions")
        audit_fields_present = (
            self.term_source is not None
            or self.term_budget is not None
            or self.server_derived
        )
        if audit_fields_present and (
            self.term_source != "canonical_question"
            or self.term_budget is None
            or not self.server_derived
        ):
            raise ValueError(
                "server-derived keyword args require source, budget, and audit marker"
            )
        return self


class EmbeddingSearchArgs(BaseSchema):
    vector_role: NonEmptyStr
    top_k: PositiveInt = 10


class ImageEvidenceArgs(BaseSchema):
    image_ids: list[NonEmptyStr] = Field(default_factory=list)
    source_step_ids: list[NonNegativeInt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_images(self) -> ImageEvidenceArgs:
        ensure_unique_strings(self.image_ids, "image_ids")
        if len(self.source_step_ids) != len(set(self.source_step_ids)):
            raise ValueError("source_step_ids must not contain duplicates")
        return self


class ContextAroundImageArgs(ImageEvidenceArgs):
    radius: NonNegativeInt = 1


class TemporalNeighborArgs(BaseSchema):
    memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    source_step_ids: list[NonNegativeInt] = Field(default_factory=list)
    radius: PositiveInt = 1

    @model_validator(mode="after")
    def _validate_memories(self) -> TemporalNeighborArgs:
        ensure_unique_strings(self.memory_ids, "memory_ids")
        if len(self.source_step_ids) != len(set(self.source_step_ids)):
            raise ValueError("source_step_ids must not contain duplicates")
        return self


class EvidenceSufficiencyArgs(BaseSchema):
    source_step_ids: list[NonNegativeInt] = Field(default_factory=list)
    required_modalities: list[Modality] = Field(default_factory=list)
    minimum_evidence: NonNegativeInt = 1
    minimum_evidence_groups: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _validate_modalities(self) -> EvidenceSufficiencyArgs:
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities must not contain duplicate values")
        if len(self.source_step_ids) != len(set(self.source_step_ids)):
            raise ValueError("source_step_ids must not contain duplicates")
        return self


class FallbackArgs(BaseSchema):
    source_step_ids: list[NonNegativeInt] = Field(default_factory=list)
    condition: NonEmptyStr = "always"
    reason_code: NonEmptyStr = "insufficient_compiled_evidence"

    @model_validator(mode="after")
    def _validate_fallback(self) -> FallbackArgs:
        if self.condition not in {"always", "if_insufficient"}:
            raise ValueError("fallback condition must be always or if_insufficient")
        if len(self.source_step_ids) != len(set(self.source_step_ids)):
            raise ValueError("source_step_ids must not contain duplicates")
        return self


class AbstainArgs(BaseSchema):
    source_step_ids: list[NonNegativeInt] = Field(default_factory=list)
    minimum_evidence: NonNegativeInt = 1
    minimum_evidence_groups: NonNegativeInt | None = None
    required_modalities: list[Modality] = Field(default_factory=list)
    reason_code: NonEmptyStr = "no_sufficient_evidence"

    @model_validator(mode="after")
    def _validate_abstain(self) -> AbstainArgs:
        if len(self.source_step_ids) != len(set(self.source_step_ids)):
            raise ValueError("source_step_ids must not contain duplicates")
        if len(self.required_modalities) != len(set(self.required_modalities)):
            raise ValueError("required_modalities must not contain duplicates")
        return self


class OperatorDataRepository(Protocol):
    """Prospective-safe data operations consumed by concrete operators."""

    async def validate_context(self, context: OperatorExecutionContext) -> None: ...

    async def validate_explicit_references(
        self,
        context: OperatorExecutionContext,
        *,
        memory_ids: list[str],
        image_ids: list[str],
        session_ids: list[str],
    ) -> None: ...

    async def search_keyword(
        self,
        context: OperatorExecutionContext,
        *,
        query: str,
        top_k: int,
        entity_mentions: list[str],
    ) -> list[OperatorEvidence]: ...

    async def search_embedding(
        self,
        context: OperatorExecutionContext,
        *,
        collection_id: str,
        vector: list[float],
        top_k: int,
        evidence_type: EvidenceType,
    ) -> list[OperatorEvidence]: ...

    async def resolve_image_text_pairs(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
    ) -> list[OperatorEvidence]: ...

    async def context_around_images(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
        radius: int,
    ) -> list[OperatorEvidence]: ...

    async def expand_temporal_neighbors(
        self,
        context: OperatorExecutionContext,
        memory_ids: list[str],
        radius: int,
    ) -> list[OperatorEvidence]: ...

    async def inspect_images(
        self,
        context: OperatorExecutionContext,
        image_ids: list[str],
    ) -> list[OperatorEvidence]: ...


class MemoryOperator(Protocol):
    spec: OperatorSpec
    args_model: type[BaseSchema]

    async def execute(
        self,
        context: OperatorExecutionContext,
        args: BaseSchema,
        previous_outputs: Sequence[OperatorOutput],
    ) -> OperatorOutput: ...


@runtime_checkable
class OperatorEvidenceMetadataRepository(Protocol):
    async def execution_evidence_metadata(
        self, context: OperatorExecutionContext, evidence: Sequence[ProgramExecutionEvidenceRef]
    ) -> dict[tuple[str, str], dict[str, object]]: ...


class OperatorRegistry:
    """Immutable registry whose hash is part of every compiler run identity."""

    def __init__(self, version: str, operators: Sequence[MemoryOperator]) -> None:
        if not version:
            raise ValueError("operator registry version must not be empty")
        names = [operator.spec.name for operator in operators]
        if len(names) != len(set(names)):
            raise ValueError("operator registry cannot contain duplicate names")
        if any(operator.spec.version != version for operator in operators):
            raise ValueError("every operator spec must use the registry version")
        self.version = version
        self._operators = {operator.spec.name: operator for operator in operators}
        self.resolved_hash = canonical_content_hash(
            {
                "version": version,
                "operators": [
                    operator.spec.model_dump(mode="json")
                    for operator in sorted(operators, key=lambda item: item.spec.name)
                ],
            }
        )

    def resolve(self, name: str) -> MemoryOperator:
        try:
            return self._operators[name]
        except KeyError as exc:
            raise KeyError(f"unknown PMMC operator: {name}") from exc

    def specs(self) -> list[OperatorSpec]:
        return [self._operators[name].spec for name in sorted(self._operators)]

    def names(self) -> tuple[str, ...]:
        return tuple(self._operators)


class ProgramValidator:
    """Perform deterministic, model-independent validation of a memory program."""

    _FORBIDDEN_ARGS = frozenset(
        {
            "snapshot_id",
            "conversation_id",
            "cutoff_session_id",
            "collection_id",
            "database_url",
            "dsn",
            "file_path",
        }
    )

    def __init__(
        self,
        registry: OperatorRegistry,
        config: CompilerFoundationConfig,
    ) -> None:
        self.registry = registry
        self.config = config

    def validate(self, program: MemoryProgram) -> ProgramValidationResult:
        issues: list[ProgramValidationIssue] = []
        if len(program.operator_sequence) > self.config.program_max_steps:
            issues.append(
                ProgramValidationIssue(
                    code="too_many_steps",
                    message="program exceeds the configured step limit",
                )
            )
        if (
            program.operator_set_version is not None
            and program.operator_set_version != self.registry.version
        ):
            issues.append(
                ProgramValidationIssue(
                    code="operator_set_mismatch",
                    message="program operator set does not match the active registry",
                )
            )
        for index, step in enumerate(program.operator_sequence):
            forbidden = sorted(self._FORBIDDEN_ARGS.intersection(step.args))
            if forbidden:
                issues.append(
                    ProgramValidationIssue(
                        code="scope_argument_forbidden",
                        message=(
                            f"operator args contain forbidden scope keys: {', '.join(forbidden)}"
                        ),
                        step_id=step.step_id,
                    )
                )
            try:
                operator = self.registry.resolve(step.op_name)
            except KeyError:
                issues.append(
                    ProgramValidationIssue(
                        code="unknown_operator",
                        message=f"operator is not registered: {step.op_name}",
                        step_id=step.step_id,
                    )
                )
                continue
            try:
                parsed = operator.args_model.model_validate(step.args)
            except ValueError as exc:
                issues.append(
                    ProgramValidationIssue(
                        code="invalid_operator_args",
                        message=f"operator args failed validation: {type(exc).__name__}",
                        step_id=step.step_id,
                    )
                )
            else:
                top_k = getattr(parsed, "top_k", None)
                if top_k is not None and top_k > self.config.operator_top_k_max:
                    issues.append(
                        ProgramValidationIssue(
                            code="operator_top_k_exceeded",
                            message="operator top_k exceeds the compiler limit",
                            step_id=step.step_id,
                        )
                    )
                source_step_ids = list(getattr(parsed, "source_step_ids", ()))
                if any(source_id >= step.step_id for source_id in source_step_ids):
                    issues.append(
                        ProgramValidationIssue(
                            code="invalid_source_step",
                            message="source_step_ids may only reference earlier steps",
                            step_id=step.step_id,
                        )
                    )
            if step.expected_output_type != operator.spec.output_type:
                issues.append(
                    ProgramValidationIssue(
                        code="output_type_mismatch",
                        message=(
                            "program step expected_output_type does not match the "
                            "registered operator contract"
                        ),
                        step_id=step.step_id,
                    )
                )
            if operator.spec.terminal and index != len(program.operator_sequence) - 1:
                issues.append(
                    ProgramValidationIssue(
                        code="terminal_operator_not_last",
                        message="terminal control operators must be the final program step",
                        step_id=step.step_id,
                    )
                )

        if (
            self.registry.version
            in {
                "pmmc-operators-v2",
                "pmmc-operators-v3",
                "pmmc-operators-v4",
                "pmmc-operators-v5",
                "pmmc-operators-v6",
                "pmmc-operators-v7",
            }
            and program.fallback_operator is not None
        ):
            issues.append(
                ProgramValidationIssue(
                    code="implicit_fallback_forbidden",
                    message=(
                        "v2/v3 programs require an explicit terminal fallback step"
                    ),
                )
            )
        elif program.fallback_operator is not None:
            try:
                fallback = self.registry.resolve(program.fallback_operator)
            except KeyError:
                issues.append(
                    ProgramValidationIssue(
                        code="unknown_fallback_operator",
                        message="fallback_operator is not registered",
                    )
                )
            else:
                if not fallback.spec.terminal:
                    issues.append(
                        ProgramValidationIssue(
                            code="fallback_operator_not_terminal",
                            message="fallback_operator must reference a terminal operator",
                        )
                    )

        actual_hash = memory_program_content_hash(program)
        if program.program_hash is not None and program.program_hash != actual_hash:
            issues.append(
                ProgramValidationIssue(
                    code="program_hash_mismatch",
                    message="program_hash does not match canonical program content",
                )
            )
        return ProgramValidationResult(
            valid=not any(issue.severity is ProgramValidationSeverity.ERROR for issue in issues),
            program_hash=actual_hash,
            operator_registry_hash=self.registry.resolved_hash,
            issues=issues,
        )


class ProgramLineageValidator:
    """Validate all explicit program scope and canonical IDs before execution."""

    def __init__(
        self,
        registry: OperatorRegistry,
        data: OperatorDataRepository,
    ) -> None:
        self.registry = registry
        self.data = data

    async def validate(
        self,
        program: MemoryProgram,
        context: OperatorExecutionContext,
    ) -> None:
        await self.data.validate_context(context)
        memory_ids: list[str] = []
        image_ids: list[str] = []
        session_ids: list[str] = []
        for step in program.operator_sequence:
            operator = self.registry.resolve(step.op_name)
            args = operator.args_model.model_validate(step.args)
            memory_ids.extend(getattr(args, "memory_ids", ()))
            image_ids.extend(getattr(args, "image_ids", ()))
            session_ids.extend(getattr(args, "session_ids", ()))
        await self.data.validate_explicit_references(
            context,
            memory_ids=list(dict.fromkeys(memory_ids)),
            image_ids=list(dict.fromkeys(image_ids)),
            session_ids=list(dict.fromkeys(session_ids)),
        )


@dataclass(frozen=True)
class ProgramExecutionResult:
    trace: ProgramExecutionTrace
    steps: tuple[ProgramExecutionStepTrace, ...]
    evidence: tuple[ProgramExecutionEvidenceRef, ...]


class LinearProgramExecutor:
    """Execute validated programs without holding a database write transaction."""

    def __init__(
        self,
        registry: OperatorRegistry,
        validator: ProgramValidator,
        data: OperatorDataRepository,
        config: CompilerFoundationConfig,
    ) -> None:
        self.registry = registry
        self.validator = validator
        self.data = data
        self.config = config
        self.lineage_validator = ProgramLineageValidator(registry, data)

    async def execute(
        self,
        *,
        execution_id: str,
        program: MemoryProgram,
        context: OperatorExecutionContext,
        purpose: OperatorExecutionPurpose = OperatorExecutionPurpose.COMPILER_DRY_RUN,
        attempt_number: int = 1,
    ) -> ProgramExecutionResult:
        validation = self.validator.validate(program)
        if not validation.valid:
            raise ValueError("memory program failed static validation")
        self._validate_execution_scope(program, context, purpose)
        await self.lineage_validator.validate(program, context)
        started_at = utc_now()
        started_clock = time.perf_counter()
        outputs: list[OperatorOutput] = []
        steps: list[ProgramExecutionStepTrace] = []
        evidence: list[ProgramExecutionEvidenceRef] = []
        seen_evidence: set[tuple[str, EvidenceType, str]] = set()
        status = TraceStatus.SUCCESS
        control_action = OperatorControlAction.CONTINUE
        execution_error: str | None = None
        v7 = self.registry.version == "pmmc-operators-v7"
        quotas = program_evidence_quotas(program.operator_sequence)[0] if v7 else {}
        if v7:
            for step in program.operator_sequence:
                quota = step.metadata.get("evidence_quota", quotas[step.step_id])
                if type(quota) is not int or quota != quotas[step.step_id]:
                    raise ValueError("v7 step evidence quota does not match frozen program args")
        evidence_positions: dict[tuple[str, EvidenceType, str], int] = {}

        for program_step in program.operator_sequence:
            operator = self.registry.resolve(program_step.op_name)
            args = operator.args_model.model_validate(program_step.args)
            step_started = time.perf_counter()
            input_hash = canonical_content_hash(
                {
                    "context": context.model_dump(mode="json"),
                    "args": args.model_dump(mode="json"),
                    "previous": [value.model_dump(mode="json") for value in outputs],
                }
            )
            try:
                async with asyncio.timeout(self.config.operator_timeout_seconds):
                    output = await operator.execute(context, args, tuple(outputs))
            except TimeoutError:
                output = OperatorOutput(
                    success=False,
                    error_type="OperatorTimeout",
                    error_summary="operator exceeded the configured timeout",
                    evidence_type=operator.spec.evidence_type,
                )
            except Exception as exc:
                output = OperatorOutput(
                    success=False,
                    error_type=type(exc).__name__,
                    error_summary="operator execution failed",
                    evidence_type=operator.spec.evidence_type,
                )
            elapsed_ms = (time.perf_counter() - step_started) * 1000
            output = output.model_copy(update={"latency_ms": elapsed_ms})
            new_evidence: list[tuple[str, EvidenceType, str]] = []
            step_evidence: dict[tuple[str, EvidenceType, str], list[str]] = {}
            for reference_kind, default_type, identifiers in (
                ("memory", EvidenceType.MEMORY, output.retrieved_memory_ids),
                ("image", EvidenceType.IMAGE, output.retrieved_image_ids),
                ("caption", EvidenceType.CAPTION, output.retrieved_caption_ids),
                ("session", EvidenceType.SESSION, output.retrieved_session_ids),
            ):
                for identifier in identifiers:
                    evidence_type = (
                        output.evidence_type
                        if output.evidence_type is not EvidenceType.UNKNOWN
                        else default_type
                    )
                    key = (reference_kind, evidence_type, identifier)
                    roles = produced_evidence_roles(
                        step=program_step, output=output, previous=outputs,
                        reference_kind=reference_kind, identifier=identifier,
                    ) if v7 else []
                    if key not in seen_evidence and key not in step_evidence:
                        new_evidence.append((reference_kind, evidence_type, identifier))
                    step_evidence[key] = roles
            quota = program_step.metadata.get("evidence_quota", quotas.get(program_step.step_id))
            quota_exceeded = v7 and len(step_evidence) > quotas[program_step.step_id]
            if quota_exceeded or len(evidence) + len(new_evidence) > self.config.total_evidence_max:
                output = output.model_copy(
                    update={
                        "success": False,
                        "error_type": "StepEvidenceQuotaExceeded" if quota_exceeded
                        else "EvidenceLimitExceeded",
                        "error_summary": "operator exceeded its typed evidence quota"
                        if quota_exceeded else "program exceeded the total evidence limit",
                        "control_action": OperatorControlAction.CONTINUE,
                    }
                )
                new_evidence = []
            if v7:
                output = output.model_copy(update={"metadata": {
                    **output.metadata,
                    "evidence_budget_policy": "planner-evidence-budget-v1",
                    "evidence_quota": quota,
                    "typed_evidence_count": len(step_evidence),
                    "new_typed_evidence_count": len(new_evidence),
                    "evidence_quota_exceeded": quota_exceeded,
                    "semantic_role_policy": EVIDENCE_ROLE_POLICY,
                }})
                if output.success:
                    for key, roles in step_evidence.items():
                        if key in evidence_positions and roles:
                            position = evidence_positions[key]
                            value = evidence[position]
                            evidence[position] = value.model_copy(update={"metadata": {
                                **value.metadata,
                                "semantic_roles": sorted(set(value.metadata.get(
                                    "semantic_roles", [])) | set(roles)),
                            }})
            outputs.append(output)
            output_hash = canonical_content_hash(output.model_dump(mode="json"))
            steps.append(
                ProgramExecutionStepTrace(
                    execution_id=execution_id,
                    run_id=context.run_id,
                    snapshot_id=context.snapshot_id,
                    query_id=context.query_id,
                    compiler_unit_id=context.compiler_unit_id,
                    plan_id=program.plan_id,
                    step_id=program_step.step_id,
                    op_name=program_step.op_name,
                    retrieved_memory_ids=output.retrieved_memory_ids,
                    retrieved_image_ids=output.retrieved_image_ids,
                    retrieved_caption_ids=output.retrieved_caption_ids,
                    retrieved_session_ids=output.retrieved_session_ids,
                    scores=output.scores,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    output=output.model_dump(mode="json"),
                    success=output.success,
                    error=output.error_summary or output.error,
                    latency_ms=elapsed_ms,
                    status=TraceStatus.SUCCESS if output.success else TraceStatus.FAILED,
                )
            )
            for reference_kind, evidence_type, identifier in new_evidence:
                key = (reference_kind, evidence_type, identifier)
                seen_evidence.add(key)
                evidence_positions[key] = len(evidence)
                evidence.append(
                    ProgramExecutionEvidenceRef(
                        evidence_ref_id=stable_compiler_id(
                            "execution-evidence",
                            [
                                execution_id,
                                program_step.step_id,
                                evidence_type.value,
                                identifier,
                            ],
                        ),
                        execution_id=execution_id,
                        step_id=program_step.step_id,
                        snapshot_id=context.snapshot_id,
                        evidence_id=identifier,
                        evidence_type=evidence_type,
                        memory_id=(identifier if reference_kind == "memory" else None),
                        image_id=(identifier if reference_kind == "image" else None),
                        caption_id=(identifier if reference_kind == "caption" else None),
                        session_id=(identifier if reference_kind == "session" else None),
                        rank=len(evidence) + 1,
                        score=output.scores.get(identifier),
                        metadata={
                            "semantic_role_policy": EVIDENCE_ROLE_POLICY,
                            "semantic_roles": step_evidence[key] if output.success else [],
                        } if v7 else {},
                    )
                )
            if not output.success:
                status = TraceStatus.FAILED
                execution_error = output.error_type or "OperatorFailed"
                break
            if output.control_action is not OperatorControlAction.CONTINUE:
                control_action = output.control_action
                break

        if v7 and evidence and isinstance(self.data, OperatorEvidenceMetadataRepository):
            metadata = await self.data.execution_evidence_metadata(context, evidence)
            enriched = []
            for value in evidence:
                for kind in ("memory", "image", "caption", "session"):
                    identifier = getattr(value, f"{kind}_id")
                    if identifier is not None:
                        value = value.model_copy(update={"metadata": {
                            **value.metadata, **metadata.get((kind, identifier), {}),
                        }})
                        break
                enriched.append(value)
            evidence = enriched
        completed_at = utc_now()
        trace = ProgramExecutionTrace(
            execution_id=execution_id,
            run_id=context.run_id,
            source_compiler_run_id=context.source_compiler_run_id,
            snapshot_id=context.snapshot_id,
            plan_id=program.plan_id,
            compiler_unit_id=context.compiler_unit_id,
            query_id=context.query_id,
            purpose=purpose,
            attempt_number=attempt_number,
            status=status,
            bindings_hash=canonical_content_hash(dict(sorted(context.collection_bindings.items()))),
            control_action=control_action,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_ms=(time.perf_counter() - started_clock) * 1000,
            error_type=execution_error,
            error_summary="program execution stopped after a failed operator"
            if execution_error
            else None,
        )
        return ProgramExecutionResult(trace, tuple(steps), tuple(evidence))

    @staticmethod
    def _validate_execution_scope(
        program: MemoryProgram,
        context: OperatorExecutionContext,
        purpose: OperatorExecutionPurpose,
    ) -> None:
        if program.run_id != context.source_compiler_run_id:
            raise ValueError("program and source compiler run do not match")
        if program.snapshot_id != context.snapshot_id:
            raise ValueError("program and execution snapshot do not match")
        if program.compiler_unit_id != context.compiler_unit_id:
            raise ValueError("program and execution compiler unit do not match")
        if purpose is OperatorExecutionPurpose.COMPILER_DRY_RUN:
            if context.query_id is not None or context.run_id != context.source_compiler_run_id:
                raise ValueError("compiler dry-run cannot bind a benchmark query or another run")
        elif context.query_id is None:
            raise ValueError("query-time execution requires a benchmark query")
