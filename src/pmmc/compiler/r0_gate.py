"""Immutable round-zero gate before any Doubter model call is permitted."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pmmc.compiler.ids import (
    canonical_content_hash,
    memory_program_content_hash,
    stable_compiler_id,
)
from pmmc.compiler.planner import (
    PlannerProgramPolicy,
    PlannerStrategyPolicyError,
    build_planner_requirement_contract,
)
from pmmc.compiler.planner_config import PlannerConfig, planner_config_hash
from pmmc.infrastructure.database.compiler_models import (
    MemoryProgramModel,
    PlannerCandidateDraftModel,
    PlannerCandidateOutcomeModel,
    PlannerGenerationBatchModel,
    ProgramDryRunAssessmentModel,
    ProgramQueryEmbeddingModel,
    ProgramValidationResultModel,
    QBankQuestionModel,
)
from pmmc.infrastructure.database.compiler_repositories import (
    PostgresMemoryProgramRepository,
    PostgresQBankCompilerRepository,
)
from pmmc.infrastructure.database.experiment_models import RuntimeModelCallModel
from pmmc.infrastructure.database.experiment_repositories import ExperimentUnitOfWork
from pmmc.infrastructure.database.planner_repositories import PostgresPlannerRepository
from pmmc.infrastructure.database.questioner_models import (
    QBankIndexManifestModel,
    QuestionerAnswerClaimModel,
    QuestionerCandidateGateResultModel,
    QuestionerCandidateVerificationModel,
    QuestionerGenerationModel,
    QuestionerVerificationBatchModel,
    RuntimeStructuredOutputItemDiagnosticModel,
)
from pmmc.schemas import (
    DataScope,
    MemoryProgram,
    PlannerCandidateOutcome,
    PlannerRequirementContract,
    QuestionCandidate,
    R0InputManifest,
)
from pmmc.schemas.base import utc_now

R0_OPERATOR_SETS = {
    "0.6.0": "pmmc-operators-v4",
    "0.7.0": "pmmc-operators-v5",
    "0.8.0": "pmmc-operators-v6",
    "0.9.0": "pmmc-operators-v7",
}


def _policy_rejection_matches(
    candidate: QuestionCandidate,
    outcome: PlannerCandidateOutcome | None,
    error: PlannerStrategyPolicyError,
    *,
    before_provider: bool,
) -> bool:
    return bool(
        outcome is not None
        and outcome.run_id == candidate.run_id
        and outcome.compiler_unit_id == candidate.compiler_unit_id
        and outcome.qbank_id == candidate.qbank_id
        and outcome.status == "policy_rejected"
        and outcome.plan_id is None
        and outcome.planner_batch_id is None
        and outcome.reason_code == (
            error.code if before_provider else "planner_strategy_policy_failed"
        )
        and outcome.metadata.get("issue_codes") == [error.code]
        and outcome.metadata.get("strategy_policy_diagnostics") == error.safe_diagnostics()
        and (not before_provider or outcome.metadata.get("provider_call_required") is False)
    )


def _fixed_candidate_matches(
    candidate: QuestionCandidate,
    requirement: PlannerRequirementContract | None,
    program: MemoryProgram | None,
    outcome: PlannerCandidateOutcome,
    config: PlannerConfig,
) -> bool:
    """Check fixed P0 facts without requiring provider drafts or native IDs."""
    if (
        outcome.run_id != candidate.run_id
        or outcome.compiler_unit_id != candidate.compiler_unit_id
        or outcome.planner_batch_id is not None
    ):
        return False
    try:
        expected_contract = build_planner_requirement_contract(
            run_id=candidate.run_id, compiler_unit_id=candidate.compiler_unit_id,
            candidate=candidate, config=config,
        )
    except PlannerStrategyPolicyError as exc:
        return requirement is None and program is None and _policy_rejection_matches(
            candidate, outcome, exc, before_provider=True
        )
    if requirement is None:
        return False
    excluded = {"requirement_contract_id", "content_hash", "created_at", "metadata"}
    if (
        requirement.model_dump(exclude=excluded) != expected_contract.model_dump(exclude=excluded)
        or any(requirement.metadata.get(k) != v for k, v in expected_contract.metadata.items())
    ):
        return False
    if program is None:
        reasons = {
            "static_rejected": {"static_validation_failed"},
            "policy_rejected": {"planner_strategy_policy_failed", "planner_policy_failed"},
        }
        codes = outcome.metadata.get("issue_codes")
        return bool(
            outcome.plan_id is None
            and outcome.reason_code in reasons.get(outcome.status, set())
            and isinstance(codes, list) and codes
            and all(isinstance(code, str) and code for code in codes)
        )
    assert config.fixed_strategy is not None
    strategy = None if requirement.refusal_only else config.fixed_strategy.model_dump(mode="json")
    return bool(
        outcome.plan_id == program.plan_id
        and outcome.status in {
            "program_ready", "program_ready_with_fallback", "program_ready_with_abstain",
            "dry_run_insufficient", "execution_failed",
        }
        and program.run_id == candidate.run_id
        and program.snapshot_id == candidate.snapshot_id
        and program.compiler_unit_id == candidate.compiler_unit_id
        and program.qbank_id == candidate.qbank_id
        and program.round_id == 0
        and program.producer_model_call_id is None
        and program.program_hash == memory_program_content_hash(program)
        and program.metadata.get("server_materialized") is True
        and program.metadata.get("planning_mode") == "fixed"
        and program.metadata.get("strategy") == strategy
        and program.metadata.get("requirement_contract_hash") == requirement.content_hash
        and PlannerProgramPolicy(config).validate(program, candidate, requirement).valid
    )


async def build_r0_input_manifest(
    sessions: async_sessionmaker[AsyncSession],
    *,
    run_id: str,
    planner_config: PlannerConfig | None = None,
) -> R0InputManifest:
    """Validate and freeze the exact r0 candidate/program/execution denominator."""

    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.compiler_runs is not None
        assert uow.memory_operator_indexes is not None
        assert uow.doubter is not None
        compiler = await uow.compiler_runs.get(run_id)
        if compiler is None:
            raise KeyError(f"compiler run does not exist: {run_id}")
        existing = await uow.doubter.get_r0_manifest(run_id)
        memory_index = await uow.memory_operator_indexes.get_for_run(run_id)
        saved_config = existing.metadata.get("planner_accounting_config") if existing else None
        if saved_config is not None:
            saved_planner = PlannerConfig.model_validate(saved_config)
            if planner_config is not None and planner_config_hash(planner_config) != (
                planner_config_hash(saved_planner)
            ):
                raise ValueError("R0 planner accounting config conflicts with frozen evidence")
            planner_config = saved_planner
        if planner_config is not None:
            planner_config = PlannerConfig.model_validate(planner_config.model_dump(mode="json"))
            assert uow.runs is not None
            run_manifest = await uow.runs.get_manifest(run_id)
            if (
                compiler.planner_config_hash != planner_config_hash(planner_config)
                or run_manifest.metadata.get("planner_config_hash") != compiler.planner_config_hash
            ):
                raise ValueError("R0 planner accounting config is not bound to the frozen run")
            if planner_config.planning_mode == "fixed":
                if (
                    compiler.compiler_schema_version != "0.9.0"
                    or compiler.operator_set_version != planner_config.operator_set_version
                ):
                    raise ValueError("R0 fixed planner requires its native compiler contract")

    fixed = planner_config is not None and planner_config.planning_mode == "fixed"
    accounting_issues: list[str] = []
    accounting_evidence = None

    async with sessions() as session:
        eligible_ids = list(
            await session.scalars(
                select(QBankQuestionModel.qbank_id)
                .join(
                    QuestionerCandidateGateResultModel,
                    QuestionerCandidateGateResultModel.qbank_id
                    == QBankQuestionModel.qbank_id,
                )
                .join(
                    QuestionerCandidateVerificationModel,
                    (
                        QuestionerCandidateVerificationModel.qbank_id
                        == QBankQuestionModel.qbank_id
                    )
                    & (
                        QuestionerCandidateVerificationModel.verification_round
                        == QBankQuestionModel.candidate_version
                    ),
                )
                .where(
                    QBankQuestionModel.run_id == run_id,
                    QuestionerCandidateGateResultModel.status
                    == "eligible_for_planning",
                    QuestionerCandidateVerificationModel.verdict == "pass",
                )
                .order_by(QBankQuestionModel.qbank_id)
            )
        )
        route_rows = list(
            (
                await session.execute(
                select(
                    QBankQuestionModel.qbank_id,
                    QBankQuestionModel.route_class,
                    QBankQuestionModel.generation_id,
                    QBankQuestionModel.compiler_unit_id,
                    )
                    .where(QBankQuestionModel.qbank_id.in_(eligible_ids))
                    .order_by(QBankQuestionModel.qbank_id)
                )
            ).all()
        )
        programs = list(
            (
                await session.execute(
                    select(
                        MemoryProgramModel.plan_id,
                        MemoryProgramModel.qbank_id,
                        MemoryProgramModel.program_hash,
                        MemoryProgramModel.operator_set_version,
                    )
                    .where(
                        MemoryProgramModel.run_id == run_id,
                        MemoryProgramModel.round_id == 0,
                    )
                    .order_by(MemoryProgramModel.qbank_id)
                )
            ).all()
        )
        plan_ids = [item.plan_id for item in programs]
        assessments = list(
            (
                await session.execute(
                    select(
                        ProgramDryRunAssessmentModel.plan_id,
                        ProgramDryRunAssessmentModel.execution_id,
                        ProgramDryRunAssessmentModel.content_hash,
                    )
                    .where(ProgramDryRunAssessmentModel.plan_id.in_(plan_ids))
                    .order_by(ProgramDryRunAssessmentModel.plan_id)
                )
            ).all()
        )
        outcomes = list(
            (
                await session.execute(
                    select(
                        PlannerCandidateOutcomeModel.qbank_id,
                        PlannerCandidateOutcomeModel.status,
                        PlannerCandidateOutcomeModel.content_hash,
                    )
                    .where(PlannerCandidateOutcomeModel.run_id == run_id)
                    .order_by(PlannerCandidateOutcomeModel.qbank_id)
                )
            ).all()
        )
        native_questioner_bad = int(
            await session.scalar(
                select(func.count())
                .select_from(QBankQuestionModel)
                .join(
                    QuestionerGenerationModel,
                    QuestionerGenerationModel.generation_id
                    == QBankQuestionModel.generation_id,
                )
                .where(
                    QBankQuestionModel.qbank_id.in_(eligible_ids),
                    QuestionerGenerationModel.repair_attempt_count != 0,
                )
            )
            or 0
        )
        terminal_generation_rows = list(
            (
                await session.execute(
                    select(
                        QuestionerGenerationModel.generation_id,
                        QuestionerGenerationModel.compiler_unit_id,
                        QuestionerGenerationModel.output_hash,
                    )
                    .where(
                        QuestionerGenerationModel.run_id == run_id,
                        QuestionerGenerationModel.parse_status == "failed",
                    )
                    .order_by(QuestionerGenerationModel.compiler_unit_id)
                )
            ).all()
        )
        verifier_rows = list(
            (
                await session.execute(
                    select(
                        QuestionerCandidateVerificationModel.qbank_id,
                        QuestionerCandidateVerificationModel.verification_round,
                        QuestionerCandidateVerificationModel.verdict,
                        QuestionerVerificationBatchModel.input_count,
                        QuestionerVerificationBatchModel.scope_qbank_id,
                        QuestionerVerificationBatchModel.metadata_json,
                    )
                    .join(
                        QuestionerVerificationBatchModel,
                        QuestionerVerificationBatchModel.verification_batch_id
                        == QuestionerCandidateVerificationModel.verification_batch_id,
                    )
                    .where(
                        QuestionerCandidateVerificationModel.qbank_id.in_(eligible_ids)
                    )
                    .order_by(QuestionerCandidateVerificationModel.qbank_id)
                )
            ).all()
        )
        candidate_versions = {
            str(qbank_id): int(candidate_version)
            for qbank_id, candidate_version in (
                await session.execute(
                    select(
                        QBankQuestionModel.qbank_id,
                        QBankQuestionModel.candidate_version,
                    ).where(QBankQuestionModel.qbank_id.in_(eligible_ids))
                )
            ).all()
        }
        verifier_by_candidate = {
            str(item.qbank_id): item
            for item in verifier_rows
            if int(item.verification_round)
            == candidate_versions.get(str(item.qbank_id), -1)
        }
        verifier_bad = sum(
            1
            for qbank_id in eligible_ids
            if (
                (item := verifier_by_candidate.get(str(qbank_id))) is None
                or item.verdict != "pass"
                or int(item.input_count) != 1
                or item.scope_qbank_id != qbank_id
                or not isinstance(item.metadata_json, dict)
                or int(item.metadata_json.get("repair_attempt_count", 0)) != 0
                or bool(item.metadata_json.get("provider_output_fallback", False))
            )
        )
        answerable_ids = [
            item.qbank_id for item in route_rows if item.route_class == "answerable"
        ]
        claimless = int(
            await session.scalar(
                select(func.count())
                .select_from(QBankQuestionModel)
                .where(
                    QBankQuestionModel.qbank_id.in_(answerable_ids),
                    ~QBankQuestionModel.qbank_id.in_(
                        select(QuestionerAnswerClaimModel.qbank_id)
                    ),
                )
            )
            or 0
        )
        draft_rows = list(
            (await session.execute(
                select(PlannerCandidateDraftModel.qbank_id, PlannerCandidateDraftModel.parsed)
                .where(PlannerCandidateDraftModel.run_id == run_id)
            )).all()
        )
        draft_ids = [item.qbank_id for item in draft_rows if item.qbank_id in answerable_ids]
        planner_native_bad = sum(
            not item.parsed for item in draft_rows if item.qbank_id in answerable_ids
        )
        accounted_ids = set(draft_ids)
        if fixed or (
            planner_config is not None and planner_config.is_v15
            and accounted_ids != set(answerable_ids)
        ):
            questions = PostgresQBankCompilerRepository(session)
            planner = PostgresPlannerRepository(session)
            program_repo = PostgresMemoryProgramRepository(session)
            contracts = await planner.list_requirement_contracts(run_id)
            terminal = await planner.list_outcomes(run_id)
            contract_by_qid = {item.qbank_id: item for item in contracts}
            outcome_by_qid = {item.qbank_id: item for item in terminal}
            candidates = [await questions.get_question(qid) for qid in eligible_ids]
            if fixed:
                assert planner_config is not None
                program_by_qid = {
                    item.qbank_id: await program_repo.get(item.plan_id) for item in programs
                }
                if (
                    len(contract_by_qid) != len(contracts)
                    or not set(contract_by_qid) <= set(eligible_ids)
                    or len(outcome_by_qid) != len(terminal)
                    or set(outcome_by_qid) != set(eligible_ids)
                    or len(program_by_qid) != len(programs)
                    or not set(program_by_qid) <= set(eligible_ids)
                ):
                    accounting_issues.append("planner_fixed_candidate_scope_mismatch")
                for candidate in candidates:
                    if candidate is None or candidate.qbank_id not in outcome_by_qid or not (
                        _fixed_candidate_matches(
                            candidate, contract_by_qid.get(candidate.qbank_id),
                            program_by_qid.get(candidate.qbank_id),
                            outcome_by_qid[candidate.qbank_id], planner_config,
                        )
                    ):
                        accounting_issues.append("planner_fixed_candidate_accounting_invalid")
                provider_facts = len(draft_rows)
                for model in (PlannerGenerationBatchModel, RuntimeModelCallModel):
                    statement = (
                        select(func.count()).select_from(model).where(model.run_id == run_id)
                    )
                    if model is RuntimeModelCallModel:
                        statement = statement.where(model.stage == "planner")
                    provider_facts += int(await session.scalar(statement) or 0)
                if provider_facts:
                    accounting_issues.append("planner_fixed_provider_facts_forbidden")
                validations = list((await session.execute(
                    select(
                        ProgramValidationResultModel.plan_id,
                        ProgramValidationResultModel.validation_kind,
                        ProgramValidationResultModel.validator_version,
                        ProgramValidationResultModel.valid,
                        ProgramValidationResultModel.content_hash,
                    ).where(
                        ProgramValidationResultModel.run_id == run_id,
                        ProgramValidationResultModel.plan_id.in_(plan_ids),
                        ProgramValidationResultModel.validation_kind.in_(("static", "policy")),
                    )
                )).all())
                if (
                    len(validations) != 2 * len(plan_ids)
                    or {(v.plan_id, v.validation_kind) for v in validations} != {
                        (pid, kind) for pid in plan_ids for kind in ("static", "policy")
                    }
                    or any(not v.valid or v.validator_version != (
                        "memory-program-static-v7" if v.validation_kind == "static"
                        else planner_config.policy_version
                    ) for v in validations)
                ):
                    accounting_issues.append("planner_fixed_validation_incomplete")
            else:
                assert planner_config is not None
                # Only a reproducible contract rejection before provider dispatch
                # accounts for a model-mode candidate with no draft.
                for candidate in candidates:
                    if (
                        candidate is None
                        or candidate.compiler_unit_id is None
                        or candidate.qbank_id not in (set(answerable_ids) - accounted_ids)
                    ):
                        continue
                    try:
                        build_planner_requirement_contract(
                            run_id=run_id, compiler_unit_id=candidate.compiler_unit_id,
                            candidate=candidate, config=planner_config,
                        )
                    except PlannerStrategyPolicyError as exc:
                        if (
                            candidate.qbank_id not in contract_by_qid
                            and candidate.qbank_id not in {p.qbank_id for p in programs}
                            and _policy_rejection_matches(
                                candidate, outcome_by_qid.get(candidate.qbank_id), exc,
                                before_provider=True,
                            )
                        ):
                            accounted_ids.add(candidate.qbank_id)
            if fixed or accounted_ids != set(draft_ids):
                accounting_evidence = {
                    "mode": planner_config.planning_mode,
                    "config_hash": planner_config_hash(planner_config),
                    "contracts": sorted((item.qbank_id, item.content_hash) for item in contracts),
                    "without_provider_draft": sorted(set(answerable_ids) - set(draft_ids)),
                    **({"validations": sorted(
                        (v.plan_id, v.validation_kind, v.content_hash) for v in validations
                    )} if fixed else {}),
                }
        planner_oracle_calls = int(
            await session.scalar(
                select(func.count())
                .select_from(RuntimeModelCallModel)
                .where(
                    RuntimeModelCallModel.run_id == run_id,
                    RuntimeModelCallModel.stage == "planner",
                    func.coalesce(
                        RuntimeModelCallModel.request_config["request_metadata"][
                            "non_oracle_input"
                        ].astext,
                        "false",
                    )
                    != "true",
                )
            )
            or 0
        )
        questioner_structured_output_bad = int(
            await session.scalar(
                select(func.count())
                .select_from(RuntimeStructuredOutputItemDiagnosticModel)
                .where(
                    RuntimeStructuredOutputItemDiagnosticModel.run_id == run_id,
                    RuntimeStructuredOutputItemDiagnosticModel.stage.in_(
                        (
                            "questioner",
                            "questioner_verifier",
                            "questioner_revision",
                        )
                    ),
                    RuntimeStructuredOutputItemDiagnosticModel.status.in_(
                        ("invalid", "repaired", "conservative_reject")
                    ),
                )
            )
            or 0
        )
        planner_structured_output_bad = int(
            await session.scalar(
                select(func.count())
                .select_from(RuntimeStructuredOutputItemDiagnosticModel)
                .where(
                    RuntimeStructuredOutputItemDiagnosticModel.run_id == run_id,
                    RuntimeStructuredOutputItemDiagnosticModel.stage == "planner",
                    RuntimeStructuredOutputItemDiagnosticModel.status.in_(
                        ("invalid", "repaired", "conservative_reject")
                    ),
                )
            )
            or 0
        )
        query_vector_count = int(
            await session.scalar(
                select(func.count())
                .select_from(ProgramQueryEmbeddingModel)
                .where(ProgramQueryEmbeddingModel.run_id == run_id)
            )
            or 0
        )
        qbank_index = await session.scalar(
            select(QBankIndexManifestModel).where(
                QBankIndexManifestModel.compiler_run_id == run_id
            )
        )

    protocol_issues: list[str] = list(accounting_issues)
    candidate_count = len(route_rows)
    answerable_count = len(answerable_ids)
    control_count = candidate_count - answerable_count
    if candidate_count != len(eligible_ids):
        protocol_issues.append("eligible_candidate_scope_duplicate")
    # Invalid or conservatively rejected candidates are scientific outcomes.
    # Only a repaired/fallback candidate leaking into the eligible denominator
    # violates the protocol-level freeze contract.
    if native_questioner_bad:
        protocol_issues.append("eligible_questioner_non_native_output")
    if verifier_bad:
        protocol_issues.append("questioner_final_verifier_incomplete")
    if claimless:
        protocol_issues.append("answerable_candidate_claim_missing")
    if not fixed and (
        len(set(draft_ids)) != len(draft_ids) or accounted_ids != set(answerable_ids)
    ):
        protocol_issues.append("planner_candidate_accounting_incomplete")
    if planner_oracle_calls:
        protocol_issues.append("planner_non_oracle_contract_failed")
    if len(outcomes) != candidate_count:
        protocol_issues.append("planner_terminal_outcome_incomplete")
    program_qbank_ids = {item.qbank_id for item in programs}
    expected_program_count = sum(
        item.status
        in {
            "program_ready",
            "program_ready_with_fallback",
            "program_ready_with_abstain",
            "dry_run_insufficient",
            "execution_failed",
        }
        for item in outcomes
    )
    if len(programs) != expected_program_count:
        protocol_issues.append("r0_program_denominator_incomplete")
    required_operator_set = R0_OPERATOR_SETS[compiler.compiler_schema_version]
    if any(item.operator_set_version != required_operator_set for item in programs):
        protocol_issues.append("frozen_operator_set_required")
    if len(assessments) != len(programs):
        protocol_issues.append("r0_dry_run_incomplete")
    if {item.plan_id for item in assessments} != {item.plan_id for item in programs}:
        protocol_issues.append("r0_execution_program_scope_mismatch")
    if not program_qbank_ids.issubset(set(eligible_ids)):
        protocol_issues.append("r0_program_candidate_scope_mismatch")
    if memory_index is None or memory_index.status != "ready":
        protocol_issues.append("memory_index_not_ready")
    if query_vector_count != candidate_count * 2:
        protocol_issues.append("program_query_vector_denominator_incomplete")
    if qbank_index is None or qbank_index.status != "ready":
        protocol_issues.append("qbank_index_not_ready")

    eligible_unit_ids = {str(item.compiler_unit_id) for item in route_rows}
    coverage_zero_units = max(0, int(compiler.unit_count) - len(eligible_unit_ids))
    assessed_plan_ids = {str(item.plan_id) for item in assessments}
    downstream_viability = any(
        str(item.qbank_id) in set(answerable_ids)
        and str(item.plan_id) in assessed_plan_ids
        for item in programs
    )
    scientific_rejections = {
        "questioner_terminal_generation": len(terminal_generation_rows),
        "questioner_structured_output": questioner_structured_output_bad,
        "planner_parse": planner_native_bad,
        "planner_structured_output": planner_structured_output_bad,
        "planner_non_program_terminal": max(0, candidate_count - len(program_qbank_ids)),
    }

    questioner_identity = {
        "config": compiler.questioner_config_hash,
        "prompt": compiler.questioner_prompt_hash,
        "renderer": compiler.questioner_renderer_version,
        "parser": compiler.questioner_parser_version,
        "gate": compiler.candidate_gate_version,
        "dedupe": compiler.dedupe_version,
    }
    planner_identity = {
        "config": compiler.planner_config_hash,
        "prompt": compiler.planner_prompt_hash,
        "renderer": compiler.planner_renderer_version,
        "parser": compiler.planner_parser_version,
        "policy": compiler.program_policy_version,
    }
    fact_projection = {
        **({"planner_accounting": accounting_evidence} if accounting_evidence is not None else {}),
        "candidates": [
            [
                item.qbank_id,
                item.route_class,
                item.generation_id,
                item.compiler_unit_id,
            ]
            for item in route_rows
        ],
        "programs": [
            [item.plan_id, item.qbank_id, item.program_hash] for item in programs
        ],
        "assessments": [
            [item.plan_id, item.execution_id, item.content_hash] for item in assessments
        ],
        "outcomes": [
            [item.qbank_id, item.status, item.content_hash] for item in outcomes
        ],
        "terminal_generations": [
            [item.generation_id, item.compiler_unit_id, item.output_hash]
            for item in terminal_generation_rows
        ],
    }
    payload = {
        "run_id": run_id,
        "snapshot_id": compiler.snapshot_id,
        "candidate_count": candidate_count,
        "answerable_candidate_count": answerable_count,
        "control_candidate_count": control_count,
        "program_count": len(programs),
        "execution_count": len(assessments),
        "terminal_outcome_count": len(outcomes),
        "questioner_identity_hash": canonical_content_hash(questioner_identity),
        "planner_identity_hash": canonical_content_hash(planner_identity),
        "operator_registry_hash": compiler.operator_registry_hash,
        "memory_index_hash": (
            memory_index.content_hash if memory_index is not None else "missing"
        ),
        "query_vector_hash": canonical_content_hash(
            [run_id, query_vector_count, sorted(program_qbank_ids)]
        ),
        "facts_hash": canonical_content_hash(fact_projection),
        "artifact_hash": canonical_content_hash(
            [
                qbank_index.content_hash if qbank_index is not None else "missing",
                memory_index.content_hash if memory_index is not None else "missing",
            ]
        ),
        "hard_gate_passed": not protocol_issues,
        "issue_codes": sorted(set(protocol_issues)),
        "protocol_integrity_issues": sorted(set(protocol_issues)),
        "candidate_scientific_rejections": scientific_rejections,
        "coverage_zero_units": coverage_zero_units,
        "downstream_viability": downstream_viability,
    }
    manifest = R0InputManifest(
        r0_manifest_id=stable_compiler_id("r0-input-manifest", payload),
        **payload,
        **({"schema_version": "0.4.0"} if compiler.compiler_schema_version == "0.9.0" else {}),
        content_hash=canonical_content_hash(payload),
        created_at=(existing.created_at if existing is not None else utc_now()),
        metadata={
            **({"planner_accounting_config": planner_config.model_dump(mode="json")}
               if accounting_evidence is not None and planner_config is not None else {}),
            **({"r0_input_policy_version": "r0-input-gate-v4"}
               if compiler.compiler_schema_version == "0.9.0" else {}),
            "query_vector_count": query_vector_count,
            "terminal_generation_rejection_count": len(
                terminal_generation_rows
            ),
            "scientific_rejection_count": sum(scientific_rejections.values()),
            "downstream_viability": downstream_viability,
            "qbank_index_id": (
                qbank_index.qbank_index_id if qbank_index is not None else None
            ),
        },
    )
    if existing is not None:
        if existing != manifest:
            raise ValueError(
                "persisted R0 input manifest no longer matches its source facts"
            )
        return existing
    async with ExperimentUnitOfWork(sessions, DataScope.RUNTIME) as uow:
        assert uow.doubter is not None
        return await uow.doubter.add_r0_manifest(manifest)


async def verify_r0_input(
    sessions: async_sessionmaker[AsyncSession], *, run_id: str,
    planner_config: PlannerConfig | None = None,
) -> R0InputManifest:
    manifest = await build_r0_input_manifest(
        sessions, run_id=run_id, planner_config=planner_config
    )
    if not manifest.hard_gate_passed:
        raise RuntimeError(
            "R0 input gate failed: " + ", ".join(manifest.issue_codes[:8])
        )
    return manifest


__all__ = ["build_r0_input_manifest", "verify_r0_input"]
