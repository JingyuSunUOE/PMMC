"""Production opt-in loader. Every reader and audit owner is evaluator-scoped."""

from __future__ import annotations

from typing import Any

from pmmc.compiler.config import load_compiler_config
from pmmc.evaluation.step8_oracle import OraclePool, build_oracle_pool
from pmmc.evaluation.step8_oracle_execution import OracleJournal, OracleProgramInput
from pmmc.evaluation.step8_oracle_hook import OracleEvaluationHook
from pmmc.evaluation.step8_process_metrics import load_process_facts
from pmmc.evaluation.step8_semantic_workflow import _require_evaluator
from pmmc.experiments.step8_artifacts import CampaignArtifactStore
from pmmc.infrastructure.database.compiler_repositories import (
    PostgresQBankRoutingRepository,
    SessionScopedOperatorDataRepository,
)
from pmmc.infrastructure.database.experiment_repositories import PostgresEmbeddingRepository
from pmmc.infrastructure.database.planner_repositories import PostgresMemoryOperatorIndexRepository
from pmmc.infrastructure.database.repositories import (
    PostgresImageRepository,
    PostgresMemoryRepository,
)
from pmmc.infrastructure.object_store import create_evaluator_object_stores
from pmmc.memory.operators import LinearProgramExecutor, ProgramValidator
from pmmc.models.factory import (
    create_generation_gateway,
    create_image_url_resolver,
    create_token_counter,
)
from pmmc.retrieval.operators import build_operator_registry
from pmmc.runtime.answer_context import PmmcEvidenceAnswerInputProvider
from pmmc.runtime.grounding import GROUNDING_MODEL_ALIAS, load_step8_prompt
from pmmc.schemas import DataScope, PlannerRequirementContract
from pmmc.schemas.step8 import OracleAt50Protocol


def select_oracle_cell(
    declaration: OracleAt50Protocol | None,
    definitions: tuple[dict[str, Any], ...],
    *,
    answer_model: str,
    treatment: str,
) -> str | None:
    if declaration is None or (answer_model, treatment) != (
        declaration.answer_model,
        declaration.treatment,
    ):
        return None
    selected = [
        cell["cell_id"]
        for cell in definitions
        if cell["method"] == "pmmc"
        and cell["source_variant"] == declaration.variant
        and cell["question_budget"] == declaration.question_budget
        and cell["round_id"] == declaration.round_id
        and cell["treatment"] == declaration.treatment
        and cell["answer_model"] == declaration.answer_model
    ]
    if len(selected) != 1:
        raise ValueError("oracle_declared_cell_assignment_missing_or_duplicate")
    return selected[0]


async def load_oracle_programs(
    *,
    pool: OraclePool,
    routing,
    indexes,
    embeddings,
    projection_id: str,
    declaration: OracleAt50Protocol,
) -> dict[str, OracleProgramInput]:
    """Read only runtime-safe frozen programs, contracts and saved query vectors."""
    projection = await routing.get_runtime_projection_v3(pool.compiler_run_id)
    if projection is None or (
        projection["run_id"],
        projection["snapshot_id"],
        projection["routing_projection_id"],
    ) != (pool.compiler_run_id, pool.snapshot_id, projection_id):
        raise ValueError("oracle_frozen_projection_mismatch")
    eligible = await routing.list_verified_v4(
        routing_projection_id=projection_id,
        compiler_run_id=pool.compiler_run_id,
        snapshot_id=pool.snapshot_id,
        dataset=pool.query.dataset.value,
        conversation_id=pool.query.conversation_id,
        cutoff_session_id=pool.query.available_after_session,
        question_budget=declaration.question_budget,
        round_id=declaration.round_id,
    )
    by_id = {item.qbank_id: item for item in eligible}
    if len(by_id) != len(eligible):
        raise ValueError("oracle_duplicate_frozen_program")
    if not pool.candidates:
        return {}
    index = await indexes.get_for_run(pool.compiler_run_id)
    if index is None or index.status != "ready" or index.snapshot_id != pool.snapshot_id:
        raise ValueError("oracle_frozen_memory_index_unavailable")
    selected = {candidate.qbank_id for candidate in pool.candidates}
    vectors: dict[str, dict[str, list[float]]] = {}
    for ref in await routing.list_runtime_program_vector_refs_v3(pool.compiler_run_id):
        qbank_id, channel = ref["qbank_id"], ref["channel"]
        if qbank_id not in selected:
            continue
        values = vectors.setdefault(qbank_id, {})
        if channel in values or channel not in {"question_text_query", "question_joint_query"}:
            raise ValueError("oracle_frozen_vector_channel_invalid")
        record = await embeddings.get_record(ref["embedding_id"])
        if record is None or (
            record.embedding_id != ref["embedding_id"]
            or record.qbank_id != qbank_id
            or record.source_id != qbank_id
            or record.metadata.get("query_role") != channel
            or record.dimension != 4096
            or len(record.vector) != 4096
        ):
            raise ValueError("oracle_frozen_vector_missing_or_mismatched")
        values[channel] = list(record.vector)
    programs = {}
    for candidate in pool.candidates:
        item = by_id.get(candidate.qbank_id)
        if item is None or item.plan_id != candidate.plan_id:
            raise ValueError("oracle_candidate_outside_frozen_projection")
        frozen = await routing.get_runtime_program_v3(item.plan_id)
        if frozen is None:
            raise ValueError("oracle_frozen_program_missing")
        program, source_hash = frozen
        if (
            source_hash != item.program_hash
            or source_hash != candidate.metadata.get("source_program_hash", source_hash)
            or item.canonical_question != pool.predicted_questions[item.qbank_id]
        ):
            raise ValueError("oracle_frozen_program_identity_mismatch")
        channels = vectors.get(item.qbank_id, {})
        if set(channels) != {"question_text_query", "question_joint_query"}:
            raise ValueError("oracle_frozen_vectors_incomplete")
        safe = item.requirement_contract
        requirement = PlannerRequirementContract(
            **{
                key: safe[key]
                for key in (
                    "requirement_contract_id",
                    "required_modalities",
                    "required_capabilities",
                    "allowed_capabilities",
                    "minimum_evidence_groups",
                    "fallback_required",
                    "refusal_only",
                    "content_hash",
                )
            },
            run_id=pool.compiler_run_id,
            qbank_id=item.qbank_id,
            compiler_unit_id=program.compiler_unit_id,
            metadata={
                key: safe[key]
                for key in (
                    "required_semantic_roles",
                    "independent_group_relation_required",
                    "relation_assessment_stage",
                )
                if key in safe
            },
        )
        value = OracleProgramInput(
            program=program,
            requirement=requirement,
            predicted_question=item.canonical_question,
            source_program_hash=source_hash,
            collection_bindings={
                "text_memory": index.text_collection_id,
                "visual_memory": index.visual_collection_id,
            },
            query_vectors=channels,
        )
        value.validate(
            compiler_run_id=pool.compiler_run_id,
            snapshot_id=pool.snapshot_id,
            qbank_id=item.qbank_id,
            plan_id=item.plan_id,
        )
        programs[item.qbank_id] = value
    return programs


async def prepare_configured_oracle(
    *,
    sessions,
    settings,
    campaign_id,
    query_run_id,
    query,
    snapshot_id,
    binding,
    declaration: OracleAt50Protocol,
    protocol_hash: str,
    registry,
) -> OracleEvaluationHook:
    async with sessions() as session:
        await _require_evaluator(session)
    facts = await load_process_facts(sessions, query=query, query_run_id=query_run_id)
    pool = build_oracle_pool(
        scope=DataScope.EVALUATION,
        query=query,
        snapshot_id=snapshot_id,
        query_run_id=query_run_id,
        binding=binding,
        facts=facts,
    )
    assert facts.run is not None
    async with sessions() as session:
        programs = await load_oracle_programs(
            pool=pool,
            routing=PostgresQBankRoutingRepository(session),
            indexes=PostgresMemoryOperatorIndexRepository(session),
            embeddings=PostgresEmbeddingRepository(session),
            projection_id=facts.run["routing_projection_id"],
            declaration=declaration,
        )
    foundation = load_compiler_config(settings.resolved_compiler_config_path)
    data = SessionScopedOperatorDataRepository(sessions)
    operators = build_operator_registry(data, foundation)
    executor = LinearProgramExecutor(
        operators, ProgramValidator(operators, foundation), data, foundation
    )
    runtime_store, _ = create_evaluator_object_stores(settings)
    resolver = create_image_url_resolver(settings, runtime_store)
    return OracleEvaluationHook(
        pool=pool,
        programs=programs,
        executor=executor,
        answer_inputs=PmmcEvidenceAnswerInputProvider(
            memories=PostgresMemoryRepository(sessions),
            images=PostgresImageRepository(sessions),
            counter=create_token_counter(settings, registry, declaration.answer_model),
            instruction=load_step8_prompt("pmmc_answer_contract_v3.txt"),
            answer_model_alias=declaration.answer_model,
            total_context_tokens=32768,
            reserved_output_tokens=512,
            timeout_seconds=settings.model_request_timeout_seconds,
            paper_mode=settings.mode.value == "paper",
            runtime_bucket=settings.object_store_bucket,
            answer_prompt_version="pmmc-answer-contract-v3",
            answer_contract_version="typed-answer-contract-v3",
            allow_memory_raw_images=True,
        ),
        answer_gateway=create_generation_gateway(
            settings, registry, declaration.answer_model, image_url_resolver=resolver
        ),
        grounding_gateway=create_generation_gateway(
            settings, registry, GROUNDING_MODEL_ALIAS, image_url_resolver=resolver
        ),
        grounding_counter=create_token_counter(settings, registry, GROUNDING_MODEL_ALIAS),
        journal=OracleJournal(
            CampaignArtifactStore(settings.resolved_runs_dir / "evaluator-oracle"),
            scope=DataScope.EVALUATION,
        ),
        protocol_identity={
            "campaign_id": campaign_id,
            "protocol_hash": protocol_hash,
            "declaration": declaration.model_dump(mode="json"),
            "cell_id": binding.cell_id,
            "cost_scope": "research_only_evaluator_oracle",
        },
    )
