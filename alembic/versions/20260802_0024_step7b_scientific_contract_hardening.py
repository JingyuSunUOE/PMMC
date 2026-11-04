# ruff: noqa: E501
"""Harden Step 7B scientific contracts and persist reranked executor facts.

Revision ID: 20260802_0024
Revises: 20260730_0023
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260802_0024"
down_revision: str | None = "20260730_0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"

_NEW_TABLES = (
    "planner_revision_patches",
    "qbank_lexical_term_facts",
    "qbank_reranker_assessments",
    "query_evidence_relevance_assessments",
    "query_evidence_relevance_items",
    "structured_answer_outputs",
    "answer_format_diagnostics",
    "answer_reuse_links",
    "query_execution_reuse_links",
    "step7b_pilots",
    "step7b_threshold_decisions",
    "step7b_freeze_manifests",
)


def _metadata_column() -> sa.Column[dict[str, object]]:
    return sa.Column(
        "metadata",
        postgresql.JSONB(astext_type=sa.Text()),
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )


def _created_at() -> sa.Column[object]:
    return sa.Column("created_at", sa.DateTime(timezone=True), nullable=False)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _drop_check_constraint_matching(
    *,
    schema: str,
    table: str,
    required_fragments: tuple[str, ...],
    contract: str,
) -> None:
    """Drop one check by its definition, including pre-Alembic raw-SQL checks."""

    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            """SELECT constraint_row.conname AS constraint_name,
                      pg_get_constraintdef(constraint_row.oid, true) AS definition
                 FROM pg_constraint constraint_row
                 JOIN pg_class relation_row
                   ON relation_row.oid = constraint_row.conrelid
                 JOIN pg_namespace namespace_row
                   ON namespace_row.oid = relation_row.relnamespace
                WHERE namespace_row.nspname = :schema_name
                  AND relation_row.relname = :table_name
                  AND constraint_row.contype = 'c'
                ORDER BY constraint_row.conname"""
        ),
        {"schema_name": schema, "table_name": table},
    ).mappings()
    candidates = list(rows)
    normalized_fragments = tuple(
        " ".join(fragment.lower().split()) for fragment in required_fragments
    )
    matches = []
    for row in candidates:
        definition = " ".join(str(row["definition"] or "").lower().split())
        if all(fragment in definition for fragment in normalized_fragments):
            matches.append(row)
    print(
        "STEP7B_MIGRATION_CONSTRAINT "
        f"contract={contract} state=resolved table={schema}.{table} "
        f"candidate_count={len(candidates)} match_count={len(matches)}",
        flush=True,
    )
    if len(matches) != 1:
        available = ",".join(str(row["constraint_name"]) for row in candidates)
        raise RuntimeError(
            f"Step 7B constraint resolution failed for {contract}: "
            f"expected one match, found {len(matches)}; available={available or 'none'}"
        )
    constraint_name = str(matches[0]["constraint_name"])
    op.drop_constraint(
        op.f(constraint_name),
        table,
        schema=schema,
        type_="check",
    )
    print(
        "STEP7B_MIGRATION_CONSTRAINT "
        f"contract={contract} state=dropped table={schema}.{table} "
        f"constraint_name={constraint_name}",
        flush=True,
    )


def _replace_guard_fragment(
    *, procedure: str, source: str, target: str, expected_count: int, contract: str
) -> None:
    print(
        "STEP7B_MIGRATION_GUARD_REWRITE "
        f"contract={contract} state=start procedure={procedure} "
        f"expected_count={expected_count}",
        flush=True,
    )
    op.execute(
        f"""DO $$
        DECLARE definition text; found_count integer;
        BEGIN
          SELECT pg_get_functiondef('{procedure}'::regprocedure) INTO definition;
          IF definition IS NULL THEN
            RAISE EXCEPTION 'missing Step 7B persistence guard: %',
              {_sql_literal(contract)};
          END IF;
          found_count := (length(definition) - length(replace(
            definition, {_sql_literal(source)}, ''))) / length({_sql_literal(source)});
          IF found_count IS DISTINCT FROM {expected_count} THEN
            RAISE EXCEPTION 'unexpected Step 7B persistence guard definition: %/%',
              {_sql_literal(contract)}, found_count;
          END IF;
          EXECUTE replace(definition, {_sql_literal(source)}, {_sql_literal(target)});
        END $$"""
    )
    print(
        "STEP7B_MIGRATION_GUARD_REWRITE "
        f"contract={contract} state=ready procedure={procedure} "
        f"expected_count={expected_count}",
        flush=True,
    )


def _extend_historical_guards(*, enable: bool) -> None:
    replacements = (
        (
            f"{RUNTIME}.validate_questioner_fact_scope()",
            "compiler_version IN ('0.6.0', '0.7.0')",
            "compiler_version IN ('0.6.0', '0.7.0', '0.8.0')",
            1,
            "questioner_generation_terminal_rejection",
        ),
        (
            f"{RUNTIME}.enforce_questioner_verifier_scope_v3()",
            "WHEN '0.7.0' THEN 'pmmc-questioner-verifier-v5'",
            "WHEN '0.7.0' THEN 'pmmc-questioner-verifier-v5'\n"
            "                WHEN '0.8.0' THEN 'pmmc-questioner-verifier-v6'",
            1,
            "questioner_verifier_prompt",
        ),
        (
            f"{RUNTIME}.enforce_questioner_verifier_scope_v3()",
            "compiler_version IN ('0.5.0', '0.6.0', '0.7.0')",
            "compiler_version IN ('0.5.0', '0.6.0', '0.7.0', '0.8.0')",
            1,
            "questioner_verifier_scope",
        ),
        (
            f"{RUNTIME}.enforce_questioner_verifier_scope_v3()",
            "compiler_version IN ('0.6.0', '0.7.0')",
            "compiler_version IN ('0.6.0', '0.7.0', '0.8.0')",
            1,
            "questioner_verifier_terminal_rejection",
        ),
        (
            f"{RUNTIME}.validate_compiler_embedding_lineage_v4()",
            "compiler.compiler_schema_version IN ('0.6.0', '0.7.0')",
            "compiler.compiler_schema_version IN ('0.6.0', '0.7.0', '0.8.0')",
            1,
            "program_query_embedding_lineage",
        ),
        (
            f"{RUNTIME}.enforce_compiler_completion_v3()",
            "compiler.compiler_schema_version NOT IN ('0.4.0', '0.5.0', '0.6.0', '0.7.0')",
            "compiler.compiler_schema_version NOT IN ('0.4.0', '0.5.0', '0.6.0', '0.7.0', '0.8.0')",
            1,
            "compiler_completion_delegation",
        ),
    )
    for procedure, old, new, count, contract in replacements:
        _replace_guard_fragment(
            procedure=procedure,
            source=old if enable else new,
            target=new if enable else old,
            expected_count=count,
            contract=contract,
        )


def _create_revision_patch_table() -> None:
    op.create_table(
        "planner_revision_patches",
        sa.Column("patch_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("qbank_id", sa.String(128), nullable=False),
        sa.Column("source_plan_id", sa.String(128), nullable=False),
        sa.Column("target_round_id", sa.Integer(), nullable=False),
        sa.Column("directive_hash", sa.String(128), nullable=False),
        sa.Column("before_strategy_hash", sa.String(128), nullable=False),
        sa.Column("after_strategy_hash", sa.String(128), nullable=False),
        sa.Column("before_program_hash", sa.String(128), nullable=False),
        sa.Column("after_program_hash", sa.String(128)),
        sa.Column("applied", sa.Boolean(), nullable=False),
        sa.Column("rejection_code", sa.String(64)),
        sa.Column("capability_added", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("capability_removed", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _created_at(),
        _metadata_column(),
        sa.ForeignKeyConstraint(
            ["source_plan_id", "run_id"],
            [f"{RUNTIME}.memory_programs.plan_id", f"{RUNTIME}.memory_programs.run_id"],
        ),
        sa.UniqueConstraint("run_id", "qbank_id", "target_round_id"),
        sa.CheckConstraint("target_round_id BETWEEN 1 AND 3", name="round_values"),
        sa.CheckConstraint(
            "(applied AND after_program_hash IS NOT NULL AND rejection_code IS NULL "
            "AND before_program_hash <> after_program_hash) OR "
            "(NOT applied AND after_program_hash IS NULL AND rejection_code IS NOT NULL)",
            name="application_consistent",
        ),
        schema=RUNTIME,
    )


def _create_match_tables() -> None:
    op.create_table(
        "qbank_lexical_term_facts",
        sa.Column("lexical_term_fact_id", sa.String(128), primary_key=True),
        sa.Column(
            "match_batch_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.qbank_match_batches.match_batch_id"),
            nullable=False,
        ),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("term_hash", sa.String(128), nullable=False),
        sa.Column("term_kind", sa.String(16), nullable=False),
        sa.Column("exact_boost", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _metadata_column(),
        sa.UniqueConstraint("match_batch_id", "ordinal"),
        sa.CheckConstraint("ordinal BETWEEN 0 AND 11", name="ordinal_values"),
        sa.CheckConstraint(
            "term_kind IN ('lexeme','entity','number','date')", name="term_kind_values"
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "qbank_reranker_assessments",
        sa.Column("reranker_assessment_id", sa.String(128), primary_key=True),
        sa.Column(
            "query_executor_run_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.query_executor_runs.query_executor_run_id"),
            nullable=False,
        ),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column(
            "qbank_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.qbank_questions.qbank_id"),
            nullable=False,
        ),
        sa.Column(
            "model_call_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.model_calls.model_call_id"),
            nullable=False,
        ),
        sa.Column("same_information_need", sa.String(16), nullable=False),
        sa.Column("candidate_can_answer_query", sa.String(16), nullable=False),
        sa.Column("answer_type_compatible", sa.String(16), nullable=False),
        sa.Column("modality_compatible", sa.String(16), nullable=False),
        sa.Column("match_level", sa.String(16), nullable=False),
        sa.Column("reranker_score", sa.Float(), nullable=False),
        sa.Column("provider_schema_conformant", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _created_at(),
        _metadata_column(),
        sa.UniqueConstraint("query_executor_run_id", "query_id", "qbank_id"),
        sa.CheckConstraint(
            "same_information_need IN ('pass','fail') AND "
            "candidate_can_answer_query IN ('pass','fail') AND "
            "answer_type_compatible IN ('pass','fail') AND "
            "modality_compatible IN ('pass','fail','not_applicable')",
            name="dimension_values",
        ),
        sa.CheckConstraint(
            "match_level IN ('exact','strong','weak','none')", name="match_level_values"
        ),
        sa.CheckConstraint("reranker_score BETWEEN 0 AND 1", name="score_valid"),
        schema=RUNTIME,
    )


def _create_relevance_tables() -> None:
    op.create_table(
        "query_evidence_relevance_assessments",
        sa.Column("relevance_assessment_id", sa.String(128), primary_key=True),
        sa.Column(
            "query_executor_run_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.query_executor_runs.query_executor_run_id"),
            nullable=False,
        ),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column(
            "query_program_binding_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.query_program_bindings.query_program_binding_id"),
            nullable=False,
        ),
        sa.Column("relevance_threshold", sa.Float(), nullable=False),
        sa.Column("required_group_count", sa.Integer(), nullable=False),
        sa.Column("relevant_group_count", sa.Integer(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _created_at(),
        _metadata_column(),
        sa.UniqueConstraint("query_program_binding_id"),
        sa.CheckConstraint(
            "relevance_threshold BETWEEN 0 AND 1 AND required_group_count > 0 "
            "AND relevant_group_count >= 0",
            name="relevance_counts_valid",
        ),
        sa.CheckConstraint(
            "passed = (relevant_group_count >= required_group_count)",
            name="relevance_server_derived",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "query_evidence_relevance_items",
        sa.Column("relevance_item_id", sa.String(128), primary_key=True),
        sa.Column(
            "relevance_assessment_id",
            sa.String(128),
            sa.ForeignKey(
                f"{RUNTIME}.query_evidence_relevance_assessments.relevance_assessment_id"
            ),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("query_executor_run_id", sa.String(128), nullable=False),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column(
            "execution_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.program_executions.execution_id"),
            nullable=False,
        ),
        sa.Column("evidence_id_hash", sa.String(128), nullable=False),
        sa.Column("evidence_group_hash", sa.String(128), nullable=False),
        sa.Column("modality", sa.String(16), nullable=False),
        sa.Column("cosine_similarity", sa.Float(), nullable=False),
        sa.Column("normalized_score", sa.Float(), nullable=False),
        sa.Column("relevant", sa.Boolean(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _metadata_column(),
        sa.UniqueConstraint("relevance_assessment_id", "ordinal"),
        sa.CheckConstraint("ordinal >= 0", name="ordinal_nonnegative"),
        sa.CheckConstraint(
            "cosine_similarity BETWEEN -1 AND 1 AND normalized_score BETWEEN 0 AND 1 "
            "AND threshold BETWEEN 0 AND 1",
            name="score_valid",
        ),
        schema=RUNTIME,
    )


def _create_answer_tables() -> None:
    op.create_table(
        "structured_answer_outputs",
        sa.Column("structured_answer_output_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column("answer_variant", sa.String(128), nullable=False),
        sa.Column("attempt_ordinal", sa.Integer(), nullable=False),
        sa.Column(
            "model_call_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.model_calls.model_call_id"),
            nullable=False,
        ),
        sa.Column(
            "answer_id", sa.String(128), sa.ForeignKey(f"{RUNTIME}.final_answers.answer_id")
        ),
        sa.Column("answer", sa.Text()),
        sa.Column("supported", sa.Boolean(), nullable=False),
        sa.Column("used_evidence_aliases", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("abstain_reason", sa.String(64)),
        sa.Column("parse_status", sa.String(16), nullable=False),
        sa.Column("provider_schema_conformant", sa.Boolean(), nullable=False),
        sa.Column("finish_reason", sa.String(32)),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _created_at(),
        _metadata_column(),
        sa.UniqueConstraint("run_id", "query_id", "answer_variant", "attempt_ordinal"),
        sa.CheckConstraint("attempt_ordinal IN (0,1)", name="attempt_ordinal_values"),
        sa.CheckConstraint("parse_status IN ('accepted','rejected')", name="parse_status_values"),
        sa.CheckConstraint(
            "(supported AND answer IS NOT NULL AND abstain_reason IS NULL "
            "AND cardinality(used_evidence_aliases) > 0) OR "
            "(NOT supported AND answer IS NULL AND abstain_reason IS NOT NULL)",
            name="support_contract",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "answer_format_diagnostics",
        sa.Column("diagnostic_id", sa.String(128), primary_key=True),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column("answer_variant", sa.String(128), nullable=False),
        sa.Column("attempt_ordinal", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column(
            "model_call_id", sa.String(128), sa.ForeignKey(f"{RUNTIME}.model_calls.model_call_id")
        ),
        sa.Column("failure_codes", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("provider_schema_conformant", sa.Boolean(), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _created_at(),
        _metadata_column(),
        sa.UniqueConstraint("run_id", "query_id", "answer_variant", "attempt_ordinal"),
        sa.CheckConstraint("attempt_ordinal IN (0,1)", name="attempt_ordinal_values"),
        sa.CheckConstraint(
            "accepted = (cardinality(failure_codes) = 0)",
            name="accepted_failure_consistency",
        ),
        schema=RUNTIME,
    )
    op.create_table(
        "answer_reuse_links",
        sa.Column("answer_reuse_link_id", sa.String(128), primary_key=True),
        sa.Column(
            "query_executor_run_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.query_executor_runs.query_executor_run_id"),
            nullable=False,
        ),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column(
            "source_answer_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.final_answers.answer_id"),
            nullable=False,
        ),
        sa.Column("target_variant", sa.String(128), nullable=False),
        sa.Column("context_hash", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _created_at(),
        _metadata_column(),
        sa.UniqueConstraint("query_executor_run_id", "query_id", "target_variant"),
        schema=RUNTIME,
    )
    op.create_table(
        "query_execution_reuse_links",
        sa.Column("execution_reuse_link_id", sa.String(128), primary_key=True),
        sa.Column(
            "query_executor_run_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.query_executor_runs.query_executor_run_id"),
            nullable=False,
        ),
        sa.Column("query_id", sa.String(256), nullable=False),
        sa.Column(
            "source_execution_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.program_executions.execution_id"),
            nullable=False,
        ),
        sa.Column("target_round_id", sa.Integer(), nullable=False),
        sa.Column("execution_identity_hash", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _created_at(),
        _metadata_column(),
        sa.UniqueConstraint("query_executor_run_id", "query_id", "target_round_id"),
        sa.CheckConstraint("target_round_id BETWEEN 0 AND 3", name="round_values"),
        schema=RUNTIME,
    )


def _create_pilot_tables() -> None:
    op.create_table(
        "step7b_pilots",
        sa.Column("pilot_id", sa.String(128), primary_key=True),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("compiler_run_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("query_executor_run_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("partition_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("paper_result", sa.Boolean(), nullable=False),
        sa.Column("protocol_integrity_passed", sa.Boolean(), nullable=False),
        sa.Column("viability_passed", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _created_at(),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        _metadata_column(),
        sa.CheckConstraint(
            "status IN ('running','failed','needs_remediation',"
            "'awaiting_threshold_review','approved','rejected')",
            name="status_values",
        ),
        sa.CheckConstraint("paper_result = false", name="development_only"),
        schema=RUNTIME,
    )
    op.create_table(
        "step7b_threshold_decisions",
        sa.Column("threshold_decision_id", sa.String(128), primary_key=True),
        sa.Column(
            "pilot_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.step7b_pilots.pilot_id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("reviewer_id", sa.String(256), nullable=False),
        sa.Column("match_threshold", sa.Float(), nullable=False),
        sa.Column("gap_threshold", sa.Float(), nullable=False),
        sa.Column("rationale_hash", sa.String(128), nullable=False),
        sa.Column("decision_content_hash", sa.String(128), nullable=False),
        _created_at(),
        _metadata_column(),
        sa.CheckConstraint("match_threshold IN (0.5,0.6,0.7,0.8)", name="match_grid"),
        sa.CheckConstraint("gap_threshold IN (0.0,0.05,0.1)", name="gap_grid"),
        sa.CheckConstraint("decision IN ('approve','reject')", name="decision_values"),
        schema=RUNTIME,
    )
    op.create_table(
        "step7b_freeze_manifests",
        sa.Column("freeze_manifest_id", sa.String(128), primary_key=True),
        sa.Column(
            "pilot_id",
            sa.String(128),
            sa.ForeignKey(f"{RUNTIME}.step7b_pilots.pilot_id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("questioner_freeze_hash", sa.String(128), nullable=False),
        sa.Column("compiler_program_freeze_hash", sa.String(128), nullable=False),
        sa.Column("verified_qbank_freeze_hash", sa.String(128), nullable=False),
        sa.Column("query_executor_freeze_hash", sa.String(128), nullable=False),
        sa.Column("answer_contract_hash", sa.String(128), nullable=False),
        sa.Column("active_round", sa.Integer(), nullable=False),
        sa.Column("question_budget", sa.Integer(), nullable=False),
        sa.Column("match_threshold", sa.Float(), nullable=False),
        sa.Column("gap_threshold", sa.Float(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        _created_at(),
        _metadata_column(),
        sa.CheckConstraint("active_round = 2 AND question_budget = 8", name="active_protocol"),
        schema=RUNTIME,
    )


def _upgrade_existing_tables() -> None:
    for column_name in (
        "query_executor_protocol_hash",
        "approved_routing_policy_hash",
        "approved_selection_hash",
    ):
        op.add_column(
            "query_executor_freeze_manifests",
            sa.Column(column_name, sa.String(128)),
            schema=RUNTIME,
        )
    op.create_check_constraint(
        "approved_routing_identity",
        "query_executor_freeze_manifests",
        "schema_version <> '0.2.0' OR "
        "(query_executor_protocol_hash ~ '^[0-9a-f]{64}$' AND "
        "approved_routing_policy_hash ~ '^[0-9a-f]{64}$' AND "
        "approved_selection_hash ~ '^[0-9a-f]{64}$')",
        schema=RUNTIME,
    )
    op.add_column(
        "evaluation_jobs",
        sa.Column("answer_variant", sa.String(128)),
        schema=EVALUATION,
    )
    op.execute(
        f"""UPDATE {EVALUATION}.evaluation_jobs job
               SET answer_variant = answer.answer_variant
              FROM {RUNTIME}.final_answers answer
             WHERE answer.answer_id = job.answer_id
               AND answer.run_id = job.run_id
               AND answer.query_id = job.query_id"""
    )
    op.alter_column(
        "evaluation_jobs",
        "answer_variant",
        nullable=False,
        schema=EVALUATION,
    )
    op.drop_constraint(
        "evaluation_jobs_evaluation_run_id_run_id_query_id_stage_key",
        "evaluation_jobs",
        schema=EVALUATION,
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_evaluation_jobs_run_query_variant_stage",
        "evaluation_jobs",
        (
            "evaluation_run_id",
            "run_id",
            "query_id",
            "answer_variant",
            "stage",
        ),
        schema=EVALUATION,
    )
    for column_name in (
        "context_evidence_contributed",
        "temporal_evidence_contributed",
    ):
        op.add_column(
            "program_dry_run_assessments",
            sa.Column(
                column_name,
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
            schema=RUNTIME,
        )
        op.alter_column(
            "program_dry_run_assessments",
            column_name,
            server_default=None,
            schema=RUNTIME,
        )
    op.add_column(
        "qbank_questions", sa.Column("answer_type", sa.String(32)), schema=RUNTIME
    )
    op.create_check_constraint(
        "answer_type_values",
        "qbank_questions",
        "answer_type IS NULL OR answer_type IN "
        "('entity','value','count','list','boolean','event','refusal')",
        schema=RUNTIME,
    )
    for name in (
        "lexical_normalized_score",
        "text_normalized_score",
        "multimodal_normalized_score",
        "available_weight_sum",
        "base_score",
        "base_reliable_score",
        "reranker_score",
        "route_score",
    ):
        op.add_column(
            "qbank_match_fused_candidates", sa.Column(name, sa.Float()), schema=RUNTIME
        )
    op.add_column(
        "qbank_match_fused_candidates",
        sa.Column("reranker_match_level", sa.String(16)),
        schema=RUNTIME,
    )
    op.create_check_constraint(
        "step7b_score_values",
        "qbank_match_fused_candidates",
        "(base_score IS NULL OR base_score BETWEEN 0 AND 1) AND "
        "(base_reliable_score IS NULL OR base_reliable_score BETWEEN 0 AND 1) AND "
        "(reranker_score IS NULL OR reranker_score BETWEEN 0 AND 1) AND "
        "(route_score IS NULL OR route_score BETWEEN 0 AND 1) AND "
        "(reranker_match_level IS NULL OR reranker_match_level IN "
        "('exact','strong','weak','none'))",
        schema=RUNTIME,
    )
    for name in (
        "structural_requirement_success",
        "query_evidence_relevance",
        "answer_support",
    ):
        op.add_column(
            "query_evidence_sufficiency_assessments",
            sa.Column(name, sa.Boolean()),
            schema=RUNTIME,
        )
    op.add_column(
        "query_evidence_sufficiency_assessments",
        sa.Column("relevant_evidence_group_count", sa.Integer()),
        schema=RUNTIME,
    )
    op.add_column(
        "query_evidence_sufficiency_assessments",
        sa.Column("relevance_threshold", sa.Float()),
        schema=RUNTIME,
    )
    op.execute(
        f"""UPDATE {RUNTIME}.query_evidence_sufficiency_assessments
               SET structural_requirement_success = sufficient,
                   query_evidence_relevance = sufficient,
                   answer_support = sufficient"""
    )
    _drop_check_constraint_matching(
        schema=RUNTIME,
        table="query_evidence_sufficiency_assessments",
        required_fragments=(
            "sufficient",
            "required_capabilities",
            "actual_capabilities",
            "required_modalities",
            "actual_modalities",
            "minimum_evidence_groups",
            "raw_image_required",
            "raw_image_reached",
        ),
        contract="runtime_sufficiency_v1",
    )
    op.create_check_constraint(
        "server_derived_v2",
        "query_evidence_sufficiency_assessments",
        "((metadata->>'sufficiency_policy_version' = 'query-runtime-evidence-v2' "
        "AND structural_requirement_success IS NOT NULL "
        "AND query_evidence_relevance IS NOT NULL AND answer_support IS NOT NULL "
        "AND relevance_threshold IS NOT NULL "
        "AND sufficient = (technical_success AND structural_requirement_success "
        "AND query_evidence_relevance AND answer_support)) OR "
        "(COALESCE(metadata->>'sufficiency_policy_version', '') <> "
        "'query-runtime-evidence-v2' AND sufficient = "
        "(technical_success AND nonempty_evidence "
        "AND required_capabilities <@ actual_capabilities "
        "AND required_modalities <@ actual_modalities "
        "AND independent_group_count >= minimum_evidence_groups "
        "AND (NOT raw_image_required OR raw_image_reached))))",
        schema=RUNTIME,
    )


def _create_routing_view() -> None:
    print(
        "STEP7B_MIGRATION_ROUTING_VIEW state=start "
        "source=verified_qbank_routing_v2 target=verified_qbank_routing_v3",
        flush=True,
    )
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.verified_qbank_routing_v3")
    print("STEP7B_MIGRATION_ROUTING_VIEW state=v3_create", flush=True)
    op.execute(
        f"""CREATE VIEW {RUNTIME}.verified_qbank_routing_v3 AS
        SELECT routing.*, questions.answer_type
          FROM {RUNTIME}.verified_qbank_routing_v2 routing
          JOIN {RUNTIME}.qbank_questions questions
            ON questions.qbank_id = routing.qbank_id
           AND questions.run_id = routing.compiler_run_id
          JOIN {RUNTIME}.compiler_runs compiler
            ON compiler.run_id = routing.compiler_run_id
         WHERE compiler.compiler_schema_version = '0.8.0'
           AND routing.operator_set_version = 'pmmc-operators-v6'
           AND questions.answer_type IS NOT NULL"""
    )
    print("STEP7B_MIGRATION_ROUTING_VIEW state=v3_ready", flush=True)
    column_names = (
        "routing_projection_id",
        "compiler_run_id",
        "snapshot_id",
        "dataset",
        "conversation_id",
        "available_after_session",
        "cutoff_session_id",
        "qbank_id",
        "generation_rank",
        "route_class",
        "question_type",
        "canonical_question",
        "paraphrases",
        "required_modalities",
        "accepted_round",
        "plan_id",
        "program_hash",
        "operator_set_version",
        "requirement_contract_id",
        "requirement_contract_hash",
        "contract_required_modalities",
        "required_capabilities",
        "allowed_capabilities",
        "minimum_evidence_groups",
        "fallback_required",
        "refusal_only",
        "lexical_collection_id",
        "text_collection_id",
        "multimodal_collection_id",
        "active_round",
        "decision_hash",
        "ordinal",
        "projection_round",
    )
    routing_columns = ", ".join(f"routing.{column}" for column in column_names)
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.verified_qbank_routing_completion")
    print(
        "STEP7B_MIGRATION_ROUTING_VIEW state=completion_create "
        f"column_count={len(column_names)}",
        flush=True,
    )
    op.execute(
        f"""CREATE VIEW {RUNTIME}.verified_qbank_routing_completion AS
        SELECT {routing_columns}
          FROM {RUNTIME}.verified_qbank_routing_v2 routing
          JOIN {RUNTIME}.compiler_runs compiler
            ON compiler.run_id = routing.compiler_run_id
         WHERE compiler.compiler_schema_version <> '0.8.0'
        UNION ALL
        SELECT {routing_columns}
          FROM {RUNTIME}.verified_qbank_routing_v3 routing"""
    )
    print("STEP7B_MIGRATION_ROUTING_VIEW state=completion_ready", flush=True)
    print("STEP7B_MIGRATION_ROUTING_VIEW state=guard_dispatch", flush=True)
    _replace_guard_fragment(
        procedure=f"{RUNTIME}.enforce_query_executor_completion()",
        source=f"{RUNTIME}.verified_qbank_routing_v2",
        target=f"{RUNTIME}.verified_qbank_routing_completion",
        expected_count=2,
        contract="query_completion_routing_dispatch",
    )
    print("STEP7B_MIGRATION_ROUTING_VIEW state=ready", flush=True)


def _create_step7b_completion_guards() -> None:
    op.create_check_constraint(
        "expected_recall_sufficient",
        "program_dry_run_assessments",
        "NOT ((metadata->>'assessment_policy') = "
        "'program-dry-run-assessment-v3' "
        "AND expected_group_count > 0 "
        "AND COALESCE(expected_evidence_recall, 0) = 0 "
        "AND evidence_sufficient)",
        schema=RUNTIME,
    )
    op.create_check_constraint(
        "context_requires_operator",
        "program_dry_run_assessments",
        "NOT context_evidence_contributed OR context_operator_used",
        schema=RUNTIME,
    )
    op.create_check_constraint(
        "temporal_requires_operator",
        "program_dry_run_assessments",
        "NOT temporal_evidence_contributed OR temporal_operator_used",
        schema=RUNTIME,
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS step7b_compiler_completion_integrity "
        f"ON {RUNTIME}.run_states"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_step7b_compiler_completion()"
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_step7b_compiler_completion()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE compiler {RUNTIME}.compiler_runs%ROWTYPE; bad integer;
                eligible integer; routing {RUNTIME}.routing_projection_manifests%ROWTYPE;
        BEGIN
          IF NEW.status != 'completed' OR OLD.status = 'completed' THEN RETURN NEW; END IF;
          SELECT * INTO compiler FROM {RUNTIME}.compiler_runs WHERE run_id=NEW.run_id;
          IF NOT FOUND OR compiler.compiler_schema_version != '0.8.0' THEN RETURN NEW; END IF;
          IF compiler.target_stage != 'verified_qbank'
             OR compiler.operator_set_version != 'pmmc-operators-v6'
             OR compiler.questioner_prompt_version != 'pmmc-questioner-v3.5'
             OR compiler.planner_prompt_version != 'pmmc-planner-v1.4'
             OR compiler.doubter_prompt_version != 'pmmc-doubter-v1.2'
             OR compiler.r0_input_policy_version != 'r0-input-gate-v3'
             OR compiler.routing_projection_version != 'verified-qbank-projection-v3'
             OR compiler.active_round != 2 OR compiler.max_round != 3 THEN
            RAISE EXCEPTION 'Step 7B compiler identity is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.r0_input_manifests r
           WHERE r.run_id=NEW.run_id AND r.hard_gate_passed
             AND cardinality(r.protocol_integrity_issues)=0 AND r.downstream_viability;
          IF bad != 1 THEN
            RAISE EXCEPTION 'Step 7B requires one viable R0 v3 manifest' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO eligible FROM {RUNTIME}.qbank_questions q
           WHERE q.run_id=NEW.run_id AND q.answer_type IS NOT NULL AND EXISTS (
             SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results g
              WHERE g.run_id=q.run_id AND g.qbank_id=q.qbank_id
                AND g.status='eligible_for_planning') AND EXISTS (
             SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications v
              WHERE v.run_id=q.run_id AND v.qbank_id=q.qbank_id
                AND v.verification_round=q.candidate_version AND v.verdict='pass'
                AND v.provider_schema_conformant);
          IF eligible = 0 THEN
            RAISE EXCEPTION 'Step 7B compiler has no native eligible candidates' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.program_dry_run_assessments a
           WHERE a.run_id=NEW.run_id
             AND a.metadata->>'assessment_policy'='program-dry-run-assessment-v3'
             AND a.expected_group_count>0
             AND COALESCE(a.expected_evidence_recall, 0)=0
             AND a.evidence_sufficient;
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7B expected evidence recall is zero but path is sufficient'
              USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.doubter_assessments d
            JOIN {RUNTIME}.program_dry_run_assessments a
              ON a.run_id=d.run_id AND a.plan_id=d.plan_id
             AND a.execution_id=d.execution_id
           WHERE d.run_id=NEW.run_id AND d.action='pass'
             AND NOT a.nonempty_evidence;
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7B empty retrieved path cannot pass Doubter'
              USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.doubter_assessments d
            JOIN {RUNTIME}.program_dry_run_assessments a
              ON a.run_id=d.run_id AND a.plan_id=d.plan_id
             AND a.execution_id=d.execution_id
            JOIN {RUNTIME}.planner_requirement_contracts c
              ON c.run_id=d.run_id AND c.qbank_id=d.qbank_id
           WHERE d.run_id=NEW.run_id AND d.action='pass' AND (
             ('context'=ANY(c.required_capabilities)
                AND NOT a.context_evidence_contributed) OR
             ('temporal'=ANY(c.required_capabilities)
                AND NOT a.temporal_evidence_contributed));
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7B extension operator did not contribute evidence'
              USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.planner_revision_lineage l
           WHERE l.run_id=NEW.run_id AND NOT EXISTS (
             SELECT 1 FROM {RUNTIME}.planner_revision_patches p
              WHERE p.run_id=l.run_id AND p.qbank_id=l.qbank_id
                AND p.target_round_id=l.target_round_id AND p.applied
                AND p.before_program_hash<>p.after_program_hash);
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7B revision lineage lacks an effective typed patch'
              USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_manifests p
           WHERE p.run_id=NEW.run_id AND p.round_id BETWEEN 0 AND 3;
          IF bad != 4 THEN
            RAISE EXCEPTION 'Step 7B requires continuous r0-r3 projections' USING ERRCODE='55000';
          END IF;
          SELECT * INTO routing FROM {RUNTIME}.routing_projection_manifests
           WHERE run_id=NEW.run_id;
          IF NOT FOUND OR routing.active_round!=2 OR routing.max_round!=3
             OR routing.member_count!=(SELECT count(*) FROM {RUNTIME}.routing_projection_members m
                                        WHERE m.routing_projection_id=routing.routing_projection_id)
             OR routing.member_count=0 THEN
            RAISE EXCEPTION 'Step 7B active-r2 routing projection is incomplete'
              USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.routing_projection_members m
           WHERE m.run_id=NEW.run_id AND NOT EXISTS (
             SELECT 1 FROM {RUNTIME}.verified_qbank_routing_v3 v
              WHERE v.routing_projection_id=m.routing_projection_id
                AND v.compiler_run_id=m.run_id AND v.qbank_id=m.qbank_id);
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7B routing contains unsafe or unresolved members'
              USING ERRCODE='55000';
          END IF;
          RETURN NEW;
        END $$"""
    )
    op.execute(
        f"""CREATE TRIGGER step7b_compiler_completion_integrity
        BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW
        EXECUTE FUNCTION {RUNTIME}.enforce_step7b_compiler_completion()"""
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS step7b_query_completion_integrity "
        f"ON {RUNTIME}.query_executor_runs"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_step7b_query_completion()"
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_step7b_query_completion()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE bad integer;
        BEGIN
          IF NEW.status!='completed' OR OLD.status='completed'
             OR NEW.schema_version!='0.2.0' THEN RETURN NEW; END IF;
          IF NOT EXISTS (SELECT 1 FROM {RUNTIME}.compiler_runs c
                          WHERE c.run_id=NEW.source_compiler_run_id
                            AND c.compiler_schema_version='0.8.0'
                            AND c.operator_set_version='pmmc-operators-v6') THEN
            RAISE EXCEPTION 'Step 7B query run has an incompatible compiler parent'
              USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.run_queries q
           WHERE q.run_id=NEW.query_executor_run_id AND NOT EXISTS (
             SELECT 1 FROM {RUNTIME}.qbank_reranker_assessments r
              WHERE r.query_executor_run_id=q.run_id AND r.query_id=q.query_id
                AND r.provider_schema_conformant);
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7B reranker denominator is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.query_route_decisions d
           WHERE d.query_executor_run_id=NEW.query_executor_run_id
             AND d.selected_qbank_id IS NOT NULL AND NOT EXISTS (
               SELECT 1 FROM {RUNTIME}.qbank_reranker_assessments r
                WHERE r.query_executor_run_id=d.query_executor_run_id
                  AND r.query_id=d.query_id AND r.qbank_id=d.selected_qbank_id
                  AND r.match_level IN ('exact','strong')
                  AND r.same_information_need='pass'
                  AND r.candidate_can_answer_query='pass'
                  AND r.answer_type_compatible='pass'
                  AND r.modality_compatible IN ('pass','not_applicable'));
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7B active route escaped bounded reranker acceptance'
              USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.query_program_bindings b
           WHERE b.query_executor_run_id=NEW.query_executor_run_id AND NOT EXISTS (
             SELECT 1 FROM {RUNTIME}.query_evidence_relevance_assessments a
              WHERE a.query_program_binding_id=b.query_program_binding_id
                AND a.query_executor_run_id=b.query_executor_run_id);
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7B query evidence relevance denominator is incomplete'
              USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.final_answers a
           WHERE a.run_id=NEW.query_executor_run_id AND a.method='pmmc'
             AND a.answer_variant LIKE 'pmmc_%' AND NOT EXISTS (
               SELECT 1 FROM {RUNTIME}.structured_answer_outputs s
                WHERE s.answer_id=a.answer_id AND s.parse_status='accepted'
                  AND s.provider_schema_conformant);
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7B PMMC answer lacks an accepted strict contract fact'
              USING ERRCODE='55000';
          END IF;
          RETURN NEW;
        END $$"""
    )
    op.execute(
        f"""CREATE TRIGGER step7b_query_completion_integrity
        BEFORE UPDATE ON {RUNTIME}.query_executor_runs FOR EACH ROW
        EXECUTE FUNCTION {RUNTIME}.enforce_step7b_query_completion()"""
    )


def _grant_permissions() -> None:
    for table in _NEW_TABLES:
        op.execute(f"ALTER TABLE {RUNTIME}.{table} OWNER TO pmmc_owner")
        op.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {RUNTIME}.{table} TO pmmc_runtime"
        )
        op.execute(f"GRANT SELECT ON {RUNTIME}.{table} TO pmmc_evaluator")
    op.execute(f"ALTER VIEW {RUNTIME}.verified_qbank_routing_v3 OWNER TO pmmc_owner")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_v3 TO pmmc_runtime")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_v3 TO pmmc_evaluator")
    op.execute(
        f"ALTER VIEW {RUNTIME}.verified_qbank_routing_completion OWNER TO pmmc_owner"
    )
    op.execute(
        f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_completion TO pmmc_runtime"
    )
    op.execute(
        f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_completion TO pmmc_evaluator"
    )
    for function in (
        "enforce_step7b_compiler_completion()",
        "enforce_step7b_query_completion()",
    ):
        op.execute(f"ALTER FUNCTION {RUNTIME}.{function} OWNER TO pmmc_owner")
        op.execute(f"REVOKE ALL ON FUNCTION {RUNTIME}.{function} FROM PUBLIC")


def upgrade() -> None:
    print("STEP7B_MIGRATION state=start revision=20260802_0024", flush=True)
    print("STEP7B_MIGRATION phase=existing_tables state=start", flush=True)
    _upgrade_existing_tables()
    print("STEP7B_MIGRATION phase=existing_tables state=done", flush=True)
    print("STEP7B_MIGRATION phase=new_fact_tables state=start", flush=True)
    _create_revision_patch_table()
    _create_match_tables()
    _create_relevance_tables()
    _create_answer_tables()
    _create_pilot_tables()
    print("STEP7B_MIGRATION phase=new_fact_tables state=done", flush=True)
    print("STEP7B_MIGRATION phase=guards_and_routing state=start", flush=True)
    _extend_historical_guards(enable=True)
    _create_routing_view()
    _create_step7b_completion_guards()
    print("STEP7B_MIGRATION phase=guards_and_routing state=done", flush=True)
    print("STEP7B_MIGRATION phase=permissions state=start", flush=True)
    _grant_permissions()
    print("STEP7B_MIGRATION phase=permissions state=done", flush=True)
    print(
        "STEP7B_MIGRATION state=ready routing=v3 reranker=true answer_contract=v2",
        flush=True,
    )


def downgrade() -> None:
    print("STEP7B_MIGRATION state=downgrade_start revision=20260802_0024", flush=True)
    connection = op.get_bind()
    if connection.execute(
        sa.text(
            f"SELECT EXISTS (SELECT 1 FROM {RUNTIME}.query_executor_freeze_manifests "
            "WHERE schema_version = '0.2.0')"
        )
    ).scalar_one():
        raise RuntimeError("cannot downgrade Step 7B while executor freeze facts exist")
    for table in (
        "step7b_freeze_manifests",
        "step7b_threshold_decisions",
        "step7b_pilots",
    ):
        if connection.execute(
            sa.text(f"SELECT EXISTS (SELECT 1 FROM {RUNTIME}.{table} LIMIT 1)")
        ).scalar_one():
            raise RuntimeError("cannot downgrade Step 7B while pilot or freeze facts exist")
    op.execute(
        f"DROP TRIGGER IF EXISTS step7b_query_completion_integrity "
        f"ON {RUNTIME}.query_executor_runs"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_step7b_query_completion()"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS step7b_compiler_completion_integrity "
        f"ON {RUNTIME}.run_states"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_step7b_compiler_completion()"
    )
    _drop_check_constraint_matching(
        schema=RUNTIME,
        table="program_dry_run_assessments",
        required_fragments=(
            "temporal_evidence_contributed",
            "temporal_operator_used",
        ),
        contract="temporal_contribution_requires_operator",
    )
    _drop_check_constraint_matching(
        schema=RUNTIME,
        table="program_dry_run_assessments",
        required_fragments=("context_evidence_contributed", "context_operator_used"),
        contract="context_contribution_requires_operator",
    )
    _drop_check_constraint_matching(
        schema=RUNTIME,
        table="program_dry_run_assessments",
        required_fragments=(
            "expected_group_count",
            "expected_evidence_recall",
            "evidence_sufficient",
        ),
        contract="expected_recall_sufficiency",
    )
    _replace_guard_fragment(
        procedure=f"{RUNTIME}.enforce_query_executor_completion()",
        source=f"{RUNTIME}.verified_qbank_routing_completion",
        target=f"{RUNTIME}.verified_qbank_routing_v2",
        expected_count=2,
        contract="query_completion_routing_dispatch",
    )
    _extend_historical_guards(enable=False)
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.verified_qbank_routing_completion")
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.verified_qbank_routing_v3")
    for table in reversed(_NEW_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {RUNTIME}.{table}")
    duplicate_variant_jobs = connection.execute(
        sa.text(
            f"""SELECT EXISTS (
                SELECT 1 FROM {EVALUATION}.evaluation_jobs
                 GROUP BY evaluation_run_id, run_id, query_id, stage
                HAVING count(*) > 1
            )"""
        )
    ).scalar_one()
    if duplicate_variant_jobs:
        raise RuntimeError(
            "cannot downgrade Step 7B while variant-aware evaluation jobs exist"
        )
    op.drop_constraint(
        "uq_evaluation_jobs_run_query_variant_stage",
        "evaluation_jobs",
        schema=EVALUATION,
        type_="unique",
    )
    op.create_unique_constraint(
        "evaluation_jobs_evaluation_run_id_run_id_query_id_stage_key",
        "evaluation_jobs",
        ("evaluation_run_id", "run_id", "query_id", "stage"),
        schema=EVALUATION,
    )
    op.drop_column("evaluation_jobs", "answer_variant", schema=EVALUATION)
    _drop_check_constraint_matching(
        schema=RUNTIME,
        table="query_executor_freeze_manifests",
        required_fragments=(
            "query_executor_protocol_hash",
            "approved_routing_policy_hash",
            "approved_selection_hash",
        ),
        contract="approved_routing_identity",
    )
    op.drop_column(
        "query_executor_freeze_manifests",
        "approved_selection_hash",
        schema=RUNTIME,
    )
    op.drop_column(
        "query_executor_freeze_manifests",
        "approved_routing_policy_hash",
        schema=RUNTIME,
    )
    op.drop_column(
        "query_executor_freeze_manifests",
        "query_executor_protocol_hash",
        schema=RUNTIME,
    )
    op.drop_column(
        "program_dry_run_assessments",
        "temporal_evidence_contributed",
        schema=RUNTIME,
    )
    op.drop_column(
        "program_dry_run_assessments",
        "context_evidence_contributed",
        schema=RUNTIME,
    )
    _drop_check_constraint_matching(
        schema=RUNTIME,
        table="query_evidence_sufficiency_assessments",
        required_fragments=(
            "sufficiency_policy_version",
            "structural_requirement_success",
            "query_evidence_relevance",
            "answer_support",
            "relevance_threshold",
        ),
        contract="runtime_sufficiency_v2",
    )
    for name in (
        "relevance_threshold",
        "relevant_evidence_group_count",
        "answer_support",
        "query_evidence_relevance",
        "structural_requirement_success",
    ):
        op.drop_column("query_evidence_sufficiency_assessments", name, schema=RUNTIME)
    op.create_check_constraint(
        "server_derived",
        "query_evidence_sufficiency_assessments",
        "sufficient = (technical_success AND nonempty_evidence "
        "AND required_capabilities <@ actual_capabilities "
        "AND required_modalities <@ actual_modalities "
        "AND independent_group_count >= minimum_evidence_groups "
        "AND (NOT raw_image_required OR raw_image_reached))",
        schema=RUNTIME,
    )
    _drop_check_constraint_matching(
        schema=RUNTIME,
        table="qbank_match_fused_candidates",
        required_fragments=(
            "base_score",
            "base_reliable_score",
            "reranker_score",
            "route_score",
            "reranker_match_level",
        ),
        contract="step7b_score_values",
    )
    for name in (
        "reranker_match_level",
        "route_score",
        "reranker_score",
        "base_reliable_score",
        "base_score",
        "available_weight_sum",
        "multimodal_normalized_score",
        "text_normalized_score",
        "lexical_normalized_score",
    ):
        op.drop_column("qbank_match_fused_candidates", name, schema=RUNTIME)
    _drop_check_constraint_matching(
        schema=RUNTIME,
        table="qbank_questions",
        required_fragments=("answer_type", "'entity'", "'refusal'"),
        contract="answer_type_values",
    )
    op.drop_column("qbank_questions", "answer_type", schema=RUNTIME)
    print("STEP7B_MIGRATION state=downgrade_ready revision=20260730_0023", flush=True)
