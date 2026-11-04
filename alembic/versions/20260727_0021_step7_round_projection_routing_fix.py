# ruff: noqa: E501
"""Make Step 7 routing an exact per-round projection.

Revision ID: 20260727_0021
Revises: 20260727_0020
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260727_0021"
down_revision: str | None = "20260727_0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _replace_function_fragment(
    *,
    procedure: str,
    source: str,
    target: str,
    contract: str,
) -> None:
    source_literal = _sql_literal(source)
    target_literal = _sql_literal(target)
    contract_literal = _sql_literal(contract)
    op.execute(
        f"""DO $$
        DECLARE function_definition text; occurrence_count integer;
        BEGIN
          SELECT pg_get_functiondef('{procedure}'::regprocedure)
            INTO function_definition;
          IF function_definition IS NULL THEN
            RAISE EXCEPTION 'missing Step 7 persistence guard: %',
              {contract_literal};
          END IF;
          occurrence_count := (
            length(function_definition)
            - length(replace(function_definition, {source_literal}, ''))
          ) / length({source_literal});
          IF occurrence_count IS DISTINCT FROM 1 THEN
            RAISE EXCEPTION 'unexpected Step 7 persistence guard definition: %',
              {contract_literal};
          END IF;
          function_definition := replace(
            function_definition,
            {source_literal},
            {target_literal}
          );
          EXECUTE function_definition;
        END $$"""
    )


def _create_round_exact_view() -> None:
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.verified_qbank_routing_v2")
    op.execute(
        f"""CREATE VIEW {RUNTIME}.verified_qbank_routing_v2 AS
        SELECT manifests.routing_projection_id,
               round_members.run_id AS compiler_run_id,
               manifests.snapshot_id,
               snapshots.dataset,
               questions.conversation_id,
               questions.created_after_session AS available_after_session,
               questions.created_after_session AS cutoff_session_id,
               round_members.qbank_id,
               questions.generation_rank,
               round_members.route_class,
               questions.question_type,
               questions.canonical_question,
               COALESCE((SELECT array_agg(p.paraphrase ORDER BY p.ordinal)
                           FROM {RUNTIME}.qbank_paraphrases p
                          WHERE p.qbank_id = questions.qbank_id
                            AND p.run_id = questions.run_id), ARRAY[]::text[]) AS paraphrases,
               questions.required_modalities,
               COALESCE(round_members.passed_at_round,
                        round_manifests.round_id) AS accepted_round,
               round_members.effective_plan_id AS plan_id,
               programs.program_hash,
               programs.operator_set_version,
               requirements.requirement_contract_id,
               requirements.content_hash AS requirement_contract_hash,
               requirements.required_modalities AS contract_required_modalities,
               requirements.required_capabilities,
               requirements.allowed_capabilities,
               requirements.minimum_evidence_groups,
               requirements.fallback_required,
               requirements.refusal_only,
               manifests.lexical_collection_id,
               manifests.text_collection_id,
               manifests.multimodal_collection_id,
               manifests.active_round,
               manifests.decision_hash,
               COALESCE(
                 active_members.ordinal,
                 row_number() OVER (
                   PARTITION BY manifests.routing_projection_id,
                                round_manifests.round_id
                   ORDER BY round_members.qbank_id
                 ) - 1
               ) AS ordinal,
               round_manifests.round_id AS projection_round
          FROM {RUNTIME}.routing_projection_manifests manifests
          JOIN {RUNTIME}.qbank_round_projection_manifests round_manifests
            ON round_manifests.run_id = manifests.run_id
           AND round_manifests.round_id BETWEEN 0 AND manifests.max_round
          JOIN {RUNTIME}.qbank_round_projection_members round_members
            ON round_members.projection_id = round_manifests.projection_id
           AND round_members.run_id = round_manifests.run_id
           AND round_members.outcome IN ('accepted','carried_forward')
           AND round_members.effective_plan_id IS NOT NULL
          LEFT JOIN {RUNTIME}.routing_projection_members active_members
            ON active_members.routing_projection_id = manifests.routing_projection_id
           AND active_members.run_id = manifests.run_id
           AND active_members.qbank_id = round_members.qbank_id
           AND round_manifests.round_id = manifests.active_round
          JOIN {RUNTIME}.qbank_questions questions
            ON questions.qbank_id = round_members.qbank_id
           AND questions.run_id = round_members.run_id
          JOIN {RUNTIME}.memory_programs programs
            ON programs.plan_id = round_members.effective_plan_id
           AND programs.run_id = round_members.run_id
          JOIN {RUNTIME}.planner_requirement_contracts requirements
            ON requirements.qbank_id = round_members.qbank_id
           AND requirements.run_id = round_members.run_id
          JOIN {RUNTIME}.dataset_snapshots snapshots
            ON snapshots.snapshot_id = manifests.snapshot_id
         WHERE EXISTS (
               SELECT 1
                 FROM {RUNTIME}.qbank_index_manifests indexes
                WHERE indexes.compiler_run_id = round_members.run_id
                  AND indexes.status = 'ready'
         )"""
    )
    op.execute(f"ALTER VIEW {RUNTIME}.verified_qbank_routing_v2 OWNER TO pmmc_owner")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_v2 TO pmmc_runtime")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_v2 TO pmmc_evaluator")


def _create_legacy_view() -> None:
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.verified_qbank_routing_v2")
    op.execute(
        f"""CREATE VIEW {RUNTIME}.verified_qbank_routing_v2 AS
        SELECT manifests.routing_projection_id,
               accepted.run_id AS compiler_run_id,
               manifests.snapshot_id,
               snapshots.dataset,
               questions.conversation_id,
               questions.created_after_session AS available_after_session,
               questions.created_after_session AS cutoff_session_id,
               accepted.qbank_id,
               questions.generation_rank,
               accepted.route_class,
               questions.question_type,
               questions.canonical_question,
               COALESCE((SELECT array_agg(p.paraphrase ORDER BY p.ordinal)
                           FROM {RUNTIME}.qbank_paraphrases p
                          WHERE p.qbank_id = questions.qbank_id
                            AND p.run_id = questions.run_id), ARRAY[]::text[]) AS paraphrases,
               questions.required_modalities,
               accepted.accepted_round,
               accepted.effective_plan_id AS plan_id,
               programs.program_hash,
               programs.operator_set_version,
               requirements.requirement_contract_id,
               requirements.content_hash AS requirement_contract_hash,
               requirements.required_modalities AS contract_required_modalities,
               requirements.required_capabilities,
               requirements.allowed_capabilities,
               requirements.minimum_evidence_groups,
               requirements.fallback_required,
               requirements.refusal_only,
               manifests.lexical_collection_id,
               manifests.text_collection_id,
               manifests.multimodal_collection_id,
               manifests.active_round,
               manifests.decision_hash,
               COALESCE(active_members.ordinal, accepted.stable_ordinal) AS ordinal
          FROM {RUNTIME}.routing_projection_manifests manifests
          JOIN LATERAL (
               SELECT DISTINCT ON (round_members.qbank_id)
                      round_members.run_id,
                      round_members.qbank_id,
                      round_members.route_class,
                      round_members.effective_plan_id,
                      COALESCE(round_members.passed_at_round,
                               round_manifests.round_id) AS accepted_round,
                      row_number() OVER (
                          ORDER BY round_members.qbank_id
                      ) - 1 AS stable_ordinal
                 FROM {RUNTIME}.qbank_round_projection_members round_members
                 JOIN {RUNTIME}.qbank_round_projection_manifests round_manifests
                   ON round_manifests.projection_id = round_members.projection_id
                  AND round_manifests.run_id = round_members.run_id
                WHERE round_members.run_id = manifests.run_id
                  AND round_members.outcome IN ('accepted','carried_forward')
                  AND round_members.effective_plan_id IS NOT NULL
                ORDER BY round_members.qbank_id,
                         COALESCE(round_members.passed_at_round,
                                  round_manifests.round_id),
                         round_manifests.round_id
          ) accepted ON TRUE
          LEFT JOIN {RUNTIME}.routing_projection_members active_members
            ON active_members.routing_projection_id = manifests.routing_projection_id
           AND active_members.run_id = manifests.run_id
           AND active_members.qbank_id = accepted.qbank_id
          JOIN {RUNTIME}.qbank_questions questions
            ON questions.qbank_id = accepted.qbank_id
           AND questions.run_id = accepted.run_id
          JOIN {RUNTIME}.memory_programs programs
            ON programs.plan_id = accepted.effective_plan_id
           AND programs.run_id = accepted.run_id
          JOIN {RUNTIME}.planner_requirement_contracts requirements
            ON requirements.qbank_id = accepted.qbank_id
           AND requirements.run_id = accepted.run_id
          JOIN {RUNTIME}.dataset_snapshots snapshots
            ON snapshots.snapshot_id = manifests.snapshot_id
         WHERE EXISTS (
               SELECT 1
                 FROM {RUNTIME}.qbank_index_manifests indexes
                WHERE indexes.compiler_run_id = accepted.run_id
                  AND indexes.status = 'ready'
         )"""
    )
    op.execute(f"ALTER VIEW {RUNTIME}.verified_qbank_routing_v2 OWNER TO pmmc_owner")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_v2 TO pmmc_runtime")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_v2 TO pmmc_evaluator")


def _replace_round_guards(*, exact: bool) -> None:
    replacements = (
        (
            f"{RUNTIME}.enforce_query_executor_completion()",
            "r.accepted_round <= NEW.active_round",
            "r.projection_round = NEW.active_round",
            "query_route_exact_round",
        ),
        (
            f"{RUNTIME}.enforce_query_executor_completion()",
            "r.accepted_round <= b.round_id",
            "r.projection_round = b.round_id",
            "query_program_exact_round",
        ),
        (
            f"{RUNTIME}.enforce_step7_compiler_completion()",
            "AND v.compiler_run_id=m.run_id AND v.qbank_id=m.qbank_id",
            "AND v.compiler_run_id=m.run_id AND v.qbank_id=m.qbank_id\n"
            "                  AND v.plan_id=m.plan_id\n"
            "                  AND v.projection_round=routing.active_round",
            "compiler_active_projection_exact_round",
        ),
    )
    for procedure, legacy, current, contract in replacements:
        _replace_function_fragment(
            procedure=procedure,
            source=legacy if exact else current,
            target=current if exact else legacy,
            contract=contract,
        )


def upgrade() -> None:
    print(
        "STEP7_ROUTING_MIGRATION state=start source=accepted_history target=exact_round",
        flush=True,
    )
    _create_round_exact_view()
    _replace_round_guards(exact=True)
    print(
        "STEP7_ROUTING_MIGRATION state=ready projection_rounds=4 active_round=2",
        flush=True,
    )


def downgrade() -> None:
    _replace_round_guards(exact=False)
    _create_legacy_view()

