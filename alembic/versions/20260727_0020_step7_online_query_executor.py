# ruff: noqa: E501
"""Add Step 7 verified routing and online query-executor facts.

Revision ID: 20260727_0020
Revises: 20260726_0019
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260727_0020"
down_revision: str | None = "20260726_0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"

_TABLES = (
    "query_executor_runs",
    "query_executor_stage_jobs",
    "query_embedding_batches",
    "query_embeddings",
    "qbank_match_batches",
    "qbank_match_channel_candidates",
    "qbank_match_fused_candidates",
    "query_route_decisions",
    "query_program_bindings",
    "query_evidence_sufficiency_assessments",
    "query_fallback_decisions",
    "query_answer_selections",
    "pmmc_fallback_run_links",
    "pmmc_fallback_query_links",
    "query_executor_freeze_manifests",
)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _replace_function_fragment(
    *,
    procedure: str,
    source: str,
    target: str,
    contract: str,
) -> None:
    """Replace one audited fragment in an existing PL/pgSQL guard."""

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


def _replace_questioner_persistence_contracts(*, enable_step7: bool) -> None:
    """Extend the Step 6D guards to the Step 7 compiler protocol."""

    generation_v06 = "compiler_version = '0.6.0'"
    generation_v07 = "compiler_version IN ('0.6.0', '0.7.0')"
    verifier_prompt_v06 = (
        "WHEN '0.6.0' THEN 'pmmc-questioner-verifier-v4'"
    )
    verifier_prompt_v07 = (
        verifier_prompt_v06
        + "\n                WHEN '0.7.0' THEN 'pmmc-questioner-verifier-v5'"
    )
    verifier_scope_v06 = "compiler_version IN ('0.5.0', '0.6.0')"
    verifier_scope_v07 = "compiler_version IN ('0.5.0', '0.6.0', '0.7.0')"
    embedding_v06 = "compiler.compiler_schema_version = '0.6.0'"
    embedding_v07 = (
        "compiler.compiler_schema_version IN ('0.6.0', '0.7.0')"
    )

    if enable_step7:
        replacements = (
            (
                f"{RUNTIME}.validate_questioner_fact_scope()",
                generation_v06,
                generation_v07,
                "questioner_generation_terminal_rejection",
            ),
            (
                f"{RUNTIME}.enforce_questioner_verifier_scope_v3()",
                verifier_prompt_v06,
                verifier_prompt_v07,
                "questioner_verifier_prompt",
            ),
            (
                f"{RUNTIME}.enforce_questioner_verifier_scope_v3()",
                verifier_scope_v06,
                verifier_scope_v07,
                "questioner_verifier_single_candidate_scope",
            ),
            (
                f"{RUNTIME}.enforce_questioner_verifier_scope_v3()",
                generation_v06,
                generation_v07,
                "questioner_verifier_terminal_rejection",
            ),
            (
                f"{RUNTIME}.validate_compiler_embedding_lineage_v4()",
                embedding_v06,
                embedding_v07,
                "program_query_embedding_lineage",
            ),
        )
    else:
        replacements = (
            (
                f"{RUNTIME}.validate_questioner_fact_scope()",
                generation_v07,
                generation_v06,
                "questioner_generation_terminal_rejection",
            ),
            (
                f"{RUNTIME}.enforce_questioner_verifier_scope_v3()",
                verifier_prompt_v07,
                verifier_prompt_v06,
                "questioner_verifier_prompt",
            ),
            (
                f"{RUNTIME}.enforce_questioner_verifier_scope_v3()",
                verifier_scope_v07,
                verifier_scope_v06,
                "questioner_verifier_single_candidate_scope",
            ),
            (
                f"{RUNTIME}.enforce_questioner_verifier_scope_v3()",
                generation_v07,
                generation_v06,
                "questioner_verifier_terminal_rejection",
            ),
            (
                f"{RUNTIME}.validate_compiler_embedding_lineage_v4()",
                embedding_v07,
                embedding_v06,
                "program_query_embedding_lineage",
            ),
        )

    for procedure, source, target, contract in replacements:
        _replace_function_fragment(
            procedure=procedure,
            source=source,
            target=target,
            contract=contract,
        )


def _replace_common_compiler_completion_contract(*, enable_step7: bool) -> None:
    """Delegate v0.6/v0.7 stage accounting to their versioned guards.

    The common guard predates native Questioner envelopes. Its legacy
    ``questioner`` count compares raw provider candidates with persisted
    schema-valid questions, which is intentionally no longer a 1:1 mapping.
    """

    step6c2_versions = "('0.4.0', '0.5.0')"
    verified_qbank_versions = "('0.4.0', '0.5.0', '0.6.0', '0.7.0')"
    source = verified_qbank_versions if not enable_step7 else step6c2_versions
    target = step6c2_versions if not enable_step7 else verified_qbank_versions
    _replace_function_fragment(
        procedure=f"{RUNTIME}.enforce_compiler_completion_v3()",
        source=f"compiler.compiler_schema_version NOT IN {source}",
        target=f"compiler.compiler_schema_version NOT IN {target}",
        contract="verified_qbank_stage_accounting_delegation",
    )


def _persistence_contract_event(*, state: str, protocol: str) -> None:
    print(
        "STEP7_MIGRATION "
        f"phase=questioner_persistence_contract state={state} protocol={protocol}",
        flush=True,
    )


def _upgrade_existing_contracts() -> None:
    op.execute(
        f"ALTER TABLE {EVALUATION}.evaluation_runs "
        "ADD COLUMN source_kind varchar(32) NOT NULL DEFAULT 'baseline_suite'"
    )
    op.execute(
        f"ALTER TABLE {EVALUATION}.evaluation_runs "
        "ADD CONSTRAINT evaluation_runs_source_kind_values "
        "CHECK (source_kind IN ('baseline_suite','pmmc_query_run'))"
    )
    op.execute(
        f"ALTER TABLE {EVALUATION}.evaluation_targets "
        "ADD COLUMN source_kind varchar(32) NOT NULL DEFAULT 'baseline_suite'"
    )
    op.execute(
        f"ALTER TABLE {EVALUATION}.evaluation_targets "
        "ADD CONSTRAINT evaluation_targets_source_kind_values "
        "CHECK (source_kind IN ('baseline_suite','pmmc_query_run'))"
    )
    op.execute(
        f"ALTER TABLE {RUNTIME}.planner_requirement_contracts "
        "ADD COLUMN allowed_capabilities text[]"
    )
    op.execute(
        f"UPDATE {RUNTIME}.planner_requirement_contracts "
        "SET allowed_capabilities = required_capabilities"
    )
    op.execute(
        f"ALTER TABLE {RUNTIME}.planner_requirement_contracts "
        "ALTER COLUMN allowed_capabilities SET NOT NULL"
    )
    op.execute(
        f"ALTER TABLE {RUNTIME}.r0_input_manifests "
        "ADD COLUMN protocol_integrity_issues text[] NOT NULL DEFAULT ARRAY[]::text[], "
        "ADD COLUMN candidate_scientific_rejections jsonb NOT NULL DEFAULT '{}'::jsonb, "
        "ADD COLUMN coverage_zero_units integer NOT NULL DEFAULT 0, "
        "ADD COLUMN downstream_viability boolean NOT NULL DEFAULT false"
    )
    op.execute(
        f"UPDATE {RUNTIME}.r0_input_manifests "
        "SET protocol_integrity_issues = issue_codes"
    )
    op.execute(
        f"ALTER TABLE {RUNTIME}.doubter_assessment_dimensions "
        "ADD COLUMN verdict varchar(32)"
    )
    op.execute(
        f"UPDATE {RUNTIME}.doubter_assessment_dimensions "
        "SET verdict = CASE WHEN passed THEN 'pass' ELSE 'fail' END"
    )
    op.execute(
        f"ALTER TABLE {RUNTIME}.doubter_assessment_dimensions "
        "ALTER COLUMN verdict SET NOT NULL, "
        "ADD CONSTRAINT ck_doubter_assessment_dimensions_verdict_values "
        "CHECK (verdict IN ('pass','fail','not_applicable'))"
    )


def _create_tables() -> None:
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_executor_runs (
            query_executor_run_id varchar(128) PRIMARY KEY,
            source_compiler_run_id varchar(128) NOT NULL,
            routing_projection_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            dataset varchar(32) NOT NULL CHECK (dataset IN ('memlens','memgallery')),
            schema_version varchar(32) NOT NULL,
            config_hash varchar(128) NOT NULL,
            active_round integer NOT NULL CHECK (active_round = 2),
            max_round integer NOT NULL CHECK (max_round = 3),
            question_budget integer NOT NULL CHECK (question_budget = 8),
            selected_query_count integer NOT NULL CHECK (selected_query_count > 0),
            pilot_projection boolean NOT NULL,
            paper_result boolean NOT NULL CHECK (NOT (paper_result AND pilot_projection)),
            status varchar(16) NOT NULL CHECK (status IN ('created','running','completed','failed','cancelled')),
            matched_count integer NOT NULL DEFAULT 0 CHECK (matched_count >= 0),
            executed_count integer NOT NULL DEFAULT 0 CHECK (executed_count >= 0),
            pmmc_answer_count integer NOT NULL DEFAULT 0 CHECK (pmmc_answer_count >= 0),
            fallback_answer_count integer NOT NULL DEFAULT 0 CHECK (fallback_answer_count >= 0),
            selected_answer_count integer NOT NULL DEFAULT 0 CHECK (selected_answer_count >= 0),
            failed_count integer NOT NULL DEFAULT 0 CHECK (failed_count >= 0),
            started_at timestamptz,
            heartbeat_at timestamptz,
            completed_at timestamptz,
            error_type varchar(256),
            error_summary text,
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (query_executor_run_id, snapshot_id),
            FOREIGN KEY (query_executor_run_id, snapshot_id)
              REFERENCES {RUNTIME}.run_snapshot_links(run_id, snapshot_id),
            FOREIGN KEY (source_compiler_run_id, snapshot_id)
              REFERENCES {RUNTIME}.compiler_runs(run_id, snapshot_id),
            FOREIGN KEY (routing_projection_id, source_compiler_run_id)
              REFERENCES {RUNTIME}.routing_projection_manifests(routing_projection_id, run_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_executor_stage_jobs (
            query_stage_job_id varchar(128) PRIMARY KEY,
            query_executor_run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            stage varchar(32) NOT NULL CHECK (stage IN ('embed','match','execute','answer','fallback','select')),
            round_id integer CHECK (round_id IS NULL OR round_id BETWEEN 0 AND 3),
            status varchar(16) NOT NULL CHECK (status IN ('pending','running','succeeded','failed','cancelled')),
            attempt_count integer NOT NULL CHECK (attempt_count >= 0),
            max_attempts integer NOT NULL CHECK (max_attempts > 0),
            claimed_by varchar(256), lease_expires_at timestamptz,
            available_at timestamptz NOT NULL, started_at timestamptz, finished_at timestamptz,
            error_type varchar(256), error_summary text,
            content_hash varchar(128) NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (query_executor_run_id, query_id, stage, round_id),
            FOREIGN KEY (query_executor_run_id, snapshot_id, query_id)
              REFERENCES {RUNTIME}.run_queries(run_id, snapshot_id, query_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_embedding_batches (
            query_embedding_batch_id varchar(128) PRIMARY KEY,
            query_executor_run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            model_call_id varchar(128) REFERENCES {RUNTIME}.model_calls(model_call_id),
            channel varchar(16) NOT NULL CHECK (channel IN ('text','multimodal')),
            batch_ordinal integer NOT NULL CHECK (batch_ordinal >= 0),
            query_ids text[] NOT NULL, expected_count integer NOT NULL CHECK (expected_count > 0),
            committed_count integer NOT NULL,
            content_hash varchar(128) NOT NULL, created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (query_embedding_batch_id, query_executor_run_id),
            UNIQUE (query_executor_run_id, channel, batch_ordinal),
            CHECK (committed_count IN (0, expected_count)),
            CHECK (
              (model_call_id IS NOT NULL
               AND NOT COALESCE((metadata->>'coverage_projection')::boolean, false))
              OR
              (model_call_id IS NULL
               AND COALESCE((metadata->>'coverage_projection')::boolean, false)
               AND COALESCE(metadata->>'source_evaluation_model_call_id', '') <> '')
            ),
            FOREIGN KEY (query_executor_run_id, snapshot_id)
              REFERENCES {RUNTIME}.query_executor_runs(query_executor_run_id, snapshot_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_embeddings (
            query_embedding_id varchar(128) PRIMARY KEY,
            query_embedding_batch_id varchar(128) NOT NULL,
            query_executor_run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, query_id varchar(256) NOT NULL,
            channel varchar(16) NOT NULL CHECK (channel IN ('text','multimodal')),
            embedding_id varchar(128) NOT NULL REFERENCES {RUNTIME}.embeddings(embedding_id),
            collection_id varchar(128) NOT NULL REFERENCES {RUNTIME}.embedding_collections(collection_id),
            vector_dimension integer NOT NULL CHECK (vector_dimension = 4096),
            input_hash varchar(128) NOT NULL, content_hash varchar(128) NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (query_executor_run_id, query_id, channel),
            FOREIGN KEY (query_embedding_batch_id, query_executor_run_id)
              REFERENCES {RUNTIME}.query_embedding_batches(query_embedding_batch_id, query_executor_run_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.qbank_match_batches (
            match_batch_id varchar(128) PRIMARY KEY,
            query_executor_run_id varchar(128) NOT NULL,
            source_compiler_run_id varchar(128) NOT NULL,
            routing_projection_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, query_id varchar(256) NOT NULL,
            conversation_id varchar(256) NOT NULL, cutoff_session_id varchar(256) NOT NULL,
            active_round integer NOT NULL CHECK (active_round BETWEEN 0 AND 3),
            question_budget integer NOT NULL CHECK (question_budget IN (1,2,4,8)),
            requested_top_k integer NOT NULL CHECK (requested_top_k = 50),
            channel_count integer NOT NULL CHECK (channel_count IN (2,3)),
            fused_count integer NOT NULL CHECK (fused_count BETWEEN 0 AND 100),
            config_hash varchar(128) NOT NULL, content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (query_executor_run_id, query_id, active_round, question_budget),
            FOREIGN KEY (query_executor_run_id, snapshot_id, query_id)
              REFERENCES {RUNTIME}.run_queries(run_id, snapshot_id, query_id),
            FOREIGN KEY (routing_projection_id, source_compiler_run_id)
              REFERENCES {RUNTIME}.routing_projection_manifests(routing_projection_id, run_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.qbank_match_channel_candidates (
            channel_candidate_id varchar(128) PRIMARY KEY,
            match_batch_id varchar(128) NOT NULL REFERENCES {RUNTIME}.qbank_match_batches(match_batch_id),
            query_id varchar(256) NOT NULL,
            qbank_id varchar(128) NOT NULL REFERENCES {RUNTIME}.qbank_questions(qbank_id),
            route_class varchar(16) NOT NULL CHECK (route_class IN ('answerable','control')),
            channel varchar(16) NOT NULL CHECK (channel IN ('lexical','text','multimodal')),
            rank integer NOT NULL CHECK (rank BETWEEN 1 AND 50), raw_score double precision NOT NULL,
            generation_rank integer NOT NULL CHECK (generation_rank BETWEEN 1 AND 8),
            accepted_round integer NOT NULL CHECK (accepted_round BETWEEN 0 AND 3),
            content_hash varchar(128) NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (match_batch_id, route_class, channel, rank),
            UNIQUE (match_batch_id, channel, qbank_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.qbank_match_fused_candidates (
            fused_candidate_id varchar(128) PRIMARY KEY,
            match_batch_id varchar(128) NOT NULL REFERENCES {RUNTIME}.qbank_match_batches(match_batch_id),
            query_id varchar(256) NOT NULL,
            qbank_id varchar(128) NOT NULL REFERENCES {RUNTIME}.qbank_questions(qbank_id),
            plan_id varchar(128) NOT NULL REFERENCES {RUNTIME}.memory_programs(plan_id),
            route_class varchar(16) NOT NULL CHECK (route_class IN ('answerable','control')),
            rank integer NOT NULL CHECK (rank BETWEEN 1 AND 50), lexical_rank integer, text_rank integer, multimodal_rank integer,
            raw_rrf double precision NOT NULL CHECK (raw_rrf >= 0),
            confidence double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
            modality_compatible boolean NOT NULL, content_hash varchar(128) NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (match_batch_id, route_class, rank), UNIQUE (match_batch_id, qbank_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_route_decisions (
            route_decision_id varchar(128) PRIMARY KEY,
            query_executor_run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.query_executor_runs(query_executor_run_id),
            match_batch_id varchar(128) NOT NULL REFERENCES {RUNTIME}.qbank_match_batches(match_batch_id),
            query_id varchar(256) NOT NULL,
            outcome varchar(32) NOT NULL CHECK (outcome IN ('program','fallback','control','terminal_failure')),
            selected_qbank_id varchar(128),
            selected_plan_id varchar(128) REFERENCES {RUNTIME}.memory_programs(plan_id),
            confidence double precision CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
            top1_top2_gap double precision CHECK (top1_top2_gap IS NULL OR top1_top2_gap BETWEEN 0 AND 1),
            match_threshold double precision NOT NULL, gap_threshold double precision NOT NULL,
            fallback_reason varchar(64), counterfactual_qbank_id varchar(128),
            content_hash varchar(128) NOT NULL, created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (query_executor_run_id, query_id),
            CHECK ((outcome = 'program') = (selected_qbank_id IS NOT NULL AND selected_plan_id IS NOT NULL)),
            CHECK ((outcome IN ('fallback','control')) = (fallback_reason IS NOT NULL))
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_program_bindings (
            query_program_binding_id varchar(128) PRIMARY KEY,
            query_executor_run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.query_executor_runs(query_executor_run_id),
            query_id varchar(256) NOT NULL,
            route_decision_id varchar(128) NOT NULL REFERENCES {RUNTIME}.query_route_decisions(route_decision_id),
            routing_projection_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL, plan_id varchar(128) NOT NULL REFERENCES {RUNTIME}.memory_programs(plan_id),
            round_id integer NOT NULL CHECK (round_id BETWEEN 0 AND 3),
            program_hash varchar(128) NOT NULL, operator_set_version varchar(128) NOT NULL,
            requirement_contract_hash varchar(128) NOT NULL, counterfactual boolean NOT NULL,
            content_hash varchar(128) NOT NULL, created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (query_executor_run_id, query_id, round_id),
            UNIQUE (query_program_binding_id, query_executor_run_id, query_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_evidence_sufficiency_assessments (
            sufficiency_assessment_id varchar(128) PRIMARY KEY,
            query_executor_run_id varchar(128) NOT NULL, query_id varchar(256) NOT NULL,
            query_program_binding_id varchar(128) NOT NULL UNIQUE REFERENCES {RUNTIME}.query_program_bindings(query_program_binding_id),
            execution_id varchar(128) NOT NULL REFERENCES {RUNTIME}.program_executions(execution_id),
            technical_success boolean NOT NULL, nonempty_evidence boolean NOT NULL,
            actual_capabilities text[] NOT NULL, required_capabilities text[] NOT NULL,
            actual_modalities text[] NOT NULL, required_modalities text[] NOT NULL,
            independent_group_count integer NOT NULL CHECK (independent_group_count >= 0),
            minimum_evidence_groups integer NOT NULL CHECK (minimum_evidence_groups > 0),
            raw_image_reached boolean NOT NULL, raw_image_required boolean NOT NULL,
            sufficient boolean NOT NULL, failure_reason varchar(64),
            content_hash varchar(128) NOT NULL, created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            FOREIGN KEY (query_program_binding_id, query_executor_run_id, query_id)
              REFERENCES {RUNTIME}.query_program_bindings(
                query_program_binding_id, query_executor_run_id, query_id),
            FOREIGN KEY (execution_id, query_executor_run_id)
              REFERENCES {RUNTIME}.program_executions(execution_id, run_id),
            CHECK (sufficient = (technical_success AND nonempty_evidence
              AND required_capabilities <@ actual_capabilities
              AND required_modalities <@ actual_modalities
              AND independent_group_count >= minimum_evidence_groups
              AND (NOT raw_image_required OR raw_image_reached))),
            CHECK (sufficient <> (failure_reason IS NOT NULL))
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_fallback_decisions (
            fallback_decision_id varchar(128) PRIMARY KEY,
            query_executor_run_id varchar(128) NOT NULL, query_id varchar(256) NOT NULL,
            fallback_run_id varchar(128) NOT NULL, fallback_query_id varchar(256) NOT NULL,
            counterfactual_completed boolean NOT NULL, active_fallback boolean NOT NULL,
            reason varchar(64), fallback_answer_id varchar(128), error_type varchar(256),
            content_hash varchar(128) NOT NULL, created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (query_executor_run_id, query_id),
            FOREIGN KEY (query_executor_run_id)
              REFERENCES {RUNTIME}.query_executor_runs(query_executor_run_id),
            FOREIGN KEY (fallback_answer_id, fallback_run_id, fallback_query_id)
              REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id),
            CHECK ((counterfactual_completed AND fallback_answer_id IS NOT NULL AND error_type IS NULL)
                OR (NOT counterfactual_completed AND fallback_answer_id IS NULL AND error_type IS NOT NULL)),
            CHECK (active_fallback = (reason IS NOT NULL))
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_answer_selections (
            answer_selection_id varchar(128) PRIMARY KEY,
            query_executor_run_id varchar(128) NOT NULL, query_id varchar(256) NOT NULL,
            active_variant varchar(64) NOT NULL CHECK (active_variant IN
              ('pmmc_r0','pmmc_r1','pmmc_r2','pmmc_r3','native_mm_counterfactual','no_fallback_counterfactual')),
            selected_answer_id varchar(128), pmmc_answer_id varchar(128), fallback_answer_id varchar(128),
            terminal_failure boolean NOT NULL, failure_type varchar(256),
            content_hash varchar(128) NOT NULL, created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (query_executor_run_id, query_id),
            FOREIGN KEY (query_executor_run_id)
              REFERENCES {RUNTIME}.query_executor_runs(query_executor_run_id),
            FOREIGN KEY (selected_answer_id, query_executor_run_id, query_id)
              REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id),
            FOREIGN KEY (pmmc_answer_id, query_executor_run_id, query_id)
              REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id),
            CHECK ((terminal_failure AND selected_answer_id IS NULL AND failure_type IS NOT NULL)
                OR (NOT terminal_failure AND selected_answer_id IS NOT NULL AND failure_type IS NULL))
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.pmmc_fallback_run_links (
            query_executor_run_id varchar(128) PRIMARY KEY REFERENCES {RUNTIME}.query_executor_runs(query_executor_run_id),
            fallback_run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.run_manifests(run_id),
            snapshot_id varchar(128) NOT NULL, method_id varchar(128) NOT NULL,
            content_hash varchar(128) NOT NULL, created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (query_executor_run_id, fallback_run_id),
            FOREIGN KEY (fallback_run_id, snapshot_id)
              REFERENCES {RUNTIME}.run_snapshot_links(run_id, snapshot_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.pmmc_fallback_query_links (
            query_executor_run_id varchar(128) NOT NULL, query_id varchar(256) NOT NULL,
            fallback_run_id varchar(128) NOT NULL, fallback_query_id varchar(256) NOT NULL,
            fallback_answer_id varchar(128) NOT NULL, content_hash varchar(128) NOT NULL, metadata jsonb NOT NULL,
            PRIMARY KEY (query_executor_run_id, query_id),
            FOREIGN KEY (query_executor_run_id, fallback_run_id)
              REFERENCES {RUNTIME}.pmmc_fallback_run_links(query_executor_run_id, fallback_run_id),
            FOREIGN KEY (fallback_answer_id, fallback_run_id, fallback_query_id)
              REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.query_executor_freeze_manifests (
            freeze_manifest_id varchar(128) PRIMARY KEY, schema_version varchar(32) NOT NULL,
            verified_qbank_freeze_manifest_id varchar(128) NOT NULL,
            query_executor_config_hash varchar(128) NOT NULL, answer_prompt_hash varchar(128) NOT NULL,
            fallback_policy_version varchar(128) NOT NULL,
            active_round integer NOT NULL CHECK (active_round = 2), question_budget integer NOT NULL CHECK (question_budget = 8),
            match_threshold double precision NOT NULL, gap_threshold double precision NOT NULL,
            threshold_grid double precision[] NOT NULL, gap_grid double precision[] NOT NULL,
            pilot_content_hash varchar(128) NOT NULL, approved_by varchar(256) NOT NULL,
            approved_at timestamptz NOT NULL, decision_content_hash varchar(128) NOT NULL,
            content_hash varchar(128) NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (verified_qbank_freeze_manifest_id, query_executor_config_hash)
        )"""
    )
    op.execute(
        f"CREATE INDEX ix_query_executor_stage_jobs_claim ON {RUNTIME}.query_executor_stage_jobs "
        "(query_executor_run_id, stage, status, available_at)"
    )
    op.execute(
        f"CREATE INDEX ix_qbank_match_channel_candidates_batch_channel_score "
        f"ON {RUNTIME}.qbank_match_channel_candidates (match_batch_id, channel, rank)"
    )


def _create_runtime_view() -> None:
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
         )
        """
    )


def _create_completion_guard() -> None:
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_query_executor_completion()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE bad integer;
        BEGIN
          IF NEW.status != 'completed' OR OLD.status = 'completed' THEN RETURN NEW; END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.run_queries q
           WHERE q.run_id = NEW.query_executor_run_id
             AND q.snapshot_id = NEW.snapshot_id;
          IF bad != NEW.selected_query_count THEN
            RAISE EXCEPTION 'query executor run-query denominator mismatch' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.query_executor_stage_jobs j
           WHERE j.query_executor_run_id = NEW.query_executor_run_id;
          IF bad != NEW.selected_query_count THEN
            RAISE EXCEPTION 'query executor stage-job denominator mismatch' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.query_executor_stage_jobs j
           WHERE j.query_executor_run_id = NEW.query_executor_run_id
             AND (j.stage != 'select' OR j.round_id IS NOT NULL OR j.status != 'succeeded');
          IF bad != 0 THEN
            RAISE EXCEPTION 'completed query executor has nonterminal stage jobs' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM (
            SELECT rq.query_id
              FROM {RUNTIME}.run_queries rq
              LEFT JOIN {RUNTIME}.query_embeddings e
                ON e.query_executor_run_id = rq.run_id
               AND e.snapshot_id = rq.snapshot_id
               AND e.query_id = rq.query_id
             WHERE rq.run_id = NEW.query_executor_run_id
             GROUP BY rq.query_id
            HAVING count(e.query_embedding_id) != 2
                OR count(*) FILTER (WHERE e.channel = 'text') != 1
                OR count(*) FILTER (WHERE e.channel = 'multimodal') != 1
          ) incomplete_embeddings;
          IF bad != 0 THEN
            RAISE EXCEPTION 'query embedding denominator or channels are incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.query_embeddings e
            JOIN {RUNTIME}.embedding_collections c ON c.collection_id=e.collection_id
           WHERE e.query_executor_run_id=NEW.query_executor_run_id
             AND (e.vector_dimension != 4096 OR c.status != 'ready'
                  OR c.dimension != 4096 OR c.snapshot_id != NEW.snapshot_id);
          IF bad != 0 THEN
            RAISE EXCEPTION 'query embeddings reference an unready or incompatible collection' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.query_embedding_batches b
           WHERE b.query_executor_run_id=NEW.query_executor_run_id
             AND (b.committed_count != b.expected_count OR b.expected_count != (
               SELECT count(*) FROM {RUNTIME}.query_embeddings e
                WHERE e.query_embedding_batch_id=b.query_embedding_batch_id
                  AND e.query_executor_run_id=b.query_executor_run_id));
          IF bad != 0 THEN
            RAISE EXCEPTION 'query embedding batch fan-out is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM (
            SELECT rq.query_id
              FROM {RUNTIME}.run_queries rq
              LEFT JOIN {RUNTIME}.qbank_match_batches b
                ON b.query_executor_run_id=rq.run_id AND b.query_id=rq.query_id
             WHERE rq.run_id=NEW.query_executor_run_id
             GROUP BY rq.query_id
            HAVING count(b.match_batch_id) != 16
                OR count(DISTINCT (b.question_budget, b.active_round)) != 16
          ) incomplete_match_grid;
          IF bad != 0 THEN
            RAISE EXCEPTION 'query matcher does not contain the complete b-by-r grid' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.qbank_match_batches b
           WHERE b.query_executor_run_id=NEW.query_executor_run_id
             AND (b.fused_count != (
                    SELECT count(*)
                      FROM {RUNTIME}.qbank_match_fused_candidates f
                     WHERE f.match_batch_id=b.match_batch_id)
                  OR EXISTS (
                    SELECT 1
                      FROM {RUNTIME}.qbank_match_fused_candidates f
                     WHERE f.match_batch_id=b.match_batch_id
                     GROUP BY f.route_class
                    HAVING min(f.rank) != 1
                        OR max(f.rank) != count(*)
                        OR count(*) > b.requested_top_k));
          IF bad != 0 THEN
            RAISE EXCEPTION 'query fused top-50 accounting is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.qbank_match_batches b
           WHERE b.query_executor_run_id=NEW.query_executor_run_id
             AND EXISTS (
               SELECT 1
                 FROM {RUNTIME}.qbank_match_channel_candidates c
                WHERE c.match_batch_id=b.match_batch_id
                GROUP BY c.route_class, c.channel
               HAVING min(c.rank) != 1
                   OR max(c.rank) != count(*)
                   OR count(*) > b.requested_top_k);
          IF bad != 0 THEN
            RAISE EXCEPTION 'query channel top-50 accounting is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT NEW.selected_query_count - count(*) INTO bad
            FROM {RUNTIME}.query_route_decisions d
           WHERE d.query_executor_run_id = NEW.query_executor_run_id;
          IF bad != 0 THEN RAISE EXCEPTION 'query route denominator incomplete' USING ERRCODE='55000'; END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.query_route_decisions d
            JOIN {RUNTIME}.qbank_match_batches b ON b.match_batch_id=d.match_batch_id
           WHERE d.query_executor_run_id=NEW.query_executor_run_id
             AND (b.query_executor_run_id != d.query_executor_run_id
                  OR b.query_id != d.query_id
                  OR b.question_budget != NEW.question_budget
                  OR b.active_round != NEW.active_round
                  OR b.routing_projection_id != NEW.routing_projection_id);
          IF bad != 0 THEN
            RAISE EXCEPTION 'active route does not bind the frozen b8-r2 match batch' USING ERRCODE='55000';
          END IF;
          SELECT NEW.selected_query_count - count(*) INTO bad
            FROM {RUNTIME}.query_fallback_decisions f
           WHERE f.query_executor_run_id = NEW.query_executor_run_id;
          IF bad != 0 THEN RAISE EXCEPTION 'fallback counterfactual denominator incomplete' USING ERRCODE='55000'; END IF;
          SELECT NEW.selected_query_count - count(*) INTO bad
            FROM {RUNTIME}.query_answer_selections s
           WHERE s.query_executor_run_id = NEW.query_executor_run_id;
          IF bad != 0 THEN RAISE EXCEPTION 'answer selection denominator incomplete' USING ERRCODE='55000'; END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.query_route_decisions d
            JOIN {RUNTIME}.run_queries rq
              ON rq.run_id=d.query_executor_run_id AND rq.query_id=d.query_id
            JOIN {RUNTIME}.benchmark_queries q
              ON q.snapshot_id=rq.snapshot_id AND q.query_id=rq.query_id
           WHERE d.query_executor_run_id = NEW.query_executor_run_id
             AND COALESCE(d.selected_qbank_id, d.counterfactual_qbank_id) IS NOT NULL
             AND NOT EXISTS (
               SELECT 1
                 FROM {RUNTIME}.verified_qbank_routing_v2 r
                 JOIN {RUNTIME}.benchmark_sessions available
                   ON available.snapshot_id=r.snapshot_id
                  AND available.conversation_id=r.conversation_id
                  AND available.session_id=r.available_after_session
                 JOIN {RUNTIME}.benchmark_sessions cutoff
                   ON cutoff.snapshot_id=q.snapshot_id
                  AND cutoff.conversation_id=q.conversation_id
                  AND cutoff.session_id=q.available_after_session
                WHERE r.routing_projection_id = NEW.routing_projection_id
                  AND r.compiler_run_id = NEW.source_compiler_run_id
                  AND r.snapshot_id=NEW.snapshot_id
                  AND r.dataset=NEW.dataset
                  AND r.conversation_id=q.conversation_id
                  AND available.session_order <= cutoff.session_order
                  AND r.generation_rank <= NEW.question_budget
                  AND r.accepted_round <= NEW.active_round
                  AND r.qbank_id = COALESCE(d.selected_qbank_id, d.counterfactual_qbank_id)
                  AND (d.selected_plan_id IS NULL OR r.plan_id=d.selected_plan_id)
             );
          IF bad != 0 THEN RAISE EXCEPTION 'query route escaped verified projection' USING ERRCODE='55000'; END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.query_program_bindings b
            JOIN {RUNTIME}.run_queries rq
              ON rq.run_id=b.query_executor_run_id AND rq.query_id=b.query_id
            JOIN {RUNTIME}.benchmark_queries q
              ON q.snapshot_id=rq.snapshot_id AND q.query_id=rq.query_id
           WHERE b.query_executor_run_id=NEW.query_executor_run_id
             AND NOT EXISTS (
               SELECT 1
                 FROM {RUNTIME}.verified_qbank_routing_v2 r
                 JOIN {RUNTIME}.benchmark_sessions available
                   ON available.snapshot_id=r.snapshot_id
                  AND available.conversation_id=r.conversation_id
                  AND available.session_id=r.available_after_session
                 JOIN {RUNTIME}.benchmark_sessions cutoff
                   ON cutoff.snapshot_id=q.snapshot_id
                  AND cutoff.conversation_id=q.conversation_id
                  AND cutoff.session_id=q.available_after_session
                WHERE r.routing_projection_id=b.routing_projection_id
                  AND r.compiler_run_id=NEW.source_compiler_run_id
                  AND r.snapshot_id=NEW.snapshot_id
                  AND r.dataset=NEW.dataset
                  AND r.conversation_id=q.conversation_id
                  AND r.qbank_id=b.qbank_id AND r.plan_id=b.plan_id
                  AND r.program_hash=b.program_hash
                  AND r.accepted_round <= b.round_id
                  AND available.session_order <= cutoff.session_order
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'query program binding escaped its frozen round projection' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.query_program_bindings b
           WHERE b.query_executor_run_id=NEW.query_executor_run_id
             AND NOT EXISTS (
               SELECT 1
                 FROM {RUNTIME}.query_evidence_sufficiency_assessments a
                 JOIN {RUNTIME}.program_executions e ON e.execution_id=a.execution_id
                WHERE a.query_program_binding_id=b.query_program_binding_id
                  AND a.query_executor_run_id=b.query_executor_run_id
                  AND a.query_id=b.query_id
                  AND e.run_id=b.query_executor_run_id
                  AND e.query_id=b.query_id
                  AND e.plan_id=b.plan_id
                  AND e.purpose='query_time'
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'query program execution or sufficiency facts are incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.pmmc_fallback_run_links l
           WHERE l.query_executor_run_id=NEW.query_executor_run_id
             AND l.snapshot_id=NEW.snapshot_id
             AND l.method_id='native_multimodal_rag';
          IF bad != 1 THEN
            RAISE EXCEPTION 'query run requires exactly one Native-MM child run link' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.query_fallback_decisions f
           WHERE f.query_executor_run_id=NEW.query_executor_run_id
             AND ((f.counterfactual_completed AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.pmmc_fallback_query_links l
                     WHERE l.query_executor_run_id=f.query_executor_run_id
                       AND l.query_id=f.query_id
                       AND l.fallback_run_id=f.fallback_run_id
                       AND l.fallback_query_id=f.fallback_query_id
                       AND l.fallback_answer_id=f.fallback_answer_id))
               OR (NOT f.counterfactual_completed AND EXISTS (
                    SELECT 1 FROM {RUNTIME}.pmmc_fallback_query_links l
                     WHERE l.query_executor_run_id=f.query_executor_run_id
                       AND l.query_id=f.query_id)));
          IF bad != 0 THEN
            RAISE EXCEPTION 'Native-MM query/answer lineage is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.query_answer_selections s
           WHERE s.query_executor_run_id=NEW.query_executor_run_id
             AND EXISTS (
               SELECT 1
                 FROM {RUNTIME}.query_fallback_decisions f
                WHERE f.query_executor_run_id=s.query_executor_run_id
                  AND f.query_id=s.query_id
                  AND s.fallback_answer_id IS DISTINCT FROM f.fallback_answer_id
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'query answer selection fallback lineage is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad
            FROM {RUNTIME}.query_answer_selections s
           WHERE s.query_executor_run_id=NEW.query_executor_run_id
             AND ((NOT s.terminal_failure AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.final_answers a
                     WHERE a.answer_id=s.selected_answer_id
                       AND a.run_id=s.query_executor_run_id
                       AND a.query_id=s.query_id
                       AND a.method='pmmc' AND a.answer_variant='active_final'))
               OR (s.pmmc_answer_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.final_answers a
                     WHERE a.answer_id=s.pmmc_answer_id
                       AND a.run_id=s.query_executor_run_id
                       AND a.query_id=s.query_id
                       AND a.method='pmmc' AND a.answer_variant='pmmc_r2')));
          IF bad != 0 THEN
            RAISE EXCEPTION 'active or PMMC source answer lineage is incomplete' USING ERRCODE='55000';
          END IF;
          IF NEW.matched_count != (
               SELECT count(*) FROM {RUNTIME}.query_route_decisions d
                WHERE d.query_executor_run_id=NEW.query_executor_run_id)
             OR NEW.executed_count != (
               SELECT count(DISTINCT b.query_id) FROM {RUNTIME}.query_program_bindings b
                WHERE b.query_executor_run_id=NEW.query_executor_run_id)
             OR NEW.pmmc_answer_count != (
               SELECT count(*) FROM {RUNTIME}.query_answer_selections s
                WHERE s.query_executor_run_id=NEW.query_executor_run_id
                  AND s.pmmc_answer_id IS NOT NULL)
             OR NEW.fallback_answer_count != (
               SELECT count(*) FROM {RUNTIME}.query_fallback_decisions f
                WHERE f.query_executor_run_id=NEW.query_executor_run_id)
             OR NEW.selected_answer_count != (
               SELECT count(*) FROM {RUNTIME}.query_answer_selections s
                WHERE s.query_executor_run_id=NEW.query_executor_run_id
                  AND NOT s.terminal_failure)
             OR NEW.failed_count != (
               SELECT count(*) FROM {RUNTIME}.query_answer_selections s
                WHERE s.query_executor_run_id=NEW.query_executor_run_id
                  AND s.terminal_failure)
             OR NEW.fallback_answer_count != NEW.selected_query_count
             OR NEW.selected_answer_count + NEW.failed_count != NEW.selected_query_count
             OR NEW.completed_at IS NULL THEN
            RAISE EXCEPTION 'query executor terminal accounting incomplete' USING ERRCODE='55000';
          END IF;
          RETURN NEW;
        END $$"""
    )
    op.execute(
        f"CREATE TRIGGER query_executor_completion_integrity BEFORE UPDATE ON "
        f"{RUNTIME}.query_executor_runs FOR EACH ROW EXECUTE FUNCTION "
        f"{RUNTIME}.enforce_query_executor_completion()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_query_executor_completion() FROM PUBLIC"
    )


def _create_compiler_completion_guard() -> None:
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_step7_compiler_completion()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE compiler {RUNTIME}.compiler_runs%ROWTYPE; bad integer;
                eligible integer; routing {RUNTIME}.routing_projection_manifests%ROWTYPE;
        BEGIN
          IF NEW.status != 'completed' OR OLD.status = 'completed' THEN RETURN NEW; END IF;
          SELECT * INTO compiler FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
          IF NOT FOUND OR compiler.compiler_schema_version != '0.7.0' THEN RETURN NEW; END IF;
          IF compiler.target_stage != 'verified_qbank'
             OR compiler.operator_set_version != 'pmmc-operators-v5'
             OR compiler.questioner_prompt_version != 'pmmc-questioner-v3.4'
             OR compiler.planner_prompt_version != 'pmmc-planner-v1.3'
             OR compiler.doubter_prompt_version != 'pmmc-doubter-v1.1'
             OR compiler.routing_projection_version != 'verified-qbank-projection-v2'
             OR compiler.active_round != 2 OR compiler.max_round != 3 THEN
            RAISE EXCEPTION 'Step 7 compiler identity is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.r0_input_manifests r
           WHERE r.run_id = NEW.run_id AND r.hard_gate_passed
             AND cardinality(r.protocol_integrity_issues) = 0
             AND r.downstream_viability;
          IF bad != 1 THEN
            RAISE EXCEPTION 'Step 7 requires one viable R0 v2 manifest' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO eligible FROM {RUNTIME}.qbank_questions q
           WHERE q.run_id = NEW.run_id AND EXISTS (
             SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results g
              WHERE g.run_id=q.run_id AND g.qbank_id=q.qbank_id
                AND g.status='eligible_for_planning'
           ) AND EXISTS (
             SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications v
              WHERE v.run_id=q.run_id AND v.qbank_id=q.qbank_id
                AND v.verification_round=q.candidate_version AND v.verdict='pass'
           );
          IF eligible = 0 THEN
            RAISE EXCEPTION 'Step 7 compiler has no native eligible candidates' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_manifests p
           WHERE p.run_id=NEW.run_id AND p.round_id BETWEEN 0 AND 3;
          IF bad != 4 THEN
            RAISE EXCEPTION 'Step 7 requires r0-r3 projections' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.qbank_round_projection_manifests p
           WHERE p.run_id=NEW.run_id AND p.member_count != eligible;
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7 round projection denominator mismatch' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.qbank_questions q
           WHERE q.run_id=NEW.run_id AND EXISTS (
             SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results g
              WHERE g.run_id=q.run_id AND g.qbank_id=q.qbank_id
                AND g.status='eligible_for_planning'
           ) AND NOT EXISTS (
             SELECT 1 FROM {RUNTIME}.doubter_assessments d
              WHERE d.run_id=q.run_id AND d.qbank_id=q.qbank_id AND d.round_id=0
           );
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7 Doubter r0 denominator is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT * INTO routing FROM {RUNTIME}.routing_projection_manifests
           WHERE run_id=NEW.run_id;
          IF NOT FOUND OR routing.active_round != 2 OR routing.max_round != 3
             OR routing.member_count != (
               SELECT count(*) FROM {RUNTIME}.routing_projection_members m
                WHERE m.routing_projection_id=routing.routing_projection_id
             ) THEN
            RAISE EXCEPTION 'Step 7 active-r2 routing projection is incomplete' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.qbank_decisions d
           WHERE d.run_id=NEW.run_id;
          IF bad != eligible THEN
            RAISE EXCEPTION 'Step 7 active-r2 decisions do not close the denominator' USING ERRCODE='55000';
          END IF;
          SELECT count(*) INTO bad FROM {RUNTIME}.routing_projection_members m
           WHERE m.run_id=NEW.run_id AND NOT EXISTS (
               SELECT 1 FROM {RUNTIME}.verified_qbank_routing_v2 v
                WHERE v.routing_projection_id=m.routing_projection_id
                  AND v.compiler_run_id=m.run_id AND v.qbank_id=m.qbank_id
             );
          IF bad != 0 THEN
            RAISE EXCEPTION 'Step 7 routing contains unsafe or unresolved members' USING ERRCODE='55000';
          END IF;
          RETURN NEW;
        END $$"""
    )
    op.execute(
        f"""CREATE TRIGGER step7_compiler_completion_integrity
        BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW
        EXECUTE FUNCTION {RUNTIME}.enforce_step7_compiler_completion()"""
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_step7_compiler_completion() FROM PUBLIC"
    )


def _grant_permissions() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {RUNTIME}.{table} OWNER TO pmmc_owner")
        op.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {RUNTIME}.{table} TO pmmc_runtime"
        )
        op.execute(f"GRANT SELECT ON {RUNTIME}.{table} TO pmmc_evaluator")
    op.execute(f"ALTER VIEW {RUNTIME}.verified_qbank_routing_v2 OWNER TO pmmc_owner")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_v2 TO pmmc_runtime")
    op.execute(f"GRANT SELECT ON {RUNTIME}.verified_qbank_routing_v2 TO pmmc_evaluator")


def upgrade() -> None:
    _persistence_contract_event(state="start", protocol="compiler-0.7.0")
    _replace_questioner_persistence_contracts(enable_step7=True)
    _replace_common_compiler_completion_contract(enable_step7=True)
    _persistence_contract_event(state="done", protocol="compiler-0.7.0")
    _upgrade_existing_contracts()
    _create_tables()
    _create_runtime_view()
    _create_completion_guard()
    _create_compiler_completion_guard()
    _grant_permissions()


def downgrade() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS step7_compiler_completion_integrity ON {RUNTIME}.run_states"
    )
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_step7_compiler_completion()")
    op.execute(
        f"DROP TRIGGER IF EXISTS query_executor_completion_integrity ON {RUNTIME}.query_executor_runs"
    )
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_query_executor_completion()")
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.verified_qbank_routing_v2")
    for table in reversed(_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {RUNTIME}.{table}")
    op.execute(
        f"ALTER TABLE {EVALUATION}.evaluation_targets "
        "DROP CONSTRAINT IF EXISTS evaluation_targets_source_kind_values"
    )
    op.execute(
        f"ALTER TABLE {EVALUATION}.evaluation_targets DROP COLUMN source_kind"
    )
    op.execute(
        f"ALTER TABLE {EVALUATION}.evaluation_runs "
        "DROP CONSTRAINT IF EXISTS evaluation_runs_source_kind_values"
    )
    op.execute(f"ALTER TABLE {EVALUATION}.evaluation_runs DROP COLUMN source_kind")
    op.execute(
        f"ALTER TABLE {RUNTIME}.doubter_assessment_dimensions "
        "DROP CONSTRAINT IF EXISTS ck_doubter_assessment_dimensions_verdict_values, "
        "DROP COLUMN IF EXISTS verdict"
    )
    op.execute(
        f"ALTER TABLE {RUNTIME}.r0_input_manifests "
        "DROP COLUMN IF EXISTS downstream_viability, "
        "DROP COLUMN IF EXISTS coverage_zero_units, "
        "DROP COLUMN IF EXISTS candidate_scientific_rejections, "
        "DROP COLUMN IF EXISTS protocol_integrity_issues"
    )
    op.execute(
        f"ALTER TABLE {RUNTIME}.planner_requirement_contracts "
        "DROP COLUMN IF EXISTS allowed_capabilities"
    )
    _persistence_contract_event(state="start", protocol="compiler-0.6.0")
    _replace_common_compiler_completion_contract(enable_step7=False)
    _replace_questioner_persistence_contracts(enable_step7=False)
    _persistence_contract_event(state="done", protocol="compiler-0.6.0")
