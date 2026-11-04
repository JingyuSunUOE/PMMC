# ruff: noqa: E501
"""Add PMMC compiler units, typed programs, QBank facts, and operator traces.

Revision ID: 20260718_0007
Revises: 20260716_0006
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260718_0007"
down_revision: str | None = "20260716_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"


def upgrade() -> None:
    _extend_canonical_search()
    _create_compiler_tables()
    _extend_shared_facts()
    _create_lineage_functions_and_triggers()
    _create_routing_and_summary_views()
    _grant_permissions()


def _extend_canonical_search() -> None:
    op.add_column(
        "memory_units",
        sa.Column(
            "search_document",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple', coalesce(text, '') || ' ' || coalesce(caption, ''))",
                persisted=True,
            ),
            nullable=False,
        ),
        schema=RUNTIME,
    )
    op.create_index(
        "ix_memory_units_search_document",
        "memory_units",
        ["search_document"],
        schema=RUNTIME,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_memory_units_entity_mentions",
        "memory_units",
        ["entity_mentions"],
        schema=RUNTIME,
        postgresql_using="gin",
    )


def _create_compiler_tables() -> None:
    statements = (
        f"""CREATE TABLE {RUNTIME}.compiler_runs (
            run_id varchar(128) PRIMARY KEY REFERENCES {RUNTIME}.run_manifests(run_id),
            snapshot_id varchar(128) NOT NULL,
            compiler_schema_version varchar(32) NOT NULL,
            target_stage varchar(32) NOT NULL CHECK (target_stage IN ('questions','programs','verified_qbank')),
            operator_set_version varchar(128) NOT NULL,
            operator_registry_hash varchar(128) NOT NULL,
            compiler_config_hash varchar(128) NOT NULL,
            unit_count integer NOT NULL CHECK (unit_count >= 0),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (run_id, snapshot_id),
            FOREIGN KEY (run_id, snapshot_id)
                REFERENCES {RUNTIME}.run_snapshot_links(run_id, snapshot_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.compiler_units (
            compiler_unit_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            conversation_id varchar(256) NOT NULL,
            available_after_session varchar(256) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            builder_version varchar(128) NOT NULL,
            input_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (compiler_unit_id, run_id),
            UNIQUE (compiler_unit_id, run_id, snapshot_id),
            UNIQUE (compiler_unit_id, run_id, snapshot_id, conversation_id),
            UNIQUE (run_id, ordinal),
            FOREIGN KEY (run_id, snapshot_id)
                REFERENCES {RUNTIME}.compiler_runs(run_id, snapshot_id),
            FOREIGN KEY (snapshot_id, conversation_id, available_after_session)
                REFERENCES {RUNTIME}.benchmark_sessions(snapshot_id, conversation_id, session_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.compiler_unit_memories (
            compiler_unit_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            conversation_id varchar(256) NOT NULL,
            memory_id varchar(256) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            role varchar(16) NOT NULL CHECK (role IN ('anchor','support')),
            metadata jsonb NOT NULL,
            PRIMARY KEY (compiler_unit_id, memory_id),
            UNIQUE (compiler_unit_id, ordinal),
            FOREIGN KEY (compiler_unit_id, run_id, snapshot_id, conversation_id)
                REFERENCES {RUNTIME}.compiler_units(
                    compiler_unit_id, run_id, snapshot_id, conversation_id
                ),
            FOREIGN KEY (snapshot_id, conversation_id, memory_id)
                REFERENCES {RUNTIME}.memory_units(snapshot_id, conversation_id, memory_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.compiler_unit_stage_results (
            stage_result_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            stage varchar(32) NOT NULL CHECK (stage IN ('questioner','planner','dry_run','doubter','decision')),
            attempt_number integer NOT NULL CHECK (attempt_number > 0),
            status varchar(16) NOT NULL CHECK (status IN ('success','failed','skipped','partial')),
            produced_count integer NOT NULL CHECK (produced_count >= 0),
            accepted_count integer NOT NULL CHECK (accepted_count >= 0),
            rejected_count integer NOT NULL CHECK (rejected_count >= 0),
            result_hash varchar(128) NOT NULL,
            started_at timestamptz NOT NULL,
            completed_at timestamptz NOT NULL,
            elapsed_ms double precision NOT NULL CHECK (elapsed_ms >= 0),
            error_type varchar(256),
            error_summary text,
            metadata jsonb NOT NULL,
            UNIQUE (compiler_unit_id, stage, attempt_number),
            FOREIGN KEY (compiler_unit_id, run_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            CHECK (accepted_count + rejected_count <= produced_count),
            CHECK ((status = 'failed' AND error_type IS NOT NULL) OR (status != 'failed' AND error_type IS NULL))
        )""",
        f"""CREATE TABLE {RUNTIME}.qbank_questions (
            qbank_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            producer_model_call_id varchar(128) REFERENCES {RUNTIME}.model_calls(model_call_id),
            question_hash varchar(128) NOT NULL,
            conversation_id varchar(256) NOT NULL,
            created_after_session varchar(256) NOT NULL,
            canonical_question text NOT NULL,
            question_type varchar(64) NOT NULL,
            expected_answer text,
            required_modalities text[] NOT NULL,
            evidence_requirement text[] NOT NULL,
            hop_count integer NOT NULL CHECK (hop_count > 0),
            generation_rank integer NOT NULL CHECK (generation_rank > 0),
            questioner_confidence double precision NOT NULL CHECK (questioner_confidence BETWEEN 0 AND 1),
            accepted boolean,
            rejection_reason text,
            search_document tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(canonical_question, ''))
            ) STORED,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (qbank_id, run_id),
            UNIQUE (compiler_unit_id, generation_rank),
            FOREIGN KEY (compiler_unit_id, run_id, snapshot_id, conversation_id)
                REFERENCES {RUNTIME}.compiler_units(
                    compiler_unit_id, run_id, snapshot_id, conversation_id
                ),
            FOREIGN KEY (snapshot_id, conversation_id, created_after_session)
                REFERENCES {RUNTIME}.benchmark_sessions(snapshot_id, conversation_id, session_id),
            CHECK (question_type IN ('textual_fact','visual_detail','image_text_binding','visual_search','multi_hop','temporal','knowledge_update','conflict_detection','answer_refusal','personal_preference','test_time_learning','visual_reasoning','multi_entity_reasoning','unknown')),
            CHECK (cardinality(required_modalities) > 0),
            CHECK (required_modalities <@ ARRAY['text','caption','image','raw_image','text_embedding','image_embedding','joint_mm_embedding','embedding','multimodal']::text[]),
            CHECK ((accepted = false AND rejection_reason IS NOT NULL) OR (accepted IS DISTINCT FROM false AND rejection_reason IS NULL))
        )""",
        f"""CREATE TABLE {RUNTIME}.qbank_paraphrases (
            paraphrase_id varchar(128) PRIMARY KEY,
            qbank_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            paraphrase text NOT NULL,
            paraphrase_hash varchar(128) NOT NULL,
            search_document tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(paraphrase, ''))
            ) STORED,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (qbank_id, ordinal),
            UNIQUE (qbank_id, paraphrase_hash),
            FOREIGN KEY (qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.qbank_evidence_refs (
            evidence_ref_id varchar(128) PRIMARY KEY,
            qbank_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            role varchar(16) NOT NULL CHECK (role IN ('source','expected')),
            evidence_id varchar(256) NOT NULL,
            evidence_type varchar(32) NOT NULL CHECK (evidence_type IN ('memory','text','caption','image','raw_image','image_text_pair','session','query','unknown')),
            memory_id varchar(256),
            image_id varchar(256),
            session_id varchar(256),
            caption_id varchar(256),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            metadata jsonb NOT NULL,
            FOREIGN KEY (qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            UNIQUE (qbank_id, role, ordinal),
            CHECK (num_nonnulls(memory_id, image_id, session_id, caption_id) = 1)
        )""",
        f"""CREATE TABLE {RUNTIME}.memory_programs (
            plan_id varchar(128) PRIMARY KEY,
            qbank_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            producer_model_call_id varchar(128) REFERENCES {RUNTIME}.model_calls(model_call_id),
            operator_set_version varchar(128) NOT NULL,
            round_id integer NOT NULL CHECK (round_id >= 0),
            fallback_operator varchar(128),
            required_modalities text[] NOT NULL,
            planner_rationale_short text NOT NULL,
            planner_confidence double precision NOT NULL CHECK (planner_confidence BETWEEN 0 AND 1),
            program_hash varchar(128) NOT NULL,
            accepted boolean NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (plan_id, run_id),
            UNIQUE (qbank_id, round_id),
            FOREIGN KEY (qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            FOREIGN KEY (compiler_unit_id, run_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            CHECK (cardinality(required_modalities) > 0),
            CHECK (required_modalities <@ ARRAY['text','caption','image','raw_image','text_embedding','image_embedding','joint_mm_embedding','embedding','multimodal']::text[])
        )""",
        f"""CREATE TABLE {RUNTIME}.program_steps (
            plan_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            step_id integer NOT NULL CHECK (step_id >= 0),
            op_name varchar(128) NOT NULL,
            args jsonb NOT NULL,
            expected_output_type varchar(128) NOT NULL,
            metadata jsonb NOT NULL,
            PRIMARY KEY (plan_id, step_id),
            FOREIGN KEY (plan_id, run_id)
                REFERENCES {RUNTIME}.memory_programs(plan_id, run_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.program_expected_evidence_refs (
            evidence_ref_id varchar(128) PRIMARY KEY,
            plan_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            step_id integer,
            evidence_id varchar(256) NOT NULL,
            evidence_type varchar(32) NOT NULL,
            memory_id varchar(256),
            image_id varchar(256),
            session_id varchar(256),
            caption_id varchar(256),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            metadata jsonb NOT NULL,
            FOREIGN KEY (plan_id, run_id)
                REFERENCES {RUNTIME}.memory_programs(plan_id, run_id),
            FOREIGN KEY (plan_id, step_id)
                REFERENCES {RUNTIME}.program_steps(plan_id, step_id),
            UNIQUE NULLS NOT DISTINCT (plan_id, step_id, ordinal),
            CHECK (evidence_type IN ('memory','text','caption','image','raw_image','image_text_pair','session','query','unknown')),
            CHECK (num_nonnulls(memory_id, image_id, session_id, caption_id) = 1)
        )""",
        f"""CREATE TABLE {RUNTIME}.program_executions (
            execution_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.run_manifests(run_id),
            source_compiler_run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            plan_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128),
            query_id varchar(256),
            purpose varchar(32) NOT NULL CHECK (purpose IN ('compiler_dry_run','query_time')),
            attempt_number integer NOT NULL CHECK (attempt_number > 0),
            status varchar(16) NOT NULL CHECK (status IN ('success','failed','skipped','partial')),
            bindings_hash varchar(128) NOT NULL,
            control_action varchar(16) NOT NULL CHECK (control_action IN ('continue','fallback','abstain')),
            started_at timestamptz NOT NULL,
            completed_at timestamptz NOT NULL,
            elapsed_ms double precision NOT NULL CHECK (elapsed_ms >= 0),
            error_type varchar(256),
            error_summary text,
            metadata jsonb NOT NULL,
            UNIQUE (execution_id, run_id),
            FOREIGN KEY (plan_id, source_compiler_run_id)
                REFERENCES {RUNTIME}.memory_programs(plan_id, run_id),
            FOREIGN KEY (compiler_unit_id, source_compiler_run_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            FOREIGN KEY (run_id, snapshot_id)
                REFERENCES {RUNTIME}.run_snapshot_links(run_id, snapshot_id),
            FOREIGN KEY (run_id, snapshot_id, query_id)
                REFERENCES {RUNTIME}.run_queries(run_id, snapshot_id, query_id),
            CHECK ((purpose = 'compiler_dry_run' AND compiler_unit_id IS NOT NULL AND query_id IS NULL) OR (purpose = 'query_time' AND query_id IS NOT NULL)),
            CHECK ((status = 'failed' AND error_type IS NOT NULL) OR (status != 'failed' AND error_type IS NULL))
        )""",
        f"""CREATE TABLE {RUNTIME}.program_execution_steps (
            execution_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128),
            query_id varchar(256),
            plan_id varchar(128) NOT NULL,
            step_id integer NOT NULL CHECK (step_id >= 0),
            op_name varchar(128) NOT NULL,
            input_hash varchar(128) NOT NULL,
            output_hash varchar(128) NOT NULL,
            output jsonb NOT NULL,
            success boolean NOT NULL,
            error text,
            latency_ms double precision CHECK (latency_ms IS NULL OR latency_ms >= 0),
            status varchar(16) NOT NULL CHECK (status IN ('success','failed','skipped','partial')),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            PRIMARY KEY (execution_id, step_id),
            FOREIGN KEY (execution_id, run_id)
                REFERENCES {RUNTIME}.program_executions(execution_id, run_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.program_execution_evidence_refs (
            evidence_ref_id varchar(128) PRIMARY KEY,
            execution_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            step_id integer NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            evidence_id varchar(256) NOT NULL,
            evidence_type varchar(32) NOT NULL,
            memory_id varchar(256),
            image_id varchar(256),
            session_id varchar(256),
            caption_id varchar(256),
            rank integer NOT NULL CHECK (rank > 0),
            score jsonb,
            metadata jsonb NOT NULL,
            UNIQUE (execution_id, rank),
            FOREIGN KEY (execution_id, run_id)
                REFERENCES {RUNTIME}.program_executions(execution_id, run_id),
            FOREIGN KEY (execution_id, step_id)
                REFERENCES {RUNTIME}.program_execution_steps(execution_id, step_id),
            CHECK (evidence_type IN ('memory','text','caption','image','raw_image','image_text_pair','session','query','unknown')),
            CHECK (num_nonnulls(memory_id, image_id, session_id, caption_id) = 1)
        )""",
        f"""CREATE TABLE {RUNTIME}.doubter_feedback (
            feedback_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            plan_id varchar(128) NOT NULL,
            program_execution_id varchar(128) REFERENCES {RUNTIME}.program_executions(execution_id),
            producer_model_call_id varchar(128) REFERENCES {RUNTIME}.model_calls(model_call_id),
            feedback_hash varchar(128) NOT NULL,
            round_id integer NOT NULL CHECK (round_id >= 0),
            verifier_pass boolean NOT NULL,
            verifier_score double precision NOT NULL CHECK (verifier_score BETWEEN 0 AND 1),
            evidence_sufficiency varchar(32) NOT NULL CHECK (evidence_sufficiency IN ('sufficient','partial','insufficient')),
            failure_reason text,
            suggested_revision text[] NOT NULL,
            missing_modalities text[] NOT NULL,
            should_abstain boolean NOT NULL,
            raw_response_asset_id varchar(128),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (qbank_id, round_id),
            FOREIGN KEY (plan_id, run_id)
                REFERENCES {RUNTIME}.memory_programs(plan_id, run_id),
            FOREIGN KEY (raw_response_asset_id, run_id)
                REFERENCES {RUNTIME}.run_artifacts(artifact_id, run_id),
            CHECK (verifier_pass OR failure_reason IS NOT NULL OR cardinality(suggested_revision) > 0)
        )""",
        f"""CREATE TABLE {RUNTIME}.qbank_decisions (
            decision_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            accepted boolean NOT NULL,
            final_plan_id varchar(128) REFERENCES {RUNTIME}.memory_programs(plan_id),
            final_feedback_id varchar(128) REFERENCES {RUNTIME}.doubter_feedback(feedback_id),
            reason_code varchar(128) NOT NULL,
            reason_summary text,
            verifier_pass boolean NOT NULL,
            verifier_score double precision CHECK (verifier_score BETWEEN 0 AND 1),
            decision_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (qbank_id),
            FOREIGN KEY (qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            CHECK ((accepted AND final_plan_id IS NOT NULL AND verifier_pass) OR (NOT accepted AND NOT verifier_pass))
        )""",
    )
    for statement in statements:
        op.execute(statement)

    op.create_index(
        "uq_compiler_unit_stage_success",
        "compiler_unit_stage_results",
        ["compiler_unit_id", "stage"],
        unique=True,
        schema=RUNTIME,
        postgresql_where=sa.text("status = 'success'"),
    )
    op.create_index(
        "ix_qbank_questions_search_document",
        "qbank_questions",
        ["search_document"],
        schema=RUNTIME,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_qbank_paraphrases_search_document",
        "qbank_paraphrases",
        ["search_document"],
        schema=RUNTIME,
        postgresql_using="gin",
    )


def _extend_shared_facts() -> None:
    op.add_column(
        "experiment_jobs",
        sa.Column("compiler_unit_id", sa.String(128)),
        schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_experiment_jobs_compiler_unit",
        "experiment_jobs",
        "compiler_units",
        ["compiler_unit_id", "run_id", "snapshot_id"],
        ["compiler_unit_id", "run_id", "snapshot_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.create_unique_constraint(
        "uq_experiment_jobs_compiler_unit_stage_variant",
        "experiment_jobs",
        ["run_id", "compiler_unit_id", "stage", "variant"],
        schema=RUNTIME,
    )
    op.create_check_constraint(
        "experiment_jobs_compiler_job_scope",
        "experiment_jobs",
        "(stage = 'compile' AND compiler_unit_id IS NOT NULL AND query_id IS NULL) OR (stage != 'compile' AND compiler_unit_id IS NULL)",
        schema=RUNTIME,
    )

    op.add_column(
        "model_calls",
        sa.Column("compiler_unit_id", sa.String(128)),
        schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_runtime_model_calls_compiler_unit",
        "model_calls",
        "compiler_units",
        ["compiler_unit_id", "run_id", "snapshot_id"],
        ["compiler_unit_id", "run_id", "snapshot_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.create_check_constraint(
        "model_calls_compiler_call_scope",
        "model_calls",
        "compiler_unit_id IS NULL OR "
        "(job_id IS NOT NULL AND stage IN ('questioner','planner','doubter'))",
        schema=RUNTIME,
    )

    for column in ("qbank_id", "paraphrase_id"):
        op.add_column(
            "embeddings", sa.Column(column, sa.String(128)), schema=RUNTIME
        )
    op.create_foreign_key(
        "fk_embeddings_qbank_question",
        "embeddings",
        "qbank_questions",
        ["qbank_id"],
        ["qbank_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_embeddings_qbank_paraphrase",
        "embeddings",
        "qbank_paraphrases",
        ["paraphrase_id"],
        ["paraphrase_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.create_check_constraint(
        "embeddings_one_typed_compiler_source",
        "embeddings",
        "num_nonnulls(retrieval_document_id, qbank_id, paraphrase_id) <= 1",
        schema=RUNTIME,
    )


def _create_lineage_functions_and_triggers() -> None:
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_compiler_run_manifest()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE manifest jsonb; manifest_method varchar(128);
        BEGIN
            SELECT manifests.manifest_json, manifests.method
              INTO manifest, manifest_method
              FROM {RUNTIME}.run_manifests manifests
             WHERE manifests.run_id = NEW.run_id;
            IF manifest IS NULL
               OR manifest_method != 'pmmc_compiler'
               OR COALESCE(jsonb_array_length(manifest -> 'dataset_snapshot_ids'), 0) != 1
               OR manifest -> 'dataset_snapshot_ids' ->> 0 != NEW.snapshot_id
               OR manifest ->> 'compiler_schema_version' != NEW.compiler_schema_version
               OR manifest ->> 'operator_set_version' != NEW.operator_set_version
               OR manifest ->> 'operator_registry_hash' != NEW.operator_registry_hash
               OR manifest ->> 'compiler_config_hash' != NEW.compiler_config_hash THEN
                RAISE EXCEPTION 'compiler run does not match its immutable run manifest' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_compiler_unit_memory_visibility()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE cutoff_order integer; memory_order integer;
        BEGIN
            SELECT session_order INTO cutoff_order
            FROM {RUNTIME}.benchmark_sessions sessions
            JOIN {RUNTIME}.compiler_units units
              ON units.snapshot_id = sessions.snapshot_id
             AND units.conversation_id = sessions.conversation_id
             AND units.available_after_session = sessions.session_id
            WHERE units.compiler_unit_id = NEW.compiler_unit_id;
            SELECT sessions.session_order INTO memory_order
            FROM {RUNTIME}.memory_units memories
            JOIN {RUNTIME}.benchmark_sessions sessions
              ON sessions.snapshot_id = memories.snapshot_id
             AND sessions.conversation_id = memories.conversation_id
             AND sessions.session_id = memories.available_after_session
            WHERE memories.snapshot_id = NEW.snapshot_id
              AND memories.conversation_id = NEW.conversation_id
              AND memories.memory_id = NEW.memory_id;
            IF cutoff_order IS NULL OR memory_order IS NULL OR memory_order > cutoff_order THEN
                RAISE EXCEPTION 'compiler unit memory violates prospective cutoff' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_compiler_evidence_visibility()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE cutoff_order integer; visible boolean; payload jsonb;
        BEGIN
            payload := to_jsonb(NEW);
            SELECT sessions.session_order INTO cutoff_order
            FROM {RUNTIME}.compiler_units units
            JOIN {RUNTIME}.benchmark_sessions sessions
              ON sessions.snapshot_id = units.snapshot_id
             AND sessions.conversation_id = units.conversation_id
             AND sessions.session_id = units.available_after_session
            WHERE units.compiler_unit_id = payload ->> 'compiler_unit_id'
              AND units.run_id = payload ->> 'run_id'
              AND units.snapshot_id = payload ->> 'snapshot_id';
            IF cutoff_order IS NULL THEN
                RAISE EXCEPTION 'compiler evidence has no matching compiler unit' USING ERRCODE = '23503';
            END IF;
            visible := false;
            IF payload ->> 'memory_id' IS NOT NULL THEN
                SELECT true INTO visible
                FROM {RUNTIME}.memory_units memories
                JOIN {RUNTIME}.benchmark_sessions sessions
                  ON sessions.snapshot_id = memories.snapshot_id
                 AND sessions.conversation_id = memories.conversation_id
                 AND sessions.session_id = memories.available_after_session
                JOIN {RUNTIME}.compiler_units units
                  ON units.compiler_unit_id = payload ->> 'compiler_unit_id'
                WHERE memories.snapshot_id = payload ->> 'snapshot_id'
                  AND memories.memory_id = payload ->> 'memory_id'
                  AND memories.conversation_id = units.conversation_id
                  AND sessions.session_order <= cutoff_order;
            ELSIF payload ->> 'image_id' IS NOT NULL THEN
                SELECT true INTO visible
                FROM {RUNTIME}.memory_unit_images links
                JOIN {RUNTIME}.memory_units memories
                  ON memories.snapshot_id = links.snapshot_id AND memories.memory_id = links.memory_id
                JOIN {RUNTIME}.benchmark_sessions sessions
                  ON sessions.snapshot_id = memories.snapshot_id
                 AND sessions.conversation_id = memories.conversation_id
                 AND sessions.session_id = memories.available_after_session
                JOIN {RUNTIME}.compiler_units units
                  ON units.compiler_unit_id = payload ->> 'compiler_unit_id'
                WHERE links.snapshot_id = payload ->> 'snapshot_id'
                  AND links.image_id = payload ->> 'image_id'
                  AND memories.conversation_id = units.conversation_id
                  AND sessions.session_order <= cutoff_order LIMIT 1;
            ELSIF payload ->> 'session_id' IS NOT NULL THEN
                SELECT true INTO visible
                FROM {RUNTIME}.benchmark_sessions sessions
                JOIN {RUNTIME}.compiler_units units
                  ON units.compiler_unit_id = payload ->> 'compiler_unit_id'
                WHERE sessions.snapshot_id = payload ->> 'snapshot_id'
                  AND sessions.session_id = payload ->> 'session_id'
                  AND sessions.conversation_id = units.conversation_id
                  AND sessions.session_order <= cutoff_order;
            ELSIF payload ->> 'caption_id' IS NOT NULL THEN
                visible := false;
            END IF;
            IF COALESCE(visible, false) = false THEN
                RAISE EXCEPTION 'compiler evidence violates snapshot, conversation, or cutoff lineage' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_qbank_embedding_lineage()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE source_kind varchar(64); collection_snapshot varchar(128); valid boolean;
        BEGIN
            SELECT source_type, snapshot_id INTO source_kind, collection_snapshot
              FROM {RUNTIME}.embedding_collections WHERE collection_id = NEW.collection_id;
            IF source_kind = 'qbank_question' THEN
                SELECT true INTO valid FROM {RUNTIME}.qbank_questions
                 WHERE qbank_id = NEW.qbank_id AND snapshot_id = collection_snapshot
                   AND NEW.source_id = qbank_id AND NEW.paraphrase_id IS NULL;
            ELSIF source_kind = 'qbank_paraphrase' THEN
                SELECT true INTO valid FROM {RUNTIME}.qbank_paraphrases
                 WHERE paraphrase_id = NEW.paraphrase_id AND snapshot_id = collection_snapshot
                   AND NEW.source_id = paraphrase_id AND NEW.qbank_id IS NULL;
            ELSE
                valid := NEW.qbank_id IS NULL AND NEW.paraphrase_id IS NULL;
            END IF;
            IF COALESCE(valid, false) = false THEN
                RAISE EXCEPTION 'embedding typed source does not match collection source type' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_compiler_fact_scope()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE payload jsonb; valid boolean; parent_qbank varchar(128);
                expected_producer_stage varchar(32);
        BEGIN
            payload := to_jsonb(NEW);
            SELECT true INTO valid FROM {RUNTIME}.compiler_units units
             WHERE units.compiler_unit_id = payload ->> 'compiler_unit_id'
               AND units.run_id = payload ->> 'run_id'
               AND units.snapshot_id = payload ->> 'snapshot_id';
            IF COALESCE(valid, false) = false THEN
                RAISE EXCEPTION 'compiler fact scope does not match its unit' USING ERRCODE = '23514';
            END IF;
            expected_producer_stage := CASE TG_TABLE_NAME
                WHEN 'qbank_questions' THEN 'questioner'
                WHEN 'memory_programs' THEN 'planner'
                WHEN 'doubter_feedback' THEN 'doubter'
                ELSE NULL
            END;
            IF payload ->> 'producer_model_call_id' IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM {RUNTIME}.model_calls calls
                 WHERE calls.model_call_id = payload ->> 'producer_model_call_id'
                   AND calls.run_id = payload ->> 'run_id'
                   AND calls.snapshot_id = payload ->> 'snapshot_id'
                   AND calls.compiler_unit_id = payload ->> 'compiler_unit_id'
                   AND calls.stage = expected_producer_stage
            ) THEN
                RAISE EXCEPTION 'compiler producer model-call lineage is inconsistent' USING ERRCODE = '23514';
            END IF;
            IF TG_TABLE_NAME = 'qbank_questions' AND NOT EXISTS (
                SELECT 1 FROM {RUNTIME}.compiler_units units
                 WHERE units.compiler_unit_id = payload ->> 'compiler_unit_id'
                   AND units.available_after_session = payload ->> 'created_after_session'
            ) THEN
                RAISE EXCEPTION 'qbank question cutoff does not match compiler unit' USING ERRCODE = '23514';
            END IF;
            IF payload ->> 'qbank_id' IS NOT NULL AND TG_TABLE_NAME != 'qbank_questions' THEN
                SELECT questions.qbank_id INTO parent_qbank FROM {RUNTIME}.qbank_questions questions
                 WHERE questions.qbank_id = payload ->> 'qbank_id'
                   AND questions.run_id = payload ->> 'run_id'
                   AND questions.snapshot_id = payload ->> 'snapshot_id'
                   AND questions.compiler_unit_id = payload ->> 'compiler_unit_id';
                IF parent_qbank IS NULL THEN
                    RAISE EXCEPTION 'compiler fact qbank lineage does not match its unit' USING ERRCODE = '23514';
                END IF;
            END IF;
            IF payload ->> 'plan_id' IS NOT NULL AND TG_TABLE_NAME NOT IN ('memory_programs','qbank_decisions') AND NOT EXISTS (
                SELECT 1 FROM {RUNTIME}.memory_programs programs
                 WHERE programs.plan_id = payload ->> 'plan_id'
                   AND programs.run_id = payload ->> 'run_id'
                   AND programs.snapshot_id = payload ->> 'snapshot_id'
                   AND programs.compiler_unit_id = payload ->> 'compiler_unit_id'
            ) THEN
                RAISE EXCEPTION 'compiler fact program lineage does not match its unit' USING ERRCODE = '23514';
            END IF;
            IF payload ->> 'program_execution_id' IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM {RUNTIME}.program_executions executions
                 WHERE executions.execution_id = payload ->> 'program_execution_id'
                   AND executions.source_compiler_run_id = payload ->> 'run_id'
                   AND executions.snapshot_id = payload ->> 'snapshot_id'
                   AND executions.compiler_unit_id = payload ->> 'compiler_unit_id'
                   AND executions.plan_id = payload ->> 'plan_id'
            ) THEN
                RAISE EXCEPTION 'doubter execution lineage is inconsistent' USING ERRCODE = '23514';
            END IF;
            IF TG_TABLE_NAME = 'qbank_decisions' THEN
                IF payload ->> 'final_plan_id' IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.memory_programs programs
                     WHERE programs.plan_id = payload ->> 'final_plan_id'
                       AND programs.qbank_id = payload ->> 'qbank_id'
                       AND programs.run_id = payload ->> 'run_id'
                       AND programs.snapshot_id = payload ->> 'snapshot_id'
                       AND programs.compiler_unit_id = payload ->> 'compiler_unit_id'
                ) THEN
                    RAISE EXCEPTION 'qbank decision final program has mismatched lineage' USING ERRCODE = '23514';
                END IF;
                IF payload ->> 'final_feedback_id' IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.doubter_feedback feedback
                     WHERE feedback.feedback_id = payload ->> 'final_feedback_id'
                       AND feedback.qbank_id = payload ->> 'qbank_id'
                       AND feedback.run_id = payload ->> 'run_id'
                       AND feedback.snapshot_id = payload ->> 'snapshot_id'
                       AND feedback.compiler_unit_id = payload ->> 'compiler_unit_id'
                ) THEN
                    RAISE EXCEPTION 'qbank decision final feedback has mismatched lineage' USING ERRCODE = '23514';
                END IF;
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_compiler_model_call_scope()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        BEGIN
            IF NEW.compiler_unit_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM {RUNTIME}.experiment_jobs jobs
                 WHERE jobs.job_id = NEW.job_id
                   AND jobs.run_id = NEW.run_id
                   AND jobs.snapshot_id = NEW.snapshot_id
                   AND jobs.compiler_unit_id = NEW.compiler_unit_id
                   AND jobs.stage = 'compile'
            ) THEN
                RAISE EXCEPTION 'compiler model call does not match its COMPILE job' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_execution_evidence_visibility()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE unit_id varchar(128); source_run varchar(128); parent_query_id varchar(256);
                parent_purpose varchar(32);
                conversation varchar(256);
                cutoff_order integer; visible boolean;
        BEGIN
            SELECT executions.compiler_unit_id, executions.source_compiler_run_id,
                   executions.query_id, executions.purpose
              INTO unit_id, source_run, parent_query_id, parent_purpose
              FROM {RUNTIME}.program_executions executions
             WHERE executions.execution_id = NEW.execution_id
               AND executions.run_id = NEW.run_id
               AND executions.snapshot_id = NEW.snapshot_id;
            IF parent_purpose = 'compiler_dry_run' AND unit_id IS NOT NULL THEN
                SELECT units.conversation_id, sessions.session_order
                  INTO conversation, cutoff_order
                  FROM {RUNTIME}.compiler_units units
                  JOIN {RUNTIME}.benchmark_sessions sessions
                    ON sessions.snapshot_id = units.snapshot_id
                   AND sessions.conversation_id = units.conversation_id
                   AND sessions.session_id = units.available_after_session
                 WHERE units.compiler_unit_id = unit_id AND units.run_id = source_run;
            ELSIF parent_purpose = 'query_time' AND parent_query_id IS NOT NULL THEN
                SELECT queries.conversation_id, sessions.session_order
                  INTO conversation, cutoff_order
                  FROM {RUNTIME}.benchmark_queries queries
                  JOIN {RUNTIME}.benchmark_sessions sessions
                    ON sessions.snapshot_id = queries.snapshot_id
                   AND sessions.conversation_id = queries.conversation_id
                   AND sessions.session_id = queries.available_after_session
                 WHERE queries.snapshot_id = NEW.snapshot_id
                   AND queries.query_id = parent_query_id;
            END IF;
            IF cutoff_order IS NULL THEN
                RAISE EXCEPTION 'program execution evidence has no prospective cutoff' USING ERRCODE = '23514';
            END IF;
            visible := false;
            IF NEW.memory_id IS NOT NULL THEN
                SELECT true INTO visible FROM {RUNTIME}.memory_units memories
                JOIN {RUNTIME}.benchmark_sessions sessions
                  ON sessions.snapshot_id = memories.snapshot_id
                 AND sessions.conversation_id = memories.conversation_id
                 AND sessions.session_id = memories.available_after_session
                 WHERE memories.snapshot_id = NEW.snapshot_id
                   AND memories.memory_id = NEW.memory_id
                   AND memories.conversation_id = conversation
                   AND sessions.session_order <= cutoff_order;
            ELSIF NEW.image_id IS NOT NULL THEN
                SELECT true INTO visible FROM {RUNTIME}.memory_unit_images links
                JOIN {RUNTIME}.memory_units memories
                  ON memories.snapshot_id = links.snapshot_id AND memories.memory_id = links.memory_id
                JOIN {RUNTIME}.benchmark_sessions sessions
                  ON sessions.snapshot_id = memories.snapshot_id
                 AND sessions.conversation_id = memories.conversation_id
                 AND sessions.session_id = memories.available_after_session
                 WHERE links.snapshot_id = NEW.snapshot_id AND links.image_id = NEW.image_id
                   AND memories.conversation_id = conversation
                   AND sessions.session_order <= cutoff_order LIMIT 1;
            ELSIF NEW.session_id IS NOT NULL THEN
                SELECT true INTO visible FROM {RUNTIME}.benchmark_sessions sessions
                 WHERE sessions.snapshot_id = NEW.snapshot_id
                   AND sessions.session_id = NEW.session_id
                   AND sessions.conversation_id = conversation
                   AND sessions.session_order <= cutoff_order;
            ELSIF NEW.caption_id IS NOT NULL THEN
                visible := false;
            END IF;
            IF COALESCE(visible, false) = false THEN
                RAISE EXCEPTION 'program execution evidence violates prospective lineage' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_program_execution_scope()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE valid boolean; parent {RUNTIME}.program_executions%ROWTYPE;
        BEGIN
            IF TG_TABLE_NAME = 'program_executions' THEN
                SELECT true INTO valid FROM {RUNTIME}.memory_programs programs
                 WHERE programs.plan_id = NEW.plan_id
                   AND programs.run_id = NEW.source_compiler_run_id
                   AND programs.snapshot_id = NEW.snapshot_id
                   AND (NEW.compiler_unit_id IS NULL OR programs.compiler_unit_id = NEW.compiler_unit_id);
                IF COALESCE(valid, false) = false THEN
                    RAISE EXCEPTION 'program execution source program scope is inconsistent' USING ERRCODE = '23514';
                END IF;
                IF NEW.purpose = 'compiler_dry_run' AND NEW.run_id != NEW.source_compiler_run_id THEN
                    RAISE EXCEPTION 'compiler dry-run must belong to its source compiler run' USING ERRCODE = '23514';
                END IF;
                IF NEW.purpose = 'query_time' AND NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.run_queries run_queries
                    JOIN {RUNTIME}.benchmark_queries queries
                      ON queries.snapshot_id = run_queries.snapshot_id
                     AND queries.query_id = run_queries.query_id
                    JOIN {RUNTIME}.qbank_questions questions
                      ON questions.qbank_id = (
                          SELECT programs.qbank_id FROM {RUNTIME}.memory_programs programs
                          WHERE programs.plan_id = NEW.plan_id
                            AND programs.run_id = NEW.source_compiler_run_id
                      )
                     AND questions.run_id = NEW.source_compiler_run_id
                    JOIN {RUNTIME}.benchmark_sessions question_cutoff
                      ON question_cutoff.snapshot_id = queries.snapshot_id
                     AND question_cutoff.conversation_id = queries.conversation_id
                     AND question_cutoff.session_id = queries.available_after_session
                    JOIN {RUNTIME}.benchmark_sessions qbank_cutoff
                      ON qbank_cutoff.snapshot_id = questions.snapshot_id
                     AND qbank_cutoff.conversation_id = questions.conversation_id
                     AND qbank_cutoff.session_id = questions.created_after_session
                    WHERE run_queries.run_id = NEW.run_id
                      AND run_queries.snapshot_id = NEW.snapshot_id
                      AND run_queries.query_id = NEW.query_id
                      AND questions.snapshot_id = NEW.snapshot_id
                      AND questions.conversation_id = queries.conversation_id
                      AND qbank_cutoff.session_order <= question_cutoff.session_order
                ) THEN
                    RAISE EXCEPTION 'query-time execution violates qbank/query cutoff lineage' USING ERRCODE = '23514';
                END IF;
            ELSE
                SELECT * INTO parent FROM {RUNTIME}.program_executions executions
                 WHERE executions.execution_id = NEW.execution_id AND executions.run_id = NEW.run_id;
                IF NOT FOUND OR parent.snapshot_id != NEW.snapshot_id
                   OR parent.plan_id != NEW.plan_id
                   OR parent.compiler_unit_id IS DISTINCT FROM NEW.compiler_unit_id
                   OR parent.query_id IS DISTINCT FROM NEW.query_id THEN
                    RAISE EXCEPTION 'program execution step scope is inconsistent' USING ERRCODE = '23514';
                END IF;
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_compiler_completion()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE compiler {RUNTIME}.compiler_runs%ROWTYPE; units integer; bad integer; required text[];
        BEGIN
            IF NEW.status != 'completed' THEN RETURN NEW; END IF;
            SELECT * INTO compiler FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
            IF NOT FOUND THEN RETURN NEW; END IF;
            IF NEW.selected_count != 0 OR NEW.succeeded_count != 0 OR NEW.failed_count != 0 THEN
                RAISE EXCEPTION 'compiler run must keep query denominator counters at zero' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO units FROM {RUNTIME}.compiler_units WHERE run_id = NEW.run_id;
            IF units != compiler.unit_count OR COALESCE((
                SELECT max(ordinal) FROM {RUNTIME}.compiler_units WHERE run_id = NEW.run_id
            ), -1) != compiler.unit_count - 1 THEN
                RAISE EXCEPTION 'compiler unit denominator is incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad FROM {RUNTIME}.compiler_units units
            WHERE units.run_id = NEW.run_id AND (
                (SELECT count(*) FROM {RUNTIME}.experiment_jobs jobs
                 WHERE jobs.run_id = NEW.run_id AND jobs.compiler_unit_id = units.compiler_unit_id
                   AND jobs.stage = 'compile' AND jobs.status = 'succeeded') != 1
            );
            IF bad != 0 THEN
                RAISE EXCEPTION 'compiler units require exactly one succeeded COMPILE job' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad FROM {RUNTIME}.compiler_units units
            WHERE units.run_id = NEW.run_id AND (
                NOT EXISTS (SELECT 1 FROM {RUNTIME}.compiler_unit_memories members
                    WHERE members.compiler_unit_id = units.compiler_unit_id AND members.role = 'anchor')
                OR COALESCE((SELECT min(ordinal) FROM {RUNTIME}.compiler_unit_memories members
                    WHERE members.compiler_unit_id = units.compiler_unit_id), -1) != 0
                OR COALESCE((SELECT max(ordinal) FROM {RUNTIME}.compiler_unit_memories members
                    WHERE members.compiler_unit_id = units.compiler_unit_id), -1) !=
                   (SELECT count(*) - 1 FROM {RUNTIME}.compiler_unit_memories members
                    WHERE members.compiler_unit_id = units.compiler_unit_id)
            );
            IF bad != 0 THEN
                RAISE EXCEPTION 'compiler unit membership is incomplete' USING ERRCODE = '55000';
            END IF;
            required := CASE compiler.target_stage
                WHEN 'questions' THEN ARRAY['questioner']::text[]
                WHEN 'programs' THEN ARRAY['questioner','planner','dry_run']::text[]
                ELSE ARRAY['questioner','planner','dry_run','doubter','decision']::text[]
            END;
            SELECT count(*) INTO bad FROM {RUNTIME}.compiler_units units
            WHERE units.run_id = NEW.run_id AND EXISTS (
                SELECT 1 FROM unnest(required) stage_name
                WHERE NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.compiler_unit_stage_results results
                    WHERE results.compiler_unit_id = units.compiler_unit_id
                      AND results.stage = stage_name AND results.status = 'success'
                )
            );
            IF bad != 0 THEN
                RAISE EXCEPTION 'compiler target stage results are incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad
            FROM {RUNTIME}.compiler_unit_stage_results results
            JOIN {RUNTIME}.compiler_units units
              ON units.compiler_unit_id = results.compiler_unit_id
             AND units.run_id = results.run_id
            WHERE units.run_id = NEW.run_id
              AND results.status = 'success'
              AND results.stage = ANY(required)
              AND results.produced_count != CASE results.stage
                WHEN 'questioner' THEN (SELECT count(*) FROM {RUNTIME}.qbank_questions facts
                    WHERE facts.run_id = NEW.run_id
                      AND facts.compiler_unit_id = units.compiler_unit_id)
                WHEN 'planner' THEN (SELECT count(*) FROM {RUNTIME}.memory_programs facts
                    WHERE facts.run_id = NEW.run_id
                      AND facts.compiler_unit_id = units.compiler_unit_id)
                WHEN 'dry_run' THEN (SELECT count(*) FROM {RUNTIME}.program_executions facts
                    WHERE facts.source_compiler_run_id = NEW.run_id
                      AND facts.compiler_unit_id = units.compiler_unit_id
                      AND facts.purpose = 'compiler_dry_run')
                WHEN 'doubter' THEN (SELECT count(*) FROM {RUNTIME}.doubter_feedback facts
                    WHERE facts.run_id = NEW.run_id
                      AND facts.compiler_unit_id = units.compiler_unit_id)
                WHEN 'decision' THEN (SELECT count(*) FROM {RUNTIME}.qbank_decisions facts
                    WHERE facts.run_id = NEW.run_id
                      AND facts.compiler_unit_id = units.compiler_unit_id)
                ELSE -1
              END;
            IF bad != 0 THEN
                RAISE EXCEPTION 'compiler stage counts differ from persisted facts' USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END; $$"""
    )

    op.execute(
        f"CREATE TRIGGER validate_compiler_run BEFORE INSERT OR UPDATE ON {RUNTIME}.compiler_runs FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_compiler_run_manifest()"
    )
    op.execute(
        f"CREATE TRIGGER validate_compiler_unit_memory BEFORE INSERT OR UPDATE ON {RUNTIME}.compiler_unit_memories FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_compiler_unit_memory_visibility()"
    )
    for table in (
        "qbank_evidence_refs",
        "program_expected_evidence_refs",
    ):
        op.execute(
            f"CREATE TRIGGER validate_compiler_evidence BEFORE INSERT OR UPDATE ON {RUNTIME}.{table} FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_compiler_evidence_visibility()"
        )
    op.execute(
        f"CREATE TRIGGER validate_qbank_embedding BEFORE INSERT OR UPDATE ON {RUNTIME}.embeddings FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_qbank_embedding_lineage()"
    )
    op.execute(
        f"CREATE TRIGGER validate_compiler_model_call BEFORE INSERT OR UPDATE ON {RUNTIME}.model_calls FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_compiler_model_call_scope()"
    )
    for table in (
        "qbank_questions",
        "qbank_paraphrases",
        "qbank_evidence_refs",
        "memory_programs",
        "program_steps",
        "program_expected_evidence_refs",
        "doubter_feedback",
        "qbank_decisions",
    ):
        op.execute(
            f"CREATE TRIGGER validate_compiler_fact_scope BEFORE INSERT OR UPDATE ON {RUNTIME}.{table} FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_compiler_fact_scope()"
        )
    op.execute(
        f"CREATE TRIGGER validate_execution_evidence BEFORE INSERT OR UPDATE ON {RUNTIME}.program_execution_evidence_refs FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_execution_evidence_visibility()"
    )
    op.execute(
        f"CREATE TRIGGER validate_program_execution_scope BEFORE INSERT OR UPDATE ON {RUNTIME}.program_executions FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_program_execution_scope()"
    )
    op.execute(
        f"CREATE TRIGGER validate_program_execution_step_scope BEFORE INSERT OR UPDATE ON {RUNTIME}.program_execution_steps FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_program_execution_scope()"
    )
    op.execute(
        f"CREATE TRIGGER compiler_completion_integrity BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_compiler_completion()"
    )
    compiler_tables = (
        "compiler_runs",
        "compiler_units",
        "compiler_unit_memories",
        "compiler_unit_stage_results",
        "qbank_questions",
        "qbank_paraphrases",
        "qbank_evidence_refs",
        "memory_programs",
        "program_steps",
        "program_expected_evidence_refs",
        "program_executions",
        "program_execution_steps",
        "program_execution_evidence_refs",
        "doubter_feedback",
        "qbank_decisions",
    )
    for table in compiler_tables:
        op.execute(
            f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.{table} FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()"
        )


def _create_routing_and_summary_views() -> None:
    op.execute(
        f"""CREATE VIEW {RUNTIME}.qbank_routing_items AS
        SELECT questions.run_id, questions.qbank_id, questions.snapshot_id,
            questions.conversation_id,
            questions.created_after_session AS available_after_session,
            questions.canonical_question, questions.question_type,
            questions.required_modalities,
            COALESCE((SELECT array_agg(paraphrase ORDER BY ordinal)
                FROM {RUNTIME}.qbank_paraphrases paraphrases
                WHERE paraphrases.qbank_id = questions.qbank_id), ARRAY[]::text[]) AS paraphrases,
            decisions.final_plan_id AS memory_program_id,
            programs.program_hash, programs.operator_set_version,
            decisions.verifier_pass AS verified, decisions.verifier_score,
            jsonb_build_object('compiler_run_id', questions.run_id) AS metadata
        FROM {RUNTIME}.qbank_questions questions
        JOIN {RUNTIME}.qbank_decisions decisions USING (qbank_id, run_id)
        JOIN {RUNTIME}.memory_programs programs
          ON programs.plan_id = decisions.final_plan_id AND programs.run_id = decisions.run_id
        JOIN {RUNTIME}.run_states states ON states.run_id = decisions.run_id
        WHERE decisions.accepted AND decisions.verifier_pass
          AND states.status = 'completed'"""
    )
    op.execute(
        f"""CREATE VIEW {RUNTIME}.qbank_routing_program_steps AS
        SELECT decisions.run_id, decisions.qbank_id, decisions.final_plan_id AS plan_id,
            steps.step_id, steps.op_name, steps.args, steps.expected_output_type,
            programs.operator_set_version, programs.program_hash
        FROM {RUNTIME}.qbank_decisions decisions
        JOIN {RUNTIME}.memory_programs programs ON programs.plan_id = decisions.final_plan_id
        JOIN {RUNTIME}.program_steps steps ON steps.plan_id = programs.plan_id
        JOIN {RUNTIME}.run_states states ON states.run_id = decisions.run_id
        WHERE decisions.accepted AND decisions.verifier_pass
          AND states.status = 'completed'"""
    )
    op.execute(
        f"""CREATE VIEW {RUNTIME}.compiler_run_costs AS
        SELECT compiler.run_id, compiler.unit_count,
            COALESCE(sum(calls.input_tokens), 0)::bigint AS input_tokens,
            COALESCE(sum(calls.output_tokens), 0)::bigint AS output_tokens,
            COALESCE(sum(calls.total_tokens), 0)::bigint AS total_tokens,
            COALESCE(sum(calls.cost_usd), 0)::double precision AS cost_usd,
            COALESCE(sum(calls.latency_ms), 0)::double precision AS model_latency_ms,
            count(calls.model_call_id)::bigint AS model_call_count
        FROM {RUNTIME}.compiler_runs compiler
        LEFT JOIN {RUNTIME}.model_calls calls ON calls.run_id = compiler.run_id
          AND calls.stage IN ('questioner','planner','doubter')
        GROUP BY compiler.run_id, compiler.unit_count"""
    )
    op.execute(
        f"""CREATE VIEW {RUNTIME}.operator_execution_summary AS
        SELECT executions.source_compiler_run_id AS run_id,
            executions.purpose, steps.op_name,
            count(*)::bigint AS execution_count,
            count(*) FILTER (WHERE steps.success)::bigint AS succeeded_count,
            count(*) FILTER (WHERE NOT steps.success)::bigint AS failed_count,
            avg(CASE WHEN steps.success THEN 1.0 ELSE 0.0 END)::double precision
                AS success_rate,
            COALESCE(sum(steps.latency_ms), 0)::double precision AS latency_ms
        FROM {RUNTIME}.program_execution_steps steps
        JOIN {RUNTIME}.program_executions executions USING (execution_id, run_id)
        GROUP BY executions.source_compiler_run_id, executions.purpose, steps.op_name"""
    )


def _grant_permissions() -> None:
    tables = (
        "compiler_runs",
        "compiler_units",
        "compiler_unit_memories",
        "compiler_unit_stage_results",
        "qbank_questions",
        "qbank_paraphrases",
        "qbank_evidence_refs",
        "memory_programs",
        "program_steps",
        "program_expected_evidence_refs",
        "program_executions",
        "program_execution_steps",
        "program_execution_evidence_refs",
        "doubter_feedback",
        "qbank_decisions",
    )
    qualified = ", ".join(f"{RUNTIME}.{table}" for table in tables)
    op.execute(f"GRANT SELECT ON {qualified} TO pmmc_runtime, pmmc_evaluator")
    op.execute(f"GRANT INSERT ON {qualified} TO pmmc_runtime")
    op.execute(
        f"GRANT SELECT ON {RUNTIME}.qbank_routing_items, {RUNTIME}.qbank_routing_program_steps, {RUNTIME}.compiler_run_costs, {RUNTIME}.operator_execution_summary TO pmmc_runtime, pmmc_evaluator"
    )
    for function in (
        "validate_compiler_run_manifest()",
        "validate_compiler_unit_memory_visibility()",
        "validate_compiler_evidence_visibility()",
        "validate_qbank_embedding_lineage()",
        "validate_compiler_model_call_scope()",
        "validate_compiler_fact_scope()",
        "validate_execution_evidence_visibility()",
        "validate_program_execution_scope()",
        "enforce_compiler_completion()",
    ):
        op.execute(f"REVOKE ALL ON FUNCTION {RUNTIME}.{function} FROM PUBLIC")


def downgrade() -> None:
    for view in (
        "operator_execution_summary",
        "compiler_run_costs",
        "qbank_routing_program_steps",
        "qbank_routing_items",
    ):
        op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.{view}")

    op.execute(
        f"DROP TRIGGER IF EXISTS compiler_completion_integrity ON {RUNTIME}.run_states"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_qbank_embedding ON {RUNTIME}.embeddings"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_compiler_model_call ON {RUNTIME}.model_calls"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_execution_evidence ON {RUNTIME}.program_execution_evidence_refs"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_program_execution_step_scope ON {RUNTIME}.program_execution_steps"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_program_execution_scope ON {RUNTIME}.program_executions"
    )
    for table in (
        "qbank_questions",
        "qbank_paraphrases",
        "qbank_evidence_refs",
        "memory_programs",
        "program_steps",
        "program_expected_evidence_refs",
        "doubter_feedback",
        "qbank_decisions",
    ):
        op.execute(
            f"DROP TRIGGER IF EXISTS validate_compiler_fact_scope ON {RUNTIME}.{table}"
        )
    for table in ("qbank_evidence_refs", "program_expected_evidence_refs"):
        op.execute(
            f"DROP TRIGGER IF EXISTS validate_compiler_evidence ON {RUNTIME}.{table}"
        )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_compiler_unit_memory ON {RUNTIME}.compiler_unit_memories"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_compiler_run ON {RUNTIME}.compiler_runs"
    )
    for function in (
        "enforce_compiler_completion()",
        "validate_compiler_run_manifest()",
        "validate_qbank_embedding_lineage()",
        "validate_compiler_model_call_scope()",
        "validate_execution_evidence_visibility()",
        "validate_program_execution_scope()",
        "validate_compiler_fact_scope()",
        "validate_compiler_evidence_visibility()",
        "validate_compiler_unit_memory_visibility()",
    ):
        op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.{function}")

    op.drop_constraint(
        "embeddings_one_typed_compiler_source",
        "embeddings",
        schema=RUNTIME,
        type_="check",
    )
    for constraint in (
        "fk_embeddings_qbank_paraphrase",
        "fk_embeddings_qbank_question",
    ):
        op.drop_constraint(constraint, "embeddings", schema=RUNTIME, type_="foreignkey")
    for column in ("paraphrase_id", "qbank_id"):
        op.drop_column("embeddings", column, schema=RUNTIME)

    op.drop_constraint(
        "model_calls_compiler_call_scope",
        "model_calls",
        schema=RUNTIME,
        type_="check",
    )
    op.drop_constraint(
        "fk_runtime_model_calls_compiler_unit",
        "model_calls",
        schema=RUNTIME,
        type_="foreignkey",
    )
    op.drop_column("model_calls", "compiler_unit_id", schema=RUNTIME)
    op.drop_constraint(
        "experiment_jobs_compiler_job_scope",
        "experiment_jobs",
        schema=RUNTIME,
        type_="check",
    )
    op.drop_constraint(
        "uq_experiment_jobs_compiler_unit_stage_variant",
        "experiment_jobs",
        schema=RUNTIME,
        type_="unique",
    )
    op.drop_constraint(
        "fk_experiment_jobs_compiler_unit",
        "experiment_jobs",
        schema=RUNTIME,
        type_="foreignkey",
    )
    op.drop_column("experiment_jobs", "compiler_unit_id", schema=RUNTIME)

    for table in (
        "qbank_decisions",
        "doubter_feedback",
        "program_execution_evidence_refs",
        "program_execution_steps",
        "program_executions",
        "program_expected_evidence_refs",
        "program_steps",
        "memory_programs",
        "qbank_evidence_refs",
        "qbank_paraphrases",
        "qbank_questions",
        "compiler_unit_stage_results",
        "compiler_unit_memories",
        "compiler_units",
        "compiler_runs",
    ):
        op.execute(f"DROP TABLE IF EXISTS {RUNTIME}.{table} CASCADE")

    op.drop_index(
        "ix_memory_units_entity_mentions", table_name="memory_units", schema=RUNTIME
    )
    op.drop_index(
        "ix_memory_units_search_document", table_name="memory_units", schema=RUNTIME
    )
    op.drop_column("memory_units", "search_document", schema=RUNTIME)
