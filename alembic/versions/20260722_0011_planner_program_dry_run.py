# ruff: noqa: E501
"""Add Planner v1, memory indexes, and program dry-run facts.

Revision ID: 20260722_0011
Revises: 20260721_0010
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260722_0011"
down_revision: str | None = "20260721_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"

_COMPILER_IDENTITY_COLUMNS = (
    "questioner_freeze_manifest_hash",
    "planner_config_hash",
    "planner_model_alias",
    "planner_model_revision",
    "planner_prompt_version",
    "planner_prompt_hash",
    "planner_renderer_version",
    "planner_parser_version",
    "planner_schema_version",
    "program_policy_version",
    "dry_run_policy_version",
    "memory_index_representation_version",
    "memory_text_model_alias",
    "memory_text_model_revision",
    "memory_multimodal_model_alias",
    "memory_multimodal_model_revision",
)

_TABLES = (
    "compiler_stage_jobs",
    "planner_generation_batches",
    "planner_candidate_outcomes",
    "planner_batch_programs",
    "program_validation_results",
    "program_validation_issues",
    "memory_operator_index_manifests",
    "program_query_embedding_batches",
    "program_query_embeddings",
    "program_dry_run_assessments",
    "program_dry_run_evidence_matches",
)


def upgrade() -> None:
    _extend_compiler_identity()
    _add_lineage_keys()
    _create_tables()
    _replace_questioner_verifier_guard()
    _replace_embedding_lineage_guard()
    _replace_compiler_completion_guard()
    _create_completion_guard()
    _grant_permissions()


def _extend_compiler_identity() -> None:
    for name in _COMPILER_IDENTITY_COLUMNS:
        length = 256 if name.endswith(("_alias", "_revision")) else 128
        op.add_column(
            "compiler_runs",
            sa.Column(name, sa.String(length=length), nullable=True),
            schema=RUNTIME,
        )


def _add_lineage_keys() -> None:
    op.create_unique_constraint(
        "uq_qbank_questions_run_unit",
        "qbank_questions",
        ["qbank_id", "run_id", "compiler_unit_id"],
        schema=RUNTIME,
    )
    op.create_unique_constraint(
        "uq_memory_programs_run_unit",
        "memory_programs",
        ["plan_id", "run_id", "compiler_unit_id"],
        schema=RUNTIME,
    )


def _create_tables() -> None:
    statements = (
        f"""CREATE TABLE {RUNTIME}.compiler_stage_jobs (
            stage_job_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            stage varchar(32) NOT NULL CHECK (stage IN ('planner','dry_run')),
            round_id integer NOT NULL CHECK (round_id >= 0),
            status varchar(16) NOT NULL CHECK (status IN ('pending','running','succeeded','failed','cancelled')),
            attempt_count integer NOT NULL CHECK (attempt_count >= 0),
            max_attempts integer NOT NULL CHECK (max_attempts > 0),
            claimed_by varchar(256),
            lease_expires_at timestamptz,
            available_at timestamptz NOT NULL,
            started_at timestamptz,
            finished_at timestamptz,
            error_type varchar(256),
            error_summary text,
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (run_id, compiler_unit_id, stage, round_id),
            UNIQUE (stage_job_id, run_id, compiler_unit_id),
            FOREIGN KEY (compiler_unit_id, run_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            CHECK ((status = 'running' AND claimed_by IS NOT NULL AND lease_expires_at IS NOT NULL)
                OR status != 'running'),
            CHECK ((status IN ('succeeded','failed','cancelled') AND finished_at IS NOT NULL)
                OR status NOT IN ('succeeded','failed','cancelled')),
            CHECK ((status = 'failed' AND error_type IS NOT NULL) OR status != 'failed')
        )""",
        f"""CREATE TABLE {RUNTIME}.planner_generation_batches (
            planner_batch_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            stage_job_id varchar(128) NOT NULL,
            model_call_id varchar(128) NOT NULL UNIQUE,
            batch_ordinal integer NOT NULL CHECK (batch_ordinal >= 0),
            candidate_qbank_ids text[] NOT NULL CHECK (cardinality(candidate_qbank_ids) BETWEEN 1 AND 4),
            parsed_count integer NOT NULL CHECK (parsed_count >= 0),
            invalid_count integer NOT NULL CHECK (invalid_count >= 0),
            prompt_hash varchar(128) NOT NULL,
            schema_hash varchar(128) NOT NULL,
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (run_id, compiler_unit_id, batch_ordinal),
            UNIQUE (planner_batch_id, run_id, compiler_unit_id),
            FOREIGN KEY (compiler_unit_id, run_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            FOREIGN KEY (stage_job_id, run_id, compiler_unit_id)
                REFERENCES {RUNTIME}.compiler_stage_jobs(stage_job_id, run_id, compiler_unit_id),
            FOREIGN KEY (model_call_id, run_id) REFERENCES {RUNTIME}.model_calls(model_call_id, run_id),
            CHECK (parsed_count + invalid_count = cardinality(candidate_qbank_ids))
        )""",
        f"""CREATE TABLE {RUNTIME}.planner_candidate_outcomes (
            planner_outcome_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            planner_batch_id varchar(128),
            plan_id varchar(128),
            status varchar(64) NOT NULL CHECK (status IN (
                'program_ready','program_ready_with_fallback','program_ready_with_abstain',
                'parse_failed','static_rejected','policy_rejected','lineage_rejected',
                'query_vector_unavailable','execution_failed','dry_run_insufficient')),
            reason_code varchar(128) NOT NULL,
            reason_summary text,
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (qbank_id),
            FOREIGN KEY (qbank_id, run_id, compiler_unit_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id, compiler_unit_id),
            FOREIGN KEY (planner_batch_id, run_id, compiler_unit_id)
                REFERENCES {RUNTIME}.planner_generation_batches(planner_batch_id, run_id, compiler_unit_id),
            FOREIGN KEY (plan_id, run_id, compiler_unit_id)
                REFERENCES {RUNTIME}.memory_programs(plan_id, run_id, compiler_unit_id),
            CHECK ((status IN ('program_ready','program_ready_with_fallback','program_ready_with_abstain',
                    'lineage_rejected','query_vector_unavailable','execution_failed',
                    'dry_run_insufficient') AND plan_id IS NOT NULL)
                OR (status NOT IN ('program_ready','program_ready_with_fallback','program_ready_with_abstain',
                    'lineage_rejected','query_vector_unavailable','execution_failed',
                    'dry_run_insufficient') AND plan_id IS NULL))
        )""",
        f"""CREATE TABLE {RUNTIME}.planner_batch_programs (
            planner_batch_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            plan_id varchar(128) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            content_hash varchar(128) NOT NULL,
            metadata jsonb NOT NULL,
            PRIMARY KEY (planner_batch_id, plan_id),
            UNIQUE (qbank_id),
            FOREIGN KEY (planner_batch_id, run_id, compiler_unit_id)
                REFERENCES {RUNTIME}.planner_generation_batches(planner_batch_id, run_id, compiler_unit_id),
            FOREIGN KEY (plan_id, run_id, compiler_unit_id)
                REFERENCES {RUNTIME}.memory_programs(plan_id, run_id, compiler_unit_id),
            FOREIGN KEY (qbank_id, run_id, compiler_unit_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id, compiler_unit_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.program_validation_results (
            validation_result_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            plan_id varchar(128) NOT NULL,
            validation_kind varchar(16) NOT NULL CHECK (validation_kind IN ('static','policy','lineage')),
            validator_version varchar(128) NOT NULL,
            valid boolean NOT NULL,
            issue_count integer NOT NULL CHECK (issue_count >= 0),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (plan_id, validation_kind),
            UNIQUE (validation_result_id, run_id),
            FOREIGN KEY (plan_id, run_id) REFERENCES {RUNTIME}.memory_programs(plan_id, run_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.program_validation_issues (
            validation_issue_id varchar(128) PRIMARY KEY,
            validation_result_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            code varchar(128) NOT NULL,
            severity varchar(16) NOT NULL CHECK (severity IN ('error','warning')),
            step_id integer CHECK (step_id IS NULL OR step_id >= 0),
            message text NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (validation_result_id, ordinal),
            FOREIGN KEY (validation_result_id, run_id)
                REFERENCES {RUNTIME}.program_validation_results(validation_result_id, run_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.memory_operator_index_manifests (
            memory_index_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            dataset varchar(32) NOT NULL,
            selection_hash varchar(128) NOT NULL,
            representation_version varchar(128) NOT NULL,
            text_collection_id varchar(128) NOT NULL,
            visual_collection_id varchar(128) NOT NULL,
            text_model_alias varchar(256) NOT NULL,
            text_model_revision varchar(256) NOT NULL,
            multimodal_model_alias varchar(256) NOT NULL,
            multimodal_model_revision varchar(256) NOT NULL,
            dimension integer NOT NULL CHECK (dimension = 4096),
            distance_metric varchar(16) NOT NULL CHECK (distance_metric = 'cosine'),
            status varchar(16) NOT NULL CHECK (status IN ('building','ready','failed')),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (run_id),
            CHECK (text_collection_id != visual_collection_id),
            FOREIGN KEY (run_id, snapshot_id) REFERENCES {RUNTIME}.compiler_runs(run_id, snapshot_id),
            FOREIGN KEY (text_collection_id) REFERENCES {RUNTIME}.embedding_collections(collection_id),
            FOREIGN KEY (visual_collection_id) REFERENCES {RUNTIME}.embedding_collections(collection_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.program_query_embedding_batches (
            query_embedding_batch_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            channel varchar(32) NOT NULL CHECK (channel IN ('question_text_query','question_joint_query')),
            model_call_id varchar(128) NOT NULL UNIQUE,
            batch_ordinal integer NOT NULL CHECK (batch_ordinal >= 0),
            qbank_ids text[] NOT NULL CHECK (cardinality(qbank_ids) > 0),
            dimension integer NOT NULL CHECK (dimension = 4096),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (run_id, channel, batch_ordinal),
            UNIQUE (query_embedding_batch_id, run_id),
            FOREIGN KEY (run_id) REFERENCES {RUNTIME}.compiler_runs(run_id),
            FOREIGN KEY (model_call_id, run_id) REFERENCES {RUNTIME}.model_calls(model_call_id, run_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.program_query_embeddings (
            program_query_embedding_id varchar(128) PRIMARY KEY,
            query_embedding_batch_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            channel varchar(32) NOT NULL CHECK (channel IN ('question_text_query','question_joint_query')),
            embedding_id varchar(128) NOT NULL,
            input_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (run_id, qbank_id, channel),
            FOREIGN KEY (query_embedding_batch_id, run_id)
                REFERENCES {RUNTIME}.program_query_embedding_batches(query_embedding_batch_id, run_id),
            FOREIGN KEY (qbank_id, run_id) REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            FOREIGN KEY (embedding_id) REFERENCES {RUNTIME}.embeddings(embedding_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.program_dry_run_assessments (
            assessment_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            plan_id varchar(128) NOT NULL,
            execution_id varchar(128) NOT NULL UNIQUE,
            status varchar(16) NOT NULL CHECK (status IN ('success','failed','skipped','partial')),
            control_action varchar(16) NOT NULL CHECK (control_action IN ('continue','fallback','abstain')),
            execution_success boolean NOT NULL,
            nonempty_evidence boolean NOT NULL,
            evidence_sufficient boolean NOT NULL,
            expected_group_count integer NOT NULL CHECK (expected_group_count >= 0),
            matched_group_count integer NOT NULL CHECK (matched_group_count >= 0),
            expected_evidence_recall double precision CHECK (expected_evidence_recall BETWEEN 0 AND 1),
            expected_evidence_complete boolean NOT NULL,
            modality_coverage jsonb NOT NULL,
            raw_image_reached boolean NOT NULL,
            context_operator_used boolean NOT NULL,
            temporal_operator_used boolean NOT NULL,
            missing_evidence_group_ids text[] NOT NULL,
            failure_code varchar(128),
            elapsed_ms double precision NOT NULL CHECK (elapsed_ms >= 0),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (plan_id),
            UNIQUE (assessment_id, run_id),
            FOREIGN KEY (qbank_id, run_id, compiler_unit_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id, compiler_unit_id),
            FOREIGN KEY (plan_id, run_id, compiler_unit_id)
                REFERENCES {RUNTIME}.memory_programs(plan_id, run_id, compiler_unit_id),
            FOREIGN KEY (execution_id, run_id) REFERENCES {RUNTIME}.program_executions(execution_id, run_id),
            FOREIGN KEY (compiler_unit_id, run_id) REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            CHECK (matched_group_count <= expected_group_count),
            CHECK ((expected_group_count = 0 AND expected_evidence_recall IS NULL)
                OR expected_group_count > 0),
            CHECK (NOT expected_evidence_complete OR matched_group_count = expected_group_count)
        )""",
        f"""CREATE TABLE {RUNTIME}.program_dry_run_evidence_matches (
            match_id varchar(128) PRIMARY KEY,
            assessment_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL,
            expected_evidence_id varchar(256) NOT NULL,
            actual_evidence_id varchar(256) NOT NULL,
            expected_evidence_type varchar(32) NOT NULL,
            actual_evidence_type varchar(32) NOT NULL,
            match_kind varchar(16) NOT NULL CHECK (match_kind IN ('direct','derived','group')),
            evidence_group_id varchar(256),
            metadata jsonb NOT NULL,
            UNIQUE (assessment_id, expected_evidence_id, actual_evidence_id),
            FOREIGN KEY (assessment_id, run_id)
                REFERENCES {RUNTIME}.program_dry_run_assessments(assessment_id, run_id)
        )""",
    )
    for statement in statements:
        op.execute(statement)
    for table in _TABLES:
        op.execute(
            f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.{table} FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()"
        )
    op.create_index(
        "ix_compiler_stage_jobs_claim",
        "compiler_stage_jobs",
        ["run_id", "stage", "status", "available_at"],
        schema=RUNTIME,
    )


def _replace_questioner_verifier_guard() -> None:
    """Version the verifier prompt guard for Questioner v3.1 program runs."""

    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_verifier "
        f"ON {RUNTIME}.questioner_verification_batches"
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_questioner_verifier_scope_v2()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE call_row record; generation_call record; generation_row record;
                compiler_version text; expected_prompt_version text;
        BEGIN
            SELECT * INTO call_row FROM {RUNTIME}.model_calls
             WHERE model_call_id = NEW.model_call_id AND run_id = NEW.run_id;
            SELECT * INTO generation_row FROM {RUNTIME}.questioner_generations
             WHERE generation_id = NEW.generation_id AND run_id = NEW.run_id;
            SELECT * INTO generation_call FROM {RUNTIME}.model_calls
             WHERE model_call_id = generation_row.model_call_id AND run_id = NEW.run_id;
            SELECT compiler_schema_version INTO compiler_version
              FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
            expected_prompt_version := CASE compiler_version
                WHEN '0.3.0' THEN 'pmmc-questioner-verifier-v1'
                WHEN '0.4.0' THEN 'pmmc-questioner-verifier-v2'
                ELSE NULL
            END;
            IF call_row.model_call_id IS NULL
               OR generation_row.generation_id IS NULL
               OR generation_call.model_call_id IS NULL
               OR expected_prompt_version IS NULL
               OR call_row.compiler_unit_id != NEW.compiler_unit_id
               OR call_row.stage != 'questioner' OR call_row.status != 'success'
               OR call_row.parent_model_call_id != generation_row.model_call_id
               OR call_row.model_alias != generation_call.model_alias
               OR call_row.actual_model_id IS DISTINCT FROM generation_call.actual_model_id
               OR call_row.job_id IS DISTINCT FROM generation_call.job_id
               OR NEW.prompt_version != expected_prompt_version
               OR COALESCE(call_row.request_config->'request_metadata'->>'task', '')
                  != 'pmmc_questioner_verifier' THEN
                RAISE EXCEPTION 'Questioner verifier batch has inconsistent model-call lineage'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_verifier BEFORE INSERT OR UPDATE "
        f"ON {RUNTIME}.questioner_verification_batches FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.enforce_questioner_verifier_scope_v2()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_questioner_verifier_scope_v2() "
        "FROM PUBLIC"
    )


def _replace_embedding_lineage_guard() -> None:
    """Allow only run-scoped Planner query vectors to reference QBank rows."""

    op.execute(
        f"DROP TRIGGER IF EXISTS validate_qbank_embedding ON {RUNTIME}.embeddings"
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_compiler_embedding_lineage_v2()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE source_kind varchar(64); collection_snapshot varchar(128);
                collection_run varchar(128); collection_metadata jsonb; valid boolean;
        BEGIN
            SELECT source_type, snapshot_id, run_id, metadata
              INTO source_kind, collection_snapshot, collection_run, collection_metadata
              FROM {RUNTIME}.embedding_collections
             WHERE collection_id = NEW.collection_id;
            IF source_kind = 'qbank_question' THEN
                SELECT true INTO valid FROM {RUNTIME}.qbank_questions
                 WHERE qbank_id = NEW.qbank_id
                   AND snapshot_id = collection_snapshot
                   AND run_id = collection_run
                   AND NEW.source_id = qbank_id
                   AND NEW.paraphrase_id IS NULL;
            ELSIF source_kind = 'qbank_paraphrase' THEN
                SELECT true INTO valid FROM {RUNTIME}.qbank_paraphrases
                 WHERE paraphrase_id = NEW.paraphrase_id
                   AND snapshot_id = collection_snapshot
                   AND run_id = collection_run
                   AND NEW.source_id = paraphrase_id
                   AND NEW.qbank_id IS NULL;
            ELSIF source_kind = 'query'
              AND collection_metadata->>'query_role' IN (
                  'question_text_query', 'question_joint_query'
              )
              AND collection_metadata->>'expected_images_allowed' = 'false'
              AND EXISTS (
                  SELECT 1 FROM {RUNTIME}.compiler_runs compiler
                   WHERE compiler.run_id = collection_run
                     AND compiler.snapshot_id = collection_snapshot
                     AND compiler.compiler_schema_version = '0.4.0'
                     AND compiler.target_stage = 'programs'
              ) THEN
                SELECT true INTO valid FROM {RUNTIME}.qbank_questions
                 WHERE qbank_id = NEW.qbank_id
                   AND snapshot_id = collection_snapshot
                   AND run_id = collection_run
                   AND NEW.source_id = qbank_id
                   AND NEW.paraphrase_id IS NULL;
            ELSE
                valid := NEW.qbank_id IS NULL AND NEW.paraphrase_id IS NULL;
            END IF;
            IF COALESCE(valid, false) = false THEN
                RAISE EXCEPTION 'embedding typed source does not match collection source type'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER validate_qbank_embedding BEFORE INSERT OR UPDATE "
        f"ON {RUNTIME}.embeddings FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.validate_compiler_embedding_lineage_v2()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.validate_compiler_embedding_lineage_v2() "
        "FROM PUBLIC"
    )


def _replace_compiler_completion_guard() -> None:
    """Keep the 0007 denominator checks while delegating v0.4 stage accounting."""

    op.execute(
        f"DROP TRIGGER IF EXISTS compiler_completion_integrity ON {RUNTIME}.run_states"
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_compiler_completion_v2()
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
                      AND results.run_id = units.run_id
                      AND results.stage = stage_name AND results.status = 'success'
                )
            );
            IF bad != 0 THEN
                RAISE EXCEPTION 'compiler target stage results are incomplete' USING ERRCODE = '55000';
            END IF;
            IF compiler.compiler_schema_version != '0.4.0' THEN
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
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER compiler_completion_integrity BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_compiler_completion_v2()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_compiler_completion_v2() FROM PUBLIC"
    )


def _create_completion_guard() -> None:
    required_identity = " OR ".join(f"compiler.{name} IS NULL" for name in _COMPILER_IDENTITY_COLUMNS)
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_planner_v1_completion()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE compiler_version text; bad integer;
        BEGIN
            IF NEW.status != 'completed' THEN RETURN NEW; END IF;
            SELECT compiler_schema_version INTO compiler_version
              FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
            IF compiler_version IS DISTINCT FROM '0.4.0' THEN RETURN NEW; END IF;
            IF EXISTS (SELECT 1 FROM {RUNTIME}.compiler_runs compiler
                WHERE compiler.run_id = NEW.run_id AND (
                    compiler.target_stage != 'programs'
                    OR compiler.operator_set_version != 'pmmc-operators-v2'
                    OR compiler.questioner_prompt_version != 'pmmc-questioner-v3.1'
                    OR compiler.questioner_renderer_version != 'questioner-safe-alias-renderer-v3.1'
                    OR compiler.questioner_parser_version != 'questioner-json-parser-v3.1'
                    OR compiler.candidate_gate_version != 'questioner-static-semantic-gate-v3.1'
                    OR compiler.planner_prompt_version != 'pmmc-planner-v1'
                    OR compiler.program_policy_version != 'planner-program-policy-v1'
                    OR compiler.dry_run_policy_version != 'evidence_driven_v2'
                    OR {required_identity})) THEN
                RAISE EXCEPTION 'Planner v1 compiler identity is incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad FROM {RUNTIME}.compiler_units units
             WHERE units.run_id = NEW.run_id AND (
                (SELECT count(*) FROM {RUNTIME}.questioner_opportunity_sets sets
                  WHERE sets.compiler_unit_id = units.compiler_unit_id
                    AND sets.run_id = units.run_id) != 1
                OR (SELECT count(*) FROM {RUNTIME}.questioner_generations generations
                  WHERE generations.compiler_unit_id = units.compiler_unit_id
                    AND generations.run_id = units.run_id
                    AND generations.parse_status != 'failed') != 1
                OR EXISTS (SELECT 1 FROM (VALUES ('questioner'),('planner'),('dry_run')) AS stages(stage)
                  WHERE (SELECT count(*) FROM {RUNTIME}.compiler_unit_stage_results results
                    WHERE results.compiler_unit_id = units.compiler_unit_id
                      AND results.run_id = units.run_id
                      AND results.stage = stages.stage
                      AND results.status = 'success') != 1)
                OR EXISTS (SELECT 1 FROM {RUNTIME}.compiler_unit_stage_results results
                  JOIN {RUNTIME}.questioner_generations generations
                    ON generations.compiler_unit_id = results.compiler_unit_id
                   AND generations.run_id = results.run_id
                   AND generations.parse_status != 'failed'
                  WHERE results.compiler_unit_id = units.compiler_unit_id
                    AND results.run_id = units.run_id
                    AND results.stage = 'questioner'
                    AND (results.produced_count != generations.raw_candidate_count
                      OR results.accepted_count != (
                          SELECT count(*) FROM {RUNTIME}.questioner_candidate_gate_results gates
                          WHERE gates.compiler_unit_id = units.compiler_unit_id
                            AND gates.run_id = units.run_id
                            AND gates.status = 'eligible_for_planning')
                      OR results.rejected_count !=
                          generations.raw_candidate_count - results.accepted_count))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions questions
                  JOIN {RUNTIME}.questioner_candidate_gate_results gates USING (qbank_id, run_id)
                  WHERE questions.compiler_unit_id = units.compiler_unit_id
                    AND gates.status = 'eligible_for_planning'
                    AND NOT EXISTS (SELECT 1 FROM {RUNTIME}.planner_candidate_outcomes outcomes
                      WHERE outcomes.qbank_id = questions.qbank_id))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions questions
                  JOIN {RUNTIME}.questioner_candidate_gate_results gates USING (qbank_id, run_id)
                  WHERE questions.compiler_unit_id = units.compiler_unit_id
                    AND gates.status = 'eligible_for_planning'
                    AND NOT EXISTS (SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications verifies
                      WHERE verifies.qbank_id = questions.qbank_id
                        AND verifies.run_id = questions.run_id
                        AND verifies.verdict = 'pass'))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.memory_programs programs
                  LEFT JOIN {RUNTIME}.planner_candidate_outcomes outcomes
                    ON outcomes.plan_id = programs.plan_id AND outcomes.run_id = programs.run_id
                  WHERE programs.compiler_unit_id = units.compiler_unit_id AND (
                    outcomes.planner_outcome_id IS NULL
                    OR (SELECT count(*) FROM {RUNTIME}.program_validation_results validations
                          WHERE validations.plan_id = programs.plan_id
                            AND validations.valid
                            AND validations.validation_kind IN ('static','policy')) != 2
                    OR (outcomes.status = 'lineage_rejected' AND (
                          (SELECT count(*) FROM {RUNTIME}.program_validation_results validations
                            WHERE validations.plan_id = programs.plan_id
                              AND NOT validations.valid
                              AND validations.validation_kind = 'lineage') != 1
                          OR EXISTS (SELECT 1 FROM {RUNTIME}.program_dry_run_assessments assessments
                            WHERE assessments.plan_id = programs.plan_id)))
                    OR (outcomes.status = 'query_vector_unavailable' AND EXISTS (
                          SELECT 1 FROM {RUNTIME}.program_dry_run_assessments assessments
                          WHERE assessments.plan_id = programs.plan_id))
                    OR (outcomes.status NOT IN ('lineage_rejected','query_vector_unavailable') AND (
                          (SELECT count(*) FROM {RUNTIME}.program_validation_results validations
                            WHERE validations.plan_id = programs.plan_id
                              AND validations.valid
                              AND validations.validation_kind = 'lineage') != 1
                          OR NOT EXISTS (SELECT 1 FROM {RUNTIME}.program_dry_run_assessments assessments
                            WHERE assessments.plan_id = programs.plan_id)))))
                OR EXISTS (SELECT 1 FROM (VALUES ('planner'),('dry_run')) AS stages(stage)
                  WHERE (SELECT count(*) FROM {RUNTIME}.compiler_stage_jobs jobs
                    WHERE jobs.compiler_unit_id = units.compiler_unit_id
                      AND jobs.run_id = units.run_id
                      AND jobs.stage = stages.stage
                      AND jobs.round_id = 0
                      AND jobs.status = 'succeeded') != 1)
                OR EXISTS (SELECT 1 FROM {RUNTIME}.compiler_unit_stage_results results
                  WHERE results.compiler_unit_id = units.compiler_unit_id
                    AND results.run_id = units.run_id
                    AND results.stage = 'planner'
                    AND (results.produced_count != (
                          SELECT count(*) FROM {RUNTIME}.qbank_questions questions
                          JOIN {RUNTIME}.questioner_candidate_gate_results gates
                            USING (qbank_id, run_id)
                          WHERE questions.compiler_unit_id = units.compiler_unit_id
                            AND gates.status = 'eligible_for_planning')
                      OR results.accepted_count != (
                          SELECT count(*) FROM {RUNTIME}.memory_programs programs
                          WHERE programs.compiler_unit_id = units.compiler_unit_id)
                      OR results.rejected_count != results.produced_count - results.accepted_count))
                OR EXISTS (SELECT 1 FROM {RUNTIME}.compiler_unit_stage_results results
                  WHERE results.compiler_unit_id = units.compiler_unit_id
                    AND results.run_id = units.run_id
                    AND results.stage = 'dry_run'
                    AND (results.produced_count != (
                          SELECT count(*) FROM {RUNTIME}.memory_programs programs
                          WHERE programs.compiler_unit_id = units.compiler_unit_id)
                      OR results.accepted_count != (
                          SELECT count(*) FROM {RUNTIME}.planner_candidate_outcomes outcomes
                          WHERE outcomes.compiler_unit_id = units.compiler_unit_id
                            AND outcomes.plan_id IS NOT NULL
                            AND outcomes.status IN ('program_ready','program_ready_with_fallback',
                              'program_ready_with_abstain'))
                      OR results.rejected_count != results.produced_count - results.accepted_count))
             );
            IF bad != 0 THEN
                RAISE EXCEPTION 'Planner v1 unit, outcome, validation, or dry-run facts are incomplete' USING ERRCODE = '55000';
            END IF;
            IF (SELECT count(*) FROM {RUNTIME}.memory_operator_index_manifests indexes
                 WHERE indexes.run_id = NEW.run_id AND indexes.status = 'ready') != 1 THEN
                RAISE EXCEPTION 'Planner v1 requires one ready memory operator index' USING ERRCODE = '55000';
            END IF;
            IF EXISTS (SELECT 1 FROM {RUNTIME}.memory_operator_index_manifests indexes
              JOIN {RUNTIME}.embedding_collections text_collection
                ON text_collection.collection_id = indexes.text_collection_id
              JOIN {RUNTIME}.embedding_collections visual_collection
                ON visual_collection.collection_id = indexes.visual_collection_id
              WHERE indexes.run_id = NEW.run_id
                AND (text_collection.status != 'ready'
                  OR visual_collection.status != 'ready'
                  OR text_collection.dimension != 4096
                  OR visual_collection.dimension != 4096)) THEN
                RAISE EXCEPTION 'Planner v1 memory collections are incomplete' USING ERRCODE = '55000';
            END IF;
            IF EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions questions
              JOIN {RUNTIME}.questioner_candidate_gate_results gates USING (qbank_id, run_id)
              WHERE questions.run_id = NEW.run_id
                AND gates.status = 'eligible_for_planning'
                AND (SELECT count(*) FROM {RUNTIME}.program_query_embeddings vectors
                  WHERE vectors.run_id = questions.run_id
                    AND vectors.qbank_id = questions.qbank_id
                    AND vectors.channel IN ('question_text_query','question_joint_query')) != 2) THEN
                RAISE EXCEPTION 'Planner v1 query-vector denominator is incomplete' USING ERRCODE = '55000';
            END IF;
            IF (SELECT count(*) FROM {RUNTIME}.qbank_index_manifests indexes
                 WHERE indexes.compiler_run_id = NEW.run_id
                   AND indexes.status = 'ready') != 1 THEN
                RAISE EXCEPTION 'Planner v1 requires one ready QBank index' USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER planner_v1_completion_integrity BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_planner_v1_completion()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_planner_v1_completion() FROM PUBLIC"
    )


def _grant_permissions() -> None:
    qualified = ", ".join(f"{RUNTIME}.{table}" for table in _TABLES)
    op.execute(f"GRANT SELECT ON {qualified} TO pmmc_runtime, pmmc_evaluator")
    op.execute(f"GRANT INSERT, UPDATE, DELETE ON {qualified} TO pmmc_runtime")


def downgrade() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS planner_v1_completion_integrity ON {RUNTIME}.run_states"
    )
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_planner_v1_completion()")
    _restore_embedding_lineage_guard()
    _restore_questioner_verifier_guard()
    _restore_compiler_completion_guard()
    for table in reversed(_TABLES):
        op.execute(f"DROP TRIGGER IF EXISTS freeze_terminal_run ON {RUNTIME}.{table}")
        op.execute(f"DROP TABLE {RUNTIME}.{table}")
    op.drop_constraint(
        "uq_memory_programs_run_unit",
        "memory_programs",
        schema=RUNTIME,
        type_="unique",
    )
    op.drop_constraint(
        "uq_qbank_questions_run_unit",
        "qbank_questions",
        schema=RUNTIME,
        type_="unique",
    )
    for name in reversed(_COMPILER_IDENTITY_COLUMNS):
        op.drop_column("compiler_runs", name, schema=RUNTIME)


def _restore_questioner_verifier_guard() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_verifier "
        f"ON {RUNTIME}.questioner_verification_batches"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_questioner_verifier_scope_v2()"
    )
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_verifier BEFORE INSERT OR UPDATE "
        f"ON {RUNTIME}.questioner_verification_batches FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.enforce_questioner_v3_scope()"
    )


def _restore_embedding_lineage_guard() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_qbank_embedding ON {RUNTIME}.embeddings"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.validate_compiler_embedding_lineage_v2()"
    )
    op.execute(
        f"CREATE TRIGGER validate_qbank_embedding BEFORE INSERT OR UPDATE "
        f"ON {RUNTIME}.embeddings FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.validate_qbank_embedding_lineage()"
    )


def _restore_compiler_completion_guard() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS compiler_completion_integrity ON {RUNTIME}.run_states"
    )
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_compiler_completion_v2()")
    op.execute(
        f"CREATE TRIGGER compiler_completion_integrity BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_compiler_completion()"
    )
