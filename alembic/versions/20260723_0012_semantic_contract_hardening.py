# ruff: noqa: E501
"""Harden Questioner, QBank semantics, and non-oracle Planner contracts.

Revision ID: 20260723_0012
Revises: 20260722_0011
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260723_0012"
down_revision: str | None = "20260722_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"
MATCH_ROLE_CHECK = "ck_qbank_coverage_gold_group_members_match_role_values"

_RUNTIME_TABLES = (
    "structured_output_item_diagnostics",
    "planner_requirement_contracts",
    "planner_candidate_drafts",
)
_EVALUATION_TABLES = (
    "qbank_semantic_match_runs",
    "qbank_semantic_match_jobs",
    "qbank_semantic_match_assessments",
    "qbank_semantic_match_metrics",
    "structured_output_item_diagnostics",
)


def upgrade() -> None:
    _extend_questioner_contract()
    _extend_planner_contract()
    _create_runtime_diagnostics()
    _create_semantic_tables()
    _create_semantic_scope_guard()
    _replace_questioner_scope_guard()
    _create_r0_completion_guard()
    _create_semantic_freeze_guard()
    _grant_permissions()


def _extend_questioner_contract() -> None:
    op.add_column(
        "questioner_opportunities",
        sa.Column(
            "speaker_scope",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("ARRAY[]::text[]"),
        ),
        schema=RUNTIME,
    )
    op.add_column(
        "questioner_opportunities",
        sa.Column(
            "evidence_group_ids",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("ARRAY[]::text[]"),
        ),
        schema=RUNTIME,
    )
    op.add_column(
        "questioner_opportunities",
        sa.Column("relation_content_hash", sa.String(length=128), nullable=True),
        schema=RUNTIME,
    )
    op.alter_column(
        "questioner_opportunities",
        "speaker_scope",
        server_default=None,
        schema=RUNTIME,
    )
    op.alter_column(
        "questioner_opportunities",
        "evidence_group_ids",
        server_default=None,
        schema=RUNTIME,
    )

    op.add_column(
        "questioner_verification_batches",
        sa.Column("scope_qbank_id", sa.String(length=128), nullable=True),
        schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_questioner_verification_batches_scope_qbank",
        "questioner_verification_batches",
        "qbank_questions",
        ["scope_qbank_id", "run_id"],
        ["qbank_id", "run_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    # 0010 used an unnamed three-column UNIQUE. Locate it by its exact key set
    # instead of relying on PostgreSQL's truncated generated name.
    op.execute(
        f"""DO $$
        DECLARE constraint_name text;
        BEGIN
          SELECT con.conname INTO constraint_name
          FROM pg_constraint con
          JOIN pg_class rel ON rel.oid = con.conrelid
          JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
          WHERE nsp.nspname = '{RUNTIME}'
            AND rel.relname = 'questioner_verification_batches'
            AND con.contype = 'u'
            AND (
              SELECT array_agg(
                att.attname::text ORDER BY key_position.ordinality
              )
              FROM unnest(con.conkey) WITH ORDINALITY AS key_position(attnum, ordinality)
              JOIN pg_attribute att
                ON att.attrelid = rel.oid AND att.attnum = key_position.attnum
            ) = ARRAY[
              'compiler_unit_id','generation_id','verification_round'
            ]::text[];
          IF constraint_name IS NOT NULL THEN
            EXECUTE format(
              'ALTER TABLE {RUNTIME}.questioner_verification_batches DROP CONSTRAINT %I',
              constraint_name
            );
          END IF;
        END $$"""
    )
    op.execute(
        f"""CREATE UNIQUE INDEX uq_questioner_verification_batches_candidate_scope
        ON {RUNTIME}.questioner_verification_batches(
          compiler_unit_id, generation_id, verification_round,
          COALESCE(scope_qbank_id, '')
        )"""
    )

    op.add_column(
        "qbank_coverage_gold_group_members",
        sa.Column(
            "match_role",
            sa.String(length=32),
            nullable=False,
            server_default="discriminative",
        ),
        schema=EVALUATION,
    )
    op.create_check_constraint(
        op.f(MATCH_ROLE_CHECK),
        "qbank_coverage_gold_group_members",
        "match_role IN ('discriminative','contextual')",
        schema=EVALUATION,
    )
    op.alter_column(
        "qbank_coverage_gold_group_members",
        "match_role",
        server_default=None,
        schema=EVALUATION,
    )
    op.create_unique_constraint(
        "uq_qbank_coverage_runs_semantic_parent",
        "qbank_coverage_runs",
        ["coverage_run_id", "compiler_run_id", "snapshot_id"],
        schema=EVALUATION,
    )


def _extend_planner_contract() -> None:
    op.execute(
        f"""CREATE TABLE {RUNTIME}.planner_requirement_contracts (
            requirement_contract_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL UNIQUE,
            required_modalities text[] NOT NULL CHECK (
                cardinality(required_modalities) > 0
                AND required_modalities <@
                  ARRAY['text','caption','image','raw_image']::text[]),
            minimum_evidence_groups integer NOT NULL CHECK (minimum_evidence_groups > 0),
            required_capabilities text[] NOT NULL,
            fallback_required boolean NOT NULL,
            refusal_only boolean NOT NULL,
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (requirement_contract_id, run_id, compiler_unit_id),
            FOREIGN KEY (qbank_id, run_id, compiler_unit_id)
              REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id, compiler_unit_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.planner_candidate_drafts (
            planner_candidate_draft_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            planner_batch_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL UNIQUE,
            requirement_contract_id varchar(128) NOT NULL,
            item_ordinal integer NOT NULL CHECK (item_ordinal >= 0),
            item_hash varchar(128) NOT NULL,
            parsed boolean NOT NULL,
            program_hash varchar(128),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (planner_batch_id, item_ordinal),
            FOREIGN KEY (planner_batch_id, run_id, compiler_unit_id)
              REFERENCES {RUNTIME}.planner_generation_batches(
                planner_batch_id, run_id, compiler_unit_id),
            FOREIGN KEY (qbank_id, run_id, compiler_unit_id)
              REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id, compiler_unit_id),
            FOREIGN KEY (requirement_contract_id, run_id, compiler_unit_id)
              REFERENCES {RUNTIME}.planner_requirement_contracts(
                requirement_contract_id, run_id, compiler_unit_id),
            CHECK ((parsed AND program_hash IS NOT NULL)
                OR (NOT parsed AND program_hash IS NULL))
        )"""
    )

    assessment_columns = (
        ("successful_step_count", "integer NOT NULL DEFAULT 0"),
        ("required_modality_sufficient", "boolean NOT NULL DEFAULT false"),
        ("discriminative_group_count", "integer NOT NULL DEFAULT 0"),
        ("matched_discriminative_group_count", "integer NOT NULL DEFAULT 0"),
        ("contextual_match_count", "integer NOT NULL DEFAULT 0"),
        ("pre_control_sufficient", "boolean NOT NULL DEFAULT false"),
        ("fallback_triggered", "boolean NOT NULL DEFAULT false"),
        ("abstain_triggered", "boolean NOT NULL DEFAULT false"),
        (
            "semantic_answer_support",
            "varchar(32) NOT NULL DEFAULT 'not_evaluated'",
        ),
    )
    for name, definition in assessment_columns:
        op.execute(
            f"ALTER TABLE {RUNTIME}.program_dry_run_assessments "
            f"ADD COLUMN {name} {definition}"
        )
    op.execute(
        f"""ALTER TABLE {RUNTIME}.program_dry_run_assessments
        ADD CHECK (successful_step_count >= 0),
        ADD CHECK (discriminative_group_count >= 0),
        ADD CHECK (matched_discriminative_group_count >= 0),
        ADD CHECK (matched_discriminative_group_count <= discriminative_group_count),
        ADD CHECK (contextual_match_count >= 0),
        ADD CHECK (NOT (fallback_triggered AND abstain_triggered)),
        ADD CHECK (semantic_answer_support IN ('not_evaluated','supported','unsupported'))"""
    )

    match_columns = (
        ("direct_match", "boolean NOT NULL DEFAULT false"),
        ("derived_match", "boolean NOT NULL DEFAULT false"),
        ("contextual_only", "boolean NOT NULL DEFAULT false"),
        ("discriminative_match", "boolean NOT NULL DEFAULT false"),
    )
    for name, definition in match_columns:
        op.execute(
            f"ALTER TABLE {RUNTIME}.program_dry_run_evidence_matches "
            f"ADD COLUMN {name} {definition}"
        )
    op.execute(
        f"""ALTER TABLE {RUNTIME}.program_dry_run_evidence_matches
        ADD CHECK (NOT (direct_match AND derived_match)),
        ADD CHECK (NOT (contextual_only AND discriminative_match))"""
    )


def _create_runtime_diagnostics() -> None:
    op.execute(
        f"""CREATE TABLE {RUNTIME}.structured_output_item_diagnostics (
            diagnostic_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            model_call_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            semantic_run_id varchar(128),
            stage varchar(32) NOT NULL CHECK (stage IN (
              'questioner','questioner_verifier','questioner_revision','planner')),
            item_ordinal integer NOT NULL CHECK (item_ordinal >= 0),
            item_hash varchar(128) NOT NULL,
            status varchar(32) NOT NULL CHECK (status IN (
              'valid','invalid','repaired','conservative_reject')),
            parser_version varchar(128) NOT NULL,
            repair_attempt_count integer NOT NULL CHECK (repair_attempt_count >= 0),
            errors jsonb NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (model_call_id, stage, item_ordinal),
            FOREIGN KEY (model_call_id, run_id)
              REFERENCES {RUNTIME}.model_calls(model_call_id, run_id),
            FOREIGN KEY (compiler_unit_id, run_id)
              REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            CHECK (semantic_run_id IS NULL)
        )"""
    )
    for table in _RUNTIME_TABLES:
        op.execute(
            f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE "
            f"ON {RUNTIME}.{table} FOR EACH ROW "
            f"EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()"
        )


def _create_semantic_tables() -> None:
    op.execute(
        f"""ALTER TABLE {EVALUATION}.model_calls
        ALTER COLUMN evaluation_run_id DROP NOT NULL,
        ADD COLUMN semantic_run_id varchar(128),
        ADD COLUMN semantic_job_id varchar(128),
        ADD CONSTRAINT ck_model_calls_exactly_one_evaluation_parent CHECK (
          ((evaluation_run_id IS NOT NULL)::integer
           + (semantic_run_id IS NOT NULL)::integer) = 1),
        ADD CONSTRAINT ck_model_calls_semantic_scope_complete CHECK (
          (semantic_run_id IS NULL AND semantic_job_id IS NULL)
          OR (semantic_run_id IS NOT NULL AND semantic_job_id IS NOT NULL)),
        ADD CONSTRAINT uq_model_calls_semantic_run
          UNIQUE (model_call_id, semantic_run_id),
        ADD CONSTRAINT uq_model_calls_semantic_scope
          UNIQUE (model_call_id, semantic_run_id, semantic_job_id)"""
    )
    op.execute(
        f"""CREATE TABLE {EVALUATION}.qbank_semantic_match_runs (
            semantic_run_id varchar(128) PRIMARY KEY,
            schema_version varchar(32) NOT NULL,
            protocol_version varchar(128) NOT NULL,
            coverage_run_id varchar(128) NOT NULL UNIQUE,
            compiler_run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            dataset varchar(32) NOT NULL CHECK (dataset IN ('memlens','memgallery','unknown')),
            judge_model_alias varchar(256) NOT NULL,
            judge_model_revision varchar(256) NOT NULL,
            prompt_version varchar(128) NOT NULL,
            prompt_hash varchar(128) NOT NULL,
            renderer_version varchar(128) NOT NULL,
            parser_version varchar(128) NOT NULL,
            config_hash varchar(128) NOT NULL,
            pair_count integer NOT NULL CHECK (pair_count >= 0),
            status varchar(16) NOT NULL CHECK (
              status IN ('created','running','completed','failed','cancelled')),
            created_at timestamptz NOT NULL,
            completed_at timestamptz,
            metadata jsonb NOT NULL,
            UNIQUE (semantic_run_id, coverage_run_id),
            FOREIGN KEY (coverage_run_id, compiler_run_id, snapshot_id)
              REFERENCES {EVALUATION}.qbank_coverage_runs(
                coverage_run_id, compiler_run_id, snapshot_id),
            CHECK ((status IN ('completed','failed','cancelled') AND completed_at IS NOT NULL)
                OR (status NOT IN ('completed','failed','cancelled') AND completed_at IS NULL))
        )"""
    )
    op.execute(
        f"""ALTER TABLE {EVALUATION}.run_artifacts
        ALTER COLUMN evaluation_run_id DROP NOT NULL,
        ADD COLUMN semantic_run_id varchar(128),
        ADD CONSTRAINT fk_run_artifacts_semantic_run
          FOREIGN KEY (semantic_run_id)
          REFERENCES {EVALUATION}.qbank_semantic_match_runs(semantic_run_id),
        ADD CONSTRAINT ck_run_artifacts_exactly_one_evaluation_parent CHECK (
          ((evaluation_run_id IS NOT NULL)::integer
           + (semantic_run_id IS NOT NULL)::integer) = 1)"""
    )
    # The 0006 compatibility trigger assigns a legacy evaluation parent whenever
    # evaluation_run_id is NULL. Semantic calls and artifacts intentionally use a
    # different parent, so keep legacy assignment only for baseline rows.
    for table in ("run_artifacts", "model_calls"):
        op.execute(
            f"DROP TRIGGER IF EXISTS a_assign_legacy_evaluation_run "
            f"ON {EVALUATION}.{table}"
        )
        op.execute(
            f"CREATE TRIGGER a_assign_legacy_evaluation_run BEFORE INSERT "
            f"ON {EVALUATION}.{table} FOR EACH ROW "
            f"WHEN (NEW.semantic_run_id IS NULL) "
            f"EXECUTE FUNCTION {EVALUATION}.assign_legacy_evaluation_run()"
        )
        op.execute(
            f"DROP TRIGGER IF EXISTS z_freeze_evaluation_run "
            f"ON {EVALUATION}.{table}"
        )
        op.execute(
            f"CREATE TRIGGER z_freeze_evaluation_run_insert "
            f"BEFORE INSERT ON {EVALUATION}.{table} "
            f"FOR EACH ROW "
            f"WHEN (NEW.semantic_run_id IS NULL) "
            f"EXECUTE FUNCTION {EVALUATION}.enforce_evaluation_run_writable()"
        )
        op.execute(
            f"CREATE TRIGGER z_freeze_evaluation_run_update "
            f"BEFORE UPDATE ON {EVALUATION}.{table} "
            f"FOR EACH ROW "
            f"WHEN (OLD.semantic_run_id IS NULL OR NEW.semantic_run_id IS NULL) "
            f"EXECUTE FUNCTION {EVALUATION}.enforce_evaluation_run_writable()"
        )
        op.execute(
            f"CREATE TRIGGER z_freeze_evaluation_run_delete "
            f"BEFORE DELETE ON {EVALUATION}.{table} "
            f"FOR EACH ROW "
            f"WHEN (OLD.semantic_run_id IS NULL) "
            f"EXECUTE FUNCTION {EVALUATION}.enforce_evaluation_run_writable()"
        )
    op.execute(
        f"""CREATE TABLE {EVALUATION}.qbank_semantic_match_jobs (
            semantic_job_id varchar(128) PRIMARY KEY,
            semantic_run_id varchar(128) NOT NULL,
            coverage_run_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            pair_hash varchar(128) NOT NULL,
            status varchar(16) NOT NULL CHECK (
              status IN ('pending','running','succeeded','failed','cancelled')),
            attempt_count integer NOT NULL CHECK (attempt_count >= 0),
            max_attempts integer NOT NULL CHECK (max_attempts > 0),
            model_call_id varchar(128),
            created_at timestamptz NOT NULL,
            finished_at timestamptz,
            metadata jsonb NOT NULL,
            UNIQUE (semantic_run_id, query_id, qbank_id),
            UNIQUE (semantic_job_id, semantic_run_id),
            UNIQUE (
              semantic_job_id, semantic_run_id, coverage_run_id, query_id, qbank_id
            ),
            FOREIGN KEY (semantic_run_id, coverage_run_id)
              REFERENCES {EVALUATION}.qbank_semantic_match_runs(
                semantic_run_id, coverage_run_id),
            FOREIGN KEY (coverage_run_id, query_id)
              REFERENCES {EVALUATION}.qbank_coverage_queries(
                coverage_run_id, query_id),
            CHECK ((status IN ('succeeded','failed','cancelled') AND finished_at IS NOT NULL)
                OR (status NOT IN ('succeeded','failed','cancelled') AND finished_at IS NULL)),
            CHECK (status != 'succeeded' OR model_call_id IS NOT NULL)
        )"""
    )
    op.create_foreign_key(
        "fk_evaluation_model_calls_semantic_job",
        "model_calls",
        "qbank_semantic_match_jobs",
        ["semantic_job_id", "semantic_run_id"],
        ["semantic_job_id", "semantic_run_id"],
        source_schema=EVALUATION,
        referent_schema=EVALUATION,
    )
    op.execute(
        f"""CREATE TABLE {EVALUATION}.qbank_semantic_match_assessments (
            semantic_assessment_id varchar(128) PRIMARY KEY,
            semantic_run_id varchar(128) NOT NULL,
            semantic_job_id varchar(128) NOT NULL UNIQUE,
            coverage_run_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            model_call_id varchar(128) NOT NULL,
            dimensions jsonb NOT NULL,
            reason_codes text[] NOT NULL,
            semantic_dimensions_passed boolean NOT NULL,
            evidence_proxy_complete boolean NOT NULL,
            semantic_match boolean NOT NULL,
            elapsed_ms double precision NOT NULL CHECK (elapsed_ms >= 0),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            FOREIGN KEY (
              semantic_job_id, semantic_run_id, coverage_run_id, query_id, qbank_id
            )
              REFERENCES {EVALUATION}.qbank_semantic_match_jobs(
                semantic_job_id, semantic_run_id, coverage_run_id, query_id, qbank_id),
            FOREIGN KEY (model_call_id, semantic_run_id, semantic_job_id)
              REFERENCES {EVALUATION}.model_calls(
                model_call_id, semantic_run_id, semantic_job_id),
            CHECK (semantic_match = (
              semantic_dimensions_passed AND evidence_proxy_complete))
        )"""
    )
    op.execute(
        f"""CREATE TABLE {EVALUATION}.qbank_semantic_match_metrics (
            semantic_metric_id varchar(128) PRIMARY KEY,
            semantic_run_id varchar(128) NOT NULL,
            query_id varchar(256),
            metric_name varchar(128) NOT NULL,
            metric_version varchar(128) NOT NULL,
            question_budget integer NOT NULL CHECK (question_budget > 0),
            k integer NOT NULL CHECK (k > 0),
            value double precision NOT NULL CHECK (value >= 0 AND value <= 1),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (semantic_run_id, query_id, metric_name, question_budget, k),
            FOREIGN KEY (semantic_run_id)
              REFERENCES {EVALUATION}.qbank_semantic_match_runs(semantic_run_id)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {EVALUATION}.structured_output_item_diagnostics (
            diagnostic_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            model_call_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128),
            semantic_run_id varchar(128) NOT NULL,
            stage varchar(32) NOT NULL CHECK (stage = 'qbank_semantic_judge'),
            item_ordinal integer NOT NULL CHECK (item_ordinal >= 0),
            item_hash varchar(128) NOT NULL,
            status varchar(32) NOT NULL CHECK (status IN (
              'valid','invalid','repaired','conservative_reject')),
            parser_version varchar(128) NOT NULL,
            repair_attempt_count integer NOT NULL CHECK (repair_attempt_count >= 0),
            errors jsonb NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (model_call_id, stage, item_ordinal),
            FOREIGN KEY (model_call_id, semantic_run_id)
              REFERENCES {EVALUATION}.model_calls(model_call_id, semantic_run_id),
            CHECK (compiler_unit_id IS NULL)
        )"""
    )


def _create_semantic_scope_guard() -> None:
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.enforce_semantic_job_candidate_scope()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {EVALUATION} AS $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1
            FROM {EVALUATION}.qbank_coverage_candidates candidate
            WHERE candidate.coverage_run_id = NEW.coverage_run_id
              AND candidate.query_id = NEW.query_id
              AND candidate.qbank_id = NEW.qbank_id
          ) THEN
            RAISE EXCEPTION
              'QBank semantic job candidate is outside its coverage scope'
              USING ERRCODE = '23514';
          END IF;
          RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER validate_semantic_job_candidate_scope "
        f"BEFORE INSERT OR UPDATE ON {EVALUATION}.qbank_semantic_match_jobs "
        f"FOR EACH ROW EXECUTE FUNCTION "
        f"{EVALUATION}.enforce_semantic_job_candidate_scope()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION "
        f"{EVALUATION}.enforce_semantic_job_candidate_scope() FROM PUBLIC"
    )


def _replace_questioner_scope_guard() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_verifier "
        f"ON {RUNTIME}.questioner_verification_batches"
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_questioner_verifier_scope_v3()
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
                WHEN '0.5.0' THEN 'pmmc-questioner-verifier-v3'
                ELSE NULL
            END;
            IF call_row.model_call_id IS NULL
               OR generation_row.generation_id IS NULL
               OR generation_call.model_call_id IS NULL
               OR expected_prompt_version IS NULL
               OR call_row.compiler_unit_id != NEW.compiler_unit_id
               OR call_row.stage != 'questioner'
               OR NOT (
                    call_row.status = 'success'
                    OR (
                      compiler_version = '0.5.0'
                      AND call_row.status = 'failed'
                      AND call_row.output_asset_id IS NOT NULL
                      AND call_row.parse_error IS NOT NULL
                      AND COALESCE(
                        (NEW.metadata->>'provider_output_fallback')::boolean,
                        false
                      )
                    )
               )
               OR call_row.parent_model_call_id != generation_row.model_call_id
               OR call_row.model_alias != generation_call.model_alias
               OR call_row.actual_model_id IS DISTINCT FROM generation_call.actual_model_id
               OR call_row.job_id IS DISTINCT FROM generation_call.job_id
               OR NEW.prompt_version != expected_prompt_version
               OR COALESCE(call_row.request_config->'request_metadata'->>'task', '')
                  != 'pmmc_questioner_verifier'
               OR (compiler_version = '0.5.0' AND (
                    NEW.scope_qbank_id IS NULL OR NEW.input_count != 1)) THEN
                RAISE EXCEPTION 'Questioner verifier batch has inconsistent candidate scope'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_verifier BEFORE INSERT OR UPDATE "
        f"ON {RUNTIME}.questioner_verification_batches FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.enforce_questioner_verifier_scope_v3()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_questioner_verifier_scope_v3() "
        "FROM PUBLIC"
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_questioner_candidate_scope_v3()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE batch_row record; compiler_version text;
        BEGIN
          SELECT * INTO batch_row
            FROM {RUNTIME}.questioner_verification_batches
            WHERE verification_batch_id = NEW.verification_batch_id
              AND run_id = NEW.run_id;
          SELECT compiler_schema_version INTO compiler_version
            FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
          IF compiler_version = '0.5.0'
             AND (batch_row.scope_qbank_id IS NULL
                  OR NEW.qbank_id != batch_row.scope_qbank_id) THEN
            RAISE EXCEPTION
              'Questioner candidate verification escapes its isolated scope'
              USING ERRCODE = '23514';
          END IF;
          RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER validate_questioner_candidate_scope_v3 "
        f"BEFORE INSERT OR UPDATE ON "
        f"{RUNTIME}.questioner_candidate_verifications FOR EACH ROW "
        f"EXECUTE FUNCTION "
        f"{RUNTIME}.enforce_questioner_candidate_scope_v3()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION "
        f"{RUNTIME}.enforce_questioner_candidate_scope_v3() FROM PUBLIC"
    )


def _create_r0_completion_guard() -> None:
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_r0_hardening_completion()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE compiler_row record;
        BEGIN
          IF NEW.status != 'completed' OR OLD.status = 'completed' THEN
            RETURN NEW;
          END IF;
          SELECT * INTO compiler_row FROM {RUNTIME}.compiler_runs
            WHERE run_id = NEW.run_id;
          IF compiler_row.compiler_schema_version != '0.5.0' THEN
            RETURN NEW;
          END IF;
          IF EXISTS (
              SELECT 1 FROM {RUNTIME}.questioner_verification_batches
              WHERE run_id = NEW.run_id
                AND (scope_qbank_id IS NULL OR input_count != 1))
             OR EXISTS (
              SELECT 1
              FROM {RUNTIME}.questioner_candidate_verifications verification
              JOIN {RUNTIME}.questioner_verification_batches batch
                USING (verification_batch_id, run_id)
              WHERE verification.run_id = NEW.run_id
                AND verification.qbank_id != batch.scope_qbank_id)
             OR EXISTS (
              SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results gate
              JOIN {RUNTIME}.qbank_questions question USING (qbank_id, run_id)
              WHERE gate.run_id = NEW.run_id
                AND gate.status = 'eligible_for_planning'
                AND NOT EXISTS (
                  SELECT 1 FROM {RUNTIME}.questioner_candidate_verifications verification
                  WHERE verification.qbank_id = question.qbank_id
                    AND verification.verdict = 'pass'))
             OR EXISTS (
              SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results gate
              WHERE gate.run_id = NEW.run_id
                AND gate.status = 'eligible_for_planning'
                AND (NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.planner_requirement_contracts contract
                    WHERE contract.qbank_id = gate.qbank_id)
                  OR NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.planner_candidate_outcomes outcome
                    WHERE outcome.qbank_id = gate.qbank_id)))
             OR EXISTS (
              SELECT 1 FROM {RUNTIME}.memory_programs program
              WHERE program.run_id = NEW.run_id
                AND NOT EXISTS (
                  SELECT 1 FROM {RUNTIME}.program_dry_run_assessments assessment
                  WHERE assessment.plan_id = program.plan_id)) THEN
            RAISE EXCEPTION 'compiler 0.5.0 r=0 semantic-contract facts are incomplete'
              USING ERRCODE = '23514';
          END IF;
          RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER r0_hardening_completion_integrity BEFORE UPDATE "
        f"ON {RUNTIME}.run_states FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.enforce_r0_hardening_completion()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_r0_hardening_completion() "
        "FROM PUBLIC"
    )


def _create_semantic_freeze_guard() -> None:
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.enforce_semantic_state_transition()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {EVALUATION} AS $$
        BEGIN
          IF OLD.status IN ('completed','failed','cancelled') THEN
            RAISE EXCEPTION 'terminal QBank semantic run state is immutable'
              USING ERRCODE = '55000';
          END IF;
          IF TG_OP = 'DELETE' THEN
            RETURN OLD;
          END IF;
          RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER freeze_terminal_semantic_state "
        f"BEFORE UPDATE OR DELETE ON {EVALUATION}.qbank_semantic_match_runs "
        f"FOR EACH ROW EXECUTE FUNCTION "
        f"{EVALUATION}.enforce_semantic_state_transition()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION "
        f"{EVALUATION}.enforce_semantic_state_transition() FROM PUBLIC"
    )
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.enforce_semantic_run_writable()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {EVALUATION} AS $$
        DECLARE parent_id text; parent_status text;
        BEGIN
          parent_id := COALESCE(NEW.semantic_run_id, OLD.semantic_run_id);
          SELECT status INTO parent_status
            FROM {EVALUATION}.qbank_semantic_match_runs
            WHERE semantic_run_id = parent_id;
          IF parent_status IN ('completed','failed','cancelled') THEN
            RAISE EXCEPTION 'terminal QBank semantic run is immutable'
              USING ERRCODE = '55000';
          END IF;
          IF TG_OP = 'DELETE' THEN
            RETURN OLD;
          END IF;
          RETURN NEW;
        END; $$"""
    )
    for table in (
        "qbank_semantic_match_jobs",
        "qbank_semantic_match_assessments",
        "qbank_semantic_match_metrics",
        "structured_output_item_diagnostics",
    ):
        op.execute(
            f"CREATE TRIGGER freeze_terminal_semantic_run "
            f"BEFORE INSERT OR UPDATE OR DELETE ON {EVALUATION}.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION "
            f"{EVALUATION}.enforce_semantic_run_writable()"
        )
    for table in ("run_artifacts", "model_calls"):
        op.execute(
            f"CREATE TRIGGER freeze_terminal_semantic_parent_insert "
            f"BEFORE INSERT ON {EVALUATION}.{table} "
            f"FOR EACH ROW "
            f"WHEN (NEW.semantic_run_id IS NOT NULL) "
            f"EXECUTE FUNCTION {EVALUATION}.enforce_semantic_run_writable()"
        )
        op.execute(
            f"CREATE TRIGGER freeze_terminal_semantic_parent_update "
            f"BEFORE UPDATE ON {EVALUATION}.{table} "
            f"FOR EACH ROW "
            f"WHEN (OLD.semantic_run_id IS NOT NULL "
            f"OR NEW.semantic_run_id IS NOT NULL) "
            f"EXECUTE FUNCTION {EVALUATION}.enforce_semantic_run_writable()"
        )
        op.execute(
            f"CREATE TRIGGER freeze_terminal_semantic_parent_delete "
            f"BEFORE DELETE ON {EVALUATION}.{table} "
            f"FOR EACH ROW "
            f"WHEN (OLD.semantic_run_id IS NOT NULL) "
            f"EXECUTE FUNCTION {EVALUATION}.enforce_semantic_run_writable()"
        )
    op.execute(
        f"REVOKE ALL ON FUNCTION {EVALUATION}.enforce_semantic_run_writable() "
        "FROM PUBLIC"
    )


def _grant_permissions() -> None:
    for table in _RUNTIME_TABLES:
        op.execute(f"ALTER TABLE {RUNTIME}.{table} OWNER TO pmmc_owner")
        op.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {RUNTIME}.{table} TO pmmc_runtime"
        )
        op.execute(f"GRANT SELECT ON {RUNTIME}.{table} TO pmmc_evaluator")
    for table in _EVALUATION_TABLES:
        op.execute(f"ALTER TABLE {EVALUATION}.{table} OWNER TO pmmc_owner")
        op.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {EVALUATION}.{table} TO pmmc_evaluator"
        )
        op.execute(f"REVOKE ALL ON {EVALUATION}.{table} FROM pmmc_runtime")
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE ON {EVALUATION}.model_calls TO pmmc_evaluator"
    )


def downgrade() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS r0_hardening_completion_integrity "
        f"ON {RUNTIME}.run_states"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_r0_hardening_completion()"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_candidate_scope_v3 "
        f"ON {RUNTIME}.questioner_candidate_verifications"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS "
        f"{RUNTIME}.enforce_questioner_candidate_scope_v3()"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_questioner_v3_verifier "
        f"ON {RUNTIME}.questioner_verification_batches"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_questioner_verifier_scope_v3()"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_semantic_job_candidate_scope "
        f"ON {EVALUATION}.qbank_semantic_match_jobs"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS "
        f"{EVALUATION}.enforce_semantic_job_candidate_scope()"
    )
    op.drop_constraint(
        "fk_evaluation_model_calls_semantic_job",
        "model_calls",
        schema=EVALUATION,
        type_="foreignkey",
    )
    for table in ("run_artifacts", "model_calls"):
        for suffix in ("insert", "update", "delete"):
            op.execute(
                f"DROP TRIGGER IF EXISTS freeze_terminal_semantic_parent_{suffix} "
                f"ON {EVALUATION}.{table}"
            )
        op.execute(
            f"DROP TRIGGER IF EXISTS a_assign_legacy_evaluation_run "
            f"ON {EVALUATION}.{table}"
        )
        for suffix in ("insert", "update", "delete"):
            op.execute(
                f"DROP TRIGGER IF EXISTS z_freeze_evaluation_run_{suffix} "
                f"ON {EVALUATION}.{table}"
            )
    op.execute(
        f"DROP TRIGGER IF EXISTS freeze_terminal_semantic_state "
        f"ON {EVALUATION}.qbank_semantic_match_runs"
    )
    # A downgrade intentionally discards 0012-only semantic evaluation facts.
    # Remove semantic artifacts before restoring the historical NOT NULL parent.
    op.execute(
        f"DELETE FROM {EVALUATION}.run_artifacts "
        "WHERE semantic_run_id IS NOT NULL"
    )
    op.execute(
        f"""ALTER TABLE {EVALUATION}.run_artifacts
        DROP CONSTRAINT IF EXISTS ck_run_artifacts_exactly_one_evaluation_parent,
        DROP CONSTRAINT IF EXISTS fk_run_artifacts_semantic_run,
        DROP COLUMN IF EXISTS semantic_run_id,
        ALTER COLUMN evaluation_run_id SET NOT NULL"""
    )
    for table in reversed(_EVALUATION_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {EVALUATION}.{table} CASCADE")
    op.execute(
        f"DROP FUNCTION IF EXISTS {EVALUATION}.enforce_semantic_run_writable()"
    )
    op.execute(
        f"DROP FUNCTION IF EXISTS "
        f"{EVALUATION}.enforce_semantic_state_transition()"
    )
    op.execute(
        f"DELETE FROM {EVALUATION}.model_calls "
        "WHERE semantic_run_id IS NOT NULL"
    )
    op.execute(
        f"""ALTER TABLE {EVALUATION}.model_calls
        DROP CONSTRAINT IF EXISTS ck_model_calls_exactly_one_evaluation_parent,
        DROP CONSTRAINT IF EXISTS ck_model_calls_semantic_scope_complete,
        DROP CONSTRAINT IF EXISTS uq_model_calls_semantic_run,
        DROP CONSTRAINT IF EXISTS uq_model_calls_semantic_scope,
        DROP COLUMN IF EXISTS semantic_job_id,
        DROP COLUMN IF EXISTS semantic_run_id,
        ALTER COLUMN evaluation_run_id SET NOT NULL"""
    )
    for table in ("run_artifacts", "model_calls"):
        op.execute(
            f"CREATE TRIGGER a_assign_legacy_evaluation_run BEFORE INSERT "
            f"ON {EVALUATION}.{table} FOR EACH ROW "
            f"EXECUTE FUNCTION {EVALUATION}.assign_legacy_evaluation_run()"
        )
        op.execute(
            f"CREATE TRIGGER z_freeze_evaluation_run "
            f"BEFORE INSERT OR UPDATE OR DELETE ON {EVALUATION}.{table} "
            f"FOR EACH ROW "
            f"EXECUTE FUNCTION {EVALUATION}.enforce_evaluation_run_writable()"
        )
    for table in reversed(_RUNTIME_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {RUNTIME}.{table} CASCADE")
    for name in (
        "successful_step_count",
        "required_modality_sufficient",
        "discriminative_group_count",
        "matched_discriminative_group_count",
        "contextual_match_count",
        "pre_control_sufficient",
        "fallback_triggered",
        "abstain_triggered",
        "semantic_answer_support",
    ):
        op.drop_column("program_dry_run_assessments", name, schema=RUNTIME)
    for name in (
        "direct_match",
        "derived_match",
        "contextual_only",
        "discriminative_match",
    ):
        op.drop_column("program_dry_run_evidence_matches", name, schema=RUNTIME)
    op.execute(
        f"DROP INDEX IF EXISTS {RUNTIME}.uq_questioner_verification_batches_candidate_scope"
    )
    op.drop_constraint(
        "fk_questioner_verification_batches_scope_qbank",
        "questioner_verification_batches",
        schema=RUNTIME,
        type_="foreignkey",
    )
    op.drop_column(
        "questioner_verification_batches", "scope_qbank_id", schema=RUNTIME
    )
    op.create_unique_constraint(
        "uq_questioner_verification_batches_legacy_scope",
        "questioner_verification_batches",
        ["compiler_unit_id", "generation_id", "verification_round"],
        schema=RUNTIME,
    )
    op.drop_constraint(
        "uq_qbank_coverage_runs_semantic_parent",
        "qbank_coverage_runs",
        schema=EVALUATION,
        type_="unique",
    )
    op.drop_constraint(
        op.f(MATCH_ROLE_CHECK),
        "qbank_coverage_gold_group_members",
        schema=EVALUATION,
        type_="check",
    )
    op.drop_column(
        "qbank_coverage_gold_group_members", "match_role", schema=EVALUATION
    )
    for name in (
        "relation_content_hash",
        "evidence_group_ids",
        "speaker_scope",
    ):
        op.drop_column("questioner_opportunities", name, schema=RUNTIME)
    # Restore the 0011 guard expected by historical compiler 0.3/0.4 runs.
    op.execute(
        f"CREATE TRIGGER validate_questioner_v3_verifier BEFORE INSERT OR UPDATE "
        f"ON {RUNTIME}.questioner_verification_batches FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.enforce_questioner_verifier_scope_v2()"
    )
