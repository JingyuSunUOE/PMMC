# ruff: noqa: E501
"""Route evaluation query lineage through the active evaluation parent.

Revision ID: 20260724_0015
Revises: 20260724_0014
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260724_0015"
down_revision: str | None = "20260724_0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"
_QUERY_SCOPED_TABLES = ("run_artifacts", "model_calls")
_FUNCTION_NAME = "validate_evaluation_query_scope_v2"
_TRIGGER_NAME = "b_validate_evaluation_query_scope"


def upgrade() -> None:
    for table in _QUERY_SCOPED_TABLES:
        _drop_runtime_query_foreign_key(table)
    _create_parent_aware_query_guard()
    for table in _QUERY_SCOPED_TABLES:
        op.execute(
            f"DROP TRIGGER IF EXISTS {_TRIGGER_NAME} "
            f"ON {EVALUATION}.{table}"
        )
        op.execute(
            f"CREATE TRIGGER {_TRIGGER_NAME} BEFORE INSERT OR UPDATE "
            f"ON {EVALUATION}.{table} FOR EACH ROW "
            f"EXECUTE FUNCTION {EVALUATION}.{_FUNCTION_NAME}()"
        )
    op.execute(
        f"REVOKE ALL ON FUNCTION {EVALUATION}.{_FUNCTION_NAME}() FROM PUBLIC"
    )


def downgrade() -> None:
    for table in _QUERY_SCOPED_TABLES:
        op.execute(
            f"DROP TRIGGER IF EXISTS {_TRIGGER_NAME} "
            f"ON {EVALUATION}.{table}"
        )
    op.execute(
        f"DROP FUNCTION IF EXISTS {EVALUATION}.{_FUNCTION_NAME}()"
    )
    for table in _QUERY_SCOPED_TABLES:
        op.create_foreign_key(
            f"fk_evaluation_{table}_runtime_query",
            table,
            "run_queries",
            ["run_id", "snapshot_id", "query_id"],
            ["run_id", "snapshot_id", "query_id"],
            source_schema=EVALUATION,
            referent_schema=RUNTIME,
        )


def _drop_runtime_query_foreign_key(table: str) -> None:
    if table not in _QUERY_SCOPED_TABLES:
        raise ValueError("unknown evaluation query-scoped table")
    op.execute(
        f"""DO $$
        DECLARE constraint_name text;
        BEGIN
          SELECT constraint_row.conname INTO constraint_name
          FROM pg_constraint constraint_row
          WHERE constraint_row.contype = 'f'
            AND constraint_row.conrelid =
                '{EVALUATION}.{table}'::regclass
            AND constraint_row.confrelid =
                '{RUNTIME}.run_queries'::regclass
            AND ARRAY(
              SELECT attribute.attname::text
              FROM unnest(constraint_row.conkey)
                   WITH ORDINALITY AS key_position(attnum, ordinal)
              JOIN pg_attribute attribute
                ON attribute.attrelid = constraint_row.conrelid
               AND attribute.attnum = key_position.attnum
              ORDER BY key_position.ordinal
            ) = ARRAY['run_id','snapshot_id','query_id']::text[];
          IF constraint_name IS NULL THEN
            RAISE EXCEPTION
              'evaluation runtime-query foreign key is missing'
              USING ERRCODE = '55000',
                    SCHEMA = '{EVALUATION}',
                    TABLE = '{table}';
          END IF;
          EXECUTE format(
            'ALTER TABLE %I.%I DROP CONSTRAINT %I',
            '{EVALUATION}',
            '{table}',
            constraint_name
          );
        END $$"""
    )


def _create_parent_aware_query_guard() -> None:
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.{_FUNCTION_NAME}()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME}, {EVALUATION} AS $$
        DECLARE payload jsonb; runtime_run varchar(128);
                snapshot varchar(128); query_identifier varchar(256);
                semantic_parent varchar(128); semantic_job varchar(128);
                qbank_identifier varchar(256); valid boolean;
                constraint_label text;
        BEGIN
            payload := to_jsonb(NEW);
            runtime_run := payload ->> 'run_id';
            snapshot := payload ->> 'snapshot_id';
            query_identifier := payload ->> 'query_id';
            semantic_parent := payload ->> 'semantic_run_id';
            semantic_job := payload ->> 'semantic_job_id';
            qbank_identifier := payload ->> 'qbank_id';

            IF semantic_parent IS NULL THEN
                IF query_identifier IS NULL THEN
                    RETURN NEW;
                END IF;
                SELECT EXISTS (
                    SELECT 1 FROM {RUNTIME}.run_queries runtime_query
                    WHERE runtime_query.run_id = runtime_run
                      AND runtime_query.snapshot_id = snapshot
                      AND runtime_query.query_id = query_identifier
                ) INTO valid;
                constraint_label := CASE TG_TABLE_NAME
                    WHEN 'run_artifacts'
                        THEN 'fk_evaluation_run_artifacts_runtime_query_scope'
                    ELSE 'fk_evaluation_model_calls_runtime_query_scope'
                END;
            ELSIF TG_TABLE_NAME = 'run_artifacts' THEN
                SELECT EXISTS (
                    SELECT 1
                    FROM {EVALUATION}.qbank_semantic_match_runs semantic_run
                    WHERE semantic_run.semantic_run_id = semantic_parent
                      AND semantic_run.compiler_run_id = runtime_run
                      AND semantic_run.snapshot_id = snapshot
                      AND (
                        query_identifier IS NULL
                        OR EXISTS (
                          SELECT 1
                          FROM {EVALUATION}.qbank_coverage_queries coverage_query
                          WHERE coverage_query.coverage_run_id =
                                semantic_run.coverage_run_id
                            AND coverage_query.query_id = query_identifier
                        )
                      )
                ) INTO valid;
                constraint_label :=
                    'fk_evaluation_run_artifacts_semantic_query_scope';
            ELSIF TG_TABLE_NAME = 'model_calls' THEN
                SELECT EXISTS (
                    SELECT 1
                    FROM {EVALUATION}.qbank_semantic_match_jobs semantic_pair
                    JOIN {EVALUATION}.qbank_semantic_match_runs semantic_run
                      ON semantic_run.semantic_run_id =
                         semantic_pair.semantic_run_id
                    WHERE semantic_pair.semantic_job_id = semantic_job
                      AND semantic_pair.semantic_run_id = semantic_parent
                      AND semantic_pair.query_id = query_identifier
                      AND semantic_pair.qbank_id = qbank_identifier
                      AND semantic_run.compiler_run_id = runtime_run
                      AND semantic_run.snapshot_id = snapshot
                ) INTO valid;
                constraint_label :=
                    'fk_evaluation_model_calls_semantic_query_scope';
            ELSE
                RAISE EXCEPTION 'unsupported evaluation query-scoped table'
                    USING ERRCODE = '55000',
                          SCHEMA = '{EVALUATION}',
                          TABLE = TG_TABLE_NAME;
            END IF;

            IF COALESCE(valid, false) = false THEN
                RAISE EXCEPTION 'evaluation query scope is invalid'
                    USING ERRCODE = '23503',
                          SCHEMA = '{EVALUATION}',
                          TABLE = TG_TABLE_NAME,
                          CONSTRAINT = constraint_label;
            END IF;
            RETURN NEW;
        END; $$"""
    )
