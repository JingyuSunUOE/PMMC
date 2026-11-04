# ruff: noqa: E501
"""Delegate Step 6C.2 stage accounting to its atomic-fact guards.

Revision ID: 20260724_0014
Revises: 20260724_0013
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260724_0014"
down_revision: str | None = "20260724_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"


def upgrade() -> None:
    _replace_compiler_completion_guard(
        function_name="enforce_compiler_completion_v3",
        delegated_schema_versions=("0.4.0", "0.5.0"),
    )


def downgrade() -> None:
    _replace_compiler_completion_guard(
        function_name="enforce_compiler_completion_v2",
        delegated_schema_versions=("0.4.0",),
    )


def _replace_compiler_completion_guard(
    *,
    function_name: str,
    delegated_schema_versions: tuple[str, ...],
) -> None:
    """Install the common denominator guard and delegate versioned stage facts."""

    allowed_functions = {
        "enforce_compiler_completion_v2",
        "enforce_compiler_completion_v3",
    }
    allowed_versions = {"0.4.0", "0.5.0"}
    if function_name not in allowed_functions:
        raise ValueError("unknown compiler completion function")
    if (
        not delegated_schema_versions
        or set(delegated_schema_versions) - allowed_versions
    ):
        raise ValueError("unknown delegated compiler schema")
    version_values = ", ".join(
        f"'{value}'" for value in delegated_schema_versions
    )

    op.execute(
        f"DROP TRIGGER IF EXISTS compiler_completion_integrity ON {RUNTIME}.run_states"
    )
    for candidate in sorted(allowed_functions):
        op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.{candidate}()")
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.{function_name}()
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
            IF compiler.compiler_schema_version NOT IN ({version_values}) THEN
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
        f"CREATE TRIGGER compiler_completion_integrity BEFORE UPDATE "
        f"ON {RUNTIME}.run_states FOR EACH ROW "
        f"EXECUTE FUNCTION {RUNTIME}.{function_name}()"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {RUNTIME}.{function_name}() FROM PUBLIC"
    )
