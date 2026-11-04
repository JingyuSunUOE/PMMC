"""Restore legacy Questioner raw-candidate accounting without rewriting facts."""

from alembic import op

revision = "20260927_0036"
down_revision = "20260927_0035"
branch_labels = None
depends_on = None

_OLD = """WHEN 'questioner' THEN (SELECT count(*) FROM pmmc_runtime.qbank_questions facts
                        WHERE facts.run_id = NEW.run_id
                          AND facts.compiler_unit_id = units.compiler_unit_id)"""
_NEW = """WHEN 'questioner' THEN CASE
                      WHEN compiler.compiler_schema_version IN ('0.2.0', '0.3.0') THEN
                        COALESCE((SELECT g.raw_candidate_count
                          FROM pmmc_runtime.questioner_generations g
                         WHERE g.run_id=NEW.run_id
                           AND g.compiler_unit_id=units.compiler_unit_id
                           AND g.parse_status!='failed'), -1)
                      ELSE (SELECT count(*) FROM pmmc_runtime.qbank_questions facts
                        WHERE facts.run_id = NEW.run_id
                          AND facts.compiler_unit_id = units.compiler_unit_id)
                    END"""


def _rewrite(*, forward: bool) -> None:
    source, target = (_OLD, _NEW) if forward else (_NEW, _OLD)
    source = "'" + source.replace("'", "''") + "'"
    target = "'" + target.replace("'", "''") + "'"
    op.execute(f"""DO $rewrite$
    DECLARE definition text;
    BEGIN
      definition := pg_get_functiondef(
        'pmmc_runtime.enforce_compiler_completion_v3()'::regprocedure);
      IF (length(definition)-length(replace(definition,{source},'')))
           / length({source}) IS DISTINCT FROM 1 THEN
        RAISE EXCEPTION 'legacy_questioner_accounting_rewrite_mismatch';
      END IF;
      EXECUTE replace(definition,{source},{target});
    END $rewrite$""")


def upgrade() -> None:
    _rewrite(forward=True)


def downgrade() -> None:
    _rewrite(forward=False)
