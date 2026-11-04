"""Repair Step 7B completion's reference to a nonexistent verifier column.

This is a DDL-only correction: existing native verification facts are retained.
"""

import sqlalchemy as sa
from alembic import op

revision: str = "20260907_0026"
down_revision: str | None = "20260803_0025"
branch_labels: str | None = None
depends_on: str | None = None

_FUNCTION = "pmmc_runtime.enforce_step7b_compiler_completion()"
_OLD_JOIN = "FROM pmmc_runtime.questioner_candidate_verifications v\n"
_NEW_JOIN = """FROM pmmc_runtime.questioner_candidate_verifications v
             JOIN pmmc_runtime.questioner_verification_batches vb
               ON vb.verification_batch_id=v.verification_batch_id
              AND vb.run_id=v.run_id AND vb.compiler_unit_id=v.compiler_unit_id
              AND vb.generation_id=v.generation_id
              AND vb.verification_round=v.verification_round
             JOIN pmmc_runtime.model_calls vc
               ON vc.model_call_id=vb.model_call_id AND vc.run_id=vb.run_id
"""
_OLD_NATIVE = "AND v.provider_schema_conformant"
_NEW_NATIVE = """AND vb.scope_qbank_id=v.qbank_id AND vb.input_count=1
                AND vb.metadata @> '{"single_candidate_scope":true,
                  "provider_output_fallback":false,"repair_attempt_count":0}'::jsonb
                AND NOT (v.metadata @> '{"provider_output_fallback":true}'::jsonb)
                AND vc.status='success' AND vc.repair_attempt_count=0
                AND vc.parse_error IS NULL"""


def _rewrite(*, forward: bool) -> None:
    connection = op.get_bind()
    definition = connection.scalar(
        sa.text("SELECT pg_get_functiondef(to_regprocedure(:name))"),
        {"name": _FUNCTION},
    )
    if not isinstance(definition, str):
        raise RuntimeError("Step 7B compiler completion function is missing")
    for old, new in ((_OLD_JOIN, _NEW_JOIN), (_OLD_NATIVE, _NEW_NATIVE)):
        source, target = (old, new) if forward else (new, old)
        if definition.count(source) != 1:
            raise RuntimeError("Step 7B verifier guard rewrite expected one match")
        definition = definition.replace(source, target, 1)
    # CREATE OR REPLACE preserves ownership, SECURITY DEFINER and existing ACLs.
    connection.exec_driver_sql(definition, execution_options={"no_parameters": True})
    print(
        "STEP7B_COMPLETION_GUARD_MIGRATION "
        f"direction={'upgrade' if forward else 'downgrade'} state=ready "
        "fact_rows_modified=0",
        flush=True,
    )


def upgrade() -> None:
    _rewrite(forward=True)


def downgrade() -> None:
    _rewrite(forward=False)
