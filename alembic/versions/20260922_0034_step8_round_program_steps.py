"""Expose frozen Step8 program steps for every exact round membership."""

from alembic import op

revision = "20260922_0034"
down_revision = "20260922_0033"
branch_labels = None
depends_on = None

RUNTIME = "pmmc_runtime"
_ONLINE_VIEW_ADDITIONS = ("query_runtime_program_steps_v3",)


def upgrade() -> None:
    # A plan may be reused by several depth projections after early stopping.
    # Deduplicate authorized plans, not steps, and never consult the active-r2
    # decision here: r0 and r3 have their own frozen membership in routing_v4.
    op.execute(f"""CREATE VIEW {RUNTIME}.query_runtime_program_steps_v3
        WITH (security_barrier = true) AS
        WITH authorized_plans AS (
            SELECT DISTINCT programs.plan_id, programs.run_id, programs.snapshot_id,
                   programs.compiler_unit_id, programs.qbank_id,
                   programs.operator_set_version, programs.program_hash
              FROM {RUNTIME}.verified_qbank_routing_v4 routing
              JOIN {RUNTIME}.memory_programs programs
                ON programs.plan_id = routing.plan_id
               AND programs.run_id = routing.compiler_run_id
               AND programs.snapshot_id = routing.snapshot_id
               AND programs.qbank_id = routing.qbank_id
               AND programs.program_hash = routing.program_hash
               AND programs.operator_set_version = routing.operator_set_version
               AND programs.round_id = routing.program_round
             WHERE routing.projection_round BETWEEN 0 AND 3
               AND programs.round_id <= routing.projection_round
        )
        SELECT authorized.run_id, authorized.qbank_id, authorized.plan_id,
               steps.step_id, steps.op_name, steps.args, steps.expected_output_type,
               authorized.operator_set_version, authorized.program_hash
          FROM authorized_plans authorized
          JOIN {RUNTIME}.program_steps steps
            ON steps.plan_id = authorized.plan_id
           AND steps.run_id = authorized.run_id
           AND steps.snapshot_id = authorized.snapshot_id
           AND steps.compiler_unit_id = authorized.compiler_unit_id""")
    for view in _ONLINE_VIEW_ADDITIONS:
        op.execute(f"ALTER VIEW {RUNTIME}.{view} OWNER TO pmmc_owner")
        op.execute(
            f"REVOKE ALL ON {RUNTIME}.{view} FROM PUBLIC,pmmc_runtime,pmmc_evaluator,pmmc_online"
        )
        op.execute(
            f"GRANT SELECT ON {RUNTIME}.{view} TO pmmc_runtime,pmmc_evaluator,pmmc_online"
        )


def downgrade() -> None:
    for view in reversed(_ONLINE_VIEW_ADDITIONS):
        op.execute(f"DROP VIEW {RUNTIME}.{view}")
