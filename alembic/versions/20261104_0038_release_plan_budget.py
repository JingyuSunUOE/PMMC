"""Bind one monetary ledger to an immutable release plan and its native scopes."""

from alembic import op

revision = "20261104_0038"
down_revision = "20261104_0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE pmmc_runtime.benchmark_budget_approvals
        ADD COLUMN plan_hash VARCHAR(64),
        ADD COLUMN plan_bindings JSONB,
        ADD COLUMN stopped_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN stopped_by VARCHAR(256),
        ADD COLUMN stop_reason VARCHAR(128),
        ALTER COLUMN limit_usd DROP NOT NULL,
        ADD CONSTRAINT uq_benchmark_budget_approvals_plan_hash UNIQUE (plan_hash),
        ADD CONSTRAINT ck_benchmark_budget_approvals_budget_authority CHECK (
          (plan_hash IS NULL AND plan_bindings IS NULL AND limit_usd IS NOT NULL)
          OR (plan_hash IS NOT NULL AND plan_bindings IS NOT NULL)
        ),
        ADD CONSTRAINT ck_benchmark_budget_approvals_stop_audit CHECK (
          (stopped_at IS NULL AND stopped_by IS NULL AND stop_reason IS NULL)
          OR (plan_hash IS NOT NULL AND stopped_at IS NOT NULL
              AND stopped_by IS NOT NULL AND stop_reason IS NOT NULL)
        )
    """)
    op.execute(
        "GRANT UPDATE (stopped_at, stopped_by, stop_reason) "
        "ON pmmc_runtime.benchmark_budget_approvals TO pmmc_runtime"
    )
    op.execute("""
        CREATE FUNCTION pmmc_runtime.keep_budget_stop() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
          IF OLD.stopped_at IS NOT NULL AND
             (NEW.stopped_at, NEW.stopped_by, NEW.stop_reason)
             IS DISTINCT FROM (OLD.stopped_at, OLD.stopped_by, OLD.stop_reason) THEN
            RAISE EXCEPTION 'release_plan_approval_permanently_stopped';
          END IF;
          RETURN NEW;
        END; $$
    """)
    op.execute("""
        CREATE TRIGGER keep_budget_stop BEFORE UPDATE OF stopped_at, stopped_by, stop_reason
        ON pmmc_runtime.benchmark_budget_approvals FOR EACH ROW
        EXECUTE FUNCTION pmmc_runtime.keep_budget_stop()
    """)


def downgrade() -> None:
    # Losing a plan binding would turn its shared budget into a legacy approval.
    op.execute("""
        ALTER TABLE pmmc_runtime.benchmark_budget_approvals
        ADD CONSTRAINT no_release_plan_approvals CHECK (plan_hash IS NULL)
    """)
    op.execute("DROP TRIGGER keep_budget_stop ON pmmc_runtime.benchmark_budget_approvals")
    op.execute("DROP FUNCTION pmmc_runtime.keep_budget_stop()")
    op.execute("""
        ALTER TABLE pmmc_runtime.benchmark_budget_approvals
        DROP CONSTRAINT no_release_plan_approvals,
        DROP CONSTRAINT ck_benchmark_budget_approvals_budget_authority,
        DROP CONSTRAINT ck_benchmark_budget_approvals_stop_audit,
        ALTER COLUMN limit_usd SET NOT NULL,
        DROP COLUMN stopped_at,
        DROP COLUMN stopped_by,
        DROP COLUMN stop_reason,
        DROP COLUMN plan_bindings,
        DROP COLUMN plan_hash
    """)
