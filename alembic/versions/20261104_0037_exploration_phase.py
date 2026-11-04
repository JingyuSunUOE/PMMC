"""Allow full-development exploration without changing the paper freeze constraint."""

from alembic import op

revision = "20261104_0037"
down_revision = "20260927_0036"
branch_labels = None
depends_on = None


def _constraint(values: str) -> None:
    op.execute("ALTER TABLE pmmc_runtime.benchmark_campaigns "
               "DROP CONSTRAINT ck_benchmark_campaigns_phase_values")
    op.execute("ALTER TABLE pmmc_runtime.benchmark_campaigns ADD CONSTRAINT "
               f"ck_benchmark_campaigns_phase_values CHECK (phase IN ({values}))")


def upgrade() -> None:
    _constraint("'development','paper','exploration'")


def downgrade() -> None:
    # PostgreSQL refuses the downgrade if exploration facts exist; never rewrite them.
    _constraint("'development','paper'")
