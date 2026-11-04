# ruff: noqa: E501
"""Represent baseline and PMMC evaluation context lineage explicitly.

Revision ID: 20260730_0023
Revises: 20260729_0022
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_0023"
down_revision: str | None = "20260729_0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EVALUATION = "pmmc_evaluation"
RUNTIME = "pmmc_runtime"

_SOURCE_CHECK = "context_source_exactly_one"
_SOURCE_CHECK_DATABASE_NAME = "ck_context_evaluations_context_source_exactly_one"
_EXECUTION_FK = "fk_context_evaluations_program_execution_id"


def upgrade() -> None:
    print(
        "STEP7_EVALUATION_CONTEXT_MIGRATION state=start "
        "source=baseline_context_only target=typed_runtime_lineage",
        flush=True,
    )
    op.add_column(
        "context_evaluations",
        sa.Column(
            "context_source_kind",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'baseline_context'"),
        ),
        schema=EVALUATION,
    )
    op.add_column(
        "context_evaluations",
        sa.Column("program_execution_id", sa.String(length=128), nullable=True),
        schema=EVALUATION,
    )
    op.alter_column(
        "context_evaluations",
        "context_assembly_id",
        existing_type=sa.String(length=128),
        nullable=True,
        schema=EVALUATION,
    )
    op.create_foreign_key(
        _EXECUTION_FK,
        "context_evaluations",
        "program_executions",
        ["program_execution_id"],
        ["execution_id"],
        source_schema=EVALUATION,
        referent_schema=RUNTIME,
    )
    op.create_check_constraint(
        _SOURCE_CHECK,
        "context_evaluations",
        "(context_source_kind = 'baseline_context' "
        "AND context_assembly_id IS NOT NULL AND program_execution_id IS NULL) OR "
        "(context_source_kind = 'program_execution' "
        "AND context_assembly_id IS NULL AND program_execution_id IS NOT NULL)",
        schema=EVALUATION,
    )
    print(
        "STEP7_EVALUATION_CONTEXT_MIGRATION state=ready "
        "baseline_context=true program_execution=true synthetic_context=false",
        flush=True,
    )


def downgrade() -> None:
    connection = op.get_bind()
    program_rows = int(
        connection.execute(
            sa.text(
                f"SELECT count(*) FROM {EVALUATION}.context_evaluations "
                "WHERE context_source_kind = 'program_execution'"
            )
        ).scalar_one()
    )
    if program_rows:
        raise RuntimeError(
            "cannot downgrade Step 7 evaluation lineage while program execution facts exist"
        )
    op.drop_constraint(
        op.f(_SOURCE_CHECK_DATABASE_NAME),
        "context_evaluations",
        schema=EVALUATION,
        type_="check",
    )
    op.drop_constraint(
        _EXECUTION_FK,
        "context_evaluations",
        schema=EVALUATION,
        type_="foreignkey",
    )
    op.alter_column(
        "context_evaluations",
        "context_assembly_id",
        existing_type=sa.String(length=128),
        nullable=False,
        schema=EVALUATION,
    )
    op.drop_column("context_evaluations", "program_execution_id", schema=EVALUATION)
    op.drop_column("context_evaluations", "context_source_kind", schema=EVALUATION)
