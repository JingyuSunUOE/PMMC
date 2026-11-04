# ruff: noqa: E501
"""Align the Questioner v3.5 rejection-code CHECK with the runtime schema.

Revision ID: 20260803_0025
Revises: 20260802_0024
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260803_0025"
down_revision: str | None = "20260802_0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
TABLE = "questioner_candidate_gate_results"
CONSTRAINT = "ck_questioner_candidate_gate_results_rejection_codes_v35"
LEGACY_CONSTRAINT = "ck_questioner_candidate_gate_results_rejection_codes_v3"

_LEGACY_REJECTION_CODES = (
    "invalid_schema",
    "unsupported_question_type",
    "future_or_foreign_reference",
    "missing_expected_answer",
    "missing_expected_evidence",
    "modality_mismatch",
    "visual_evidence_missing",
    "multi_hop_evidence_missing",
    "duplicate",
    "budget_exceeded",
    "internal_identifier_leak",
    "system_metadata_reference",
    "assistant_follow_up",
    "historical_question_replay",
    "generic_external_question",
    "placeholder_expected_answer",
    "opportunity_mismatch",
    "taxonomy_invariant_failed",
    "reasoning_invariant_failed",
    "evidence_not_entailed",
    "semantic_revision_required",
    "semantic_verification_failed",
    "revision_exhausted",
)

_STEP7B_REJECTION_CODES = (
    *_LEGACY_REJECTION_CODES,
    "question_answer_closure_failed",
    "answer_type_mismatch",
    "question_as_answer",
    "ai_disclaimer",
    "incomplete_answer",
    "source_unnatural",
)


def _array(values: tuple[str, ...]) -> str:
    literals = ",".join("'" + value.replace("'", "''") + "'" for value in values)
    return f"ARRAY[{literals}]::text[]"


def _drop_rejection_code_constraint() -> str:
    connection = op.get_bind()
    rows = list(
        connection.execute(
            sa.text(
                """SELECT constraint_row.conname AS constraint_name,
                          pg_get_constraintdef(constraint_row.oid, true) AS definition
                     FROM pg_constraint constraint_row
                     JOIN pg_class relation_row
                       ON relation_row.oid = constraint_row.conrelid
                     JOIN pg_namespace namespace_row
                       ON namespace_row.oid = relation_row.relnamespace
                    WHERE namespace_row.nspname = :schema_name
                      AND relation_row.relname = :table_name
                      AND constraint_row.contype = 'c'
                    ORDER BY constraint_row.conname"""
            ),
            {"schema_name": RUNTIME, "table_name": TABLE},
        ).mappings()
    )
    matches = [
        row
        for row in rows
        if "rejection_codes" in str(row["definition"] or "")
        and "<@" in str(row["definition"] or "")
    ]
    print(
        "STEP7B_GATE_CODE_MIGRATION "
        f"state=constraint_resolved candidate_count={len(rows)} "
        f"match_count={len(matches)}",
        flush=True,
    )
    if len(matches) != 1:
        raise RuntimeError(
            "Step 7B candidate rejection-code constraint resolution failed: "
            f"expected one match, found {len(matches)}"
        )
    name = str(matches[0]["constraint_name"])
    op.drop_constraint(op.f(name), TABLE, schema=RUNTIME, type_="check")
    return name


def _create_rejection_code_constraint(
    *, name: str, values: tuple[str, ...], name_is_final: bool
) -> None:
    op.create_check_constraint(
        op.f(name) if name_is_final else name,
        TABLE,
        f"rejection_codes <@ {_array(values)}",
        schema=RUNTIME,
    )


def upgrade() -> None:
    print(
        "STEP7B_GATE_CODE_MIGRATION "
        "state=start revision=20260803_0025 protocol=questioner-v3.5",
        flush=True,
    )
    previous_name = _drop_rejection_code_constraint()
    _create_rejection_code_constraint(
        name=CONSTRAINT,
        values=_STEP7B_REJECTION_CODES,
        name_is_final=True,
    )
    print(
        "STEP7B_GATE_CODE_MIGRATION "
        f"state=ready previous_constraint={previous_name} "
        f"allowed_code_count={len(_STEP7B_REJECTION_CODES)}",
        flush=True,
    )


def downgrade() -> None:
    connection = op.get_bind()
    incompatible_rows = int(
        connection.execute(
            sa.text(
                f"SELECT count(*) FROM {RUNTIME}.{TABLE} "
                f"WHERE NOT (rejection_codes <@ {_array(_LEGACY_REJECTION_CODES)})"
            )
        ).scalar_one()
    )
    print(
        "STEP7B_GATE_CODE_MIGRATION "
        f"state=downgrade_preflight incompatible_row_count={incompatible_rows}",
        flush=True,
    )
    if incompatible_rows:
        raise RuntimeError(
            "cannot downgrade Step 7B rejection-code contract while v3.5 gate facts exist; "
            f"incompatible_row_count={incompatible_rows}"
        )
    _drop_rejection_code_constraint()
    _create_rejection_code_constraint(
        name=LEGACY_CONSTRAINT,
        values=_LEGACY_REJECTION_CODES,
        # Reproduce migration 0010's naming-convention input so a later
        # downgrade through 0010 resolves the same physical constraint name.
        name_is_final=False,
    )
    print(
        "STEP7B_GATE_CODE_MIGRATION "
        f"state=downgrade_ready revision={down_revision} "
        f"allowed_code_count={len(_LEGACY_REJECTION_CODES)}",
        flush=True,
    )
