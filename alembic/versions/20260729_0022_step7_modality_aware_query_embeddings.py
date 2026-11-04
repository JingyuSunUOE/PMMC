# ruff: noqa: E501
"""Make Step 7 query-embedding completion modality-aware.

Revision ID: 20260729_0022
Revises: 20260727_0021
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260729_0022"
down_revision: str | None = "20260727_0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"

_LEGACY_QUERY_EMBEDDING_DENOMINATOR = """HAVING count(e.query_embedding_id) != 2
                OR count(*) FILTER (WHERE e.channel = 'text') != 1
                OR count(*) FILTER (WHERE e.channel = 'multimodal') != 1"""

_MODALITY_AWARE_QUERY_EMBEDDING_DENOMINATOR = f"""HAVING count(e.query_embedding_id) != CASE
                  WHEN EXISTS (
                    SELECT 1 FROM {RUNTIME}.query_images qi
                     WHERE qi.snapshot_id = NEW.snapshot_id
                       AND qi.query_id = rq.query_id
                  ) THEN 2 ELSE 1 END
                OR count(*) FILTER (WHERE e.channel = 'text') != 1
                OR count(*) FILTER (WHERE e.channel = 'multimodal') != CASE
                  WHEN EXISTS (
                    SELECT 1 FROM {RUNTIME}.query_images qi
                     WHERE qi.snapshot_id = NEW.snapshot_id
                       AND qi.query_id = rq.query_id
                  ) THEN 1 ELSE 0 END"""


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _replace_query_embedding_denominator(*, modality_aware: bool) -> None:
    source = (
        _LEGACY_QUERY_EMBEDDING_DENOMINATOR
        if modality_aware
        else _MODALITY_AWARE_QUERY_EMBEDDING_DENOMINATOR
    )
    target = (
        _MODALITY_AWARE_QUERY_EMBEDDING_DENOMINATOR
        if modality_aware
        else _LEGACY_QUERY_EMBEDDING_DENOMINATOR
    )
    source_literal = _sql_literal(source)
    target_literal = _sql_literal(target)
    op.execute(
        f"""DO $$
        DECLARE function_definition text; occurrence_count integer;
        BEGIN
          SELECT pg_get_functiondef(
                   '{RUNTIME}.enforce_query_executor_completion()'::regprocedure
                 )
            INTO function_definition;
          IF function_definition IS NULL THEN
            RAISE EXCEPTION 'missing Step 7 query completion guard';
          END IF;
          occurrence_count := (
            length(function_definition)
            - length(replace(function_definition, {source_literal}, ''))
          ) / length({source_literal});
          IF occurrence_count IS DISTINCT FROM 1 THEN
            RAISE EXCEPTION 'unexpected Step 7 query-embedding denominator guard';
          END IF;
          function_definition := replace(
            function_definition,
            {source_literal},
            {target_literal}
          );
          EXECUTE function_definition;
        END $$"""
    )


def upgrade() -> None:
    print(
        "STEP7_QUERY_EMBEDDING_MIGRATION state=start "
        "source=all_queries_two_channels target=query_modality",
        flush=True,
    )
    _replace_query_embedding_denominator(modality_aware=True)
    print(
        "STEP7_QUERY_EMBEDDING_MIGRATION state=ready "
        "text_required=all multimodal_required=query_image_only",
        flush=True,
    )


def downgrade() -> None:
    _replace_query_embedding_denominator(modality_aware=False)
