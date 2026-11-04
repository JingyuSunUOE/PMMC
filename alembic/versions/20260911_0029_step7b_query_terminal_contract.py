"""Version query completion and validate the bounded reranker pair universe.

DDL only: do not rewrite answers, reranker assessments, or completed runs.
"""

import sqlalchemy as sa
from alembic import op

revision: str = "20260911_0029"
down_revision: str | None = "20260907_0028"
branch_labels: str | None = None
depends_on: str | None = None

_OLD_ANSWER = "a.method='pmmc' AND a.answer_variant='active_final'"
_NEW_ANSWER = """a.method='pmmc' AND a.answer_variant=(
                         CASE WHEN NEW.schema_version='0.2.0'
                              THEN 'active_final_provisional' ELSE 'active_final' END)"""
_OLD_MM = """WHEN EXISTS (
                    SELECT 1 FROM pmmc_runtime.query_images qi"""
_NEW_MM = """WHEN NEW.schema_version='0.2.0' OR EXISTS (
                    SELECT 1 FROM pmmc_runtime.query_images qi"""
_OLD_PAIRS = """SELECT count(*) INTO bad FROM pmmc_runtime.run_queries q
           WHERE q.run_id=NEW.query_executor_run_id AND NOT EXISTS (
             SELECT 1 FROM pmmc_runtime.qbank_reranker_assessments r
              WHERE r.query_executor_run_id=q.run_id AND r.query_id=q.query_id
                AND r.provider_schema_conformant);"""
# Reconstruct the frozen base top-20 before reranking, not the final route ranking.
# An empty grid has no logical pairs; the base guard still requires all 16 batches,
# a route, a fallback decision and a terminal selection for every selected query.
_NEW_PAIRS = """SELECT count(*) INTO bad FROM (
            SELECT DISTINCT query_id, qbank_id FROM (
              SELECT b.query_id, f.qbank_id,
                     row_number() OVER (
                       PARTITION BY b.match_batch_id
                       ORDER BY CASE f.route_class WHEN 'answerable' THEN 0 ELSE 1 END,
                                f.base_reliable_score DESC, f.raw_rrf DESC, f.qbank_id
                     ) AS base_rank
                FROM pmmc_runtime.qbank_match_batches b
                JOIN pmmc_runtime.qbank_match_fused_candidates f
                  ON f.match_batch_id=b.match_batch_id
               WHERE b.query_executor_run_id=NEW.query_executor_run_id
            ) ranked WHERE base_rank <= 20
          ) required_pairs
          WHERE NOT EXISTS (
            SELECT 1 FROM pmmc_runtime.qbank_reranker_assessments r
             WHERE r.query_executor_run_id=NEW.query_executor_run_id
               AND r.query_id=required_pairs.query_id
               AND r.qbank_id=required_pairs.qbank_id
               AND r.provider_schema_conformant);"""
_OLD_STRICT = """SELECT 1 FROM pmmc_runtime.structured_answer_outputs s
                WHERE s.answer_id=a.answer_id AND s.parse_status='accepted'
                  AND s.provider_schema_conformant"""
_NEW_STRICT = """SELECT 1 FROM pmmc_runtime.structured_answer_outputs s
                JOIN pmmc_runtime.final_answers source ON source.answer_id=s.answer_id
                WHERE s.run_id=a.run_id AND s.query_id=a.query_id
                  AND s.parse_status='accepted' AND s.provider_schema_conformant
                  AND s.supported AND s.answer=a.final_answer
                  AND s.finish_reason IS DISTINCT FROM 'length'
                  AND source.run_id=a.run_id AND source.query_id=a.query_id
                  AND source.snapshot_id=a.snapshot_id AND source.method=a.method
                  AND source.final_answer=a.final_answer
                  AND s.model_call_id=source.model_call_id
                  AND s.answer_variant=source.answer_variant
                  AND source.model_call_id=a.model_call_id
                  AND source.answer_model_alias=a.answer_model_alias
                  AND source.actual_model_id IS NOT DISTINCT FROM a.actual_model_id
                  AND (s.answer_id=a.answer_id OR EXISTS (
                    SELECT 1 FROM pmmc_runtime.answer_reuse_links l
                     WHERE l.query_executor_run_id=a.run_id AND l.query_id=a.query_id
                       AND l.target_variant=a.answer_variant
                       AND l.source_answer_id=source.answer_id
                       AND a.metadata->>'source_answer_id'=source.answer_id
                       AND a.metadata @> '{"answer_reused":true}'::jsonb
                       AND l.context_hash=a.metadata->>'answer_context_hash'))"""


def _rewrite(*, forward: bool) -> None:
    connection = op.get_bind()
    for name, old, new, count in (
        ("enforce_query_executor_completion", _OLD_MM, _NEW_MM, 2),
        ("enforce_query_executor_completion", _OLD_ANSWER, _NEW_ANSWER, 1),
        ("enforce_step7b_query_completion", _OLD_PAIRS, _NEW_PAIRS, 1),
        ("enforce_step7b_query_completion", _OLD_STRICT, _NEW_STRICT, 1),
    ):
        definition = connection.scalar(
            sa.text("SELECT pg_get_functiondef(to_regprocedure(:name))"),
            {"name": f"pmmc_runtime.{name}()"},
        )
        source, target = (old, new) if forward else (new, old)
        if not isinstance(definition, str) or definition.count(source) != count:
            raise RuntimeError("Step 7B query completion rewrite match count differs")
        connection.exec_driver_sql(
            definition.replace(source, target, count), execution_options={"no_parameters": True}
        )
    print(
        "STEP7B_QUERY_TERMINAL_MIGRATION "
        f"direction={'upgrade' if forward else 'downgrade'} "
        "state=ready answer_variant=schema_dispatch reranker=base_top20_union "
        "embedding_channels=schema_dispatch answer_reuse=strict_lineage "
        "fact_rows_modified=0",
        flush=True,
    )


def upgrade() -> None:
    _rewrite(forward=True)


def downgrade() -> None:
    _rewrite(forward=False)
