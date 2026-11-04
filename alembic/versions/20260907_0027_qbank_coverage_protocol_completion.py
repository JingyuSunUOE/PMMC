"""Align coverage completion with versioned proxy metrics and v6 query channels.

DDL only: no historical metric, embedding or model-call facts are rewritten.
"""

import sqlalchemy as sa
from alembic import op

revision: str = "20260907_0027"
down_revision: str | None = "20260907_0026"
branch_labels: str | None = None
depends_on: str | None = None

_FUNCTION = "pmmc_evaluation.enforce_qbank_coverage_completion()"
_HARDENED = (
    "NEW.protocol_version IN ('qbank-coverage-v3','qbank-coverage-v4',"
    "'qbank-coverage-v5','qbank-coverage-v6')"
)
_LEGACY_METRICS = (
    "ARRAY['qbank_evidence_recall_at_k','qbank_cover_at_k',"
    "'qbank_match_at_k','projected_fallback_at_k']"
)
_PROXY_METRICS = (
    "ARRAY['qbank_evidence_recall_at_k','qbank_evidence_proxy_cover_at_k',"
    "'qbank_evidence_proxy_match_at_k','proxy_projected_fallback_at_k']"
)
_LEGACY_ORACLES = "ARRAY['oracle_qbank_cover','oracle_qbank_match']"
_PROXY_ORACLES = "ARRAY['oracle_qbank_evidence_proxy_cover','oracle_qbank_evidence_proxy_match']"
_LEGACY_CHANNELS = "CASE WHEN queries.has_query_image THEN 2 ELSE 1 END"
_VERSIONED_CHANNELS = """CASE WHEN queries.has_query_image
                         OR NEW.protocol_version='qbank-coverage-v6'
                         THEN 2 ELSE 1 END
                OR NOT EXISTS (SELECT 1
                   FROM pmmc_evaluation.qbank_coverage_query_embeddings embeddings
                  WHERE embeddings.coverage_run_id=NEW.coverage_run_id
                    AND embeddings.query_id=queries.query_id
                    AND embeddings.channel='text_semantic')"""


def _rewrite(*, forward: bool) -> None:
    connection = op.get_bind()
    definition = connection.scalar(
        sa.text("SELECT pg_get_functiondef(to_regprocedure(:name))"), {"name": _FUNCTION}
    )
    if not isinstance(definition, str):
        raise RuntimeError("QBank coverage completion function is missing")
    replacements = (
        (
            _LEGACY_METRICS,
            f"CASE WHEN {_HARDENED} THEN {_PROXY_METRICS} ELSE {_LEGACY_METRICS} END",
            1,
        ),
        (
            _LEGACY_ORACLES,
            f"CASE WHEN {_HARDENED} THEN {_PROXY_ORACLES} ELSE {_LEGACY_ORACLES} END",
            1,
        ),
        (_LEGACY_CHANNELS, _VERSIONED_CHANNELS, 1),
        # Qualify the outer UNNEST column: an unqualified name binds to the inner
        # metrics table, turning the required-name comparison into a tautology.
        (
            ") metric_name\n",
            ") AS required_metric(metric_name)\n",
            2,
        ),
        (
            "metrics.metric_name = metric_name",
            "metrics.metric_name = required_metric.metric_name",
            2,
        ),
    )
    for old, new, count in replacements:
        source, target = (old, new) if forward else (new, old)
        if definition.count(source) != count:
            raise RuntimeError("QBank coverage guard rewrite match count differs")
        definition = definition.replace(source, target, count)
    connection.exec_driver_sql(definition, execution_options={"no_parameters": True})
    print(
        "QBANK_COMPLETION_GUARD_MIGRATION "
        f"direction={'upgrade' if forward else 'downgrade'} state=ready fact_rows_modified=0",
        flush=True,
    )


def upgrade() -> None:
    _rewrite(forward=True)


def downgrade() -> None:
    _rewrite(forward=False)
