"""Versioned completion-function extension for Step8 method-scoped evaluation.

The 0032 completion trigger remains installed. Bootstrap and the maintenance installer
apply this same extension without rewriting historical Alembic migrations or
component receipts. All original source, reference and completion guards remain.
"""

import hashlib

VERSION = "step8-evaluation-method-scope-v3"
FUNCTION = "pmmc_evaluation.enforce_step8_evaluation_completion()"
_MARKER = "      -- step8-evaluation-method-scope-v2\n"
_OLD_MARKER = "      -- step8-evaluation-method-scope-v1\n"
_OLD_SCOPE = """      IF COALESCE(evaluation.metadata->>'method_scope','all') NOT IN ('all','pmmc')
         OR (evaluation.metadata ? 'method_scope' AND (
              jsonb_typeof(evaluation.metadata->'method_scope') IS DISTINCT FROM 'string'
              OR report->'method_scope' IS DISTINCT FROM evaluation.metadata->'method_scope'))
         OR (NOT (evaluation.metadata ? 'method_scope') AND report ? 'method_scope')
         OR (evaluation.metadata->>'method_scope'='pmmc'
             AND report->'all_methods_evaluated' IS DISTINCT FROM 'false'::jsonb) THEN
        RAISE EXCEPTION 'step8_evaluation_method_scope_invalid' USING ERRCODE='55000';
      END IF;
"""
_SCOPE = _OLD_SCOPE.replace("('all','pmmc')", "('all','pmmc','pmmc_main','paired_main')").replace(
    "evaluation.metadata->>'method_scope'='pmmc'",
    "evaluation.metadata->>'method_scope' IN ('pmmc','pmmc_main','paired_main')",
) + """      IF (COALESCE(evaluation.metadata->>'method_scope','all')='pmmc_main') != EXISTS (
            SELECT 1 FROM pmmc_runtime.query_executor_runs r
            WHERE r.query_executor_run_id=evaluation.legacy_source_run_id
              AND r.metadata->>'readout'='exploration-primary-b8-r2')
         OR (evaluation.metadata->>'method_scope'='pmmc_main' AND NOT EXISTS (
            SELECT 1 FROM pmmc_runtime.query_executor_runs r
            JOIN pmmc_runtime.benchmark_campaigns campaign
              ON campaign.campaign_id=evaluation.metadata->>'campaign_id'
            WHERE r.query_executor_run_id=evaluation.legacy_source_run_id
              AND campaign.phase='exploration' AND r.metadata->>'treatment'='main'
              AND r.metadata->'execute_full_grid'='false'::jsonb
              AND r.metadata->'declared_cells'='[[8,2]]'::jsonb
              AND r.metadata->>'expected_cell_count'='3'))
         OR (evaluation.metadata->>'method_scope'='paired_main' AND NOT EXISTS (
            SELECT 1 FROM pmmc_runtime.query_executor_runs r
            JOIN pmmc_runtime.benchmark_campaigns campaign
              ON campaign.campaign_id=evaluation.metadata->>'campaign_id'
            WHERE r.query_executor_run_id=evaluation.legacy_source_run_id
              AND campaign.phase='exploration' AND r.metadata->>'treatment'='main')) THEN
        RAISE EXCEPTION 'step8_evaluation_primary_scope_invalid' USING ERRCODE='55000';
      END IF;
"""
_OLD_FILTER = "AND (COALESCE(evaluation.metadata->>'method_scope','all')='all' OR c.method='pmmc')"
_FILTER = """AND (COALESCE(evaluation.metadata->>'method_scope','all') IN ('all','paired_main') OR c.method='pmmc')
        AND (COALESCE(evaluation.metadata->>'method_scope','all') NOT IN ('pmmc_main','paired_main')
             OR (c.payload->>'treatment'='main'
                 AND c.payload->>'source_variant'='active'
                 AND c.payload->>'question_budget'='8' AND c.payload->>'round_id'='2'))"""
_JUDGE = """AND (c.evaluation_run_id=NEW.evaluation_run_id OR EXISTS (
                   SELECT 1 FROM pmmc_evaluation.evaluation_runs origin
                    WHERE origin.evaluation_run_id=c.evaluation_run_id
                      AND origin.protocol_version=evaluation.protocol_version
                      AND origin.legacy_source_run_id=evaluation.legacy_source_run_id
                      AND origin.judge_model_alias=evaluation.judge_model_alias
                      AND origin.judge_prompt_hash=evaluation.judge_prompt_hash
                      AND origin.judge_rubric_version=evaluation.judge_rubric_version
                      AND NOT EXISTS (
                        SELECT 1 FROM unnest(ARRAY['campaign_id','protocol_hash','query_hash',
                          'gold_hash','snapshot_id','judge_config','judge_prompt_hash',
                          'judge_model_hash','adapter_source_hash']) key
                         WHERE origin.metadata->key IS NULL
                            OR origin.metadata->key IS DISTINCT FROM evaluation.metadata->key)))"""


def _changes(*, legacy=False):
    marker, scope, cell_filter = (
        (_OLD_MARKER, _OLD_SCOPE, _OLD_FILTER) if legacy else (_MARKER, _SCOPE, _FILTER)
    )
    return (
        ("      SELECT count(*) INTO expected FROM pmmc_runtime.benchmark_cell_assignments a",
         marker + scope
         + "      SELECT count(*) INTO expected FROM pmmc_runtime.benchmark_cell_assignments a"),
        ("AND c.answer_model=r.metadata->>'answer_model_alias'\n        AND ((",
         "AND c.answer_model=r.metadata->>'answer_model_alias'\n        " + cell_filter
         + "\n        AND (("),
        ("AND c.dataset=r.dataset AND c.answer_model=r.metadata->>'answer_model_alias'"
         "\n                  AND ((",
         "AND c.dataset=r.dataset AND c.answer_model=r.metadata->>'answer_model_alias'"
         "\n                  "
         + cell_filter + "\n                  AND (("),
        ("AND c.evaluation_run_id=NEW.evaluation_run_id", _JUDGE),
    )


def _v2_definition(definition: str) -> str:
    """Exact, reversible edits of the installed body, including idempotent replay."""
    if _OLD_MARKER in definition or _MARKER in definition:
        legacy = _OLD_MARKER in definition
        original = definition
        for before, after in reversed(_changes(legacy=legacy)):
            if original.count(after) != 1:
                raise ValueError("step8_evaluation_scope_extension_drift")
            original = original.replace(after, before, 1)
        updated = _v2_definition(original)
        if not legacy and updated != definition:
            raise ValueError("step8_evaluation_scope_extension_drift")
        return updated
    for before, after in _changes():
        if definition.count(before) != 1:
            raise ValueError("step8_evaluation_completion_contract_mismatch")
        definition = definition.replace(before, after, 1)
    return definition


# Present a completed standalone child as an evaluator source, never as a row in
# query_executor_runs. The original count, assignment, source-fact, Judge, gold
# and object receipt guards below all run on both paths.
_STANDALONE_SOURCE = """(
            SELECT r.query_executor_run_id,r.schema_version,r.status,r.snapshot_id,r.dataset,r.metadata
              FROM pmmc_runtime.query_executor_runs r
             WHERE COALESCE(evaluation.metadata->>'method_scope','all')!='baseline'
            UNION ALL
            SELECT b.run_id,NULL::text,s.status,u.snapshot_id,u.dataset,
                   jsonb_build_object('answer_model_alias',u.answer_model_alias,'treatment','main')
              FROM pmmc_runtime.baseline_suite_runs b
              JOIN pmmc_runtime.baseline_suites u ON u.suite_id=b.suite_id
              JOIN pmmc_runtime.run_states s ON s.run_id=b.run_id
              JOIN pmmc_runtime.benchmark_campaigns camp ON camp.campaign_id=evaluation.metadata->>'campaign_id'
             WHERE evaluation.metadata->>'method_scope'='baseline'
               AND evaluation.source_kind='baseline_suite' AND evaluation.suite_id=u.suite_id
               AND u.suite_id=evaluation.metadata->>'suite_id' AND camp.phase='exploration'
               AND u.metadata->>'release_plan_hash'=camp.payload->>'release_plan_hash'
               AND u.metadata->>'release_campaign_id'=camp.campaign_id
               AND camp.payload->'snapshot_ids'->>u.dataset=u.snapshot_id
               AND u.metadata->>'method_protocol_hash'=camp.protocol_hash
               AND u.metadata->>'partition_role'='development' AND u.metadata->'paper_result'='false'::jsonb
               AND u.metadata->>'answer_contract_version'='typed-answer-contract-v3'
               AND u.metadata->>'standalone_method'=b.method AND cardinality(u.methods)=1
               AND u.methods[1]=b.method AND b.method=evaluation.metadata->>'standalone_method'
               AND EXISTS (SELECT 1 FROM pmmc_runtime.baseline_suite_queries sq
                            WHERE sq.suite_id=u.suite_id AND sq.query_id=evaluation.metadata->>'query_id')
        ) r"""


_SELECTED_SCOPE = """      IF EXISTS (
            SELECT 1 FROM pmmc_runtime.benchmark_campaigns camp
             WHERE camp.campaign_id=evaluation.metadata->>'campaign_id' AND camp.payload ? 'release_plan_hash'
               AND NOT (COALESCE(evaluation.metadata->>'method_scope','all')='baseline' OR (
                 COALESCE(evaluation.metadata->>'method_scope','all') IN ('pmmc','pmmc_main')
                 AND EXISTS (SELECT 1 FROM pmmc_runtime.query_executor_runs r
                   JOIN pmmc_runtime.baseline_suite_runs b ON b.run_id=r.metadata->>'fallback_run_id'
                   JOIN pmmc_runtime.baseline_suites u ON u.suite_id=b.suite_id
                  WHERE r.query_executor_run_id=evaluation.legacy_source_run_id
                    AND evaluation.source_kind='pmmc_query_run' AND u.suite_id=evaluation.metadata->>'suite_id'
                    AND b.method='native_multimodal_rag' AND cardinality(u.methods)=1
                    AND u.metadata->'internal_pmmc_fallback'='true'::jsonb
                    AND u.metadata->>'release_campaign_id'=camp.campaign_id
                    AND u.metadata->>'release_plan_hash'=camp.payload->>'release_plan_hash'
                    AND u.metadata->>'method_protocol_hash'=camp.protocol_hash)))) THEN
        RAISE EXCEPTION 'step8_selected_evaluation_scope_invalid' USING ERRCODE='55000';
      END IF;
"""


def scoped_completion_function(definition: str) -> str:
    marker = "      -- " + VERSION + "\n"
    replacements = [
        (_MARKER, marker + _SELECTED_SCOPE),
        ("('all','pmmc','pmmc_main','paired_main')", "('all','pmmc','pmmc_main','paired_main','baseline')"),
        ("IN ('pmmc','pmmc_main','paired_main')", "IN ('pmmc','pmmc_main','paired_main','baseline')"),
        ("report->>'query_executor_run_id' IS DISTINCT FROM evaluation.legacy_source_run_id",
         """(CASE WHEN evaluation.metadata->>'method_scope'='baseline' THEN report->>'baseline_run_id'
                  ELSE report->>'query_executor_run_id' END) IS DISTINCT FROM evaluation.legacy_source_run_id
         OR (evaluation.metadata->>'method_scope'='baseline' AND
             report->'query_executor_run_id' IS DISTINCT FROM 'null'::jsonb)"""),
        ("FROM pmmc_runtime.query_executor_runs r\n             JOIN pmmc_runtime.run_queries",
         "FROM " + _STANDALONE_SOURCE + "\n             JOIN pmmc_runtime.run_queries"),
        ("AND r.schema_version='0.3.0' AND r.status='completed'",
         "AND (r.schema_version='0.3.0' OR evaluation.metadata->>'method_scope'='baseline') AND r.status='completed'"),
    ]
    for indent in ("        ", "                 "):
        on = "\n" + indent + "ON r.query_executor_run_id=evaluation.legacy_source_run_id"
        replacements.append(("JOIN pmmc_runtime.query_executor_runs r" + on,
                             "JOIN " + _STANDALONE_SOURCE + on))
    v3_filter = _FILTER.replace("OR c.method='pmmc')", """OR
            (evaluation.metadata->>'method_scope'='baseline' AND c.method=evaluation.metadata->>'standalone_method') OR
            (COALESCE(evaluation.metadata->>'method_scope','all')!='baseline' AND c.method='pmmc'))""")
    replacements.append((_FILTER, v3_filter))
    installed = marker in definition
    if installed:
        for before, after in reversed(replacements):
            if definition.count(after) != (2 if before == _FILTER else 1):
                raise ValueError("step8_evaluation_scope_extension_drift")
            definition = definition.replace(after, before)
    definition = _v2_definition(definition)
    for before, after in replacements:
        if definition.count(before) != (2 if before == _FILTER else 1):
            raise ValueError("step8_evaluation_completion_contract_mismatch")
        definition = definition.replace(before, after)
    return definition


def install_evaluation_scope(connection) -> dict:
    """Run inside an owner transaction; no answer, gold, score or ledger mutations."""
    row = connection.execute(
        "SELECT pg_get_functiondef(%s::regprocedure) AS definition", (FUNCTION,)
    ).fetchone()
    before = row["definition"]
    after = scoped_completion_function(before)
    if before != after:
        connection.execute(after)
    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    return {"version": VERSION, "function": FUNCTION, "before_sha256": digest(before),
            "after_sha256": digest(after), "changed": before != after,
            "table_schema_changed": False, "scientific_rows_changed": False}
