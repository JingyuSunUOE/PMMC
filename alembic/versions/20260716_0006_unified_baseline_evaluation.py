# ruff: noqa: E501
"""Add paired baseline suites and independent evaluation runs.

Revision ID: 20260716_0006
Revises: 20260715_0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260716_0006"
down_revision: str | None = "20260715_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"

BAD_CASE_VALUES = (
    "retrieval_top50_miss",
    "retrieval_top5_miss",
    "context_budget_exclusion",
    "visual_evidence_miss",
    "text_evidence_miss",
    "evidence_found_answer_wrong",
    "answer_format_mismatch",
    "judge_metric_disagreement",
    "model_call_failure",
    "gold_unresolvable",
    "questioner_coverage_failure",
    "matching_failure",
    "planner_failure",
    "doubter_failure",
    "evidence_insufficiency",
    "answer_generation_failure",
    "update_conflict_failure",
    "none",
    "unknown",
)
LEGACY_BAD_CASE_VALUES = BAD_CASE_VALUES[10:]
LEGACY_JUDGE_UNIQUE_CONSTRAINT = (
    "judge_scores_answer_id_judge_model_alias_rubric_version_rep_key"
)
BAD_CASE_CHECK_CONSTRAINTS = (
    (
        "judge_scores",
        "bad_case_category",
        "judge_scores_bad_case_category_check",
        "ck_judge_scores_judge_scores_bad_case_category_check",
        "judge_scores_bad_case_category_values_v2",
        "ck_judge_scores_judge_scores_bad_case_category_values_v2",
        "ck_judge_scores_bad_case_category_values",
    ),
    (
        "bad_case_labels",
        "category",
        "bad_case_labels_category_check",
        "ck_bad_case_labels_bad_case_labels_category_check",
        "bad_case_labels_category_values_v2",
        "ck_bad_case_labels_bad_case_labels_category_values_v2",
        "ck_bad_case_labels_category_values",
    ),
)


def _values(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{value}'" for value in values) + ")"


def _replace_bad_case_check_constraints(
    values: tuple[str, ...],
    *,
    use_v2_names: bool,
) -> None:
    """Replace category checks without depending on Alembic name expansion.

    Migration 0002 created anonymous checks through raw SQL, while an Alembic
    downgrade creates named checks under the SQLAlchemy naming convention. Both
    physical names must therefore be removed before recreating the constraint.
    """

    for (
        table,
        column,
        legacy_name,
        convention_legacy_name,
        historical_v2_name,
        convention_historical_v2_name,
        stable_v2_name,
    ) in BAD_CASE_CHECK_CONSTRAINTS:
        for constraint_name in (
            legacy_name,
            convention_legacy_name,
            historical_v2_name,
            convention_historical_v2_name,
            stable_v2_name,
        ):
            op.execute(
                f"ALTER TABLE {EVALUATION}.{table} "
                f"DROP CONSTRAINT IF EXISTS {constraint_name}"
            )
        target_name = stable_v2_name if use_v2_names else legacy_name
        op.create_check_constraint(
            op.f(target_name),
            table,
            f"{column} IN {_values(values)}",
            schema=EVALUATION,
        )


def upgrade() -> None:
    _create_suite_tables()
    _create_evaluation_lifecycle_tables()
    _extend_existing_evaluation_tables()
    _backfill_legacy_evaluation_runs()
    _create_evaluation_fact_tables()
    _replace_freeze_boundaries()
    _create_cost_views()
    _grant_permissions()


def _create_suite_tables() -> None:
    statements = (
        f"""CREATE TABLE {RUNTIME}.baseline_suites (
            suite_id varchar(128) PRIMARY KEY,
            dataset varchar(32) NOT NULL CHECK (dataset IN ('memlens','memgallery','unknown')),
            snapshot_id varchar(128) NOT NULL REFERENCES {RUNTIME}.dataset_snapshots(snapshot_id),
            selection_hash varchar(128) NOT NULL,
            methods text[] NOT NULL,
            answer_model_alias varchar(256) NOT NULL,
            answer_prompt_hash varchar(128) NOT NULL,
            decoding_config jsonb NOT NULL,
            total_context_tokens integer NOT NULL CHECK (total_context_tokens > 0),
            reserved_output_tokens integer NOT NULL CHECK (
                reserved_output_tokens > 0 AND reserved_output_tokens < total_context_tokens
            ),
            query_image_policy varchar(128) NOT NULL,
            evaluation_policy_version varchar(128) NOT NULL,
            seed integer NOT NULL CHECK (seed >= 0),
            paper_mode boolean NOT NULL,
            config_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (suite_id, snapshot_id),
            CHECK (cardinality(methods) > 0)
        )""",
        f"""CREATE TABLE {RUNTIME}.baseline_suite_states (
            suite_id varchar(128) PRIMARY KEY REFERENCES {RUNTIME}.baseline_suites(suite_id),
            status varchar(16) NOT NULL CHECK (status IN ('created','running','ready','failed','cancelled')),
            selected_count integer NOT NULL CHECK (selected_count > 0),
            completed_run_count integer NOT NULL CHECK (completed_run_count >= 0),
            started_at timestamptz,
            heartbeat_at timestamptz,
            completed_at timestamptz,
            error_type varchar(256),
            error_summary text,
            metadata jsonb NOT NULL,
            CHECK (status != 'failed' OR error_type IS NOT NULL)
        )""",
        f"""CREATE TABLE {RUNTIME}.baseline_suite_runs (
            suite_id varchar(128) NOT NULL REFERENCES {RUNTIME}.baseline_suites(suite_id),
            method varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.run_manifests(run_id),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            PRIMARY KEY (suite_id, method),
            UNIQUE (suite_id, run_id),
            UNIQUE (suite_id, method, run_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.baseline_suite_queries (
            suite_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            metadata jsonb NOT NULL,
            PRIMARY KEY (suite_id, snapshot_id, query_id),
            UNIQUE (suite_id, ordinal),
            FOREIGN KEY (suite_id, snapshot_id)
                REFERENCES {RUNTIME}.baseline_suites(suite_id, snapshot_id),
            FOREIGN KEY (snapshot_id, query_id)
                REFERENCES {RUNTIME}.benchmark_queries(snapshot_id, query_id)
        )""",
    )
    for statement in statements:
        op.execute(statement)


def _create_evaluation_lifecycle_tables() -> None:
    statements = (
        f"""CREATE TABLE {EVALUATION}.evaluation_runs (
            evaluation_run_id varchar(128) PRIMARY KEY,
            suite_id varchar(128) REFERENCES {RUNTIME}.baseline_suites(suite_id),
            legacy_source_run_id varchar(128) REFERENCES {RUNTIME}.run_manifests(run_id),
            protocol_version varchar(128) NOT NULL,
            normalizer_version varchar(128) NOT NULL,
            evidence_policy_version varchar(128) NOT NULL,
            bad_case_rule_version varchar(128) NOT NULL,
            judge_model_alias varchar(256) NOT NULL,
            judge_prompt_hash varchar(128) NOT NULL,
            judge_rubric_version varchar(128) NOT NULL,
            judge_repeat_count integer NOT NULL CHECK (judge_repeat_count > 0),
            judge_score_values double precision[] NOT NULL,
            secondary_judge_enabled boolean NOT NULL,
            metric_versions jsonb NOT NULL,
            config_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (suite_id, config_hash),
            CHECK (suite_id IS NOT NULL OR legacy_source_run_id IS NOT NULL),
            CHECK (judge_score_values = ARRAY[0.0,0.5,1.0]::double precision[])
        )""",
        f"""CREATE TABLE {EVALUATION}.evaluation_run_states (
            evaluation_run_id varchar(128) PRIMARY KEY
                REFERENCES {EVALUATION}.evaluation_runs(evaluation_run_id),
            status varchar(16) NOT NULL CHECK (status IN ('created','running','completed','failed','cancelled')),
            selected_count integer NOT NULL CHECK (selected_count > 0),
            target_count integer NOT NULL CHECK (target_count > 0),
            succeeded_count integer NOT NULL CHECK (succeeded_count >= 0),
            failed_count integer NOT NULL CHECK (failed_count >= 0),
            started_at timestamptz,
            heartbeat_at timestamptz,
            completed_at timestamptz,
            error_type varchar(256),
            error_summary text,
            metadata jsonb NOT NULL,
            CHECK (succeeded_count + failed_count <= selected_count * target_count),
            CHECK (status != 'failed' OR error_type IS NOT NULL)
        )""",
        f"""CREATE TABLE {EVALUATION}.evaluation_targets (
            evaluation_run_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.evaluation_runs(evaluation_run_id),
            suite_id varchar(128),
            run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.run_manifests(run_id),
            method varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL REFERENCES {RUNTIME}.dataset_snapshots(snapshot_id),
            selected_count integer NOT NULL CHECK (selected_count > 0),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            PRIMARY KEY (evaluation_run_id, run_id),
            UNIQUE (evaluation_run_id, method),
            UNIQUE (evaluation_run_id, run_id, snapshot_id),
            FOREIGN KEY (suite_id, method, run_id)
                REFERENCES {RUNTIME}.baseline_suite_runs(suite_id, method, run_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.evaluation_jobs (
            evaluation_job_id varchar(128) PRIMARY KEY,
            evaluation_run_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.evaluation_runs(evaluation_run_id),
            run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            answer_id varchar(128) NOT NULL REFERENCES {RUNTIME}.final_answers(answer_id),
            stage varchar(32) NOT NULL CHECK (stage IN ('deterministic','judge','analyze')),
            status varchar(16) NOT NULL CHECK (status IN ('pending','running','succeeded','failed','cancelled')),
            priority integer NOT NULL CHECK (priority >= 0),
            attempt_count integer NOT NULL CHECK (attempt_count >= 0),
            max_attempts integer NOT NULL CHECK (max_attempts > 0),
            claimed_by varchar(256),
            lease_expires_at timestamptz,
            available_at timestamptz NOT NULL,
            created_at timestamptz NOT NULL,
            started_at timestamptz,
            finished_at timestamptz,
            error_type varchar(256),
            error_summary text,
            metadata jsonb NOT NULL,
            UNIQUE (evaluation_job_id, evaluation_run_id),
            UNIQUE (evaluation_run_id, run_id, query_id, stage),
            FOREIGN KEY (evaluation_run_id, run_id, snapshot_id)
                REFERENCES {EVALUATION}.evaluation_targets(
                    evaluation_run_id, run_id, snapshot_id
                ),
            FOREIGN KEY (answer_id, run_id, query_id)
                REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id),
            CHECK (status != 'running' OR (claimed_by IS NOT NULL AND lease_expires_at IS NOT NULL)),
            CHECK (status NOT IN ('succeeded','failed','cancelled') OR finished_at IS NOT NULL),
            CHECK (status != 'failed' OR error_type IS NOT NULL)
        )""",
        f"""CREATE TABLE {EVALUATION}.evaluation_job_attempts (
            attempt_id varchar(128) PRIMARY KEY,
            evaluation_job_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.evaluation_jobs(evaluation_job_id),
            attempt_number integer NOT NULL CHECK (attempt_number > 0),
            worker_id varchar(256) NOT NULL,
            status varchar(16) NOT NULL CHECK (status IN ('pending','running','succeeded','failed','cancelled')),
            started_at timestamptz NOT NULL,
            finished_at timestamptz,
            error_type varchar(256),
            error_summary text,
            metadata jsonb NOT NULL,
            UNIQUE (evaluation_job_id, attempt_number)
        )""",
    )
    for statement in statements:
        op.execute(statement)


def _extend_existing_evaluation_tables() -> None:
    op.execute(
        f"ALTER TABLE {EVALUATION}.model_calls "
        "DROP CONSTRAINT IF EXISTS model_calls_job_id_run_id_fkey"
    )
    for table in (
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
    ):
        op.add_column(
            table,
            sa.Column("evaluation_run_id", sa.String(128), nullable=True),
            schema=EVALUATION,
        )
        op.create_foreign_key(
            f"fk_{table}_evaluation_run",
            table,
            "evaluation_runs",
            ["evaluation_run_id"],
            ["evaluation_run_id"],
            source_schema=EVALUATION,
            referent_schema=EVALUATION,
        )

    op.add_column(
        "metric_scores",
        sa.Column("metric_family", sa.String(32), nullable=False, server_default="answer"),
        schema=EVALUATION,
    )
    op.add_column("metric_scores", sa.Column("k", sa.Integer()), schema=EVALUATION)
    op.add_column(
        "metric_scores",
        sa.Column("applicable", sa.Boolean(), nullable=False, server_default=sa.true()),
        schema=EVALUATION,
    )
    op.add_column(
        "metric_scores", sa.Column("unavailable_reason", sa.String(256)), schema=EVALUATION
    )
    op.alter_column("metric_scores", "value", schema=EVALUATION, nullable=True)
    op.add_column("judge_scores", sa.Column("verdict", sa.String(32)), schema=EVALUATION)
    op.add_column(
        "model_calls",
        sa.Column("evaluation_job_id", sa.String(128), nullable=True),
        schema=EVALUATION,
    )
    op.create_foreign_key(
        "fk_evaluation_model_calls_evaluation_job",
        "model_calls",
        "evaluation_jobs",
        ["evaluation_job_id", "evaluation_run_id"],
        ["evaluation_job_id", "evaluation_run_id"],
        source_schema=EVALUATION,
        referent_schema=EVALUATION,
    )
    op.add_column(
        "bad_case_labels", sa.Column("rule_version", sa.String(128)), schema=EVALUATION
    )
    op.add_column(
        "bad_case_labels",
        sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.false()),
        schema=EVALUATION,
    )


def _legacy_run_union() -> str:
    return " UNION ".join(
        f"SELECT run_id FROM {EVALUATION}.{table}"
        for table in (
            "run_artifacts",
            "model_calls",
            "retrieval_evaluations",
            "metric_scores",
            "judge_scores",
            "bad_case_labels",
        )
    )


def _backfill_legacy_evaluation_runs() -> None:
    run_union = _legacy_run_union()
    op.execute(
        f"""INSERT INTO {EVALUATION}.evaluation_runs (
                evaluation_run_id, suite_id, legacy_source_run_id,
                protocol_version, normalizer_version, evidence_policy_version,
                bad_case_rule_version, judge_model_alias, judge_prompt_hash,
                judge_rubric_version, judge_repeat_count, judge_score_values,
                secondary_judge_enabled, metric_versions, config_hash, created_at, metadata
            )
            SELECT 'legacy-eval-' || substr(md5(run_id), 1, 24), NULL, run_id,
                'legacy-v0.4', 'legacy-v0.4', 'legacy-v0.4', 'legacy-v0.4',
                'legacy', repeat('0', 64), 'legacy-v0.4', 1,
                ARRAY[0.0,0.5,1.0]::double precision[], false, '{{}}'::jsonb,
                md5('legacy:' || run_id), now(), jsonb_build_object('legacy', true)
            FROM ({run_union}) AS source_runs
            ON CONFLICT (evaluation_run_id) DO NOTHING"""
    )
    op.execute(
        f"""INSERT INTO {EVALUATION}.evaluation_run_states (
                evaluation_run_id, status, selected_count, target_count,
                succeeded_count, failed_count, started_at, metadata
            )
            SELECT 'legacy-eval-' || substr(md5(run_id), 1, 24), 'running',
                GREATEST((SELECT count(*) FROM {RUNTIME}.run_queries q WHERE q.run_id = source.run_id), 1),
                1, 0, 0, now(), jsonb_build_object('legacy', true)
            FROM ({run_union}) AS source
            ON CONFLICT (evaluation_run_id) DO NOTHING"""
    )
    op.execute(
        f"""INSERT INTO {EVALUATION}.evaluation_targets (
                evaluation_run_id, suite_id, run_id, method, snapshot_id,
                selected_count, created_at, metadata
            )
            SELECT 'legacy-eval-' || substr(md5(source.run_id), 1, 24), NULL,
                source.run_id, manifests.method, links.snapshot_id,
                GREATEST((SELECT count(*) FROM {RUNTIME}.run_queries q WHERE q.run_id = source.run_id), 1),
                now(), jsonb_build_object('legacy', true)
            FROM ({run_union}) AS source
            JOIN {RUNTIME}.run_manifests manifests ON manifests.run_id = source.run_id
            JOIN LATERAL (
                SELECT snapshot_id FROM {RUNTIME}.run_snapshot_links
                WHERE run_id = source.run_id ORDER BY snapshot_id LIMIT 1
            ) links ON true
            ON CONFLICT (evaluation_run_id, run_id) DO NOTHING"""
    )
    for table in (
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
    ):
        op.execute(
            f"""UPDATE {EVALUATION}.{table}
                SET evaluation_run_id = 'legacy-eval-' || substr(md5(run_id), 1, 24)
                WHERE evaluation_run_id IS NULL"""
        )
        op.alter_column(table, "evaluation_run_id", schema=EVALUATION, nullable=False)

    op.execute(
        f"ALTER TABLE {EVALUATION}.retrieval_evaluations "
        "DROP CONSTRAINT IF EXISTS retrieval_evaluations_retrieval_trace_id_key"
    )
    op.create_unique_constraint(
        "uq_retrieval_evaluations_run_trace",
        "retrieval_evaluations",
        ["evaluation_run_id", "retrieval_trace_id"],
        schema=EVALUATION,
    )
    op.execute(
        f"ALTER TABLE {EVALUATION}.metric_scores "
        "DROP CONSTRAINT IF EXISTS metric_scores_value_check"
    )
    op.create_check_constraint(
        "metric_scores_value_probability",
        "metric_scores",
        "value IS NULL OR (value >= 0 AND value <= 1)",
        schema=EVALUATION,
    )
    op.create_check_constraint(
        "metric_scores_family_values",
        "metric_scores",
        "metric_family IN ('answer','retrieval','context','efficiency')",
        schema=EVALUATION,
    )
    op.create_check_constraint(
        "metric_scores_k_positive",
        "metric_scores",
        "k IS NULL OR k > 0",
        schema=EVALUATION,
    )
    op.create_check_constraint(
        "metric_scores_applicability",
        "metric_scores",
        "(applicable AND value IS NOT NULL AND unavailable_reason IS NULL) OR "
        "(NOT applicable AND value IS NULL AND unavailable_reason IS NOT NULL)",
        schema=EVALUATION,
    )
    op.create_unique_constraint(
        "uq_metric_scores_evaluation_answer_metric",
        "metric_scores",
        ["evaluation_run_id", "answer_id", "metric_name", "metric_version"],
        schema=EVALUATION,
    )
    op.execute(
        f"ALTER TABLE {EVALUATION}.judge_scores DROP CONSTRAINT IF EXISTS "
        f"{LEGACY_JUDGE_UNIQUE_CONSTRAINT}"
    )
    op.create_unique_constraint(
        "uq_judge_scores_evaluation_answer_judge",
        "judge_scores",
        [
            "evaluation_run_id",
            "answer_id",
            "judge_model_alias",
            "rubric_version",
            "repeat_index",
        ],
        schema=EVALUATION,
    )
    op.create_check_constraint(
        "judge_scores_verdict_values",
        "judge_scores",
        "verdict IS NULL OR verdict IN ('incorrect','partially_correct','correct')",
        schema=EVALUATION,
    )
    _replace_bad_case_check_constraints(BAD_CASE_VALUES, use_v2_names=True)
    op.create_unique_constraint(
        "uq_bad_case_labels_evaluation_answer_rule",
        "bad_case_labels",
        ["evaluation_run_id", "answer_id", "category", "label_source", "rule_version"],
        schema=EVALUATION,
    )


def _create_evaluation_fact_tables() -> None:
    statements = (
        f"""CREATE TABLE {EVALUATION}.answer_normalizations (
            normalization_id varchar(128) PRIMARY KEY,
            evaluation_run_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.evaluation_runs(evaluation_run_id),
            answer_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            run_id varchar(128) NOT NULL,
            normalizer_version varchar(128) NOT NULL,
            candidate_hash varchar(64) NOT NULL,
            reference_hash varchar(64) NOT NULL,
            normalized_candidate text NOT NULL,
            normalized_reference text NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (evaluation_run_id, answer_id, normalizer_version),
            FOREIGN KEY (answer_id, run_id, query_id)
                REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.gold_evidence_groups (
            gold_group_id varchar(128) PRIMARY KEY,
            evaluation_run_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.evaluation_runs(evaluation_run_id),
            snapshot_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            granularity varchar(32) NOT NULL CHECK (
                granularity IN ('generic','session','memory','image','text')
            ),
            source_gold_id varchar(256) NOT NULL,
            resolution_policy_version varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (evaluation_run_id, query_id, source_gold_id),
            FOREIGN KEY (snapshot_id, query_id)
                REFERENCES {EVALUATION}.query_gold(snapshot_id, query_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.gold_evidence_group_members (
            gold_group_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.gold_evidence_groups(gold_group_id),
            member_type varchar(32) NOT NULL CHECK (
                member_type IN ('evidence','session','memory','image')
            ),
            member_id varchar(256) NOT NULL,
            derived boolean NOT NULL,
            metadata jsonb NOT NULL,
            PRIMARY KEY (gold_group_id, member_type, member_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.context_evaluations (
            context_evaluation_id varchar(128) PRIMARY KEY,
            evaluation_run_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.evaluation_runs(evaluation_run_id),
            answer_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            run_id varchar(128) NOT NULL,
            context_assembly_id varchar(128) NOT NULL
                REFERENCES {RUNTIME}.context_assemblies(context_assembly_id),
            granularity varchar(32) NOT NULL CHECK (
                granularity IN ('generic','session','memory','image','text')
            ),
            applicable boolean NOT NULL,
            gold_group_count integer NOT NULL CHECK (gold_group_count >= 0),
            policy_version varchar(128) NOT NULL,
            unavailable_reason varchar(256),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (evaluation_run_id, answer_id, granularity),
            FOREIGN KEY (answer_id, run_id, query_id)
                REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id),
            CHECK (
                (applicable AND gold_group_count > 0 AND unavailable_reason IS NULL) OR
                (NOT applicable AND unavailable_reason IS NOT NULL)
            )
        )""",
        f"""CREATE TABLE {EVALUATION}.context_evaluation_matches (
            context_evaluation_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.context_evaluations(context_evaluation_id),
            gold_group_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.gold_evidence_groups(gold_group_id),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (context_evaluation_id, gold_group_id),
            UNIQUE (context_evaluation_id, ordinal)
        )""",
    )
    for statement in statements:
        op.execute(statement)


def _replace_freeze_boundaries() -> None:
    old_tables = (
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "retrieval_evaluation_matches",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
    )
    for table in old_tables:
        op.execute(f"DROP TRIGGER IF EXISTS freeze_terminal_run ON {EVALUATION}.{table}")

    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.reject_baseline_suite_manifest_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'baseline suite manifests are immutable; create a new suite'
                USING ERRCODE = '55000';
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_baseline_suite_state_transition()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE linked_runs integer; completed_runs integer; query_count integer; method_count integer;
        BEGIN
            IF OLD.status IN ('ready','failed','cancelled') THEN
                RAISE EXCEPTION 'terminal baseline suite state is immutable' USING ERRCODE = '55000';
            END IF;
            IF NEW.status = 'ready' THEN
                SELECT cardinality(methods) INTO method_count FROM {RUNTIME}.baseline_suites
                    WHERE suite_id = NEW.suite_id;
                SELECT count(*), count(*) FILTER (WHERE states.status = 'completed')
                    INTO linked_runs, completed_runs
                    FROM {RUNTIME}.baseline_suite_runs links
                    JOIN {RUNTIME}.run_states states USING (run_id)
                    WHERE links.suite_id = NEW.suite_id;
                SELECT count(*) INTO query_count FROM {RUNTIME}.baseline_suite_queries
                    WHERE suite_id = NEW.suite_id;
                IF linked_runs != method_count OR completed_runs != linked_runs OR
                   NEW.completed_run_count != linked_runs OR query_count != NEW.selected_count OR
                   NEW.completed_at IS NULL THEN
                    RAISE EXCEPTION 'baseline suite is incomplete' USING ERRCODE = '55000';
                END IF;
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_baseline_suite_writable()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE payload jsonb; suite varchar(128); current_status varchar(16);
        BEGIN
            payload := CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) ELSE to_jsonb(NEW) END;
            suite := payload ->> 'suite_id';
            SELECT status INTO current_status FROM {RUNTIME}.baseline_suite_states
                WHERE suite_id = suite FOR SHARE;
            IF current_status IS NULL THEN
                RAISE EXCEPTION 'baseline suite is not registered' USING ERRCODE = '55000';
            END IF;
            IF TG_TABLE_NAME = 'baseline_suite_queries' AND current_status != 'created' THEN
                RAISE EXCEPTION 'baseline suite denominator is frozen' USING ERRCODE = '55000';
            END IF;
            IF TG_TABLE_NAME = 'baseline_suite_runs' AND
               current_status NOT IN ('created','running') THEN
                RAISE EXCEPTION 'baseline suite run links are frozen' USING ERRCODE = '55000';
            END IF;
            RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER immutable_baseline_suite BEFORE UPDATE OR DELETE ON {RUNTIME}.baseline_suites "
        f"FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.reject_baseline_suite_manifest_mutation()"
    )
    op.execute(
        f"CREATE TRIGGER validate_baseline_suite_state BEFORE UPDATE OR DELETE ON {RUNTIME}.baseline_suite_states "
        f"FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_baseline_suite_state_transition()"
    )
    for table in ("baseline_suite_runs", "baseline_suite_queries"):
        op.execute(
            f"CREATE TRIGGER freeze_baseline_suite_child BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_baseline_suite_writable()"
        )

    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.reject_evaluation_manifest_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'evaluation manifests are immutable; create a new evaluation run'
                USING ERRCODE = '55000';
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.assign_legacy_evaluation_run()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME}, {EVALUATION} AS $$
        DECLARE legacy_id varchar(128); selected integer; source_method varchar(128); source_snapshot varchar(128);
        BEGIN
            IF NEW.evaluation_run_id IS NOT NULL THEN RETURN NEW; END IF;
            legacy_id := 'legacy-eval-' || substr(md5(NEW.run_id), 1, 24);
            SELECT GREATEST(count(*), 1) INTO selected FROM {RUNTIME}.run_queries WHERE run_id = NEW.run_id;
            SELECT method INTO source_method FROM {RUNTIME}.run_manifests WHERE run_id = NEW.run_id;
            SELECT snapshot_id INTO source_snapshot FROM {RUNTIME}.run_snapshot_links
                WHERE run_id = NEW.run_id ORDER BY snapshot_id LIMIT 1;
            INSERT INTO {EVALUATION}.evaluation_runs (
                evaluation_run_id, legacy_source_run_id, protocol_version, normalizer_version,
                evidence_policy_version, bad_case_rule_version, judge_model_alias,
                judge_prompt_hash, judge_rubric_version, judge_repeat_count,
                judge_score_values, secondary_judge_enabled, metric_versions,
                config_hash, created_at, metadata
            ) VALUES (
                legacy_id, NEW.run_id, 'legacy-v0.4', 'legacy-v0.4', 'legacy-v0.4',
                'legacy-v0.4', 'legacy', repeat('0',64), 'legacy-v0.4', 1,
                ARRAY[0.0,0.5,1.0]::double precision[], false, '{{}}'::jsonb,
                md5('legacy:' || NEW.run_id), now(), jsonb_build_object('legacy', true)
            ) ON CONFLICT (evaluation_run_id) DO NOTHING;
            INSERT INTO {EVALUATION}.evaluation_run_states (
                evaluation_run_id, status, selected_count, target_count,
                succeeded_count, failed_count, started_at, metadata
            ) VALUES (
                legacy_id, 'running', selected, 1, 0, 0, now(),
                jsonb_build_object('legacy', true)
            )
                ON CONFLICT (evaluation_run_id) DO NOTHING;
            INSERT INTO {EVALUATION}.evaluation_targets (
                evaluation_run_id, run_id, method, snapshot_id, selected_count, created_at, metadata
            ) VALUES (
                legacy_id, NEW.run_id, source_method, source_snapshot, selected, now(),
                jsonb_build_object('legacy', true)
            )
                ON CONFLICT (evaluation_run_id, run_id) DO NOTHING;
            NEW.evaluation_run_id := legacy_id;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.enforce_evaluation_run_writable()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {EVALUATION} AS $$
        DECLARE payload jsonb; target varchar(128); current_status varchar(16);
        BEGIN
            payload := CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) ELSE to_jsonb(NEW) END;
            IF TG_TABLE_NAME = 'retrieval_evaluation_matches' THEN
                SELECT evaluation_run_id INTO target FROM {EVALUATION}.retrieval_evaluations
                    WHERE retrieval_evaluation_id = payload ->> 'retrieval_evaluation_id';
            ELSIF TG_TABLE_NAME = 'evaluation_job_attempts' THEN
                SELECT evaluation_run_id INTO target FROM {EVALUATION}.evaluation_jobs
                    WHERE evaluation_job_id = payload ->> 'evaluation_job_id';
            ELSIF TG_TABLE_NAME = 'gold_evidence_group_members' THEN
                SELECT evaluation_run_id INTO target FROM {EVALUATION}.gold_evidence_groups
                    WHERE gold_group_id = payload ->> 'gold_group_id';
            ELSIF TG_TABLE_NAME = 'context_evaluation_matches' THEN
                SELECT evaluation_run_id INTO target FROM {EVALUATION}.context_evaluations
                    WHERE context_evaluation_id = payload ->> 'context_evaluation_id';
            ELSE
                target := payload ->> 'evaluation_run_id';
            END IF;
            SELECT status INTO current_status FROM {EVALUATION}.evaluation_run_states
                WHERE evaluation_run_id = target FOR SHARE;
            IF current_status IS NULL THEN
                RAISE EXCEPTION 'evaluation run is not registered' USING ERRCODE = '55000';
            END IF;
            IF current_status IN ('completed','failed','cancelled') THEN
                RAISE EXCEPTION 'terminal evaluation run is immutable' USING ERRCODE = '55000';
            END IF;
            RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.enforce_evaluation_state_transition()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {EVALUATION} AS $$
        DECLARE unfinished integer;
        BEGIN
            IF OLD.status IN ('completed','failed','cancelled') THEN
                RAISE EXCEPTION 'terminal evaluation run state is immutable' USING ERRCODE = '55000';
            END IF;
            IF NEW.status = 'completed' THEN
                SELECT count(*) INTO unfinished FROM {EVALUATION}.evaluation_jobs
                    WHERE evaluation_run_id = NEW.evaluation_run_id AND status != 'succeeded';
                IF unfinished != 0 OR NEW.failed_count != 0 OR
                   NEW.succeeded_count != NEW.selected_count * NEW.target_count OR
                   NEW.completed_at IS NULL THEN
                    RAISE EXCEPTION 'evaluation run is incomplete' USING ERRCODE = '55000';
                END IF;
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER immutable_evaluation_run BEFORE UPDATE OR DELETE ON {EVALUATION}.evaluation_runs "
        f"FOR EACH ROW EXECUTE FUNCTION {EVALUATION}.reject_evaluation_manifest_mutation()"
    )
    op.execute(
        f"CREATE TRIGGER validate_evaluation_state BEFORE UPDATE OR DELETE ON {EVALUATION}.evaluation_run_states "
        f"FOR EACH ROW EXECUTE FUNCTION {EVALUATION}.enforce_evaluation_state_transition()"
    )

    direct_tables = (
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
    )
    for table in direct_tables:
        op.execute(
            f"CREATE TRIGGER a_assign_legacy_evaluation_run BEFORE INSERT ON {EVALUATION}.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION {EVALUATION}.assign_legacy_evaluation_run()"
        )
    freeze_tables = (
        "evaluation_targets",
        "evaluation_jobs",
        "evaluation_job_attempts",
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "retrieval_evaluation_matches",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
        "answer_normalizations",
        "gold_evidence_groups",
        "gold_evidence_group_members",
        "context_evaluations",
        "context_evaluation_matches",
    )
    for table in freeze_tables:
        op.execute(
            f"CREATE TRIGGER z_freeze_evaluation_run BEFORE INSERT OR UPDATE OR DELETE ON {EVALUATION}.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION {EVALUATION}.enforce_evaluation_run_writable()"
        )


def _create_cost_views() -> None:
    op.execute(
        f"""CREATE VIEW {EVALUATION}.baseline_production_cost_summary AS
        SELECT * FROM {RUNTIME}.run_cost_summary"""
    )
    op.execute(
        f"""CREATE VIEW {EVALUATION}.evaluation_cost_summary AS
        SELECT evaluation_run_id,
            COALESCE(sum(input_tokens), 0)::bigint AS input_tokens,
            COALESCE(sum(output_tokens), 0)::bigint AS output_tokens,
            COALESCE(sum(total_tokens), 0)::bigint AS total_tokens,
            COALESCE(sum(cost_usd), 0)::double precision AS cost_usd,
            COALESCE(sum(latency_ms), 0)::double precision AS latency_ms,
            count(*)::bigint AS model_call_count
        FROM {EVALUATION}.model_calls GROUP BY evaluation_run_id"""
    )
    op.execute(
        f"""CREATE VIEW {EVALUATION}.combined_research_cost_summary AS
        SELECT targets.evaluation_run_id,
            COALESCE(sum(production.input_tokens), 0)::bigint AS production_input_tokens,
            COALESCE(sum(production.output_tokens), 0)::bigint AS production_output_tokens,
            COALESCE(sum(production.cost_usd), 0)::double precision AS production_cost_usd,
            COALESCE(evaluation.input_tokens, 0)::bigint AS evaluation_input_tokens,
            COALESCE(evaluation.output_tokens, 0)::bigint AS evaluation_output_tokens,
            COALESCE(evaluation.cost_usd, 0)::double precision AS evaluation_cost_usd
        FROM {EVALUATION}.evaluation_targets targets
        LEFT JOIN {RUNTIME}.run_cost_summary production USING (run_id)
        LEFT JOIN {EVALUATION}.evaluation_cost_summary evaluation USING (evaluation_run_id)
        GROUP BY targets.evaluation_run_id, evaluation.input_tokens,
            evaluation.output_tokens, evaluation.cost_usd"""
    )


def _grant_permissions() -> None:
    runtime_tables = (
        "baseline_suites",
        "baseline_suite_states",
        "baseline_suite_runs",
        "baseline_suite_queries",
    )
    op.execute(
        f"GRANT SELECT, INSERT ON {RUNTIME}.baseline_suites, {RUNTIME}.baseline_suite_runs, "
        f"{RUNTIME}.baseline_suite_queries TO pmmc_runtime"
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE ON {RUNTIME}.baseline_suite_states TO pmmc_runtime"
    )
    op.execute(
        "GRANT SELECT ON "
        + ", ".join(f"{RUNTIME}.{table}" for table in runtime_tables)
        + " TO pmmc_evaluator"
    )
    eval_tables = (
        "evaluation_runs",
        "evaluation_run_states",
        "evaluation_targets",
        "evaluation_jobs",
        "evaluation_job_attempts",
        "answer_normalizations",
        "gold_evidence_groups",
        "gold_evidence_group_members",
        "context_evaluations",
        "context_evaluation_matches",
    )
    op.execute(
        "GRANT SELECT, INSERT, UPDATE ON "
        + ", ".join(f"{EVALUATION}.{table}" for table in eval_tables)
        + " TO pmmc_evaluator"
    )
    op.execute(
        f"GRANT SELECT ON {EVALUATION}.baseline_production_cost_summary, "
        f"{EVALUATION}.evaluation_cost_summary, {EVALUATION}.combined_research_cost_summary "
        "TO pmmc_evaluator"
    )
    op.execute(f"REVOKE ALL ON SCHEMA {EVALUATION} FROM pmmc_runtime")
    for function in (
        f"{RUNTIME}.reject_baseline_suite_manifest_mutation()",
        f"{RUNTIME}.enforce_baseline_suite_state_transition()",
        f"{RUNTIME}.enforce_baseline_suite_writable()",
        f"{EVALUATION}.reject_evaluation_manifest_mutation()",
        f"{EVALUATION}.assign_legacy_evaluation_run()",
        f"{EVALUATION}.enforce_evaluation_run_writable()",
        f"{EVALUATION}.enforce_evaluation_state_transition()",
    ):
        op.execute(f"REVOKE ALL ON FUNCTION {function} FROM PUBLIC")


def downgrade() -> None:
    op.execute(
        f"""DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM {EVALUATION}.metric_scores WHERE value IS NULL) THEN
                RAISE EXCEPTION 'cannot downgrade 0006 while not-applicable metrics exist';
            END IF;
        END $$"""
    )
    for view in (
        "combined_research_cost_summary",
        "evaluation_cost_summary",
        "baseline_production_cost_summary",
    ):
        op.execute(f"DROP VIEW IF EXISTS {EVALUATION}.{view}")
    for table in (
        "evaluation_targets",
        "evaluation_jobs",
        "evaluation_job_attempts",
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "retrieval_evaluation_matches",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
        "answer_normalizations",
        "gold_evidence_groups",
        "gold_evidence_group_members",
        "context_evaluations",
        "context_evaluation_matches",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS z_freeze_evaluation_run ON {EVALUATION}.{table}")
    for table in (
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS a_assign_legacy_evaluation_run ON {EVALUATION}.{table}")
    op.execute(f"DROP TRIGGER IF EXISTS immutable_evaluation_run ON {EVALUATION}.evaluation_runs")
    op.execute(f"DROP TRIGGER IF EXISTS validate_evaluation_state ON {EVALUATION}.evaluation_run_states")
    op.execute(f"DROP FUNCTION IF EXISTS {EVALUATION}.enforce_evaluation_state_transition()")
    op.execute(f"DROP FUNCTION IF EXISTS {EVALUATION}.enforce_evaluation_run_writable()")
    op.execute(f"DROP FUNCTION IF EXISTS {EVALUATION}.assign_legacy_evaluation_run()")
    op.execute(f"DROP FUNCTION IF EXISTS {EVALUATION}.reject_evaluation_manifest_mutation()")

    for table in (
        "context_evaluation_matches",
        "context_evaluations",
        "gold_evidence_group_members",
        "gold_evidence_groups",
        "answer_normalizations",
        "evaluation_job_attempts",
        "evaluation_jobs",
    ):
        op.execute(f"DROP TABLE IF EXISTS {EVALUATION}.{table} CASCADE")

    op.drop_constraint(
        "uq_bad_case_labels_evaluation_answer_rule",
        "bad_case_labels",
        schema=EVALUATION,
        type_="unique",
    )
    op.drop_constraint(
        "uq_judge_scores_evaluation_answer_judge",
        "judge_scores",
        schema=EVALUATION,
        type_="unique",
    )
    op.drop_constraint(
        "judge_scores_verdict_values", "judge_scores", schema=EVALUATION, type_="check"
    )
    op.drop_constraint(
        "uq_metric_scores_evaluation_answer_metric",
        "metric_scores",
        schema=EVALUATION,
        type_="unique",
    )
    for name in (
        "metric_scores_applicability",
        "metric_scores_k_positive",
        "metric_scores_family_values",
        "metric_scores_value_probability",
    ):
        op.drop_constraint(name, "metric_scores", schema=EVALUATION, type_="check")
    op.drop_constraint(
        "uq_retrieval_evaluations_run_trace",
        "retrieval_evaluations",
        schema=EVALUATION,
        type_="unique",
    )
    _replace_bad_case_check_constraints(
        LEGACY_BAD_CASE_VALUES,
        use_v2_names=False,
    )

    op.alter_column("metric_scores", "value", schema=EVALUATION, nullable=False)
    op.execute(
        f"ALTER TABLE {EVALUATION}.model_calls DROP CONSTRAINT IF EXISTS "
        "fk_evaluation_model_calls_evaluation_job"
    )
    op.drop_column("model_calls", "evaluation_job_id", schema=EVALUATION)
    op.create_foreign_key(
        "model_calls_job_id_run_id_fkey",
        "model_calls",
        "experiment_jobs",
        ["job_id", "run_id"],
        ["job_id", "run_id"],
        source_schema=EVALUATION,
        referent_schema=RUNTIME,
    )
    for table, columns in (
        ("metric_scores", ("unavailable_reason", "applicable", "k", "metric_family")),
        ("judge_scores", ("verdict",)),
        ("bad_case_labels", ("is_primary", "rule_version")),
    ):
        for column in columns:
            op.drop_column(table, column, schema=EVALUATION)

    for table in (
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
    ):
        op.drop_constraint(
            f"fk_{table}_evaluation_run", table, schema=EVALUATION, type_="foreignkey"
        )
        op.drop_column(table, "evaluation_run_id", schema=EVALUATION)

    op.create_unique_constraint(
        "retrieval_evaluations_retrieval_trace_id_key",
        "retrieval_evaluations",
        ["retrieval_trace_id"],
        schema=EVALUATION,
    )
    op.create_unique_constraint(
        LEGACY_JUDGE_UNIQUE_CONSTRAINT,
        "judge_scores",
        ["answer_id", "judge_model_alias", "rubric_version", "repeat_index"],
        schema=EVALUATION,
    )

    op.execute(f"DROP TABLE IF EXISTS {EVALUATION}.evaluation_targets CASCADE")
    op.execute(f"DROP TABLE IF EXISTS {EVALUATION}.evaluation_run_states CASCADE")
    op.execute(f"DROP TABLE IF EXISTS {EVALUATION}.evaluation_runs CASCADE")

    for table in ("baseline_suite_runs", "baseline_suite_queries"):
        op.execute(f"DROP TRIGGER IF EXISTS freeze_baseline_suite_child ON {RUNTIME}.{table}")
    op.execute(f"DROP TRIGGER IF EXISTS validate_baseline_suite_state ON {RUNTIME}.baseline_suite_states")
    op.execute(f"DROP TRIGGER IF EXISTS immutable_baseline_suite ON {RUNTIME}.baseline_suites")
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_baseline_suite_writable()")
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_baseline_suite_state_transition()")
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.reject_baseline_suite_manifest_mutation()")
    for table in (
        "baseline_suite_queries",
        "baseline_suite_runs",
        "baseline_suite_states",
        "baseline_suites",
    ):
        op.execute(f"DROP TABLE IF EXISTS {RUNTIME}.{table} CASCADE")

    for table in (
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "retrieval_evaluation_matches",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
    ):
        op.execute(
            f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {EVALUATION}.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()"
        )
