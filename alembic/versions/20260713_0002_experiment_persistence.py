# ruff: noqa: E501
"""Add experiment facts, vectors, artifacts, and evaluation persistence.

Revision ID: 20260713_0002
Revises: 20260713_0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260713_0002"
down_revision: str | None = "20260713_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"


def upgrade() -> None:
    op.add_column(
        "run_manifests",
        sa.Column(
            "experiment_schema_version",
            sa.String(32),
            nullable=False,
            server_default="0.1.0",
        ),
        schema=RUNTIME,
    )
    op.add_column("run_manifests", sa.Column("selection_hash", sa.String(128)), schema=RUNTIME)
    op.add_column("run_manifests", sa.Column("seed", sa.Integer()), schema=RUNTIME)
    op.add_column("run_manifests", sa.Column("retrieval_topk_max", sa.Integer()), schema=RUNTIME)
    op.add_column("run_manifests", sa.Column("config_hash", sa.String(128)), schema=RUNTIME)
    op.alter_column(
        "run_manifests",
        "experiment_schema_version",
        schema=RUNTIME,
        server_default=None,
    )

    for statement in _table_statements():
        op.execute(statement)
    for statement in _index_and_view_statements():
        op.execute(statement)

    op.execute(
        f"""
        INSERT INTO {RUNTIME}.run_states
            (run_id, status, selected_count, succeeded_count, failed_count, metadata)
        SELECT run_id, 'created', 0, 0, 0, '{{}}'::jsonb
        FROM {RUNTIME}.run_manifests
        ON CONFLICT (run_id) DO NOTHING
        """
    )
    for statement in _grant_statements():
        op.execute(statement)


def downgrade() -> None:
    op.execute(f"DROP TRIGGER IF EXISTS freeze_terminal_run ON {RUNTIME}.run_snapshot_links")
    op.execute(f"DROP VIEW IF EXISTS {EVALUATION}.run_cost_summary")
    op.execute(f"DROP VIEW IF EXISTS {EVALUATION}.query_cost_summary")
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.run_cost_summary")
    op.execute(f"DROP VIEW IF EXISTS {RUNTIME}.query_cost_summary")
    for schema, table in reversed(_created_tables()):
        op.execute(f"DROP TABLE IF EXISTS {schema}.{table} CASCADE")
    op.execute(f"DROP TRIGGER IF EXISTS immutable_run_manifest ON {RUNTIME}.run_manifests")
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.reject_run_manifest_mutation()")
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_run_state_transition()")
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_run_writable()")
    for column in (
        "config_hash",
        "retrieval_topk_max",
        "seed",
        "selection_hash",
        "experiment_schema_version",
    ):
        op.drop_column("run_manifests", column, schema=RUNTIME)


def _created_tables() -> list[tuple[str, str]]:
    return [
        (RUNTIME, "run_states"),
        (RUNTIME, "run_queries"),
        (RUNTIME, "run_stage_summaries"),
        (RUNTIME, "experiment_jobs"),
        (RUNTIME, "job_attempts"),
        (RUNTIME, "run_artifacts"),
        (EVALUATION, "run_artifacts"),
        (RUNTIME, "model_calls"),
        (EVALUATION, "model_calls"),
        (RUNTIME, "embedding_collections"),
        (RUNTIME, "embeddings"),
        (RUNTIME, "retrieval_batches"),
        (RUNTIME, "retrieval_traces"),
        (RUNTIME, "final_answers"),
        (RUNTIME, "final_answer_evidence_ids"),
        (RUNTIME, "final_answer_image_ids"),
        (EVALUATION, "retrieval_evaluations"),
        (EVALUATION, "retrieval_evaluation_matches"),
        (EVALUATION, "metric_scores"),
        (EVALUATION, "judge_scores"),
        (EVALUATION, "bad_case_labels"),
    ]


def _table_statements() -> tuple[str, ...]:
    return (
        f"""CREATE TABLE {RUNTIME}.run_states (
            run_id varchar(128) PRIMARY KEY REFERENCES {RUNTIME}.run_manifests(run_id),
            status varchar(16) NOT NULL CHECK (status IN ('created','running','completed','failed','cancelled')),
            started_at timestamptz, heartbeat_at timestamptz, completed_at timestamptz,
            error_type varchar(256), error_summary text, error_artifact_id varchar(128),
            selected_count integer NOT NULL CHECK (selected_count >= 0),
            succeeded_count integer NOT NULL CHECK (succeeded_count >= 0),
            failed_count integer NOT NULL CHECK (failed_count >= 0),
            metadata jsonb NOT NULL,
            CHECK (succeeded_count + failed_count <= selected_count)
        )""",
        f"""CREATE TABLE {RUNTIME}.run_queries (
            run_id varchar(128) NOT NULL, snapshot_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL, ordinal integer NOT NULL CHECK (ordinal >= 0),
            selection_hash varchar(128) NOT NULL,
            shard_count integer NOT NULL CHECK (shard_count > 0),
            shard_index integer NOT NULL CHECK (shard_index >= 0 AND shard_index < shard_count),
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            PRIMARY KEY (run_id, snapshot_id, query_id), UNIQUE (run_id, ordinal),
            FOREIGN KEY (run_id, snapshot_id) REFERENCES {RUNTIME}.run_snapshot_links(run_id, snapshot_id),
            FOREIGN KEY (snapshot_id, query_id) REFERENCES {RUNTIME}.benchmark_queries(snapshot_id, query_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.run_stage_summaries (
            run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.run_manifests(run_id),
            stage varchar(64) NOT NULL, success boolean NOT NULL,
            started_at timestamptz NOT NULL, completed_at timestamptz NOT NULL,
            elapsed_ms double precision NOT NULL CHECK (elapsed_ms >= 0),
            processed integer NOT NULL, total integer, succeeded integer NOT NULL, failed integer NOT NULL,
            metadata jsonb NOT NULL, PRIMARY KEY (run_id, stage),
            CHECK (processed >= 0 AND succeeded >= 0 AND failed >= 0)
        )""",
        f"""CREATE TABLE {RUNTIME}.experiment_jobs (
            job_id varchar(128) PRIMARY KEY, run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, query_id varchar(256),
            stage varchar(64) NOT NULL CHECK (stage IN
                ('compile','questioner','planner','doubter','match','retrieve','execute','answer','judge','analyze','baseline','embed')),
            variant varchar(128) NOT NULL,
            status varchar(16) NOT NULL CHECK (status IN ('pending','running','succeeded','failed','cancelled')),
            priority integer NOT NULL CHECK (priority >= 0),
            attempt_count integer NOT NULL, max_attempts integer NOT NULL,
            claimed_by varchar(256), lease_expires_at timestamptz,
            available_at timestamptz NOT NULL, created_at timestamptz NOT NULL,
            started_at timestamptz, finished_at timestamptz,
            error_type varchar(256), error_summary text, error_artifact_id varchar(128), metadata jsonb NOT NULL,
            UNIQUE (job_id, run_id),
            UNIQUE (run_id, snapshot_id, query_id, stage, variant),
            CHECK (attempt_count >= 0 AND max_attempts > 0),
            CHECK (status != 'running' OR (claimed_by IS NOT NULL AND lease_expires_at IS NOT NULL)),
            CHECK (status NOT IN ('succeeded','failed','cancelled') OR finished_at IS NOT NULL),
            CHECK (status != 'failed' OR error_type IS NOT NULL),
            FOREIGN KEY (run_id, snapshot_id) REFERENCES {RUNTIME}.run_snapshot_links(run_id, snapshot_id),
            FOREIGN KEY (run_id, snapshot_id, query_id) REFERENCES {RUNTIME}.run_queries(run_id, snapshot_id, query_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.job_attempts (
            attempt_id varchar(128) PRIMARY KEY,
            job_id varchar(128) NOT NULL REFERENCES {RUNTIME}.experiment_jobs(job_id),
            attempt_number integer NOT NULL CHECK (attempt_number > 0), worker_id varchar(256) NOT NULL,
            status varchar(16) NOT NULL CHECK (status IN ('pending','running','succeeded','failed','cancelled')),
            started_at timestamptz NOT NULL, finished_at timestamptz,
            error_type varchar(256), error_summary text, error_artifact_id varchar(128), metadata jsonb NOT NULL,
            UNIQUE (job_id, attempt_number),
            CHECK (status NOT IN ('succeeded','failed','cancelled') OR finished_at IS NOT NULL),
            CHECK (status != 'failed' OR error_type IS NOT NULL)
        )""",
        _artifact_table(RUNTIME),
        _artifact_table(EVALUATION),
        _model_call_table(RUNTIME),
        _model_call_table(EVALUATION),
        f"""CREATE TABLE {RUNTIME}.embedding_collections (
            collection_id varchar(128) PRIMARY KEY,
            snapshot_id varchar(128) NOT NULL REFERENCES {RUNTIME}.dataset_snapshots(snapshot_id),
            run_id varchar(128) REFERENCES {RUNTIME}.run_manifests(run_id),
            source_type varchar(64) NOT NULL CHECK (source_type IN
                ('memory_text','memory_caption','memory_image','memory_image_text_pair','qbank_question','qbank_paraphrase','query')),
            provider varchar(128), model_alias varchar(256) NOT NULL,
            actual_model_id varchar(512), model_revision varchar(256),
            dimension integer NOT NULL CHECK (dimension > 0),
            distance_metric varchar(32) NOT NULL CHECK (distance_metric IN ('cosine','inner_product','l2')),
            normalized boolean NOT NULL, normalization_method varchar(128),
            preprocessing_hash varchar(128) NOT NULL, index_type varchar(64), index_params jsonb NOT NULL,
            status varchar(16) NOT NULL CHECK (status IN ('building','ready','failed')),
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (collection_id, dimension),
            UNIQUE (collection_id, snapshot_id),
            CHECK ((normalized AND normalization_method IS NOT NULL) OR
                (NOT normalized AND normalization_method IS NULL))
        )""",
        f"""CREATE TABLE {RUNTIME}.embeddings (
            embedding_id varchar(128) PRIMARY KEY, collection_id varchar(128) NOT NULL,
            source_id varchar(256) NOT NULL, input_hash varchar(128) NOT NULL,
            dimension integer NOT NULL CHECK (dimension > 0), embedding vector NOT NULL,
            model_call_id varchar(128) REFERENCES {RUNTIME}.model_calls(model_call_id),
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            FOREIGN KEY (collection_id, dimension)
                REFERENCES {RUNTIME}.embedding_collections(collection_id, dimension),
            UNIQUE (collection_id, source_id), CHECK (vector_dims(embedding) = dimension)
        )""",
        f"""CREATE TABLE {RUNTIME}.retrieval_batches (
            retrieval_batch_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL, snapshot_id varchar(128) NOT NULL, query_id varchar(256) NOT NULL,
            stage varchar(64) NOT NULL CHECK (stage IN
                ('compile','questioner','planner','doubter','match','retrieve','execute','answer','judge','analyze','baseline','embed')),
            method varchar(128) NOT NULL, retriever_name varchar(256) NOT NULL,
            index_name varchar(256), chunking_strategy varchar(256),
            requested_top_k integer NOT NULL CHECK (requested_top_k > 0), returned_count integer NOT NULL,
            embedding_collection_id varchar(128),
            reranker_model_call_id varchar(128),
            plan_id varchar(256), step_id integer,
            embed_latency_ms double precision, search_latency_ms double precision,
            rerank_latency_ms double precision, total_latency_ms double precision,
            status varchar(16) NOT NULL CHECK (status IN ('success','failed','skipped','partial')),
            error_type varchar(256), created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            FOREIGN KEY (run_id, snapshot_id, query_id)
                REFERENCES {RUNTIME}.run_queries(run_id, snapshot_id, query_id),
            FOREIGN KEY (embedding_collection_id, snapshot_id)
                REFERENCES {RUNTIME}.embedding_collections(collection_id, snapshot_id),
            FOREIGN KEY (reranker_model_call_id, run_id, query_id)
                REFERENCES {RUNTIME}.model_calls(model_call_id, run_id, query_id),
            UNIQUE (retrieval_batch_id, run_id, query_id, method, retriever_name),
            CHECK (returned_count >= 0 AND returned_count <= requested_top_k),
            CHECK (step_id IS NULL OR step_id >= 0),
            CHECK (embed_latency_ms IS NULL OR embed_latency_ms >= 0),
            CHECK (search_latency_ms IS NULL OR search_latency_ms >= 0),
            CHECK (rerank_latency_ms IS NULL OR rerank_latency_ms >= 0),
            CHECK (total_latency_ms IS NULL OR total_latency_ms >= 0),
            CHECK (status != 'failed' OR error_type IS NOT NULL),
            CHECK (error_type IS NULL OR status != 'success')
        )""",
        f"""CREATE TABLE {RUNTIME}.retrieval_traces (
            retrieval_trace_id varchar(128) PRIMARY KEY,
            retrieval_batch_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL, run_id varchar(128) NOT NULL,
            method varchar(128) NOT NULL, retriever_name varchar(256) NOT NULL,
            index_name varchar(256), chunking_strategy varchar(256), rank integer NOT NULL CHECK (rank > 0),
            retrieved_id varchar(256) NOT NULL, retrieved_type varchar(64) NOT NULL CHECK
                (retrieved_type IN ('memory','text','caption','image','raw_image','image_text_pair','session','query','unknown')),
            score_type varchar(128), score_raw_value double precision, score_normalized_value double precision,
            score_direction varchar(32), normalization_method varchar(128), normalization_version varchar(128),
            score_metadata jsonb NOT NULL, created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (retrieval_batch_id, rank),
            UNIQUE (retrieval_trace_id, run_id, query_id),
            FOREIGN KEY (retrieval_batch_id, run_id, query_id, method, retriever_name)
                REFERENCES {RUNTIME}.retrieval_batches
                    (retrieval_batch_id, run_id, query_id, method, retriever_name),
            CHECK (score_direction IS NULL OR score_direction IN
                ('higher_is_better','lower_is_better')),
            CHECK ((score_type IS NULL AND score_raw_value IS NULL AND
                score_normalized_value IS NULL AND score_direction IS NULL AND
                normalization_method IS NULL AND normalization_version IS NULL) OR
                (score_type IS NOT NULL AND score_raw_value IS NOT NULL AND
                 score_direction IS NOT NULL)),
            CHECK ((score_normalized_value IS NULL AND normalization_method IS NULL) OR
                (score_normalized_value IS NOT NULL AND normalization_method IS NOT NULL)),
            CHECK (normalization_version IS NULL OR normalization_method IS NOT NULL),
            CHECK (score_normalized_value IS NULL OR
                (score_normalized_value >= 0 AND score_normalized_value <= 1))
        )""",
        f"""CREATE TABLE {RUNTIME}.final_answers (
            answer_id varchar(128) PRIMARY KEY, snapshot_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL, run_id varchar(128) NOT NULL,
            answer_variant varchar(128) NOT NULL, method varchar(128) NOT NULL,
            answer_model_alias varchar(256) NOT NULL, actual_model_id varchar(512),
            final_answer text NOT NULL, normalized_answer text,
            input_tokens integer, output_tokens integer, latency_ms double precision,
            model_call_id varchar(128),
            langfuse_trace_id varchar(256), created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            FOREIGN KEY (run_id, snapshot_id, query_id)
                REFERENCES {RUNTIME}.run_queries(run_id, snapshot_id, query_id),
            FOREIGN KEY (model_call_id, run_id, query_id)
                REFERENCES {RUNTIME}.model_calls(model_call_id, run_id, query_id),
            UNIQUE (run_id, query_id, answer_variant),
            UNIQUE (answer_id, run_id, query_id),
            CHECK (input_tokens IS NULL OR input_tokens >= 0),
            CHECK (output_tokens IS NULL OR output_tokens >= 0),
            CHECK (latency_ms IS NULL OR latency_ms >= 0)
        )""",
        f"""CREATE TABLE {RUNTIME}.final_answer_evidence_ids (
            answer_id varchar(128) NOT NULL REFERENCES {RUNTIME}.final_answers(answer_id),
            evidence_id varchar(256) NOT NULL, ordinal integer NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (answer_id, evidence_id), UNIQUE (answer_id, ordinal)
        )""",
        f"""CREATE TABLE {RUNTIME}.final_answer_image_ids (
            answer_id varchar(128) NOT NULL REFERENCES {RUNTIME}.final_answers(answer_id),
            image_id varchar(256) NOT NULL, ordinal integer NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (answer_id, image_id), UNIQUE (answer_id, ordinal)
        )""",
        f"""CREATE TABLE {EVALUATION}.retrieval_evaluations (
            retrieval_evaluation_id varchar(128) PRIMARY KEY,
            retrieval_trace_id varchar(128) NOT NULL UNIQUE,
            query_id varchar(256) NOT NULL, run_id varchar(128) NOT NULL,
            is_gold_evidence boolean NOT NULL, created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            FOREIGN KEY (retrieval_trace_id, run_id, query_id)
                REFERENCES {RUNTIME}.retrieval_traces(retrieval_trace_id, run_id, query_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.retrieval_evaluation_matches (
            retrieval_evaluation_id varchar(128) NOT NULL
                REFERENCES {EVALUATION}.retrieval_evaluations(retrieval_evaluation_id),
            gold_evidence_id varchar(256) NOT NULL, ordinal integer NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (retrieval_evaluation_id, gold_evidence_id),
            UNIQUE (retrieval_evaluation_id, ordinal)
        )""",
        f"""CREATE TABLE {EVALUATION}.metric_scores (
            metric_score_id varchar(128) PRIMARY KEY,
            answer_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL, run_id varchar(128) NOT NULL,
            metric_name varchar(128) NOT NULL, metric_version varchar(128) NOT NULL,
            scorer varchar(256), value double precision NOT NULL CHECK (value >= 0 AND value <= 1),
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            FOREIGN KEY (answer_id, run_id, query_id)
                REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.judge_scores (
            judge_score_id varchar(128) PRIMARY KEY,
            answer_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL, run_id varchar(128) NOT NULL,
            judge_model_alias varchar(256) NOT NULL, judge_actual_model_id varchar(512),
            judge_prompt_hash varchar(128),
            model_call_id varchar(128),
            rubric_version varchar(128) NOT NULL, repeat_index integer NOT NULL CHECK (repeat_index >= 0),
            f1 double precision, bleu1 double precision, exact_match boolean,
            llm_judge_score double precision, judge_reason text,
            bad_case_category varchar(128) NOT NULL CHECK (bad_case_category IN
                ('questioner_coverage_failure','matching_failure','planner_failure','doubter_failure','evidence_insufficiency','answer_generation_failure','update_conflict_failure','none','unknown')),
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (answer_id, judge_model_alias, rubric_version, repeat_index),
            CHECK (f1 IS NULL OR (f1 >= 0 AND f1 <= 1)),
            CHECK (bleu1 IS NULL OR (bleu1 >= 0 AND bleu1 <= 1)),
            CHECK (llm_judge_score IS NULL OR (llm_judge_score >= 0 AND llm_judge_score <= 1)),
            FOREIGN KEY (answer_id, run_id, query_id)
                REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id),
            FOREIGN KEY (model_call_id, run_id, query_id)
                REFERENCES {EVALUATION}.model_calls(model_call_id, run_id, query_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.bad_case_labels (
            bad_case_label_id varchar(128) PRIMARY KEY,
            answer_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL, run_id varchar(128) NOT NULL,
            category varchar(128) NOT NULL CHECK (category IN
                ('questioner_coverage_failure','matching_failure','planner_failure','doubter_failure','evidence_insufficiency','answer_generation_failure','update_conflict_failure','none','unknown')),
            label_source varchar(128) NOT NULL,
            annotator_id varchar(256), confidence double precision,
            rationale_asset_id varchar(128),
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
            FOREIGN KEY (answer_id, run_id, query_id)
                REFERENCES {RUNTIME}.final_answers(answer_id, run_id, query_id),
            FOREIGN KEY (rationale_asset_id, run_id)
                REFERENCES {EVALUATION}.run_artifacts(artifact_id, run_id)
        )""",
        f"""ALTER TABLE {RUNTIME}.run_states
            ADD CONSTRAINT fk_run_states_error_artifact
            FOREIGN KEY (error_artifact_id, run_id)
            REFERENCES {RUNTIME}.run_artifacts(artifact_id, run_id)""",
        f"""ALTER TABLE {RUNTIME}.experiment_jobs
            ADD CONSTRAINT fk_experiment_jobs_error_artifact
            FOREIGN KEY (error_artifact_id, run_id)
            REFERENCES {RUNTIME}.run_artifacts(artifact_id, run_id)""",
        f"""ALTER TABLE {RUNTIME}.job_attempts
            ADD CONSTRAINT fk_job_attempts_error_artifact
            FOREIGN KEY (error_artifact_id)
            REFERENCES {RUNTIME}.run_artifacts(artifact_id)""",
    )


def _artifact_table(schema: str) -> str:
    return f"""CREATE TABLE {schema}.run_artifacts (
        artifact_id varchar(128) PRIMARY KEY,
        run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.run_manifests(run_id),
        snapshot_id varchar(128), query_id varchar(256),
        artifact_type varchar(64) NOT NULL CHECK (artifact_type IN
            ('raw_image','resized_image','prompt','model_response','trace_export','embedding_dump','dataset_snapshot','other')),
        state varchar(16) NOT NULL CHECK (state IN ('pending','ready','failed')),
        object_key text NOT NULL, storage_uri text, sha256 varchar(64) NOT NULL,
        size_bytes bigint CHECK (size_bytes IS NULL OR size_bytes >= 0), mime_type varchar(256),
        created_at timestamptz NOT NULL, ready_at timestamptz, error_type varchar(256), metadata jsonb NOT NULL,
        UNIQUE (artifact_id, run_id),
        CHECK (sha256 ~ '^[0-9a-f]{{64}}$'),
        CHECK (state != 'ready' OR
            (storage_uri IS NOT NULL AND size_bytes IS NOT NULL AND ready_at IS NOT NULL)),
        CHECK ((state = 'failed' AND error_type IS NOT NULL) OR
            (state != 'failed' AND error_type IS NULL)),
        CHECK (query_id IS NULL OR snapshot_id IS NOT NULL),
        FOREIGN KEY (run_id, snapshot_id)
            REFERENCES {RUNTIME}.run_snapshot_links(run_id, snapshot_id),
        FOREIGN KEY (run_id, snapshot_id, query_id)
            REFERENCES {RUNTIME}.run_queries(run_id, snapshot_id, query_id)
    )"""


def _model_call_table(schema: str) -> str:
    return f"""CREATE TABLE {schema}.model_calls (
        model_call_id varchar(128) PRIMARY KEY,
        run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.run_manifests(run_id),
        snapshot_id varchar(128), query_id varchar(256), conversation_id varchar(256),
        job_id varchar(128),
        qbank_id varchar(256), plan_id varchar(256), step_id integer CHECK (step_id IS NULL OR step_id >= 0),
        parent_model_call_id varchar(128), stage varchar(64) NOT NULL CHECK (stage IN
            ('compile','questioner','planner','doubter','match','retrieve','execute','answer','judge','analyze','baseline','embed')),
        provider varchar(128),
        model_alias varchar(256) NOT NULL, actual_model_id varchar(512), prompt_hash varchar(128),
        input_asset_id varchar(128),
        output_asset_id varchar(128),
        input_tokens integer CHECK (input_tokens IS NULL OR input_tokens >= 0),
        output_tokens integer CHECK (output_tokens IS NULL OR output_tokens >= 0),
        total_tokens integer CHECK (total_tokens IS NULL OR total_tokens >= 0),
        cost_usd double precision CHECK (cost_usd IS NULL OR cost_usd >= 0),
        latency_ms double precision CHECK (latency_ms IS NULL OR latency_ms >= 0),
        temperature double precision, top_p double precision, seed integer,
        parsed_output jsonb, parse_error text, repair_attempt_count integer NOT NULL,
        cache_hit boolean NOT NULL, retry_count integer NOT NULL, fallback_used boolean NOT NULL,
        litellm_call_id varchar(256), vllm_request_id varchar(256), langfuse_trace_id varchar(256),
        provider_request_id varchar(256), request_config jsonb NOT NULL, error_type varchar(256),
        status varchar(16) NOT NULL CHECK (status IN ('success','failed','skipped','partial')),
        created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
        UNIQUE (model_call_id, run_id),
        UNIQUE (model_call_id, run_id, query_id),
        FOREIGN KEY (run_id, snapshot_id)
            REFERENCES {RUNTIME}.run_snapshot_links(run_id, snapshot_id),
        FOREIGN KEY (run_id, snapshot_id, query_id)
            REFERENCES {RUNTIME}.run_queries(run_id, snapshot_id, query_id),
        FOREIGN KEY (job_id, run_id)
            REFERENCES {RUNTIME}.experiment_jobs(job_id, run_id),
        FOREIGN KEY (parent_model_call_id, run_id)
            REFERENCES {schema}.model_calls(model_call_id, run_id),
        FOREIGN KEY (input_asset_id, run_id)
            REFERENCES {schema}.run_artifacts(artifact_id, run_id),
        FOREIGN KEY (output_asset_id, run_id)
            REFERENCES {schema}.run_artifacts(artifact_id, run_id),
        CHECK (query_id IS NULL OR snapshot_id IS NOT NULL),
        CHECK (parent_model_call_id IS NULL OR parent_model_call_id != model_call_id),
        CHECK (temperature IS NULL OR temperature >= 0),
        CHECK (top_p IS NULL OR (top_p >= 0 AND top_p <= 1)),
        CHECK (seed IS NULL OR seed >= 0),
        CHECK (repair_attempt_count >= 0 AND retry_count >= 0),
        CHECK (input_tokens IS NULL OR output_tokens IS NULL OR total_tokens IS NULL OR
            total_tokens = input_tokens + output_tokens),
        CHECK (status != 'failed' OR error_type IS NOT NULL),
        CHECK (error_type IS NULL OR status != 'success')
    )"""


def _index_and_view_statements() -> tuple[str, ...]:
    return (
        _run_manifest_immutability_function(),
        f"CREATE TRIGGER immutable_run_manifest BEFORE UPDATE OR DELETE ON "
        f"{RUNTIME}.run_manifests FOR EACH ROW EXECUTE FUNCTION "
        f"{RUNTIME}.reject_run_manifest_mutation()",
        _run_freeze_function(),
        _run_state_transition_function(),
        f"CREATE TRIGGER enforce_run_state_transition BEFORE UPDATE OR DELETE ON "
        f"{RUNTIME}.run_states FOR EACH ROW EXECUTE FUNCTION "
        f"{RUNTIME}.enforce_run_state_transition()",
        *_run_freeze_triggers(),
        f"CREATE INDEX ix_jobs_claim ON {RUNTIME}.experiment_jobs(status, available_at, priority)",
        f"CREATE INDEX ix_model_calls_run_stage ON {RUNTIME}.model_calls(run_id, stage)",
        f"CREATE INDEX ix_retrieval_batches_run_query ON {RUNTIME}.retrieval_batches(run_id, query_id)",
        f"CREATE INDEX ix_retrieval_traces_batch_rank ON {RUNTIME}.retrieval_traces(retrieval_batch_id, rank)",
        f"CREATE INDEX ix_final_answers_run_query ON {RUNTIME}.final_answers(run_id, query_id)",
        f"""CREATE VIEW {RUNTIME}.query_cost_summary AS
            WITH calls AS (
                SELECT run_id, query_id,
                    COALESCE(sum(input_tokens), 0)::bigint AS input_tokens,
                    COALESCE(sum(output_tokens), 0)::bigint AS output_tokens,
                    COALESCE(sum(total_tokens), 0)::bigint AS total_tokens,
                    COALESCE(sum(cost_usd), 0)::double precision AS cost_usd,
                    COALESCE(sum(latency_ms), 0)::double precision AS model_call_latency_ms,
                    count(*)::bigint AS model_call_count,
                    count(*) FILTER (WHERE fallback_used)::bigint AS fallback_call_count
                FROM {RUNTIME}.model_calls WHERE query_id IS NOT NULL
                GROUP BY run_id, query_id
            ), retrieval AS (
                SELECT run_id, query_id,
                    COALESCE(sum(total_latency_ms), 0)::double precision AS retrieval_latency_ms,
                    count(*)::bigint AS retrieval_batch_count
                FROM {RUNTIME}.retrieval_batches GROUP BY run_id, query_id
            ), queries AS (
                SELECT DISTINCT run_id, query_id FROM {RUNTIME}.run_queries
            )
            SELECT queries.run_id, queries.query_id,
                COALESCE(calls.input_tokens, 0)::bigint AS input_tokens,
                COALESCE(calls.output_tokens, 0)::bigint AS output_tokens,
                COALESCE(calls.total_tokens, 0)::bigint AS total_tokens,
                COALESCE(calls.cost_usd, 0)::double precision AS cost_usd,
                COALESCE(calls.model_call_latency_ms, 0)::double precision AS model_call_latency_ms,
                COALESCE(retrieval.retrieval_latency_ms, 0)::double precision AS retrieval_latency_ms,
                (COALESCE(calls.model_call_latency_ms, 0) +
                 COALESCE(retrieval.retrieval_latency_ms, 0))::double precision AS observed_latency_ms,
                COALESCE(calls.model_call_count, 0)::bigint AS model_call_count,
                COALESCE(calls.fallback_call_count, 0)::bigint AS fallback_call_count,
                COALESCE(retrieval.retrieval_batch_count, 0)::bigint AS retrieval_batch_count
            FROM queries
            LEFT JOIN calls USING (run_id, query_id)
            LEFT JOIN retrieval USING (run_id, query_id)""",
        f"""CREATE VIEW {RUNTIME}.run_cost_summary AS
            WITH calls AS (
                SELECT run_id,
                    COALESCE(sum(input_tokens), 0)::bigint AS input_tokens,
                    COALESCE(sum(output_tokens), 0)::bigint AS output_tokens,
                    COALESCE(sum(total_tokens), 0)::bigint AS total_tokens,
                    COALESCE(sum(cost_usd), 0)::double precision AS cost_usd,
                    COALESCE(sum(latency_ms), 0)::double precision AS model_call_latency_ms,
                    count(*)::bigint AS model_call_count,
                    count(*) FILTER (WHERE fallback_used)::bigint AS fallback_call_count
                FROM {RUNTIME}.model_calls GROUP BY run_id
            ), retrieval AS (
                SELECT run_id,
                    COALESCE(sum(total_latency_ms), 0)::double precision AS retrieval_latency_ms,
                    count(*)::bigint AS retrieval_batch_count
                FROM {RUNTIME}.retrieval_batches GROUP BY run_id
            )
            SELECT manifests.run_id,
                COALESCE(calls.input_tokens, 0)::bigint AS input_tokens,
                COALESCE(calls.output_tokens, 0)::bigint AS output_tokens,
                COALESCE(calls.total_tokens, 0)::bigint AS total_tokens,
                COALESCE(calls.cost_usd, 0)::double precision AS cost_usd,
                COALESCE(calls.model_call_latency_ms, 0)::double precision AS model_call_latency_ms,
                COALESCE(retrieval.retrieval_latency_ms, 0)::double precision AS retrieval_latency_ms,
                (COALESCE(calls.model_call_latency_ms, 0) +
                 COALESCE(retrieval.retrieval_latency_ms, 0))::double precision AS observed_latency_ms,
                COALESCE(calls.model_call_count, 0)::bigint AS model_call_count,
                COALESCE(calls.fallback_call_count, 0)::bigint AS fallback_call_count,
                COALESCE(retrieval.retrieval_batch_count, 0)::bigint AS retrieval_batch_count
            FROM {RUNTIME}.run_manifests AS manifests
            LEFT JOIN calls USING (run_id)
            LEFT JOIN retrieval USING (run_id)""",
        f"""CREATE VIEW {EVALUATION}.query_cost_summary AS
            WITH call_facts AS (
                SELECT * FROM {RUNTIME}.model_calls
                UNION ALL
                SELECT * FROM {EVALUATION}.model_calls
            ), calls AS (
                SELECT run_id, query_id,
                    COALESCE(sum(input_tokens), 0)::bigint AS input_tokens,
                    COALESCE(sum(output_tokens), 0)::bigint AS output_tokens,
                    COALESCE(sum(total_tokens), 0)::bigint AS total_tokens,
                    COALESCE(sum(cost_usd), 0)::double precision AS cost_usd,
                    COALESCE(sum(latency_ms), 0)::double precision AS model_call_latency_ms,
                    count(*)::bigint AS model_call_count,
                    count(*) FILTER (WHERE fallback_used)::bigint AS fallback_call_count
                FROM call_facts WHERE query_id IS NOT NULL GROUP BY run_id, query_id
            ), retrieval AS (
                SELECT run_id, query_id,
                    COALESCE(sum(total_latency_ms), 0)::double precision AS retrieval_latency_ms,
                    count(*)::bigint AS retrieval_batch_count
                FROM {RUNTIME}.retrieval_batches GROUP BY run_id, query_id
            ), queries AS (
                SELECT DISTINCT run_id, query_id FROM {RUNTIME}.run_queries
            )
            SELECT queries.run_id, queries.query_id,
                COALESCE(calls.input_tokens, 0)::bigint AS input_tokens,
                COALESCE(calls.output_tokens, 0)::bigint AS output_tokens,
                COALESCE(calls.total_tokens, 0)::bigint AS total_tokens,
                COALESCE(calls.cost_usd, 0)::double precision AS cost_usd,
                COALESCE(calls.model_call_latency_ms, 0)::double precision AS model_call_latency_ms,
                COALESCE(retrieval.retrieval_latency_ms, 0)::double precision AS retrieval_latency_ms,
                (COALESCE(calls.model_call_latency_ms, 0) +
                 COALESCE(retrieval.retrieval_latency_ms, 0))::double precision AS observed_latency_ms,
                COALESCE(calls.model_call_count, 0)::bigint AS model_call_count,
                COALESCE(calls.fallback_call_count, 0)::bigint AS fallback_call_count,
                COALESCE(retrieval.retrieval_batch_count, 0)::bigint AS retrieval_batch_count
            FROM queries
            LEFT JOIN calls USING (run_id, query_id)
            LEFT JOIN retrieval USING (run_id, query_id)""",
        f"""CREATE VIEW {EVALUATION}.run_cost_summary AS
            WITH call_facts AS (
                SELECT * FROM {RUNTIME}.model_calls
                UNION ALL
                SELECT * FROM {EVALUATION}.model_calls
            ), calls AS (
                SELECT run_id,
                    COALESCE(sum(input_tokens), 0)::bigint AS input_tokens,
                    COALESCE(sum(output_tokens), 0)::bigint AS output_tokens,
                    COALESCE(sum(total_tokens), 0)::bigint AS total_tokens,
                    COALESCE(sum(cost_usd), 0)::double precision AS cost_usd,
                    COALESCE(sum(latency_ms), 0)::double precision AS model_call_latency_ms,
                    count(*)::bigint AS model_call_count,
                    count(*) FILTER (WHERE fallback_used)::bigint AS fallback_call_count
                FROM call_facts GROUP BY run_id
            ), retrieval AS (
                SELECT run_id,
                    COALESCE(sum(total_latency_ms), 0)::double precision AS retrieval_latency_ms,
                    count(*)::bigint AS retrieval_batch_count
                FROM {RUNTIME}.retrieval_batches GROUP BY run_id
            )
            SELECT manifests.run_id,
                COALESCE(calls.input_tokens, 0)::bigint AS input_tokens,
                COALESCE(calls.output_tokens, 0)::bigint AS output_tokens,
                COALESCE(calls.total_tokens, 0)::bigint AS total_tokens,
                COALESCE(calls.cost_usd, 0)::double precision AS cost_usd,
                COALESCE(calls.model_call_latency_ms, 0)::double precision AS model_call_latency_ms,
                COALESCE(retrieval.retrieval_latency_ms, 0)::double precision AS retrieval_latency_ms,
                (COALESCE(calls.model_call_latency_ms, 0) +
                 COALESCE(retrieval.retrieval_latency_ms, 0))::double precision AS observed_latency_ms,
                COALESCE(calls.model_call_count, 0)::bigint AS model_call_count,
                COALESCE(calls.fallback_call_count, 0)::bigint AS fallback_call_count,
                COALESCE(retrieval.retrieval_batch_count, 0)::bigint AS retrieval_batch_count
            FROM {RUNTIME}.run_manifests AS manifests
            LEFT JOIN calls USING (run_id)
            LEFT JOIN retrieval USING (run_id)""",
    )


def _run_manifest_immutability_function() -> str:
    return f"""CREATE FUNCTION {RUNTIME}.reject_run_manifest_mutation()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'run manifests are immutable; create a new run'
                USING ERRCODE = '55000';
        END;
        $$"""


def _run_freeze_function() -> str:
    return f"""CREATE FUNCTION {RUNTIME}.enforce_run_writable()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME}, {EVALUATION}
        AS $$
        DECLARE
            payload jsonb;
            target_run varchar(128);
            current_status varchar(16);
            target_config_hash varchar(128);
        BEGIN
            payload := CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) ELSE to_jsonb(NEW) END;
            IF TG_TABLE_NAME = 'job_attempts' THEN
                SELECT run_id INTO target_run FROM {RUNTIME}.experiment_jobs
                    WHERE job_id = payload ->> 'job_id';
            ELSIF TG_TABLE_NAME = 'embeddings' THEN
                SELECT run_id INTO target_run FROM {RUNTIME}.embedding_collections
                    WHERE collection_id = payload ->> 'collection_id';
            ELSIF TG_TABLE_NAME IN ('final_answer_evidence_ids', 'final_answer_image_ids') THEN
                SELECT run_id INTO target_run FROM {RUNTIME}.final_answers
                    WHERE answer_id = payload ->> 'answer_id';
            ELSIF TG_TABLE_NAME = 'retrieval_evaluation_matches' THEN
                SELECT run_id INTO target_run FROM {EVALUATION}.retrieval_evaluations
                    WHERE retrieval_evaluation_id = payload ->> 'retrieval_evaluation_id';
            ELSE
                target_run := payload ->> 'run_id';
            END IF;
            IF target_run IS NOT NULL THEN
                SELECT states.status, manifests.config_hash
                    INTO current_status, target_config_hash
                    FROM {RUNTIME}.run_states AS states
                    JOIN {RUNTIME}.run_manifests AS manifests USING (run_id)
                    WHERE states.run_id = target_run
                    FOR SHARE OF states;
                IF current_status IS NULL AND session_user != 'pmmc_owner' THEN
                    RAISE EXCEPTION 'experiment run is not registered'
                        USING ERRCODE = '55000';
                END IF;
                IF target_config_hash IS NULL AND session_user != 'pmmc_owner' THEN
                    RAISE EXCEPTION 'experiment run has no immutable config hash'
                        USING ERRCODE = '55000';
                END IF;
                IF current_status IN ('completed', 'failed', 'cancelled') THEN
                    RAISE EXCEPTION 'terminal experiment run is immutable'
                        USING ERRCODE = '55000';
                END IF;
                IF TG_TABLE_NAME IN ('run_snapshot_links', 'run_queries')
                    AND current_status != 'created' THEN
                    RAISE EXCEPTION 'run snapshot links and query denominator are frozen'
                        USING ERRCODE = '55000';
                END IF;
            END IF;
            IF TG_OP = 'DELETE' THEN
                RETURN OLD;
            END IF;
            RETURN NEW;
        END;
        $$"""


def _run_state_transition_function() -> str:
    return f"""CREATE FUNCTION {RUNTIME}.enforce_run_state_transition()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME}
        AS $$
        DECLARE
            persisted_queries integer;
            covered_queries integer;
            manifest_selection_hash varchar(128);
            manifest_config_hash varchar(128);
        BEGIN
            IF OLD.status IN ('completed', 'failed', 'cancelled') THEN
                RAISE EXCEPTION 'terminal experiment run state is immutable'
                    USING ERRCODE = '55000';
            END IF;
            IF TG_OP = 'DELETE' THEN
                RETURN OLD;
            END IF;
            IF NEW.run_id != OLD.run_id OR NEW.selected_count != OLD.selected_count THEN
                RAISE EXCEPTION 'run identity and selected denominator are immutable'
                    USING ERRCODE = '55000';
            END IF;
            IF NEW.status = OLD.status AND
                (NEW.succeeded_count != OLD.succeeded_count OR
                 NEW.failed_count != OLD.failed_count) THEN
                RAISE EXCEPTION 'run result counts may only change with a state transition'
                    USING ERRCODE = '55000';
            END IF;
            IF NEW.status != OLD.status AND NOT (
                (OLD.status = 'created' AND NEW.status IN ('running','failed','cancelled')) OR
                (OLD.status = 'running' AND NEW.status IN ('completed','failed','cancelled'))
            ) THEN
                RAISE EXCEPTION 'invalid experiment run state transition'
                    USING ERRCODE = '55000';
            END IF;
            IF NEW.status = 'running' AND NEW.started_at IS NULL THEN
                RAISE EXCEPTION 'running experiment run requires started_at'
                    USING ERRCODE = '23514';
            END IF;
            IF NEW.status IN ('completed','failed','cancelled') AND NEW.completed_at IS NULL THEN
                RAISE EXCEPTION 'terminal experiment run requires completed_at'
                    USING ERRCODE = '23514';
            END IF;
            IF NEW.status = 'failed' AND NEW.error_type IS NULL THEN
                RAISE EXCEPTION 'failed experiment run requires error_type'
                    USING ERRCODE = '23514';
            END IF;
            IF NEW.started_at IS NOT NULL AND NEW.completed_at IS NOT NULL AND
                NEW.completed_at < NEW.started_at THEN
                RAISE EXCEPTION 'completed_at cannot precede started_at'
                    USING ERRCODE = '23514';
            END IF;

            IF OLD.status = 'created' AND NEW.status = 'running' THEN
                SELECT selection_hash, config_hash
                    INTO manifest_selection_hash, manifest_config_hash
                    FROM {RUNTIME}.run_manifests WHERE run_id = NEW.run_id;
                SELECT count(*)::integer INTO persisted_queries
                    FROM {RUNTIME}.run_queries WHERE run_id = NEW.run_id;
                IF manifest_config_hash IS NULL OR manifest_selection_hash IS NULL THEN
                    RAISE EXCEPTION 'registered experiment run requires immutable hashes'
                        USING ERRCODE = '55000';
                END IF;
                IF persisted_queries != NEW.selected_count OR EXISTS (
                    SELECT 1 FROM {RUNTIME}.run_queries
                    WHERE run_id = NEW.run_id AND selection_hash != manifest_selection_hash
                ) OR (
                    NEW.selected_count > 0 AND NOT EXISTS (
                        SELECT 1 FROM {RUNTIME}.run_queries
                        WHERE run_id = NEW.run_id AND ordinal = 0
                    )
                ) OR COALESCE((
                    SELECT max(ordinal) FROM {RUNTIME}.run_queries WHERE run_id = NEW.run_id
                ), -1) != NEW.selected_count - 1 THEN
                    RAISE EXCEPTION 'run query denominator is incomplete or inconsistent'
                        USING ERRCODE = '55000';
                END IF;
            END IF;

            IF NEW.status = 'completed' THEN
                SELECT count(*)::integer INTO persisted_queries
                    FROM {RUNTIME}.run_queries WHERE run_id = NEW.run_id;
                SELECT count(DISTINCT query_id)::integer INTO covered_queries
                    FROM {RUNTIME}.experiment_jobs
                    WHERE run_id = NEW.run_id AND query_id IS NOT NULL AND status = 'succeeded';
                IF persisted_queries != NEW.selected_count OR covered_queries != NEW.selected_count OR
                    EXISTS (
                        SELECT 1 FROM {RUNTIME}.experiment_jobs
                        WHERE run_id = NEW.run_id AND status != 'succeeded'
                    ) OR EXISTS (
                        SELECT 1 FROM {RUNTIME}.run_artifacts
                        WHERE run_id = NEW.run_id AND state != 'ready'
                    ) OR EXISTS (
                        SELECT 1 FROM {RUNTIME}.embedding_collections
                        WHERE run_id = NEW.run_id AND status != 'ready'
                    ) OR EXISTS (
                        SELECT 1
                        FROM {RUNTIME}.retrieval_batches AS batches
                        LEFT JOIN {RUNTIME}.retrieval_traces AS traces
                            USING (retrieval_batch_id)
                        WHERE batches.run_id = NEW.run_id
                        GROUP BY batches.retrieval_batch_id, batches.returned_count
                        HAVING count(traces.retrieval_trace_id) != batches.returned_count OR
                            COALESCE(min(traces.rank), 0) !=
                                CASE WHEN batches.returned_count = 0 THEN 0 ELSE 1 END OR
                            COALESCE(max(traces.rank), 0) != batches.returned_count
                    ) OR EXISTS (
                        SELECT 1
                        FROM {RUNTIME}.retrieval_batches AS batches
                        JOIN {RUNTIME}.run_manifests AS manifests
                            ON manifests.run_id = batches.run_id
                        WHERE batches.run_id = NEW.run_id AND
                            manifests.retrieval_topk_max IS NOT NULL AND (
                                batches.requested_top_k > manifests.retrieval_topk_max OR
                                (manifests.paper_mode AND batches.status = 'success' AND
                                 batches.requested_top_k != manifests.retrieval_topk_max)
                            )
                    ) OR NEW.succeeded_count != NEW.selected_count OR NEW.failed_count != 0 THEN
                    RAISE EXCEPTION 'completed experiment run failed integrity requirements'
                        USING ERRCODE = '55000';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$"""


def _run_freeze_triggers() -> tuple[str, ...]:
    runtime_tables = (
        "run_snapshot_links",
        "run_queries",
        "run_stage_summaries",
        "experiment_jobs",
        "job_attempts",
        "run_artifacts",
        "embedding_collections",
        "embeddings",
        "model_calls",
        "retrieval_batches",
        "retrieval_traces",
        "final_answers",
        "final_answer_evidence_ids",
        "final_answer_image_ids",
    )
    evaluation_tables = (
        "run_artifacts",
        "model_calls",
        "retrieval_evaluations",
        "retrieval_evaluation_matches",
        "metric_scores",
        "judge_scores",
        "bad_case_labels",
    )
    return tuple(
        f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON "
        f"{schema}.{table} FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()"
        for schema, tables in (
            (RUNTIME, runtime_tables),
            (EVALUATION, evaluation_tables),
        )
        for table in tables
    )


def _grant_statements() -> tuple[str, ...]:
    runtime_insert = (
        "run_manifests, run_snapshot_links, run_states, run_queries, run_stage_summaries, "
        "experiment_jobs, job_attempts, run_artifacts, embedding_collections, embeddings, "
        "model_calls, retrieval_batches, retrieval_traces, final_answers, "
        "final_answer_evidence_ids, final_answer_image_ids"
    )
    evaluation_insert = (
        "run_artifacts, model_calls, retrieval_evaluations, retrieval_evaluation_matches, "
        "metric_scores, judge_scores, bad_case_labels"
    )
    return (
        f"REVOKE ALL ON FUNCTION {RUNTIME}.reject_run_manifest_mutation() FROM PUBLIC",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_run_writable() FROM PUBLIC",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_run_state_transition() FROM PUBLIC",
        "GRANT SELECT ON public.alembic_version TO pmmc_runtime, pmmc_evaluator",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA {RUNTIME} TO pmmc_runtime, pmmc_evaluator",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA {EVALUATION} TO pmmc_evaluator",
        f"GRANT INSERT ON {', '.join(f'{RUNTIME}.{name.strip()}' for name in runtime_insert.split(','))} TO pmmc_runtime",
        f"GRANT UPDATE (status, started_at, heartbeat_at, completed_at, error_type, error_summary, error_artifact_id, succeeded_count, failed_count) ON {RUNTIME}.run_states TO pmmc_runtime",
        f"GRANT UPDATE (status, claimed_by, lease_expires_at, available_at, attempt_count, started_at, finished_at, error_type, error_summary, error_artifact_id) ON {RUNTIME}.experiment_jobs TO pmmc_runtime",
        f"GRANT UPDATE (status, finished_at, error_type, error_summary, error_artifact_id) ON {RUNTIME}.job_attempts TO pmmc_runtime",
        f"GRANT UPDATE (state, storage_uri, size_bytes, ready_at, error_type) ON {RUNTIME}.run_artifacts TO pmmc_runtime",
        f"GRANT UPDATE (status, index_type, index_params) ON {RUNTIME}.embedding_collections TO pmmc_runtime",
        f"GRANT INSERT ON {', '.join(f'{EVALUATION}.{name.strip()}' for name in evaluation_insert.split(','))} TO pmmc_evaluator",
        f"GRANT UPDATE (state, storage_uri, size_bytes, ready_at, error_type) ON {EVALUATION}.run_artifacts TO pmmc_evaluator",
    )
