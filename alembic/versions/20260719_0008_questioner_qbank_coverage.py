# ruff: noqa: E501
"""Add Questioner production, frozen QBank indexes, and coverage evaluation.

Revision ID: 20260719_0008
Revises: 20260718_0007
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260719_0008"
down_revision: str | None = "20260718_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"


def upgrade() -> None:
    _extend_compiler_tables()
    _create_runtime_tables()
    _create_evaluation_tables()
    _create_runtime_guards()
    _replace_compiler_completion_guard()
    _create_evaluation_guards()
    _grant_permissions()


def _extend_compiler_tables() -> None:
    for name, length in (
        ("questioner_config_hash", 128),
        ("questioner_model_alias", 256),
        ("questioner_model_revision", 256),
        ("questioner_prompt_version", 128),
        ("questioner_prompt_hash", 128),
        ("questioner_renderer_version", 128),
        ("questioner_parser_version", 128),
        ("unit_builder_version", 128),
        ("candidate_gate_version", 128),
        ("dedupe_version", 128),
        ("qbank_matcher_version", 128),
    ):
        op.add_column(
            "compiler_runs",
            sa.Column(name, sa.String(length=length), nullable=True),
            schema=RUNTIME,
        )

    op.add_column(
        "qbank_questions",
        sa.Column("generation_id", sa.String(length=128), nullable=True),
        schema=RUNTIME,
    )
    op.add_column(
        "qbank_questions",
        sa.Column("reasoning_type", sa.String(length=32), nullable=True),
        schema=RUNTIME,
    )
    op.add_column(
        "qbank_questions",
        sa.Column("candidate_content_hash", sa.String(length=128), nullable=True),
        schema=RUNTIME,
    )
    op.execute(
        f"UPDATE {RUNTIME}.qbank_questions "
        "SET reasoning_type = 'single_hop', candidate_content_hash = question_hash "
        "WHERE reasoning_type IS NULL"
    )
    op.create_check_constraint(
        op.f("ck_qbank_questions_reasoning_type_values"),
        "qbank_questions",
        "reasoning_type IS NULL OR reasoning_type IN "
        "('single_hop','multi_hop','temporal','update','conflict','refusal')",
        schema=RUNTIME,
    )


def _create_runtime_tables() -> None:
    statements = (
        f"""CREATE TABLE {RUNTIME}.compiler_unit_links (
            link_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            conversation_id varchar(256) NOT NULL,
            anchor_memory_id varchar(256) NOT NULL,
            support_memory_id varchar(256) NOT NULL,
            relation_type varchar(32) NOT NULL CHECK (relation_type IN ('adjacent_session','shared_entity','shared_image','profile_preference','temporal_neighbor')),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            relation_key_hash varchar(128) NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (compiler_unit_id, ordinal),
            UNIQUE (compiler_unit_id, anchor_memory_id, support_memory_id, relation_type),
            FOREIGN KEY (compiler_unit_id, run_id, snapshot_id, conversation_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id, snapshot_id, conversation_id),
            FOREIGN KEY (compiler_unit_id, anchor_memory_id)
                REFERENCES {RUNTIME}.compiler_unit_memories(compiler_unit_id, memory_id),
            FOREIGN KEY (compiler_unit_id, support_memory_id)
                REFERENCES {RUNTIME}.compiler_unit_memories(compiler_unit_id, memory_id),
            CHECK (anchor_memory_id != support_memory_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.questioner_generations (
            generation_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            job_id varchar(128) NOT NULL,
            model_call_id varchar(128) NOT NULL UNIQUE,
            attempt_number integer NOT NULL CHECK (attempt_number > 0),
            prompt_version varchar(128) NOT NULL,
            renderer_version varchar(128) NOT NULL,
            parser_version varchar(128) NOT NULL,
            prompt_hash varchar(128) NOT NULL,
            input_hash varchar(128) NOT NULL,
            output_hash varchar(128),
            parse_status varchar(16) NOT NULL CHECK (parse_status IN ('parsed','repaired','partial','failed')),
            raw_candidate_count integer NOT NULL CHECK (raw_candidate_count >= 0),
            valid_candidate_count integer NOT NULL CHECK (valid_candidate_count >= 0),
            invalid_candidate_count integer NOT NULL CHECK (invalid_candidate_count >= 0),
            repair_attempt_count integer NOT NULL CHECK (repair_attempt_count >= 0),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (generation_id, run_id),
            UNIQUE (compiler_unit_id, attempt_number),
            FOREIGN KEY (compiler_unit_id, run_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            FOREIGN KEY (job_id, run_id)
                REFERENCES {RUNTIME}.experiment_jobs(job_id, run_id),
            FOREIGN KEY (model_call_id, run_id)
                REFERENCES {RUNTIME}.model_calls(model_call_id, run_id),
            CHECK (valid_candidate_count + invalid_candidate_count = raw_candidate_count),
            CHECK (parse_status != 'failed' OR valid_candidate_count = 0)
        )""",
    )
    for statement in statements:
        op.execute(statement)

    op.create_foreign_key(
        "fk_qbank_questions_generation",
        "qbank_questions",
        "questioner_generations",
        ["generation_id"],
        ["generation_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )

    statements = (
        f"""CREATE TABLE {RUNTIME}.questioner_candidate_gate_results (
            gate_result_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            compiler_unit_id varchar(128) NOT NULL,
            generation_id varchar(128) NOT NULL,
            qbank_id varchar(128) NOT NULL,
            gate_version varchar(128) NOT NULL,
            status varchar(32) NOT NULL CHECK (status IN ('eligible_for_planning','rejected')),
            rejection_codes text[] NOT NULL,
            duplicate_of_qbank_id varchar(128),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (qbank_id, gate_version),
            FOREIGN KEY (compiler_unit_id, run_id)
                REFERENCES {RUNTIME}.compiler_units(compiler_unit_id, run_id),
            FOREIGN KEY (generation_id, run_id)
                REFERENCES {RUNTIME}.questioner_generations(generation_id, run_id),
            FOREIGN KEY (qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            FOREIGN KEY (duplicate_of_qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            CHECK (rejection_codes <@ ARRAY['invalid_schema','unsupported_question_type','future_or_foreign_reference','missing_expected_answer','missing_expected_evidence','modality_mismatch','visual_evidence_missing','multi_hop_evidence_missing','duplicate','budget_exceeded']::text[]),
            CHECK ((status = 'eligible_for_planning' AND cardinality(rejection_codes) = 0)
                OR (status = 'rejected' AND cardinality(rejection_codes) > 0)),
            CHECK (duplicate_of_qbank_id IS NULL OR 'duplicate' = ANY(rejection_codes))
        )""",
        f"""CREATE TABLE {RUNTIME}.qbank_duplicate_links (
            duplicate_link_id varchar(128) PRIMARY KEY,
            run_id varchar(128) NOT NULL,
            duplicate_qbank_id varchar(128) NOT NULL,
            primary_qbank_id varchar(128) NOT NULL,
            method varchar(128) NOT NULL,
            score jsonb,
            hard_duplicate boolean NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (duplicate_qbank_id, method),
            FOREIGN KEY (duplicate_qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            FOREIGN KEY (primary_qbank_id, run_id)
                REFERENCES {RUNTIME}.qbank_questions(qbank_id, run_id),
            CHECK (duplicate_qbank_id != primary_qbank_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.qbank_index_manifests (
            qbank_index_id varchar(128) PRIMARY KEY,
            compiler_run_id varchar(128) NOT NULL UNIQUE,
            snapshot_id varchar(128) NOT NULL,
            gate_version varchar(128) NOT NULL,
            matcher_version varchar(128) NOT NULL,
            text_question_collection_id varchar(128) NOT NULL REFERENCES {RUNTIME}.embedding_collections(collection_id),
            text_paraphrase_collection_id varchar(128) NOT NULL REFERENCES {RUNTIME}.embedding_collections(collection_id),
            multimodal_question_collection_id varchar(128) NOT NULL REFERENCES {RUNTIME}.embedding_collections(collection_id),
            multimodal_paraphrase_collection_id varchar(128) NOT NULL REFERENCES {RUNTIME}.embedding_collections(collection_id),
            eligible_question_count integer NOT NULL CHECK (eligible_question_count >= 0),
            paraphrase_count integer NOT NULL CHECK (paraphrase_count >= 0),
            rrf_constant integer NOT NULL CHECK (rrf_constant > 0),
            per_channel_top_k integer NOT NULL CHECK (per_channel_top_k > 0),
            status varchar(16) NOT NULL CHECK (status IN ('building','ready','failed')),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (qbank_index_id, compiler_run_id),
            FOREIGN KEY (compiler_run_id, snapshot_id)
                REFERENCES {RUNTIME}.compiler_runs(run_id, snapshot_id)
        )""",
    )
    for statement in statements:
        op.execute(statement)
    op.create_index(
        "ix_questioner_candidate_gate_results_status",
        "questioner_candidate_gate_results",
        ["run_id", "status"],
        schema=RUNTIME,
    )


def _create_evaluation_tables() -> None:
    statements = (
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_runs (
            coverage_run_id varchar(128) PRIMARY KEY,
            compiler_run_id varchar(128) NOT NULL,
            qbank_index_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            dataset varchar(32) NOT NULL CHECK (dataset IN ('memlens','memgallery','unknown')),
            protocol_version varchar(128) NOT NULL,
            selection_hash varchar(128) NOT NULL,
            config_hash varchar(128) NOT NULL,
            question_budgets integer[] NOT NULL CHECK (cardinality(question_budgets) > 0),
            match_ks integer[] NOT NULL CHECK (cardinality(match_ks) > 0),
            gate_mode varchar(16) NOT NULL CHECK (gate_mode IN ('report_only','enforce')),
            status varchar(16) NOT NULL CHECK (status IN ('created','running','completed','failed')),
            created_at timestamptz NOT NULL,
            completed_at timestamptz,
            metadata jsonb NOT NULL,
            FOREIGN KEY (compiler_run_id, snapshot_id)
                REFERENCES {RUNTIME}.compiler_runs(run_id, snapshot_id),
            FOREIGN KEY (qbank_index_id, compiler_run_id)
                REFERENCES {RUNTIME}.qbank_index_manifests(qbank_index_id, compiler_run_id),
            CHECK ((status = 'completed' AND completed_at IS NOT NULL)
                OR (status != 'completed' AND completed_at IS NULL))
        )""",
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_queries (
            coverage_run_id varchar(128) NOT NULL REFERENCES {EVALUATION}.qbank_coverage_runs(coverage_run_id),
            query_id varchar(256) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            conversation_id varchar(256) NOT NULL,
            cutoff_session_id varchar(256) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            has_query_image boolean NOT NULL,
            applicable boolean NOT NULL,
            gold_group_count integer NOT NULL CHECK (gold_group_count >= 0),
            unavailable_reason varchar(256),
            metadata jsonb NOT NULL,
            PRIMARY KEY (coverage_run_id, query_id),
            UNIQUE (coverage_run_id, ordinal),
            FOREIGN KEY (snapshot_id, query_id)
                REFERENCES {RUNTIME}.benchmark_queries(snapshot_id, query_id),
            FOREIGN KEY (snapshot_id, conversation_id, cutoff_session_id)
                REFERENCES {RUNTIME}.benchmark_sessions(snapshot_id, conversation_id, session_id),
            CHECK ((applicable AND gold_group_count > 0 AND unavailable_reason IS NULL)
                OR (NOT applicable AND unavailable_reason IS NOT NULL))
        )""",
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_query_embeddings (
            query_embedding_id varchar(128) PRIMARY KEY,
            coverage_run_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            channel varchar(32) NOT NULL CHECK (channel IN ('text_semantic','multimodal_semantic')),
            model_alias varchar(256) NOT NULL,
            model_revision varchar(256) NOT NULL,
            input_hash varchar(128) NOT NULL,
            dimension integer NOT NULL CHECK (dimension > 0),
            embedding vector NOT NULL,
            model_call_id varchar(128),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (coverage_run_id, query_id, channel),
            FOREIGN KEY (coverage_run_id, query_id)
                REFERENCES {EVALUATION}.qbank_coverage_queries(coverage_run_id, query_id),
            CHECK (vector_dims(embedding) = dimension)
        )""",
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_candidates (
            coverage_candidate_id varchar(128) PRIMARY KEY,
            coverage_run_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            qbank_id varchar(128) NOT NULL REFERENCES {RUNTIME}.qbank_questions(qbank_id),
            question_budget integer NOT NULL CHECK (question_budget > 0),
            rank integer NOT NULL CHECK (rank > 0),
            lexical_rank integer CHECK (lexical_rank > 0),
            text_semantic_rank integer CHECK (text_semantic_rank > 0),
            multimodal_semantic_rank integer CHECK (multimodal_semantic_rank > 0),
            lexical_score jsonb,
            text_semantic_score jsonb,
            multimodal_semantic_score jsonb,
            final_score jsonb NOT NULL,
            matched_gold_group_ids text[] NOT NULL,
            individually_sufficient boolean NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (coverage_run_id, query_id, question_budget, rank),
            UNIQUE (coverage_run_id, query_id, question_budget, qbank_id),
            FOREIGN KEY (coverage_run_id, query_id)
                REFERENCES {EVALUATION}.qbank_coverage_queries(coverage_run_id, query_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_gold_groups (
            gold_group_id varchar(128) PRIMARY KEY,
            coverage_run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            granularity varchar(32) NOT NULL CHECK (granularity IN ('generic','session','memory','image','text')),
            source_gold_id varchar(256) NOT NULL,
            resolution_policy_version varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (coverage_run_id, query_id, source_gold_id),
            FOREIGN KEY (coverage_run_id, query_id)
                REFERENCES {EVALUATION}.qbank_coverage_queries(coverage_run_id, query_id),
            FOREIGN KEY (snapshot_id, query_id)
                REFERENCES {EVALUATION}.query_gold(snapshot_id, query_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_gold_group_members (
            gold_group_id varchar(128) NOT NULL REFERENCES {EVALUATION}.qbank_coverage_gold_groups(gold_group_id),
            member_type varchar(32) NOT NULL CHECK (member_type IN ('evidence','session','memory','image')),
            member_id varchar(256) NOT NULL,
            derived boolean NOT NULL,
            metadata jsonb NOT NULL,
            PRIMARY KEY (gold_group_id, member_type, member_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_matches (
            coverage_candidate_id varchar(128) NOT NULL REFERENCES {EVALUATION}.qbank_coverage_candidates(coverage_candidate_id),
            gold_group_id varchar(128) NOT NULL REFERENCES {EVALUATION}.qbank_coverage_gold_groups(gold_group_id),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            matched_member_ids text[] NOT NULL,
            PRIMARY KEY (coverage_candidate_id, gold_group_id),
            UNIQUE (coverage_candidate_id, ordinal)
        )""",
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_metrics (
            metric_id varchar(128) PRIMARY KEY,
            coverage_run_id varchar(128) NOT NULL REFERENCES {EVALUATION}.qbank_coverage_runs(coverage_run_id),
            query_id varchar(256),
            metric_name varchar(128) NOT NULL,
            metric_version varchar(128) NOT NULL,
            question_budget integer CHECK (question_budget > 0),
            k integer CHECK (k > 0),
            applicable boolean NOT NULL,
            value double precision CHECK (value BETWEEN 0 AND 1),
            unavailable_reason varchar(256),
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            FOREIGN KEY (coverage_run_id, query_id)
                REFERENCES {EVALUATION}.qbank_coverage_queries(coverage_run_id, query_id),
            CHECK ((applicable AND value IS NOT NULL AND unavailable_reason IS NULL)
                OR (NOT applicable AND value IS NULL AND unavailable_reason IS NOT NULL))
        )""",
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_labels (
            label_id varchar(128) PRIMARY KEY,
            coverage_run_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            category varchar(64) NOT NULL CHECK (category IN ('questioner_coverage_failure','matching_failure','gold_unresolvable')),
            question_budget integer CHECK (question_budget > 0),
            k integer CHECK (k > 0),
            is_primary boolean NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            FOREIGN KEY (coverage_run_id, query_id)
                REFERENCES {EVALUATION}.qbank_coverage_queries(coverage_run_id, query_id)
        )""",
        f"""CREATE TABLE {EVALUATION}.qbank_coverage_gate_reports (
            gate_report_id varchar(128) PRIMARY KEY,
            coverage_run_id varchar(128) NOT NULL UNIQUE REFERENCES {EVALUATION}.qbank_coverage_runs(coverage_run_id),
            gate_version varchar(128) NOT NULL,
            mode varchar(16) NOT NULL CHECK (mode IN ('report_only','enforce')),
            hard_integrity_passed boolean NOT NULL,
            scientific_thresholds_passed boolean,
            query_count integer NOT NULL CHECK (query_count >= 0),
            applicable_query_count integer NOT NULL CHECK (applicable_query_count >= 0),
            metric_summary jsonb NOT NULL,
            threshold_summary jsonb NOT NULL,
            failure_counts jsonb NOT NULL,
            elapsed_ms double precision NOT NULL CHECK (elapsed_ms >= 0),
            content_hash varchar(128) NOT NULL,
            created_at timestamptz NOT NULL,
            metadata jsonb NOT NULL,
            CHECK (applicable_query_count <= query_count),
            CHECK ((mode = 'report_only' AND scientific_thresholds_passed IS NULL)
                OR (mode = 'enforce' AND scientific_thresholds_passed IS NOT NULL))
        )""",
    )
    for statement in statements:
        op.execute(statement)
    op.create_index(
        "ix_qbank_coverage_candidates_query_budget",
        "qbank_coverage_candidates",
        ["coverage_run_id", "query_id", "question_budget", "rank"],
        schema=EVALUATION,
    )


def _create_runtime_guards() -> None:
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_questioner_fact_scope()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE anchor_role text; support_role text; generation record; question record; call record;
        BEGIN
            IF TG_TABLE_NAME = 'compiler_unit_links' THEN
                SELECT role INTO anchor_role FROM {RUNTIME}.compiler_unit_memories
                 WHERE compiler_unit_id = NEW.compiler_unit_id AND memory_id = NEW.anchor_memory_id;
                SELECT role INTO support_role FROM {RUNTIME}.compiler_unit_memories
                 WHERE compiler_unit_id = NEW.compiler_unit_id AND memory_id = NEW.support_memory_id;
                IF anchor_role != 'anchor' OR support_role != 'support' THEN
                    RAISE EXCEPTION 'compiler unit links must connect anchor to support' USING ERRCODE = '23514';
                END IF;
            ELSIF TG_TABLE_NAME = 'questioner_generations' THEN
                SELECT calls.* INTO call FROM {RUNTIME}.model_calls calls
                 WHERE calls.model_call_id = NEW.model_call_id AND calls.run_id = NEW.run_id;
                IF NOT FOUND OR call.compiler_unit_id != NEW.compiler_unit_id
                   OR call.job_id != NEW.job_id OR call.stage != 'questioner'
                   OR call.status != 'success' THEN
                    RAISE EXCEPTION 'Questioner generation has inconsistent model-call lineage' USING ERRCODE = '23514';
                END IF;
                IF NOT EXISTS (SELECT 1 FROM {RUNTIME}.experiment_jobs jobs
                    WHERE jobs.job_id = NEW.job_id AND jobs.run_id = NEW.run_id
                      AND jobs.compiler_unit_id = NEW.compiler_unit_id AND jobs.stage = 'compile') THEN
                    RAISE EXCEPTION 'Questioner generation has no matching COMPILE job' USING ERRCODE = '23514';
                END IF;
            ELSIF TG_TABLE_NAME = 'questioner_candidate_gate_results' THEN
                SELECT * INTO generation FROM {RUNTIME}.questioner_generations
                 WHERE generation_id = NEW.generation_id AND run_id = NEW.run_id;
                SELECT * INTO question FROM {RUNTIME}.qbank_questions
                 WHERE qbank_id = NEW.qbank_id AND run_id = NEW.run_id;
                IF generation.compiler_unit_id != NEW.compiler_unit_id
                   OR question.compiler_unit_id != NEW.compiler_unit_id
                   OR question.generation_id != NEW.generation_id
                   OR question.candidate_content_hash != NEW.content_hash THEN
                    RAISE EXCEPTION 'Questioner gate result has inconsistent candidate lineage' USING ERRCODE = '23514';
                END IF;
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.validate_qbank_index_manifest()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE current_collection_id text; source_kind text; collection_snapshot text;
                collection_status text; collection_dimension integer; metric text;
                is_normalized boolean; actual_count integer; expected_count integer;
                collection_run text; collection_alias text; expected_alias text;
                expected_source_kind text;
                position integer;
        BEGIN
            IF NEW.status != 'ready' THEN RETURN NEW; END IF;
            IF (SELECT count(DISTINCT value) FROM unnest(ARRAY[
                    NEW.text_question_collection_id, NEW.text_paraphrase_collection_id,
                    NEW.multimodal_question_collection_id,
                    NEW.multimodal_paraphrase_collection_id
                ]) AS items(value)) != 4 THEN
                RAISE EXCEPTION 'QBank index requires four distinct collections' USING ERRCODE = '23514';
            END IF;
            FOR current_collection_id, position IN
                SELECT value, ordinality FROM unnest(ARRAY[
                    NEW.text_question_collection_id, NEW.text_paraphrase_collection_id,
                    NEW.multimodal_question_collection_id, NEW.multimodal_paraphrase_collection_id
                ]) WITH ORDINALITY AS items(value, ordinality)
            LOOP
                SELECT source_type, snapshot_id, status, dimension, distance_metric,
                       normalized, run_id, model_alias
                  INTO source_kind, collection_snapshot, collection_status,
                       collection_dimension, metric, is_normalized,
                       collection_run, collection_alias
                  FROM {RUNTIME}.embedding_collections collections
                 WHERE collections.collection_id = current_collection_id;
                expected_alias := CASE WHEN position IN (1,2)
                    THEN NEW.metadata ->> 'text_model_alias'
                    ELSE NEW.metadata ->> 'multimodal_model_alias' END;
                expected_source_kind := CASE WHEN position IN (1,3)
                    THEN 'qbank_question' ELSE 'qbank_paraphrase' END;
                IF collection_snapshot IS DISTINCT FROM NEW.snapshot_id
                   OR collection_run IS DISTINCT FROM NEW.compiler_run_id
                   OR collection_status != 'ready' OR collection_dimension != 4096
                   OR metric != 'cosine' OR NOT is_normalized
                   OR collection_alias IS DISTINCT FROM expected_alias
                   OR source_kind IS DISTINCT FROM expected_source_kind THEN
                    RAISE EXCEPTION 'QBank index collection identity is invalid' USING ERRCODE = '23514';
                END IF;
                SELECT count(*) INTO actual_count FROM {RUNTIME}.embeddings embeddings
                 WHERE embeddings.collection_id = current_collection_id;
                expected_count := CASE WHEN position IN (1,3)
                    THEN NEW.eligible_question_count ELSE NEW.paraphrase_count END;
                IF actual_count != expected_count THEN
                    RAISE EXCEPTION 'QBank index collection count is incomplete' USING ERRCODE = '55000';
                END IF;
            END LOOP;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_qbank_index_writable()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE source_run text; state text;
        BEGIN
            source_run := COALESCE(NEW.compiler_run_id, OLD.compiler_run_id);
            SELECT status INTO state FROM {RUNTIME}.run_states WHERE run_id = source_run;
            IF state IN ('completed','failed','cancelled') THEN
                RAISE EXCEPTION 'terminal compiler run is immutable' USING ERRCODE = '55000';
            END IF;
            IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
            RETURN NEW;
        END; $$"""
    )
    for table in (
        "compiler_unit_links",
        "questioner_generations",
        "questioner_candidate_gate_results",
    ):
        op.execute(
            f"CREATE TRIGGER validate_questioner_fact BEFORE INSERT OR UPDATE ON {RUNTIME}.{table} FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_questioner_fact_scope()"
        )
    op.execute(
        f"CREATE TRIGGER validate_qbank_index BEFORE INSERT OR UPDATE ON {RUNTIME}.qbank_index_manifests FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_qbank_index_manifest()"
    )
    for table in (
        "compiler_unit_links",
        "questioner_generations",
        "questioner_candidate_gate_results",
        "qbank_duplicate_links",
    ):
        op.execute(
            f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.{table} FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()"
        )
    op.execute(
        f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.qbank_index_manifests FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_qbank_index_writable()"
    )


def _replace_compiler_completion_guard() -> None:
    op.execute(f"DROP TRIGGER compiler_completion_integrity ON {RUNTIME}.run_states")
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_questioner_compiler_completion()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE compiler {RUNTIME}.compiler_runs%ROWTYPE; units integer; bad integer; required text[];
        BEGIN
            IF NEW.status != 'completed' THEN RETURN NEW; END IF;
            SELECT * INTO compiler FROM {RUNTIME}.compiler_runs WHERE run_id = NEW.run_id;
            IF NOT FOUND THEN RETURN NEW; END IF;
            IF NEW.selected_count != 0 OR NEW.succeeded_count != 0 OR NEW.failed_count != 0 THEN
                RAISE EXCEPTION 'compiler run must keep query denominator counters at zero' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO units FROM {RUNTIME}.compiler_units WHERE run_id = NEW.run_id;
            IF units != compiler.unit_count OR COALESCE((SELECT max(ordinal)
                FROM {RUNTIME}.compiler_units WHERE run_id = NEW.run_id), -1) != compiler.unit_count - 1 THEN
                RAISE EXCEPTION 'compiler unit denominator is incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad FROM {RUNTIME}.compiler_units units
             WHERE units.run_id = NEW.run_id AND (
                (SELECT count(*) FROM {RUNTIME}.experiment_jobs jobs
                 WHERE jobs.run_id = NEW.run_id AND jobs.compiler_unit_id = units.compiler_unit_id
                   AND jobs.stage = 'compile' AND jobs.status = 'succeeded') != 1
                OR NOT EXISTS (SELECT 1 FROM {RUNTIME}.compiler_unit_memories members
                    WHERE members.compiler_unit_id = units.compiler_unit_id AND members.role = 'anchor')
                OR COALESCE((SELECT min(ordinal) FROM {RUNTIME}.compiler_unit_memories members
                    WHERE members.compiler_unit_id = units.compiler_unit_id), -1) != 0
                OR COALESCE((SELECT max(ordinal) FROM {RUNTIME}.compiler_unit_memories members
                    WHERE members.compiler_unit_id = units.compiler_unit_id), -1) !=
                   (SELECT count(*) - 1 FROM {RUNTIME}.compiler_unit_memories members
                    WHERE members.compiler_unit_id = units.compiler_unit_id));
            IF bad != 0 THEN
                RAISE EXCEPTION 'compiler jobs or unit membership are incomplete' USING ERRCODE = '55000';
            END IF;
            required := CASE compiler.target_stage
                WHEN 'questions' THEN ARRAY['questioner']::text[]
                WHEN 'programs' THEN ARRAY['questioner','planner','dry_run']::text[]
                ELSE ARRAY['questioner','planner','dry_run','doubter','decision']::text[] END;
            SELECT count(*) INTO bad FROM {RUNTIME}.compiler_units units
             WHERE units.run_id = NEW.run_id AND EXISTS (
                SELECT 1 FROM unnest(required) stage_name WHERE NOT EXISTS (
                    SELECT 1 FROM {RUNTIME}.compiler_unit_stage_results results
                     WHERE results.compiler_unit_id = units.compiler_unit_id
                       AND results.stage = stage_name AND results.status = 'success'));
            IF bad != 0 THEN
                RAISE EXCEPTION 'compiler target stage results are incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad
              FROM {RUNTIME}.compiler_unit_stage_results results
              JOIN {RUNTIME}.compiler_units units USING (compiler_unit_id, run_id)
             WHERE units.run_id = NEW.run_id AND results.status = 'success'
               AND results.stage = ANY(required)
               AND (
                  (results.stage = 'questioner' AND compiler.compiler_schema_version = '0.1.0'
                    AND results.produced_count != (SELECT count(*) FROM {RUNTIME}.qbank_questions facts
                        WHERE facts.run_id = NEW.run_id AND facts.compiler_unit_id = units.compiler_unit_id))
                  OR (results.stage = 'questioner' AND compiler.compiler_schema_version = '0.2.0'
                    AND (results.produced_count, results.accepted_count, results.rejected_count) != (
                        SELECT generations.raw_candidate_count,
                            (SELECT count(*) FROM {RUNTIME}.questioner_candidate_gate_results gates
                              WHERE gates.run_id = NEW.run_id AND gates.compiler_unit_id = units.compiler_unit_id
                                AND gates.status = 'eligible_for_planning'),
                            generations.invalid_candidate_count +
                            (SELECT count(*) FROM {RUNTIME}.questioner_candidate_gate_results gates
                              WHERE gates.run_id = NEW.run_id AND gates.compiler_unit_id = units.compiler_unit_id
                                AND gates.status = 'rejected')
                          FROM {RUNTIME}.questioner_generations generations
                         WHERE generations.run_id = NEW.run_id
                           AND generations.compiler_unit_id = units.compiler_unit_id
                           AND generations.parse_status != 'failed'))
                  OR (results.stage = 'planner' AND results.produced_count !=
                        (SELECT count(*) FROM {RUNTIME}.memory_programs facts
                          WHERE facts.run_id = NEW.run_id AND facts.compiler_unit_id = units.compiler_unit_id))
                  OR (results.stage = 'dry_run' AND results.produced_count !=
                        (SELECT count(*) FROM {RUNTIME}.program_executions facts
                          WHERE facts.source_compiler_run_id = NEW.run_id
                            AND facts.compiler_unit_id = units.compiler_unit_id
                            AND facts.purpose = 'compiler_dry_run'))
                  OR (results.stage = 'doubter' AND results.produced_count !=
                        (SELECT count(*) FROM {RUNTIME}.doubter_feedback facts
                          WHERE facts.run_id = NEW.run_id AND facts.compiler_unit_id = units.compiler_unit_id))
                  OR (results.stage = 'decision' AND results.produced_count !=
                        (SELECT count(*) FROM {RUNTIME}.qbank_decisions facts
                          WHERE facts.run_id = NEW.run_id AND facts.compiler_unit_id = units.compiler_unit_id)));
            IF bad != 0 THEN
                RAISE EXCEPTION 'compiler stage counts differ from persisted facts' USING ERRCODE = '55000';
            END IF;
            IF compiler.compiler_schema_version = '0.2.0' THEN
                IF compiler.questioner_config_hash IS NULL OR compiler.questioner_model_alias IS NULL
                   OR compiler.questioner_model_revision IS NULL OR compiler.questioner_prompt_hash IS NULL
                   OR compiler.candidate_gate_version IS NULL OR compiler.qbank_matcher_version IS NULL THEN
                    RAISE EXCEPTION 'Questioner compiler identity is incomplete' USING ERRCODE = '55000';
                END IF;
                SELECT count(*) INTO bad FROM {RUNTIME}.compiler_units units
                 WHERE units.run_id = NEW.run_id AND (
                    (SELECT count(*) FROM {RUNTIME}.questioner_generations generations
                      WHERE generations.run_id = NEW.run_id
                        AND generations.compiler_unit_id = units.compiler_unit_id
                        AND generations.parse_status != 'failed') != 1
                    OR EXISTS (SELECT 1 FROM {RUNTIME}.qbank_questions questions
                      WHERE questions.run_id = NEW.run_id
                        AND questions.compiler_unit_id = units.compiler_unit_id
                        AND NOT EXISTS (SELECT 1 FROM {RUNTIME}.questioner_candidate_gate_results gates
                          WHERE gates.qbank_id = questions.qbank_id AND gates.run_id = questions.run_id))
                    OR (SELECT count(*) FROM {RUNTIME}.compiler_unit_links links
                         WHERE links.compiler_unit_id = units.compiler_unit_id) !=
                       (SELECT count(*) FROM {RUNTIME}.compiler_unit_memories members
                         WHERE members.compiler_unit_id = units.compiler_unit_id AND members.role = 'support'));
                IF bad != 0 THEN
                    RAISE EXCEPTION 'Questioner generation, gate, or unit-link facts are incomplete' USING ERRCODE = '55000';
                END IF;
                IF (SELECT count(*) FROM {RUNTIME}.qbank_index_manifests indexes
                    WHERE indexes.compiler_run_id = NEW.run_id AND indexes.status = 'ready') != 1 THEN
                    RAISE EXCEPTION 'Questioner compiler requires one ready QBank index manifest' USING ERRCODE = '55000';
                END IF;
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER compiler_completion_integrity BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_questioner_compiler_completion()"
    )


def _create_evaluation_guards() -> None:
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.enforce_qbank_coverage_writable()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {EVALUATION} AS $$
        DECLARE source_id text; state text; payload jsonb;
        BEGIN
            payload := CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) ELSE to_jsonb(NEW) END;
            source_id := payload ->> 'coverage_run_id';
            IF source_id IS NULL AND payload ->> 'gold_group_id' IS NOT NULL THEN
                SELECT coverage_run_id INTO source_id
                  FROM {EVALUATION}.qbank_coverage_gold_groups
                 WHERE gold_group_id = payload ->> 'gold_group_id';
            END IF;
            IF source_id IS NULL AND payload ->> 'coverage_candidate_id' IS NOT NULL THEN
                SELECT coverage_run_id INTO source_id
                  FROM {EVALUATION}.qbank_coverage_candidates
                 WHERE coverage_candidate_id = payload ->> 'coverage_candidate_id';
            END IF;
            SELECT status INTO state FROM {EVALUATION}.qbank_coverage_runs
             WHERE coverage_run_id = source_id;
            IF state = 'completed' THEN
                RAISE EXCEPTION 'completed QBank coverage run is immutable' USING ERRCODE = '55000';
            END IF;
            IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.validate_qbank_coverage_candidate()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {EVALUATION}, {RUNTIME} AS $$
        DECLARE compiler_run text; run_snapshot text; query_conversation text;
                query_cutoff integer; question_cutoff integer; question_run text;
                question_snapshot text; question_conversation text; question_rank integer;
        BEGIN
            SELECT runs.compiler_run_id, runs.snapshot_id, queries.conversation_id,
                   cutoff.session_order
              INTO compiler_run, run_snapshot, query_conversation, query_cutoff
              FROM {EVALUATION}.qbank_coverage_runs runs
              JOIN {EVALUATION}.qbank_coverage_queries queries
                ON queries.coverage_run_id = runs.coverage_run_id
              JOIN {RUNTIME}.benchmark_sessions cutoff
                ON cutoff.snapshot_id = queries.snapshot_id
               AND cutoff.conversation_id = queries.conversation_id
               AND cutoff.session_id = queries.cutoff_session_id
             WHERE runs.coverage_run_id = NEW.coverage_run_id
               AND queries.query_id = NEW.query_id;
            SELECT questions.run_id, questions.snapshot_id, questions.conversation_id,
                   questions.generation_rank, available.session_order
              INTO question_run, question_snapshot, question_conversation,
                   question_rank, question_cutoff
              FROM {RUNTIME}.qbank_questions questions
              JOIN {RUNTIME}.benchmark_sessions available
                ON available.snapshot_id = questions.snapshot_id
               AND available.conversation_id = questions.conversation_id
               AND available.session_id = questions.created_after_session
              JOIN {RUNTIME}.questioner_candidate_gate_results gates
                ON gates.qbank_id = questions.qbank_id AND gates.run_id = questions.run_id
             WHERE questions.qbank_id = NEW.qbank_id
               AND gates.status = 'eligible_for_planning';
            IF compiler_run IS NULL OR question_run != compiler_run
               OR question_snapshot != run_snapshot OR question_conversation != query_conversation
               OR question_cutoff > query_cutoff OR question_rank > NEW.question_budget THEN
                RAISE EXCEPTION 'QBank coverage candidate violates run, conversation, cutoff, or budget' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {EVALUATION}.enforce_qbank_coverage_completion()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {EVALUATION} AS $$
        DECLARE bad integer; query_total integer; applicable_total integer; report record;
        BEGIN
            IF OLD.status = 'completed' THEN
                RAISE EXCEPTION 'completed QBank coverage run is immutable' USING ERRCODE = '55000';
            END IF;
            IF NEW.status != 'completed' THEN RETURN NEW; END IF;
            SELECT count(*), count(*) FILTER (WHERE applicable)
              INTO query_total, applicable_total
              FROM {EVALUATION}.qbank_coverage_queries WHERE coverage_run_id = NEW.coverage_run_id;
            IF query_total = 0 OR COALESCE((SELECT max(ordinal)
                FROM {EVALUATION}.qbank_coverage_queries WHERE coverage_run_id = NEW.coverage_run_id), -1)
                != query_total - 1 THEN
                RAISE EXCEPTION 'QBank coverage query denominator is incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT * INTO report FROM {EVALUATION}.qbank_coverage_gate_reports
             WHERE coverage_run_id = NEW.coverage_run_id;
            IF NOT FOUND OR NOT report.hard_integrity_passed
               OR report.query_count != query_total
               OR report.applicable_query_count != applicable_total
               OR (NEW.gate_mode = 'enforce' AND NOT report.scientific_thresholds_passed) THEN
                RAISE EXCEPTION 'QBank coverage gate report is incomplete or failed' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad FROM {EVALUATION}.qbank_coverage_queries queries
             CROSS JOIN unnest(NEW.question_budgets) budget
             CROSS JOIN unnest(NEW.match_ks) metric_k
             WHERE queries.coverage_run_id = NEW.coverage_run_id AND EXISTS (
                SELECT 1 FROM unnest(ARRAY['qbank_evidence_recall_at_k','qbank_cover_at_k','qbank_match_at_k','projected_fallback_at_k']) metric_name
                 WHERE NOT EXISTS (SELECT 1 FROM {EVALUATION}.qbank_coverage_metrics metrics
                    WHERE metrics.coverage_run_id = NEW.coverage_run_id
                      AND metrics.query_id = queries.query_id
                      AND metrics.question_budget = budget AND metrics.k = metric_k
                      AND metrics.metric_name = metric_name));
            IF bad != 0 THEN
                RAISE EXCEPTION 'QBank coverage metric grid is incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad FROM {EVALUATION}.qbank_coverage_queries queries
             CROSS JOIN unnest(NEW.question_budgets) budget
             WHERE queries.coverage_run_id = NEW.coverage_run_id AND EXISTS (
                SELECT 1 FROM unnest(ARRAY['oracle_qbank_cover','oracle_qbank_match']) metric_name
                 WHERE NOT EXISTS (SELECT 1 FROM {EVALUATION}.qbank_coverage_metrics metrics
                    WHERE metrics.coverage_run_id = NEW.coverage_run_id
                      AND metrics.query_id = queries.query_id
                      AND metrics.question_budget = budget AND metrics.k IS NULL
                      AND metrics.metric_name = metric_name));
            IF bad != 0 THEN
                RAISE EXCEPTION 'QBank oracle metric grid is incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad FROM {EVALUATION}.qbank_coverage_queries queries
             WHERE queries.coverage_run_id = NEW.coverage_run_id AND (
                (SELECT count(DISTINCT embeddings.channel)
                   FROM {EVALUATION}.qbank_coverage_query_embeddings embeddings
                  WHERE embeddings.coverage_run_id = NEW.coverage_run_id
                    AND embeddings.query_id = queries.query_id
                    AND embeddings.dimension = 4096) !=
                    CASE WHEN queries.has_query_image THEN 2 ELSE 1 END
                OR EXISTS (SELECT 1
                   FROM {EVALUATION}.qbank_coverage_query_embeddings embeddings
                  WHERE embeddings.coverage_run_id = NEW.coverage_run_id
                    AND embeddings.query_id = queries.query_id
                    AND (embeddings.dimension != 4096
                      OR embeddings.channel NOT IN ('text_semantic','multimodal_semantic')))
                OR (SELECT count(*)
                      FROM {EVALUATION}.qbank_coverage_gold_groups groups
                     WHERE groups.coverage_run_id = NEW.coverage_run_id
                       AND groups.query_id = queries.query_id) != queries.gold_group_count
                OR EXISTS (SELECT 1 FROM {EVALUATION}.qbank_coverage_metrics metrics
                     WHERE metrics.coverage_run_id = NEW.coverage_run_id
                       AND metrics.query_id = queries.query_id
                       AND metrics.applicable != queries.applicable));
            IF bad != 0 THEN
                RAISE EXCEPTION 'QBank coverage embedding, gold, or applicability facts are incomplete' USING ERRCODE = '55000';
            END IF;
            SELECT count(*) INTO bad FROM (
                SELECT coverage_run_id, query_id, question_budget, count(*) AS count,
                    min(rank) AS min_rank, max(rank) AS max_rank
                FROM {EVALUATION}.qbank_coverage_candidates
                WHERE coverage_run_id = NEW.coverage_run_id
                GROUP BY coverage_run_id, query_id, question_budget
            ) ranked WHERE min_rank != 1 OR max_rank != count OR count > 50;
            IF bad != 0 THEN
                RAISE EXCEPTION 'QBank coverage candidate ranks are not contiguous' USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END; $$"""
    )
    op.execute(
        f"CREATE TRIGGER validate_qbank_coverage_candidate BEFORE INSERT OR UPDATE ON {EVALUATION}.qbank_coverage_candidates FOR EACH ROW EXECUTE FUNCTION {EVALUATION}.validate_qbank_coverage_candidate()"
    )
    op.execute(
        f"CREATE TRIGGER qbank_coverage_completion BEFORE UPDATE ON {EVALUATION}.qbank_coverage_runs FOR EACH ROW EXECUTE FUNCTION {EVALUATION}.enforce_qbank_coverage_completion()"
    )
    for table in (
        "qbank_coverage_queries",
        "qbank_coverage_query_embeddings",
        "qbank_coverage_candidates",
        "qbank_coverage_gold_groups",
        "qbank_coverage_gold_group_members",
        "qbank_coverage_matches",
        "qbank_coverage_metrics",
        "qbank_coverage_labels",
        "qbank_coverage_gate_reports",
    ):
        op.execute(
            f"CREATE TRIGGER freeze_completed_coverage BEFORE INSERT OR UPDATE OR DELETE ON {EVALUATION}.{table} FOR EACH ROW EXECUTE FUNCTION {EVALUATION}.enforce_qbank_coverage_writable()"
        )


def _grant_permissions() -> None:
    runtime_tables = (
        "compiler_unit_links",
        "questioner_generations",
        "questioner_candidate_gate_results",
        "qbank_duplicate_links",
        "qbank_index_manifests",
    )
    qualified_runtime = ", ".join(f"{RUNTIME}.{table}" for table in runtime_tables)
    op.execute(f"GRANT SELECT ON {qualified_runtime} TO pmmc_runtime, pmmc_evaluator")
    op.execute(f"GRANT INSERT ON {qualified_runtime} TO pmmc_runtime")
    op.execute(
        f"GRANT UPDATE (status) ON {RUNTIME}.qbank_index_manifests TO pmmc_runtime"
    )
    evaluation_tables = (
        "qbank_coverage_runs",
        "qbank_coverage_queries",
        "qbank_coverage_query_embeddings",
        "qbank_coverage_candidates",
        "qbank_coverage_gold_groups",
        "qbank_coverage_gold_group_members",
        "qbank_coverage_matches",
        "qbank_coverage_metrics",
        "qbank_coverage_labels",
        "qbank_coverage_gate_reports",
    )
    qualified_evaluation = ", ".join(
        f"{EVALUATION}.{table}" for table in evaluation_tables
    )
    op.execute(f"GRANT SELECT, INSERT ON {qualified_evaluation} TO pmmc_evaluator")
    op.execute(
        f"GRANT UPDATE (status, completed_at) ON {EVALUATION}.qbank_coverage_runs TO pmmc_evaluator"
    )
    for function in (
        f"{RUNTIME}.validate_questioner_fact_scope()",
        f"{RUNTIME}.validate_qbank_index_manifest()",
        f"{RUNTIME}.enforce_qbank_index_writable()",
        f"{RUNTIME}.enforce_questioner_compiler_completion()",
        f"{EVALUATION}.enforce_qbank_coverage_writable()",
        f"{EVALUATION}.validate_qbank_coverage_candidate()",
        f"{EVALUATION}.enforce_qbank_coverage_completion()",
    ):
        op.execute(f"REVOKE ALL ON FUNCTION {function} FROM PUBLIC")


def downgrade() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS compiler_completion_integrity ON {RUNTIME}.run_states"
    )
    op.execute(
        f"CREATE TRIGGER compiler_completion_integrity BEFORE UPDATE ON {RUNTIME}.run_states FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_compiler_completion()"
    )
    for table in (
        "qbank_coverage_gate_reports",
        "qbank_coverage_labels",
        "qbank_coverage_metrics",
        "qbank_coverage_matches",
        "qbank_coverage_gold_group_members",
        "qbank_coverage_gold_groups",
        "qbank_coverage_candidates",
        "qbank_coverage_query_embeddings",
        "qbank_coverage_queries",
    ):
        op.execute(
            f"DROP TRIGGER IF EXISTS freeze_completed_coverage ON {EVALUATION}.{table}"
        )
    op.execute(
        f"DROP TRIGGER IF EXISTS qbank_coverage_completion ON {EVALUATION}.qbank_coverage_runs"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_qbank_coverage_candidate ON {EVALUATION}.qbank_coverage_candidates"
    )
    for function in (
        "enforce_qbank_coverage_completion()",
        "validate_qbank_coverage_candidate()",
        "enforce_qbank_coverage_writable()",
    ):
        op.execute(f"DROP FUNCTION IF EXISTS {EVALUATION}.{function}")
    op.execute(
        f"DROP FUNCTION IF EXISTS {RUNTIME}.enforce_questioner_compiler_completion()"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS freeze_terminal_run ON {RUNTIME}.qbank_index_manifests"
    )
    for table in (
        "qbank_duplicate_links",
        "questioner_candidate_gate_results",
        "questioner_generations",
        "compiler_unit_links",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS freeze_terminal_run ON {RUNTIME}.{table}")
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_qbank_index ON {RUNTIME}.qbank_index_manifests"
    )
    for table in (
        "questioner_candidate_gate_results",
        "questioner_generations",
        "compiler_unit_links",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS validate_questioner_fact ON {RUNTIME}.{table}")
    for function in (
        "enforce_qbank_index_writable()",
        "validate_qbank_index_manifest()",
        "validate_questioner_fact_scope()",
    ):
        op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.{function}")

    for table in (
        "qbank_coverage_gate_reports",
        "qbank_coverage_labels",
        "qbank_coverage_metrics",
        "qbank_coverage_matches",
        "qbank_coverage_gold_group_members",
        "qbank_coverage_gold_groups",
        "qbank_coverage_candidates",
        "qbank_coverage_query_embeddings",
        "qbank_coverage_queries",
        "qbank_coverage_runs",
    ):
        op.execute(f"DROP TABLE IF EXISTS {EVALUATION}.{table}")

    op.drop_constraint(
        "fk_qbank_questions_generation",
        "qbank_questions",
        schema=RUNTIME,
        type_="foreignkey",
    )
    for table in (
        "qbank_index_manifests",
        "qbank_duplicate_links",
        "questioner_candidate_gate_results",
        "questioner_generations",
        "compiler_unit_links",
    ):
        op.execute(f"DROP TABLE IF EXISTS {RUNTIME}.{table}")
    op.drop_constraint(
        op.f("ck_qbank_questions_reasoning_type_values"),
        "qbank_questions",
        schema=RUNTIME,
        type_="check",
    )
    for column in ("candidate_content_hash", "reasoning_type", "generation_id"):
        op.drop_column("qbank_questions", column, schema=RUNTIME)
    for column in (
        "qbank_matcher_version",
        "dedupe_version",
        "candidate_gate_version",
        "unit_builder_version",
        "questioner_parser_version",
        "questioner_renderer_version",
        "questioner_prompt_hash",
        "questioner_prompt_version",
        "questioner_model_revision",
        "questioner_model_alias",
        "questioner_config_hash",
    ):
        op.drop_column("compiler_runs", column, schema=RUNTIME)
