# ruff: noqa: E501
"""Add baseline corpus, context assembly, and retrieval-document lineage.

Revision ID: 20260714_0003
Revises: 20260713_0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260714_0003"
down_revision: str | None = "20260713_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"
EMBEDDING_SOURCE_CHECK_0002 = "embedding_collections_source_type_check"
EMBEDDING_SOURCE_CHECK_0003 = "ck_embedding_collections_source_values"
EMBEDDING_SOURCE_VALUES_0002 = (
    "memory_text",
    "memory_caption",
    "memory_image",
    "memory_image_text_pair",
    "qbank_question",
    "qbank_paraphrase",
    "query",
)
EMBEDDING_SOURCE_VALUES_0003 = EMBEDDING_SOURCE_VALUES_0002 + (
    "retrieval_document_text",
    "retrieval_document_image",
    "retrieval_document_multimodal",
)


def upgrade() -> None:
    _extend_canonical_lineage_keys()
    for statement in _table_statements():
        op.execute(statement)

    _extend_experiment_tables()
    _extend_embedding_source_constraint()

    for statement in _function_and_trigger_statements():
        op.execute(statement)
    for statement in _index_statements():
        op.execute(statement)
    for statement in _grant_statements():
        op.execute(statement)


def downgrade() -> None:
    for schema, table in (
        (RUNTIME, "context_assemblies"),
    ):
        op.execute(f"DROP TRIGGER IF EXISTS freeze_terminal_run ON {schema}.{table}")
    for table in ("context_assembly_memory_units", "context_assembly_images"):
        op.execute(f"DROP TRIGGER IF EXISTS freeze_context_links ON {RUNTIME}.{table}")
    op.execute(f"DROP TRIGGER IF EXISTS freeze_embedding_records ON {RUNTIME}.embeddings")
    op.execute(
        f"DROP TRIGGER IF EXISTS enforce_embedding_document_lineage ON {RUNTIME}.embeddings"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS enforce_retrieval_document_lineage ON {RUNTIME}.retrieval_traces"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS enforce_retrieval_batch_lineage ON {RUNTIME}.retrieval_batches"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS freeze_ready_embedding_collection ON {RUNTIME}.embedding_collections"
    )
    for table in ("retrieval_document_images", "retrieval_documents", "memory_window_units", "memory_windows"):
        op.execute(f"DROP TRIGGER IF EXISTS require_building_corpus ON {RUNTIME}.{table}")
    op.execute(f"DROP TRIGGER IF EXISTS freeze_ready_corpus ON {RUNTIME}.retrieval_corpora")

    for constraint, table, schema in (
        ("fk_final_answer_image_snapshot", "final_answer_image_ids", RUNTIME),
        ("fk_retrieval_trace_document", "retrieval_traces", RUNTIME),
        ("fk_retrieval_batch_corpus", "retrieval_batches", RUNTIME),
        ("fk_evaluation_model_call_context", "model_calls", EVALUATION),
        ("fk_runtime_model_call_context", "model_calls", RUNTIME),
        ("fk_embedding_document", "embeddings", RUNTIME),
        ("fk_embedding_collection_corpus", "embedding_collections", RUNTIME),
    ):
        op.execute(f"ALTER TABLE {schema}.{table} DROP CONSTRAINT IF EXISTS {constraint}")

    for table, column, schema in (
        ("final_answer_image_ids", "snapshot_id", RUNTIME),
        ("retrieval_traces", "retrieval_document_id", RUNTIME),
        ("retrieval_batches", "corpus_id", RUNTIME),
        ("model_calls", "context_assembly_id", EVALUATION),
        ("model_calls", "context_assembly_id", RUNTIME),
        ("embeddings", "retrieval_document_id", RUNTIME),
        ("embedding_collections", "corpus_id", RUNTIME),
    ):
        op.drop_column(table, column, schema=schema)

    for table in (
        "context_assembly_images",
        "context_assembly_memory_units",
        "context_assemblies",
        "retrieval_document_images",
        "retrieval_documents",
        "memory_window_units",
        "memory_windows",
        "retrieval_corpora",
    ):
        op.execute(f"DROP TABLE IF EXISTS {RUNTIME}.{table} CASCADE")

    op.drop_constraint(
        "uq_benchmark_queries_snapshot_query_conversation",
        "benchmark_queries",
        schema=RUNTIME,
        type_="unique",
    )
    op.drop_constraint(
        "uq_memory_units_snapshot_conversation_memory",
        "memory_units",
        schema=RUNTIME,
        type_="unique",
    )

    for function in (
        "enforce_embedding_records_building",
        "validate_embedding_document_lineage",
        "validate_retrieval_document_lineage",
        "validate_retrieval_batch_lineage",
        "freeze_ready_embedding_collection",
        "enforce_corpus_building",
        "freeze_ready_corpus",
        "enforce_context_links_writable",
    ):
        op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.{function}()")
    _restore_embedding_source_constraint()


def _extend_canonical_lineage_keys() -> None:
    op.create_unique_constraint(
        "uq_memory_units_snapshot_conversation_memory",
        "memory_units",
        ["snapshot_id", "conversation_id", "memory_id"],
        schema=RUNTIME,
    )
    op.create_unique_constraint(
        "uq_benchmark_queries_snapshot_query_conversation",
        "benchmark_queries",
        ["snapshot_id", "query_id", "conversation_id"],
        schema=RUNTIME,
    )


def _table_statements() -> tuple[str, ...]:
    return (
        f"""CREATE TABLE {RUNTIME}.retrieval_corpora (
            corpus_id varchar(128) PRIMARY KEY,
            snapshot_id varchar(128) NOT NULL REFERENCES {RUNTIME}.dataset_snapshots(snapshot_id),
            dataset varchar(32) NOT NULL CHECK (dataset IN ('memlens','memgallery','unknown')),
            status varchar(16) NOT NULL CHECK (status IN ('building','ready','failed')),
            content_hash varchar(128) NOT NULL, windows_hash varchar(128) NOT NULL,
            documents_hash varchar(128) NOT NULL,
            tokenizer_alias varchar(256) NOT NULL, tokenizer_revision varchar(256) NOT NULL,
            renderer_version varchar(128) NOT NULL, window_strategy varchar(128) NOT NULL,
            max_text_tokens integer NOT NULL CHECK (max_text_tokens > 0),
            max_images integer NOT NULL CHECK (max_images > 0),
            overlap_memory_units integer NOT NULL CHECK (overlap_memory_units >= 0),
            cross_session boolean NOT NULL, representations text[] NOT NULL,
            window_count integer NOT NULL, document_count integer NOT NULL,
            image_document_count integer NOT NULL, unique_memory_count integer NOT NULL,
            unique_image_count integer NOT NULL,
            spec_json jsonb NOT NULL, manifest_json jsonb NOT NULL,
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (corpus_id, snapshot_id),
            CHECK (window_count >= 0 AND document_count >= 0 AND image_document_count >= 0),
            CHECK (unique_memory_count >= 0 AND unique_image_count >= 0),
            CHECK (cardinality(representations) > 0),
            CHECK (representations <@ ARRAY['text_caption','text','image','multimodal']::text[])
        )""",
        f"""CREATE TABLE {RUNTIME}.memory_windows (
            window_id varchar(128) PRIMARY KEY, corpus_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, dataset varchar(32) NOT NULL,
            conversation_id varchar(256) NOT NULL, session_id varchar(256) NOT NULL,
            window_ordinal integer NOT NULL CHECK (window_ordinal >= 0),
            start_timestamp_order integer NOT NULL, end_timestamp_order integer NOT NULL,
            text_caption_token_count integer NOT NULL CHECK (text_caption_token_count >= 0),
            available_after_session varchar(256) NOT NULL,
            oversized_single_unit boolean NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (window_id, corpus_id),
            UNIQUE (window_id, snapshot_id, conversation_id),
            UNIQUE (corpus_id, session_id, window_ordinal),
            FOREIGN KEY (corpus_id, snapshot_id)
                REFERENCES {RUNTIME}.retrieval_corpora(corpus_id, snapshot_id),
            FOREIGN KEY (snapshot_id, conversation_id, session_id)
                REFERENCES {RUNTIME}.benchmark_sessions(snapshot_id, conversation_id, session_id),
            FOREIGN KEY (snapshot_id, conversation_id, available_after_session)
                REFERENCES {RUNTIME}.benchmark_sessions(snapshot_id, conversation_id, session_id),
            CHECK (start_timestamp_order >= 0 AND end_timestamp_order >= start_timestamp_order)
        )""",
        f"""CREATE TABLE {RUNTIME}.memory_window_units (
            window_id varchar(128) NOT NULL, snapshot_id varchar(128) NOT NULL,
            conversation_id varchar(256) NOT NULL, memory_id varchar(256) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (window_id, memory_id), UNIQUE (window_id, ordinal),
            FOREIGN KEY (window_id, snapshot_id, conversation_id)
                REFERENCES {RUNTIME}.memory_windows(window_id, snapshot_id, conversation_id),
            FOREIGN KEY (snapshot_id, conversation_id, memory_id)
                REFERENCES {RUNTIME}.memory_units(snapshot_id, conversation_id, memory_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.retrieval_documents (
            document_id varchar(128) PRIMARY KEY, corpus_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, dataset varchar(32) NOT NULL,
            conversation_id varchar(256) NOT NULL, session_id varchar(256) NOT NULL,
            parent_window_id varchar(128) NOT NULL,
            representation varchar(32) NOT NULL CHECK
                (representation IN ('text_caption','text','image','multimodal')),
            memory_ids text[] NOT NULL, rendered_text text, token_count integer,
            available_after_session varchar(256) NOT NULL, content_hash varchar(128) NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (document_id, snapshot_id),
            FOREIGN KEY (corpus_id, snapshot_id)
                REFERENCES {RUNTIME}.retrieval_corpora(corpus_id, snapshot_id),
            FOREIGN KEY (parent_window_id, corpus_id)
                REFERENCES {RUNTIME}.memory_windows(window_id, corpus_id),
            CHECK (cardinality(memory_ids) > 0),
            CHECK (token_count IS NULL OR token_count >= 0)
        )""",
        f"""CREATE TABLE {RUNTIME}.retrieval_document_images (
            document_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, image_id varchar(256) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (document_id, image_id), UNIQUE (document_id, ordinal),
            FOREIGN KEY (document_id, snapshot_id)
                REFERENCES {RUNTIME}.retrieval_documents(document_id, snapshot_id),
            FOREIGN KEY (snapshot_id, image_id) REFERENCES {RUNTIME}.images(snapshot_id, image_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.context_assemblies (
            context_assembly_id varchar(128) PRIMARY KEY, run_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, query_id varchar(256) NOT NULL,
            conversation_id varchar(256) NOT NULL, job_id varchar(128),
            method varchar(128) NOT NULL, representation varchar(32) NOT NULL,
            truncation_policy varchar(128) NOT NULL,
            tokenizer_alias varchar(256) NOT NULL, tokenizer_revision varchar(256) NOT NULL,
            total_context_tokens integer NOT NULL CHECK (total_context_tokens > 0),
            reserved_output_tokens integer NOT NULL CHECK (reserved_output_tokens > 0),
            max_input_tokens integer NOT NULL, instruction_tokens integer NOT NULL,
            query_tokens integer NOT NULL, query_image_tokens integer NOT NULL,
            memory_text_tokens integer NOT NULL, memory_image_tokens integer NOT NULL,
            request_token_adjustment integer NOT NULL,
            input_tokens integer NOT NULL, estimated boolean NOT NULL,
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            UNIQUE (context_assembly_id, run_id),
            UNIQUE (context_assembly_id, snapshot_id),
            UNIQUE (context_assembly_id, snapshot_id, conversation_id),
            FOREIGN KEY (run_id, snapshot_id, query_id)
                REFERENCES {RUNTIME}.run_queries(run_id, snapshot_id, query_id),
            FOREIGN KEY (job_id, run_id)
                REFERENCES {RUNTIME}.experiment_jobs(job_id, run_id),
            FOREIGN KEY (snapshot_id, query_id, conversation_id)
                REFERENCES {RUNTIME}.benchmark_queries(snapshot_id, query_id, conversation_id),
            CHECK (max_input_tokens = total_context_tokens - reserved_output_tokens),
            CHECK (instruction_tokens >= 0 AND query_tokens >= 0 AND query_image_tokens >= 0),
            CHECK (memory_text_tokens >= 0 AND memory_image_tokens >= 0),
            CHECK (input_tokens = instruction_tokens + query_tokens + query_image_tokens
                + memory_text_tokens + memory_image_tokens + request_token_adjustment),
            CHECK (input_tokens >= 0 AND input_tokens <= max_input_tokens)
        )""",
        f"""CREATE TABLE {RUNTIME}.context_assembly_memory_units (
            context_assembly_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, conversation_id varchar(256) NOT NULL,
            memory_id varchar(256) NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0), included boolean NOT NULL,
            text_tokens integer NOT NULL CHECK (text_tokens >= 0),
            image_tokens integer NOT NULL CHECK (image_tokens >= 0),
            PRIMARY KEY (context_assembly_id, memory_id),
            UNIQUE (context_assembly_id, ordinal),
            FOREIGN KEY (context_assembly_id, snapshot_id, conversation_id)
                REFERENCES {RUNTIME}.context_assemblies(
                    context_assembly_id, snapshot_id, conversation_id),
            FOREIGN KEY (snapshot_id, conversation_id, memory_id)
                REFERENCES {RUNTIME}.memory_units(snapshot_id, conversation_id, memory_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.context_assembly_images (
            context_assembly_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, image_id varchar(256) NOT NULL,
            role varchar(16) NOT NULL CHECK (role IN ('included','truncated','query')),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (context_assembly_id, image_id, role),
            UNIQUE (context_assembly_id, role, ordinal),
            FOREIGN KEY (context_assembly_id, snapshot_id)
                REFERENCES {RUNTIME}.context_assemblies(context_assembly_id, snapshot_id),
            FOREIGN KEY (snapshot_id, image_id) REFERENCES {RUNTIME}.images(snapshot_id, image_id)
        )""",
    )


def _extend_experiment_tables() -> None:
    op.add_column("embedding_collections", sa.Column("corpus_id", sa.String(128)), schema=RUNTIME)
    op.add_column("embeddings", sa.Column("retrieval_document_id", sa.String(128)), schema=RUNTIME)
    op.add_column("model_calls", sa.Column("context_assembly_id", sa.String(128)), schema=RUNTIME)
    op.add_column("model_calls", sa.Column("context_assembly_id", sa.String(128)), schema=EVALUATION)
    op.add_column("retrieval_batches", sa.Column("corpus_id", sa.String(128)), schema=RUNTIME)
    op.add_column("retrieval_traces", sa.Column("retrieval_document_id", sa.String(128)), schema=RUNTIME)
    op.add_column("final_answer_image_ids", sa.Column("snapshot_id", sa.String(128)), schema=RUNTIME)
    op.execute(
        f"""UPDATE {RUNTIME}.final_answer_image_ids AS links
            SET snapshot_id = answers.snapshot_id
            FROM {RUNTIME}.final_answers AS answers
            WHERE answers.answer_id = links.answer_id"""
    )
    op.alter_column("final_answer_image_ids", "snapshot_id", nullable=False, schema=RUNTIME)

    op.create_foreign_key(
        "fk_embedding_collection_corpus",
        "embedding_collections",
        "retrieval_corpora",
        ["corpus_id", "snapshot_id"],
        ["corpus_id", "snapshot_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_embedding_document",
        "embeddings",
        "retrieval_documents",
        ["retrieval_document_id"],
        ["document_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    for schema, name in (
        (RUNTIME, "fk_runtime_model_call_context"),
        (EVALUATION, "fk_evaluation_model_call_context"),
    ):
        op.create_foreign_key(
            name,
            "model_calls",
            "context_assemblies",
            ["context_assembly_id", "run_id"],
            ["context_assembly_id", "run_id"],
            source_schema=schema,
            referent_schema=RUNTIME,
        )
    op.create_foreign_key(
        "fk_retrieval_batch_corpus",
        "retrieval_batches",
        "retrieval_corpora",
        ["corpus_id", "snapshot_id"],
        ["corpus_id", "snapshot_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_retrieval_trace_document",
        "retrieval_traces",
        "retrieval_documents",
        ["retrieval_document_id"],
        ["document_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.create_check_constraint(
        "retrieval_document_matches_retrieved",
        "retrieval_traces",
        "retrieval_document_id IS NULL OR retrieval_document_id = retrieved_id",
        schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_final_answer_image_snapshot",
        "final_answer_image_ids",
        "images",
        ["snapshot_id", "image_id"],
        ["snapshot_id", "image_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )


def _extend_embedding_source_constraint() -> None:
    _drop_embedding_source_constraints()
    allowed_values = ",".join(f"'{value}'" for value in EMBEDDING_SOURCE_VALUES_0003)
    op.execute(
        f"ALTER TABLE {RUNTIME}.embedding_collections "
        f"ADD CONSTRAINT {EMBEDDING_SOURCE_CHECK_0003} "
        f"CHECK (source_type IN ({allowed_values}))"
    )


def _restore_embedding_source_constraint() -> None:
    _drop_embedding_source_constraints()
    allowed_values = ",".join(f"'{value}'" for value in EMBEDDING_SOURCE_VALUES_0002)
    op.execute(
        f"ALTER TABLE {RUNTIME}.embedding_collections "
        f"ADD CONSTRAINT {EMBEDDING_SOURCE_CHECK_0002} "
        f"CHECK (source_type IN ({allowed_values})) NOT VALID"
    )
    op.execute(
        f"""DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM {RUNTIME}.embedding_collections
                WHERE source_type NOT IN ({allowed_values})
            ) THEN
                ALTER TABLE {RUNTIME}.embedding_collections
                    VALIDATE CONSTRAINT {EMBEDDING_SOURCE_CHECK_0002};
            END IF;
        END $$"""
    )


def _drop_embedding_source_constraints() -> None:
    op.execute(
        f"""DO $$
        DECLARE constraint_row record;
        BEGIN
            FOR constraint_row IN
                SELECT con.conname
                FROM pg_constraint AS con
                JOIN pg_class AS rel ON rel.oid = con.conrelid
                JOIN pg_namespace AS nsp ON nsp.oid = rel.relnamespace
                WHERE nsp.nspname = '{RUNTIME}'
                  AND rel.relname = 'embedding_collections'
                  AND con.contype = 'c'
                  AND pg_get_constraintdef(con.oid) LIKE '%source_type%'
            LOOP
                EXECUTE format(
                    'ALTER TABLE {RUNTIME}.embedding_collections DROP CONSTRAINT %I',
                    constraint_row.conname
                );
            END LOOP;
        END $$"""
    )


def _function_and_trigger_statements() -> tuple[str, ...]:
    corpus_tables = (
        "memory_windows",
        "memory_window_units",
        "retrieval_documents",
        "retrieval_document_images",
    )
    return (
        f"""CREATE FUNCTION {RUNTIME}.freeze_ready_corpus()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                IF OLD.status = 'ready' THEN
                    RAISE EXCEPTION 'ready retrieval corpus is immutable'
                        USING ERRCODE = '55000';
                END IF;
                RETURN OLD;
            END IF;
            IF OLD.status = 'ready' THEN
                RAISE EXCEPTION 'ready retrieval corpus is immutable' USING ERRCODE = '55000';
            END IF;
            IF OLD.status != 'building' OR NEW.status NOT IN ('building','ready','failed') THEN
                RAISE EXCEPTION 'invalid retrieval corpus state transition' USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END $$""",
        f"CREATE TRIGGER freeze_ready_corpus BEFORE UPDATE OR DELETE ON {RUNTIME}.retrieval_corpora FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.freeze_ready_corpus()",
        f"""CREATE FUNCTION {RUNTIME}.enforce_corpus_building()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE target_corpus varchar(128); current_status varchar(16);
        BEGIN
            IF TG_OP = 'DELETE' THEN
                IF TG_TABLE_NAME IN ('memory_window_units','retrieval_document_images') THEN
                    IF TG_TABLE_NAME = 'memory_window_units' THEN
                        SELECT corpus_id INTO target_corpus FROM {RUNTIME}.memory_windows
                            WHERE window_id = OLD.window_id;
                    ELSE
                        SELECT corpus_id INTO target_corpus FROM {RUNTIME}.retrieval_documents
                            WHERE document_id = OLD.document_id;
                    END IF;
                ELSE
                    target_corpus := OLD.corpus_id;
                END IF;
            ELSIF TG_TABLE_NAME IN ('memory_window_units','retrieval_document_images') THEN
                IF TG_TABLE_NAME = 'memory_window_units' THEN
                    SELECT corpus_id INTO target_corpus FROM {RUNTIME}.memory_windows
                        WHERE window_id = NEW.window_id;
                ELSE
                    SELECT corpus_id INTO target_corpus FROM {RUNTIME}.retrieval_documents
                        WHERE document_id = NEW.document_id;
                END IF;
            ELSE
                target_corpus := NEW.corpus_id;
            END IF;
            SELECT status INTO current_status FROM {RUNTIME}.retrieval_corpora
                WHERE corpus_id = target_corpus FOR SHARE;
            IF TG_OP = 'DELETE' AND current_status IN ('building','failed') THEN
                RETURN OLD;
            END IF;
            IF current_status != 'building' THEN
                RAISE EXCEPTION 'retrieval corpus rows are writable only while building'
                    USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END $$""",
        *tuple(
            f"CREATE TRIGGER require_building_corpus BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.{table} FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_corpus_building()"
            for table in corpus_tables
        ),
        f"""CREATE FUNCTION {RUNTIME}.freeze_ready_embedding_collection()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                IF OLD.status = 'ready' THEN
                    RAISE EXCEPTION 'ready embedding collection is immutable'
                        USING ERRCODE = '55000';
                END IF;
                RETURN OLD;
            END IF;
            IF OLD.status = 'ready' THEN
                RAISE EXCEPTION 'ready embedding collection is immutable' USING ERRCODE = '55000';
            END IF;
            IF OLD.status != 'building' OR NEW.status NOT IN ('building','ready','failed') THEN
                RAISE EXCEPTION 'invalid embedding collection state transition' USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END $$""",
        f"CREATE TRIGGER freeze_ready_embedding_collection BEFORE UPDATE OR DELETE ON {RUNTIME}.embedding_collections FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.freeze_ready_embedding_collection()",
        f"""CREATE FUNCTION {RUNTIME}.enforce_embedding_records_building()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE target_collection varchar(128); current_status varchar(16);
        BEGIN
            IF TG_OP = 'DELETE' THEN
                target_collection := OLD.collection_id;
            ELSE
                target_collection := NEW.collection_id;
            END IF;
            SELECT status INTO current_status FROM {RUNTIME}.embedding_collections
                WHERE collection_id = target_collection FOR SHARE;
            IF TG_OP = 'DELETE' AND current_status IN ('building','failed') THEN
                RETURN OLD;
            END IF;
            IF current_status != 'building' THEN
                RAISE EXCEPTION 'embedding records are writable only while collection is building'
                    USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END $$""",
        f"CREATE TRIGGER freeze_embedding_records BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.embeddings FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_embedding_records_building()",
        f"""CREATE FUNCTION {RUNTIME}.validate_embedding_document_lineage()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE collection_source varchar(64); collection_corpus varchar(128);
                document_corpus varchar(128);
        BEGIN
            SELECT source_type, corpus_id INTO collection_source, collection_corpus
            FROM {RUNTIME}.embedding_collections
            WHERE collection_id = NEW.collection_id;
            IF collection_source LIKE 'retrieval_document_%' THEN
                IF NEW.retrieval_document_id IS NULL
                   OR NEW.source_id != NEW.retrieval_document_id THEN
                    RAISE EXCEPTION 'retrieval-document embedding requires matching source lineage'
                        USING ERRCODE = '23514';
                END IF;
                SELECT corpus_id INTO document_corpus FROM {RUNTIME}.retrieval_documents
                    WHERE document_id = NEW.retrieval_document_id;
                IF document_corpus IS NULL OR document_corpus != collection_corpus THEN
                    RAISE EXCEPTION 'embedding document belongs to a different corpus'
                        USING ERRCODE = '23514';
                END IF;
            ELSIF NEW.retrieval_document_id IS NOT NULL THEN
                RAISE EXCEPTION 'non-document embedding cannot link retrieval document'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$""",
        f"CREATE TRIGGER enforce_embedding_document_lineage BEFORE INSERT OR UPDATE ON {RUNTIME}.embeddings FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_embedding_document_lineage()",
        f"""CREATE FUNCTION {RUNTIME}.validate_retrieval_batch_lineage()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE collection_corpus varchar(128); collection_snapshot varchar(128);
                collection_status varchar(16);
        BEGIN
            IF NEW.embedding_collection_id IS NULL THEN RETURN NEW; END IF;
            SELECT corpus_id, snapshot_id, status
                INTO collection_corpus, collection_snapshot, collection_status
            FROM {RUNTIME}.embedding_collections
            WHERE collection_id = NEW.embedding_collection_id;
            IF collection_status IS NULL OR collection_status != 'ready' THEN
                RAISE EXCEPTION 'retrieval batch requires a ready embedding collection'
                    USING ERRCODE = '23514';
            END IF;
            IF collection_snapshot != NEW.snapshot_id THEN
                RAISE EXCEPTION 'retrieval batch and embedding collection snapshots differ'
                    USING ERRCODE = '23514';
            END IF;
            IF collection_corpus IS NOT NULL
               AND collection_corpus IS DISTINCT FROM NEW.corpus_id THEN
                RAISE EXCEPTION 'retrieval batch and embedding collection corpora differ'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$""",
        f"CREATE TRIGGER enforce_retrieval_batch_lineage BEFORE INSERT OR UPDATE ON {RUNTIME}.retrieval_batches FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_retrieval_batch_lineage()",
        f"""CREATE FUNCTION {RUNTIME}.validate_retrieval_document_lineage()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE batch_corpus varchar(128); document_corpus varchar(128);
        BEGIN
            IF NEW.retrieval_document_id IS NULL THEN RETURN NEW; END IF;
            SELECT corpus_id INTO batch_corpus FROM {RUNTIME}.retrieval_batches
                WHERE retrieval_batch_id = NEW.retrieval_batch_id;
            SELECT corpus_id INTO document_corpus FROM {RUNTIME}.retrieval_documents
                WHERE document_id = NEW.retrieval_document_id;
            IF batch_corpus IS NULL OR document_corpus IS NULL
               OR batch_corpus != document_corpus THEN
                RAISE EXCEPTION 'retrieval document belongs to a different batch corpus'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$""",
        f"CREATE TRIGGER enforce_retrieval_document_lineage BEFORE INSERT OR UPDATE ON {RUNTIME}.retrieval_traces FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_retrieval_document_lineage()",
        f"""CREATE FUNCTION {RUNTIME}.enforce_context_links_writable()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE target_context varchar(128); current_status varchar(16);
        BEGIN
            IF TG_OP = 'DELETE' THEN
                target_context := OLD.context_assembly_id;
            ELSE
                target_context := NEW.context_assembly_id;
            END IF;
            SELECT states.status INTO current_status
            FROM {RUNTIME}.context_assemblies AS contexts
            JOIN {RUNTIME}.run_states AS states ON states.run_id = contexts.run_id
            WHERE contexts.context_assembly_id = target_context
            FOR SHARE OF states;
            IF current_status IS NULL THEN
                RAISE EXCEPTION 'context assembly does not exist'
                    USING ERRCODE = '23503';
            END IF;
            IF current_status NOT IN ('created','running') THEN
                RAISE EXCEPTION 'terminal run context links are immutable'
                    USING ERRCODE = '55000';
            END IF;
            IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
            RETURN NEW;
        END $$""",
        f"CREATE TRIGGER freeze_context_links BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.context_assembly_memory_units FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_context_links_writable()",
        f"CREATE TRIGGER freeze_context_links BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.context_assembly_images FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_context_links_writable()",
        f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.context_assemblies FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()",
    )


def _index_statements() -> tuple[str, ...]:
    return (
        f"CREATE INDEX ix_retrieval_corpora_snapshot_status ON {RUNTIME}.retrieval_corpora(snapshot_id, status)",
        f"CREATE INDEX ix_memory_windows_conversation_session ON {RUNTIME}.memory_windows(corpus_id, conversation_id, session_id, window_ordinal)",
        f"CREATE INDEX ix_retrieval_documents_filter ON {RUNTIME}.retrieval_documents(corpus_id, conversation_id, representation, available_after_session)",
        f"CREATE INDEX ix_context_assemblies_run_query ON {RUNTIME}.context_assemblies(run_id, query_id)",
        f"CREATE INDEX ix_embeddings_retrieval_document ON {RUNTIME}.embeddings(retrieval_document_id)",
    )


def _grant_statements() -> tuple[str, ...]:
    return (
        f"REVOKE ALL ON FUNCTION {RUNTIME}.freeze_ready_corpus() FROM PUBLIC",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_corpus_building() FROM PUBLIC",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.freeze_ready_embedding_collection() FROM PUBLIC",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_embedding_records_building() FROM PUBLIC",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.validate_embedding_document_lineage() FROM PUBLIC",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.validate_retrieval_batch_lineage() FROM PUBLIC",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.validate_retrieval_document_lineage() FROM PUBLIC",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.enforce_context_links_writable() FROM PUBLIC",
        f"GRANT SELECT ON {RUNTIME}.retrieval_corpora, {RUNTIME}.memory_windows, {RUNTIME}.memory_window_units, {RUNTIME}.retrieval_documents, {RUNTIME}.retrieval_document_images, {RUNTIME}.context_assemblies, {RUNTIME}.context_assembly_memory_units, {RUNTIME}.context_assembly_images TO pmmc_runtime, pmmc_evaluator",
        f"GRANT INSERT ON {RUNTIME}.context_assemblies, {RUNTIME}.context_assembly_memory_units, {RUNTIME}.context_assembly_images TO pmmc_runtime",
    )
