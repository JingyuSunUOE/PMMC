# ruff: noqa: E501
"""Add window-level RAG retrieval, query vectors, and context lineage.

Revision ID: 20260715_0005
Revises: 20260714_0004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260715_0005"
down_revision: str | None = "20260714_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"

EMBEDDING_SOURCE_CHECK = "ck_embedding_collections_source_values"
MANAGED_BACKEND_CHECK_0004 = "managed_runtime_services_backend_check"
MANAGED_BACKEND_CHECK_0005 = "ck_managed_runtime_services_backend_values"
RETRIEVAL_TARGET_CHECK = "ck_retrieval_traces_retrieval_target_consistent"


def upgrade() -> None:
    _extend_enum_checks()
    _add_columns_and_constraints()
    for statement in _table_statements():
        op.execute(statement)
    for statement in _trigger_statements():
        op.execute(statement)
    for statement in _index_and_grant_statements():
        op.execute(statement)


def downgrade() -> None:
    for table in (
        "retrieval_batch_query_embeddings",
        "retrieval_trace_components",
        "context_assembly_retrieval_candidates",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS freeze_terminal_run ON {RUNTIME}.{table}")
    op.execute(
        f"DROP TRIGGER IF EXISTS freeze_context_links ON "
        f"{RUNTIME}.context_assembly_retrieval_candidates"
    )
    op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.validate_rag_retrieval_lineage() CASCADE")

    for table in (
        "context_assembly_retrieval_candidates",
        "retrieval_trace_components",
        "retrieval_batch_query_embeddings",
    ):
        op.execute(f"DROP TABLE IF EXISTS {RUNTIME}.{table} CASCADE")

    op.drop_constraint(
        "fk_context_assembly_retrieval_batch",
        "context_assemblies",
        schema=RUNTIME,
        type_="foreignkey",
    )
    op.drop_constraint(
        "uq_context_assembly_rag_identity",
        "context_assemblies",
        schema=RUNTIME,
        type_="unique",
    )
    op.drop_column("context_assemblies", "retrieval_batch_id", schema=RUNTIME)

    op.drop_constraint(
        "fk_retrieval_trace_window",
        "retrieval_traces",
        schema=RUNTIME,
        type_="foreignkey",
    )
    op.drop_constraint(
        op.f(RETRIEVAL_TARGET_CHECK),
        "retrieval_traces",
        schema=RUNTIME,
        type_="check",
    )
    op.drop_column("retrieval_traces", "retrieval_window_id", schema=RUNTIME)
    op.create_check_constraint(
        "retrieval_document_matches_retrieved",
        "retrieval_traces",
        "retrieval_document_id IS NULL OR retrieval_document_id = retrieved_id",
        schema=RUNTIME,
    )

    op.drop_constraint(
        "fk_retrieval_batch_query_collection",
        "retrieval_batches",
        schema=RUNTIME,
        type_="foreignkey",
    )
    op.drop_constraint(
        "uq_retrieval_batch_rag_identity",
        "retrieval_batches",
        schema=RUNTIME,
        type_="unique",
    )
    op.drop_column("retrieval_batches", "query_embedding_collection_id", schema=RUNTIME)
    op.drop_constraint(
        "uq_embeddings_embedding_collection",
        "embeddings",
        schema=RUNTIME,
        type_="unique",
    )

    _restore_lineage_functions()
    _restore_enum_checks()


def _extend_enum_checks() -> None:
    op.drop_constraint(
        op.f(EMBEDDING_SOURCE_CHECK),
        "embedding_collections",
        schema=RUNTIME,
        type_="check",
    )
    op.create_check_constraint(
        op.f(EMBEDDING_SOURCE_CHECK),
        "embedding_collections",
        "source_type IN ('memory_text','memory_caption','memory_image',"
        "'memory_image_text_pair','qbank_question','qbank_paraphrase','query',"
        "'retrieval_document_text_caption','retrieval_document_text',"
        "'retrieval_document_image','retrieval_document_multimodal',"
        "'retrieval_document_contrastive')",
        schema=RUNTIME,
    )
    op.drop_constraint(
        op.f(MANAGED_BACKEND_CHECK_0004),
        "managed_runtime_services",
        schema=RUNTIME,
        type_="check",
    )
    op.create_check_constraint(
        op.f(MANAGED_BACKEND_CHECK_0005),
        "managed_runtime_services",
        "backend IN ('vllm_chat','vllm_embedding','vllm_multimodal_embedding',"
        "'litellm_proxy','transformers_contrastive')",
        schema=RUNTIME,
    )


def _restore_enum_checks() -> None:
    op.drop_constraint(
        op.f(EMBEDDING_SOURCE_CHECK),
        "embedding_collections",
        schema=RUNTIME,
        type_="check",
    )
    op.create_check_constraint(
        op.f(EMBEDDING_SOURCE_CHECK),
        "embedding_collections",
        "source_type IN ('memory_text','memory_caption','memory_image',"
        "'memory_image_text_pair','qbank_question','qbank_paraphrase','query',"
        "'retrieval_document_text','retrieval_document_image',"
        "'retrieval_document_multimodal')",
        schema=RUNTIME,
    )
    op.drop_constraint(
        op.f(MANAGED_BACKEND_CHECK_0005),
        "managed_runtime_services",
        schema=RUNTIME,
        type_="check",
    )
    op.create_check_constraint(
        op.f(MANAGED_BACKEND_CHECK_0004),
        "managed_runtime_services",
        "backend IN ('vllm_chat','vllm_embedding','vllm_multimodal_embedding',"
        "'litellm_proxy')",
        schema=RUNTIME,
    )


def _add_columns_and_constraints() -> None:
    op.create_unique_constraint(
        "uq_embeddings_embedding_collection",
        "embeddings",
        ["embedding_id", "collection_id"],
        schema=RUNTIME,
    )
    op.add_column(
        "retrieval_batches",
        sa.Column("query_embedding_collection_id", sa.String(128)),
        schema=RUNTIME,
    )
    op.create_unique_constraint(
        "uq_retrieval_batch_rag_identity",
        "retrieval_batches",
        ["retrieval_batch_id", "run_id", "snapshot_id", "query_id"],
        schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_retrieval_batch_query_collection",
        "retrieval_batches",
        "embedding_collections",
        ["query_embedding_collection_id", "snapshot_id"],
        ["collection_id", "snapshot_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )

    op.drop_constraint(
        "retrieval_document_matches_retrieved",
        "retrieval_traces",
        schema=RUNTIME,
        type_="check",
    )
    op.add_column(
        "retrieval_traces",
        sa.Column("retrieval_window_id", sa.String(128)),
        schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_retrieval_trace_window",
        "retrieval_traces",
        "memory_windows",
        ["retrieval_window_id"],
        ["window_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )
    op.create_check_constraint(
        op.f(RETRIEVAL_TARGET_CHECK),
        "retrieval_traces",
        "(retrieval_window_id IS NOT NULL AND retrieval_document_id IS NOT NULL "
        "AND retrieved_id = retrieval_window_id) OR "
        "(retrieval_window_id IS NULL AND "
        "(retrieval_document_id IS NULL OR retrieved_id = retrieval_document_id))",
        schema=RUNTIME,
    )

    op.add_column(
        "context_assemblies",
        sa.Column("retrieval_batch_id", sa.String(128)),
        schema=RUNTIME,
    )
    op.create_unique_constraint(
        "uq_context_assembly_rag_identity",
        "context_assemblies",
        [
            "context_assembly_id",
            "run_id",
            "snapshot_id",
            "query_id",
            "conversation_id",
        ],
        schema=RUNTIME,
    )
    op.create_foreign_key(
        "fk_context_assembly_retrieval_batch",
        "context_assemblies",
        "retrieval_batches",
        ["retrieval_batch_id", "run_id", "snapshot_id", "query_id"],
        ["retrieval_batch_id", "run_id", "snapshot_id", "query_id"],
        source_schema=RUNTIME,
        referent_schema=RUNTIME,
    )


def _table_statements() -> tuple[str, ...]:
    return (
        f"""CREATE TABLE {RUNTIME}.retrieval_batch_query_embeddings (
            retrieval_batch_id varchar(128) NOT NULL,
            embedding_id varchar(128) NOT NULL,
            query_embedding_collection_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL, snapshot_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            role varchar(16) NOT NULL CHECK (role IN ('text','image','joint','fused')),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            used_for_search boolean NOT NULL, metadata jsonb NOT NULL,
            PRIMARY KEY (retrieval_batch_id, embedding_id),
            UNIQUE (retrieval_batch_id, role, ordinal),
            FOREIGN KEY (retrieval_batch_id, run_id, snapshot_id, query_id)
                REFERENCES {RUNTIME}.retrieval_batches(
                    retrieval_batch_id, run_id, snapshot_id, query_id),
            FOREIGN KEY (embedding_id, query_embedding_collection_id)
                REFERENCES {RUNTIME}.embeddings(embedding_id, collection_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.retrieval_trace_components (
            retrieval_trace_id varchar(128) NOT NULL,
            retrieval_document_id varchar(128) NOT NULL,
            embedding_collection_id varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL, run_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL,
            representation varchar(32) NOT NULL CHECK
                (representation IN ('text_caption','text','image','multimodal')),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            score_type varchar(128) NOT NULL, score_raw_value double precision NOT NULL,
            score_normalized_value double precision,
            score_direction varchar(32) NOT NULL CHECK
                (score_direction IN ('higher_is_better','lower_is_better')),
            normalization_method varchar(128), normalization_version varchar(128),
            score_metadata jsonb NOT NULL, is_winner boolean NOT NULL,
            metadata jsonb NOT NULL,
            PRIMARY KEY (retrieval_trace_id, retrieval_document_id),
            UNIQUE (retrieval_trace_id, ordinal),
            FOREIGN KEY (retrieval_trace_id, run_id, query_id)
                REFERENCES {RUNTIME}.retrieval_traces(retrieval_trace_id, run_id, query_id),
            FOREIGN KEY (retrieval_document_id)
                REFERENCES {RUNTIME}.retrieval_documents(document_id),
            FOREIGN KEY (embedding_collection_id, snapshot_id)
                REFERENCES {RUNTIME}.embedding_collections(collection_id, snapshot_id),
            CHECK ((score_normalized_value IS NULL AND normalization_method IS NULL) OR
                (score_normalized_value IS NOT NULL AND normalization_method IS NOT NULL)),
            CHECK (score_normalized_value IS NULL OR
                (score_normalized_value >= 0 AND score_normalized_value <= 1))
        )""",
        f"""CREATE TABLE {RUNTIME}.context_assembly_retrieval_candidates (
            context_assembly_id varchar(128) NOT NULL,
            retrieval_trace_id varchar(128) NOT NULL,
            run_id varchar(128) NOT NULL, snapshot_id varchar(128) NOT NULL,
            query_id varchar(256) NOT NULL, conversation_id varchar(256) NOT NULL,
            window_id varchar(128) NOT NULL, rank integer NOT NULL CHECK (rank > 0),
            disposition varchar(32) NOT NULL CHECK
                (disposition IN ('included','deduplicated','budget_excluded')),
            contributed_memory_ids text[] NOT NULL, metadata jsonb NOT NULL,
            PRIMARY KEY (context_assembly_id, retrieval_trace_id),
            UNIQUE (context_assembly_id, rank),
            FOREIGN KEY (context_assembly_id, run_id, snapshot_id, query_id, conversation_id)
                REFERENCES {RUNTIME}.context_assemblies(
                    context_assembly_id, run_id, snapshot_id, query_id, conversation_id),
            FOREIGN KEY (retrieval_trace_id, run_id, query_id)
                REFERENCES {RUNTIME}.retrieval_traces(retrieval_trace_id, run_id, query_id),
            FOREIGN KEY (window_id, snapshot_id, conversation_id)
                REFERENCES {RUNTIME}.memory_windows(window_id, snapshot_id, conversation_id),
            CHECK ((disposition = 'included' AND cardinality(contributed_memory_ids) > 0) OR
                (disposition != 'included' AND cardinality(contributed_memory_ids) = 0))
        )""",
    )


def _trigger_statements() -> tuple[str, ...]:
    return (
        f"""CREATE OR REPLACE FUNCTION {RUNTIME}.validate_retrieval_batch_lineage()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE document_corpus varchar(128); document_snapshot varchar(128);
                document_status varchar(16); query_snapshot varchar(128);
                query_status varchar(16); query_run varchar(128);
        BEGIN
            IF NEW.embedding_collection_id IS NOT NULL THEN
                SELECT corpus_id, snapshot_id, status
                    INTO document_corpus, document_snapshot, document_status
                FROM {RUNTIME}.embedding_collections
                WHERE collection_id = NEW.embedding_collection_id;
                IF document_status IS NULL OR document_status != 'ready' THEN
                    RAISE EXCEPTION 'retrieval batch requires a ready document collection'
                        USING ERRCODE = '23514';
                END IF;
                IF document_snapshot != NEW.snapshot_id OR
                   document_corpus IS DISTINCT FROM NEW.corpus_id THEN
                    RAISE EXCEPTION 'retrieval batch document collection lineage differs'
                        USING ERRCODE = '23514';
                END IF;
            END IF;
            IF NEW.query_embedding_collection_id IS NOT NULL THEN
                SELECT snapshot_id, status, run_id
                    INTO query_snapshot, query_status, query_run
                FROM {RUNTIME}.embedding_collections
                WHERE collection_id = NEW.query_embedding_collection_id;
                IF query_status IS NULL OR query_status NOT IN ('building','ready') OR
                   query_snapshot != NEW.snapshot_id OR query_run != NEW.run_id THEN
                    RAISE EXCEPTION 'retrieval batch query collection lineage differs'
                        USING ERRCODE = '23514';
                END IF;
            END IF;
            RETURN NEW;
        END $$""",
        f"""CREATE OR REPLACE FUNCTION {RUNTIME}.validate_retrieval_document_lineage()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE batch_corpus varchar(128); document_corpus varchar(128);
                document_window varchar(128);
        BEGIN
            IF NEW.retrieval_document_id IS NULL THEN RETURN NEW; END IF;
            SELECT corpus_id INTO batch_corpus FROM {RUNTIME}.retrieval_batches
                WHERE retrieval_batch_id = NEW.retrieval_batch_id;
            SELECT corpus_id, parent_window_id INTO document_corpus, document_window
                FROM {RUNTIME}.retrieval_documents
                WHERE document_id = NEW.retrieval_document_id;
            IF batch_corpus IS NULL OR document_corpus IS NULL OR
               batch_corpus != document_corpus THEN
                RAISE EXCEPTION 'retrieval document belongs to a different batch corpus'
                    USING ERRCODE = '23514';
            END IF;
            IF NEW.retrieval_window_id IS NOT NULL AND
               NEW.retrieval_window_id != document_window THEN
                RAISE EXCEPTION 'winning document does not belong to retrieval window'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$""",
        f"""CREATE FUNCTION {RUNTIME}.validate_rag_retrieval_lineage()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE trace_window varchar(128); document_window varchar(128);
                trace_collection varchar(128);
        BEGIN
            IF TG_TABLE_NAME = 'retrieval_trace_components' THEN
                SELECT retrieval_window_id INTO trace_window
                    FROM {RUNTIME}.retrieval_traces
                    WHERE retrieval_trace_id = NEW.retrieval_trace_id;
                SELECT parent_window_id INTO document_window
                    FROM {RUNTIME}.retrieval_documents
                    WHERE document_id = NEW.retrieval_document_id;
                SELECT embedding_collection_id INTO trace_collection
                    FROM {RUNTIME}.retrieval_batches AS batches
                    JOIN {RUNTIME}.retrieval_traces AS traces USING (retrieval_batch_id)
                    WHERE traces.retrieval_trace_id = NEW.retrieval_trace_id;
                IF trace_window IS NULL OR trace_window != document_window OR
                   trace_collection != NEW.embedding_collection_id THEN
                    RAISE EXCEPTION 'retrieval component lineage differs from its window batch'
                        USING ERRCODE = '23514';
                END IF;
            END IF;
            RETURN NEW;
        END $$""",
        f"CREATE TRIGGER enforce_rag_component_lineage BEFORE INSERT OR UPDATE ON {RUNTIME}.retrieval_trace_components FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.validate_rag_retrieval_lineage()",
        f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.retrieval_batch_query_embeddings FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()",
        f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.retrieval_trace_components FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()",
        f"CREATE TRIGGER freeze_terminal_run BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.context_assembly_retrieval_candidates FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_run_writable()",
        f"CREATE TRIGGER freeze_context_links BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.context_assembly_retrieval_candidates FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_context_links_writable()",
    )


def _index_and_grant_statements() -> tuple[str, ...]:
    tables = (
        f"{RUNTIME}.retrieval_batch_query_embeddings, "
        f"{RUNTIME}.retrieval_trace_components, "
        f"{RUNTIME}.context_assembly_retrieval_candidates"
    )
    return (
        f"CREATE UNIQUE INDEX uq_rag_query_effective_vector ON {RUNTIME}.retrieval_batch_query_embeddings(retrieval_batch_id) WHERE used_for_search",
        f"CREATE UNIQUE INDEX uq_rag_trace_winner ON {RUNTIME}.retrieval_trace_components(retrieval_trace_id) WHERE is_winner",
        f"CREATE INDEX ix_rag_trace_components_document ON {RUNTIME}.retrieval_trace_components(retrieval_document_id)",
        f"CREATE INDEX ix_rag_context_candidates_rank ON {RUNTIME}.context_assembly_retrieval_candidates(context_assembly_id, rank)",
        f"REVOKE ALL ON FUNCTION {RUNTIME}.validate_rag_retrieval_lineage() FROM PUBLIC",
        f"GRANT SELECT ON {tables} TO pmmc_runtime, pmmc_evaluator",
        f"GRANT INSERT ON {tables} TO pmmc_runtime",
        f"REVOKE UPDATE, DELETE ON {tables} FROM pmmc_runtime, pmmc_evaluator",
    )


def _restore_lineage_functions() -> None:
    op.execute(
        f"""CREATE OR REPLACE FUNCTION {RUNTIME}.validate_retrieval_batch_lineage()
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
            IF collection_snapshot != NEW.snapshot_id OR
               (collection_corpus IS NOT NULL AND
                collection_corpus IS DISTINCT FROM NEW.corpus_id) THEN
                RAISE EXCEPTION 'retrieval batch lineage differs'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$"""
    )
    op.execute(
        f"""CREATE OR REPLACE FUNCTION {RUNTIME}.validate_retrieval_document_lineage()
        RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {RUNTIME} AS $$
        DECLARE batch_corpus varchar(128); document_corpus varchar(128);
        BEGIN
            IF NEW.retrieval_document_id IS NULL THEN RETURN NEW; END IF;
            SELECT corpus_id INTO batch_corpus FROM {RUNTIME}.retrieval_batches
                WHERE retrieval_batch_id = NEW.retrieval_batch_id;
            SELECT corpus_id INTO document_corpus FROM {RUNTIME}.retrieval_documents
                WHERE document_id = NEW.retrieval_document_id;
            IF batch_corpus IS NULL OR document_corpus IS NULL OR
               batch_corpus != document_corpus THEN
                RAISE EXCEPTION 'retrieval document belongs to a different batch corpus'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$"""
    )
