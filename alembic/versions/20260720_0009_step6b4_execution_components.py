# ruff: noqa: E501
"""Add independently persisted baseline family components.

Revision ID: 20260720_0009
Revises: 20260719_0008
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260720_0009"
down_revision: str | None = "20260719_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"


def upgrade() -> None:
    _create_tables()
    _create_guards()
    _grant_permissions()


def _create_tables() -> None:
    op.execute(
        f"""CREATE TABLE {RUNTIME}.baseline_suite_families (
            family_id varchar(128) PRIMARY KEY,
            schema_version varchar(32) NOT NULL,
            partition_id varchar(128) NOT NULL,
            partition_assignment_hash varchar(128) NOT NULL,
            datasets text[] NOT NULL CHECK (cardinality(datasets) > 0),
            methods text[] NOT NULL CHECK (cardinality(methods) > 0),
            snapshot_ids jsonb NOT NULL,
            selection_hashes jsonb NOT NULL,
            selected_counts jsonb NOT NULL,
            answer_model_alias varchar(256) NOT NULL,
            answer_model_revision varchar(256) NOT NULL,
            judge_model_alias varchar(256) NOT NULL,
            judge_model_revision varchar(256) NOT NULL,
            answer_prompt_hash varchar(128) NOT NULL,
            judge_prompt_hash varchar(128) NOT NULL,
            decoding_config jsonb NOT NULL,
            total_context_tokens integer NOT NULL CHECK (total_context_tokens > 0),
            reserved_output_tokens integer NOT NULL CHECK (
                reserved_output_tokens > 0 AND reserved_output_tokens < total_context_tokens
            ),
            retrieval_top_k integer NOT NULL CHECK (retrieval_top_k > 0),
            answer_top_k integer NOT NULL CHECK (
                answer_top_k > 0 AND answer_top_k <= retrieval_top_k
            ),
            protocol_versions jsonb NOT NULL,
            config_hash varchar(128) NOT NULL,
            paper_mode boolean NOT NULL,
            status varchar(16) NOT NULL CHECK (
                status IN ('registered','assembling','ready','failed')
            ),
            expected_component_count integer NOT NULL CHECK (expected_component_count > 0),
            CHECK (
                expected_component_count = cardinality(datasets) * cardinality(methods)
            ),
            verified_component_count integer NOT NULL CHECK (
                verified_component_count >= 0
                AND verified_component_count <= expected_component_count
            ),
            created_at timestamptz NOT NULL,
            updated_at timestamptz NOT NULL,
            completed_at timestamptz,
            error_type varchar(256),
            error_summary text,
            metadata jsonb NOT NULL,
            CHECK (status != 'failed' OR error_type IS NOT NULL),
            CHECK (status != 'ready' OR completed_at IS NOT NULL),
            CHECK (NOT paper_mode OR (
                datasets @> ARRAY['memlens','memgallery']::text[]
                AND cardinality(datasets) = 2
                AND cardinality(methods) = 5
                AND expected_component_count = 10
                AND COALESCE((metadata->>'paper_result')::boolean, false) = false
            ))
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.baseline_suite_family_components (
            component_id varchar(128) PRIMARY KEY,
            family_id varchar(128) NOT NULL
                REFERENCES {RUNTIME}.baseline_suite_families(family_id),
            schema_version varchar(32) NOT NULL,
            dataset varchar(32) NOT NULL CHECK (dataset IN ('memlens','memgallery')),
            method varchar(128) NOT NULL,
            snapshot_id varchar(128) NOT NULL
                REFERENCES {RUNTIME}.dataset_snapshots(snapshot_id),
            selection_hash varchar(128) NOT NULL,
            selected_count integer NOT NULL CHECK (selected_count > 0),
            suite_id varchar(128) REFERENCES {RUNTIME}.baseline_suites(suite_id),
            run_id varchar(128) REFERENCES {RUNTIME}.run_manifests(run_id),
            corpus_id varchar(128) REFERENCES {RUNTIME}.retrieval_corpora(corpus_id),
            scientific_config_hash varchar(128) NOT NULL,
            component_config_hash varchar(128) NOT NULL,
            paper_component boolean NOT NULL,
            paper_result boolean NOT NULL CHECK (paper_result = false),
            status varchar(32) NOT NULL CHECK (status IN (
                'registered','retrieval_ready','answer_ready','exported',
                'imported','verified','failed'
            )),
            transition_ordinal integer NOT NULL CHECK (transition_ordinal >= 0),
            bundle_export_id varchar(128),
            bundle_sha256 varchar(128),
            created_at timestamptz NOT NULL,
            updated_at timestamptz NOT NULL,
            completed_at timestamptz,
            error_type varchar(256),
            error_summary text,
            metadata jsonb NOT NULL,
            UNIQUE (family_id, dataset, method),
            UNIQUE (family_id, component_id),
            CHECK (status != 'failed' OR error_type IS NOT NULL),
            CHECK (paper_component = true),
            CHECK (status NOT IN ('exported','imported','verified') OR (
                bundle_export_id IS NOT NULL AND bundle_sha256 IS NOT NULL
            )),
            CHECK (status != 'verified' OR completed_at IS NOT NULL)
        )"""
    )
    op.execute(
        f"""CREATE TABLE {RUNTIME}.baseline_component_imports (
            import_id varchar(128) PRIMARY KEY,
            family_id varchar(128) NOT NULL,
            component_id varchar(128) NOT NULL,
            bundle_export_id varchar(128) NOT NULL,
            bundle_sha256 varchar(128) NOT NULL,
            source_alembic_revision varchar(64) NOT NULL,
            importer_version varchar(128) NOT NULL,
            imported_at timestamptz NOT NULL,
            imported_by varchar(256) NOT NULL,
            metadata jsonb NOT NULL,
            UNIQUE (component_id, bundle_export_id),
            FOREIGN KEY (family_id, component_id)
                REFERENCES {RUNTIME}.baseline_suite_family_components(family_id, component_id)
        )"""
    )


def _create_guards() -> None:
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_baseline_family_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.status != 'registered' OR NEW.verified_component_count != 0 THEN
                    RAISE EXCEPTION 'new baseline families must be registered' USING ERRCODE = '55000';
                END IF;
                RETURN NEW;
            END IF;
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'baseline family manifests are immutable' USING ERRCODE = '55000';
            END IF;
            IF OLD.status IN ('ready','failed') THEN
                RAISE EXCEPTION 'terminal baseline family is immutable' USING ERRCODE = '55000';
            END IF;
            IF ROW(
                NEW.schema_version, NEW.partition_id, NEW.partition_assignment_hash,
                NEW.datasets, NEW.methods, NEW.snapshot_ids, NEW.selection_hashes,
                NEW.selected_counts, NEW.answer_model_alias, NEW.answer_model_revision,
                NEW.judge_model_alias, NEW.judge_model_revision, NEW.answer_prompt_hash,
                NEW.judge_prompt_hash, NEW.decoding_config, NEW.total_context_tokens,
                NEW.reserved_output_tokens, NEW.retrieval_top_k, NEW.answer_top_k,
                NEW.protocol_versions, NEW.config_hash, NEW.paper_mode, NEW.created_at,
                NEW.metadata
            ) IS DISTINCT FROM ROW(
                OLD.schema_version, OLD.partition_id, OLD.partition_assignment_hash,
                OLD.datasets, OLD.methods, OLD.snapshot_ids, OLD.selection_hashes,
                OLD.selected_counts, OLD.answer_model_alias, OLD.answer_model_revision,
                OLD.judge_model_alias, OLD.judge_model_revision, OLD.answer_prompt_hash,
                OLD.judge_prompt_hash, OLD.decoding_config, OLD.total_context_tokens,
                OLD.reserved_output_tokens, OLD.retrieval_top_k, OLD.answer_top_k,
                OLD.protocol_versions, OLD.config_hash, OLD.paper_mode, OLD.created_at,
                OLD.metadata
            ) THEN
                RAISE EXCEPTION 'baseline family scientific identity is immutable' USING ERRCODE = '55000';
            END IF;
            IF NOT (
                (OLD.status = NEW.status) OR
                (OLD.status = 'registered' AND NEW.status IN ('assembling','failed')) OR
                (OLD.status = 'assembling' AND NEW.status IN ('ready','failed'))
            ) THEN
                RAISE EXCEPTION 'invalid baseline family state transition' USING ERRCODE = '55000';
            END IF;
            IF NEW.status = 'ready' THEN
                SELECT count(*) INTO NEW.verified_component_count
                FROM {RUNTIME}.baseline_suite_family_components c
                WHERE c.family_id = NEW.family_id AND c.status = 'verified';
                IF NEW.verified_component_count != NEW.expected_component_count OR EXISTS (
                    SELECT 1 FROM {RUNTIME}.baseline_suite_family_components c
                    WHERE c.family_id = NEW.family_id AND c.status != 'verified'
                ) THEN
                    RAISE EXCEPTION 'baseline family is missing verified components' USING ERRCODE = '55000';
                END IF;
                IF NEW.paper_mode AND (
                    (SELECT count(*) FROM {RUNTIME}.baseline_suite_family_components c
                     WHERE c.family_id = NEW.family_id) != 10 OR
                    (SELECT count(DISTINCT c.dataset) FROM {RUNTIME}.baseline_suite_family_components c
                     WHERE c.family_id = NEW.family_id) != 2 OR
                    (SELECT count(DISTINCT c.method) FROM {RUNTIME}.baseline_suite_family_components c
                     WHERE c.family_id = NEW.family_id) != 5
                ) THEN
                    RAISE EXCEPTION 'paper baseline family must contain a complete 2x5 matrix' USING ERRCODE = '55000';
                END IF;
            END IF;
            RETURN NEW;
        END $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_baseline_component_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE family_row {RUNTIME}.baseline_suite_families%ROWTYPE;
        BEGIN
            IF TG_OP = 'DELETE' THEN
                SELECT * INTO family_row
                FROM {RUNTIME}.baseline_suite_families
                WHERE family_id = OLD.family_id;
            ELSE
                SELECT * INTO family_row
                FROM {RUNTIME}.baseline_suite_families
                WHERE family_id = NEW.family_id;
            END IF;
            IF family_row.status IN ('ready','failed') THEN
                RAISE EXCEPTION 'terminal baseline family components are immutable' USING ERRCODE = '55000';
            END IF;
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'baseline family components cannot be deleted' USING ERRCODE = '55000';
            END IF;
            IF TG_OP = 'INSERT' THEN
                IF NEW.status != 'registered' OR NEW.transition_ordinal != 0 THEN
                    RAISE EXCEPTION 'new baseline components must be registered' USING ERRCODE = '55000';
                END IF;
                IF NOT (NEW.dataset = ANY(family_row.datasets)) OR
                   NOT (NEW.method = ANY(family_row.methods)) OR
                   family_row.snapshot_ids->>NEW.dataset IS DISTINCT FROM NEW.snapshot_id OR
                   family_row.selection_hashes->>NEW.dataset IS DISTINCT FROM NEW.selection_hash OR
                   (family_row.selected_counts->>NEW.dataset)::integer IS DISTINCT FROM NEW.selected_count OR
                   family_row.config_hash IS DISTINCT FROM NEW.scientific_config_hash OR
                   NOT NEW.paper_component OR NEW.paper_result THEN
                    RAISE EXCEPTION 'baseline component identity differs from family' USING ERRCODE = '55000';
                END IF;
                RETURN NEW;
            END IF;
            IF ROW(
                NEW.family_id, NEW.schema_version, NEW.dataset, NEW.method,
                NEW.snapshot_id, NEW.selection_hash, NEW.selected_count,
                NEW.scientific_config_hash, NEW.component_config_hash,
                NEW.paper_component, NEW.paper_result, NEW.created_at
            ) IS DISTINCT FROM ROW(
                OLD.family_id, OLD.schema_version, OLD.dataset, OLD.method,
                OLD.snapshot_id, OLD.selection_hash, OLD.selected_count,
                OLD.scientific_config_hash, OLD.component_config_hash,
                OLD.paper_component, OLD.paper_result, OLD.created_at
            ) THEN
                RAISE EXCEPTION 'baseline component identity is immutable' USING ERRCODE = '55000';
            END IF;
            IF OLD.status IN ('verified','failed') THEN
                RAISE EXCEPTION 'terminal baseline component is immutable' USING ERRCODE = '55000';
            END IF;
            IF NEW.suite_id IS NOT NULL AND (
                NEW.run_id IS NULL OR NOT EXISTS (
                    SELECT 1
                    FROM {RUNTIME}.baseline_suites s
                    WHERE s.suite_id = NEW.suite_id
                      AND s.dataset = NEW.dataset
                      AND s.snapshot_id = NEW.snapshot_id
                      AND s.selection_hash = NEW.selection_hash
                      AND cardinality(s.methods) = 1
                      AND NEW.method = ANY(s.methods)
                      AND COALESCE((s.metadata->>'paper_component')::boolean, false)
                      AND COALESCE(s.metadata->>'baseline_family_id', '') = NEW.family_id
                      AND COALESCE(s.metadata->>'baseline_component_id', '') = NEW.component_id
                )
            ) THEN
                RAISE EXCEPTION 'component suite identity differs from family component' USING ERRCODE = '55000';
            END IF;
            IF (NEW.suite_id IS NULL) != (NEW.run_id IS NULL) THEN
                RAISE EXCEPTION 'component suite and run lineage must be bound together' USING ERRCODE = '55000';
            END IF;
            IF NEW.status = OLD.status
               AND NEW.transition_ordinal = OLD.transition_ordinal
               AND OLD.status IN ('registered','retrieval_ready')
               AND NEW.bundle_export_id IS NOT DISTINCT FROM OLD.bundle_export_id
               AND NEW.bundle_sha256 IS NOT DISTINCT FROM OLD.bundle_sha256
               AND NEW.error_type IS NOT DISTINCT FROM OLD.error_type
               AND NEW.error_summary IS NOT DISTINCT FROM OLD.error_summary
               AND NEW.metadata IS NOT DISTINCT FROM OLD.metadata THEN
                RETURN NEW;
            END IF;
            IF NOT (
                (OLD.status = 'registered' AND NEW.status IN ('retrieval_ready','answer_ready','failed')) OR
                (OLD.status = 'retrieval_ready' AND NEW.status IN ('answer_ready','failed')) OR
                (OLD.status = 'answer_ready' AND NEW.status IN ('exported','failed')) OR
                (OLD.status = 'exported' AND NEW.status IN ('imported','failed')) OR
                (OLD.status = 'imported' AND NEW.status IN ('verified','failed'))
            ) OR NEW.transition_ordinal != OLD.transition_ordinal + 1 THEN
                RAISE EXCEPTION 'invalid baseline component state transition' USING ERRCODE = '55000';
            END IF;
            IF NEW.status IN ('answer_ready','exported','imported','verified') AND (
                NEW.suite_id IS NULL OR NEW.run_id IS NULL OR NOT EXISTS (
                    SELECT 1
                    FROM {RUNTIME}.baseline_suite_runs sr
                    JOIN {RUNTIME}.run_states rs ON rs.run_id = sr.run_id
                    WHERE sr.suite_id = NEW.suite_id
                      AND sr.method = NEW.method
                      AND sr.run_id = NEW.run_id
                      AND rs.status = 'completed'
                )
            ) THEN
                RAISE EXCEPTION 'answer-ready component requires a completed linked run' USING ERRCODE = '55000';
            END IF;
            IF NEW.status = 'verified' AND NOT EXISTS (
                SELECT 1 FROM {RUNTIME}.baseline_component_imports i
                WHERE i.component_id = NEW.component_id
                  AND i.family_id = NEW.family_id
                  AND i.bundle_export_id = NEW.bundle_export_id
                  AND i.bundle_sha256 = NEW.bundle_sha256
            ) THEN
                RAISE EXCEPTION 'verified component requires a matching import record' USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END $$"""
    )
    op.execute(
        f"""CREATE FUNCTION {RUNTIME}.enforce_baseline_component_import()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE component_row {RUNTIME}.baseline_suite_family_components%ROWTYPE;
        BEGIN
            IF TG_OP != 'INSERT' THEN
                RAISE EXCEPTION 'baseline component import records are immutable' USING ERRCODE = '55000';
            END IF;
            SELECT * INTO component_row
            FROM {RUNTIME}.baseline_suite_family_components
            WHERE component_id = NEW.component_id AND family_id = NEW.family_id;
            IF component_row.status NOT IN ('exported','imported') OR
               component_row.bundle_export_id != NEW.bundle_export_id OR
               component_row.bundle_sha256 != NEW.bundle_sha256 THEN
                RAISE EXCEPTION 'component import does not match exported component' USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END $$"""
    )
    op.execute(
        f"CREATE TRIGGER guard_baseline_family BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.baseline_suite_families FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_baseline_family_mutation()"
    )
    op.execute(
        f"CREATE TRIGGER guard_baseline_component BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.baseline_suite_family_components FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_baseline_component_mutation()"
    )
    op.execute(
        f"CREATE TRIGGER validate_baseline_component_import BEFORE INSERT OR UPDATE OR DELETE ON {RUNTIME}.baseline_component_imports FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.enforce_baseline_component_import()"
    )


def _grant_permissions() -> None:
    tables = (
        "baseline_suite_families",
        "baseline_suite_family_components",
        "baseline_component_imports",
    )
    qualified = ", ".join(f"{RUNTIME}.{table}" for table in tables)
    op.execute(f"GRANT SELECT ON {qualified} TO pmmc_runtime, pmmc_evaluator")
    op.execute(f"GRANT INSERT ON {qualified} TO pmmc_runtime")
    op.execute(
        f"GRANT UPDATE (status, verified_component_count, updated_at, completed_at, error_type, error_summary) ON {RUNTIME}.baseline_suite_families TO pmmc_runtime"
    )
    op.execute(
        f"GRANT UPDATE (suite_id, run_id, corpus_id, status, transition_ordinal, bundle_export_id, bundle_sha256, updated_at, completed_at, error_type, error_summary, metadata) ON {RUNTIME}.baseline_suite_family_components TO pmmc_runtime"
    )
    for function in (
        "enforce_baseline_family_mutation()",
        "enforce_baseline_component_mutation()",
        "enforce_baseline_component_import()",
    ):
        op.execute(f"REVOKE ALL ON FUNCTION {RUNTIME}.{function} FROM PUBLIC")


def downgrade() -> None:
    op.execute(
        f"DROP TRIGGER IF EXISTS validate_baseline_component_import ON {RUNTIME}.baseline_component_imports"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS guard_baseline_component ON {RUNTIME}.baseline_suite_family_components"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS guard_baseline_family ON {RUNTIME}.baseline_suite_families"
    )
    for function in (
        "enforce_baseline_component_import()",
        "enforce_baseline_component_mutation()",
        "enforce_baseline_family_mutation()",
    ):
        op.execute(f"DROP FUNCTION IF EXISTS {RUNTIME}.{function}")
    op.execute(f"DROP TABLE {RUNTIME}.baseline_component_imports")
    op.execute(f"DROP TABLE {RUNTIME}.baseline_suite_family_components")
    op.execute(f"DROP TABLE {RUNTIME}.baseline_suite_families")
