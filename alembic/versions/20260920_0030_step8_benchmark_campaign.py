"""Add Step 8 campaign facts and restricted online projections.

DDL is frozen here, independently of future ORM changes. Completion/lineage guards
and historical schema dispatch belong to 0031. No scientific facts are rewritten.
Creating the NOLOGIN role requires a migration principal with CREATEROLE; deployment
bootstrap separately provisions login credentials. Downgrade leaves that role in
place because credentials and memberships may be deployment-owned.
"""

from alembic import op

revision: str = "20260920_0030"
down_revision: str | None = "20260911_0029"
branch_labels: str | None = None
depends_on: str | None = None

RUNTIME = "pmmc_runtime"

# SQLAlchemy's frozen PostgreSQL names include its deterministic 63-byte truncation.
_TABLE_DDL = {
    "method_protocol_freezes": """
        freeze_hash VARCHAR(64) NOT NULL,
        protocol_hash VARCHAR(64) NOT NULL,
        decision VARCHAR(16) NOT NULL,
        reviewer_id VARCHAR(256) NOT NULL,
        payload JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_method_protocol_freezes PRIMARY KEY (freeze_hash),
        CONSTRAINT ck_method_protocol_freezes_decision_values
            CHECK (decision IN ('approved','rejected'))
    """,
    "benchmark_campaigns": """
        campaign_id VARCHAR(128) NOT NULL,
        protocol_hash VARCHAR(64) NOT NULL,
        freeze_hash VARCHAR(64),
        phase VARCHAR(16) NOT NULL,
        status VARCHAR(32) NOT NULL,
        manifest_hash VARCHAR(64) NOT NULL,
        payload JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_benchmark_campaigns PRIMARY KEY (campaign_id),
        CONSTRAINT ck_benchmark_campaigns_phase_values CHECK (phase IN ('development','paper')),
        CONSTRAINT ck_benchmark_campaigns_status_values
            CHECK (status IN ('planned','running','failed','awaiting_approval','completed')),
        CONSTRAINT ck_benchmark_campaigns_paper_freeze
            CHECK (phase <> 'paper' OR freeze_hash IS NOT NULL),
        CONSTRAINT fk_benchmark_campaigns_freeze_hash_method_protocol_freezes
            FOREIGN KEY (freeze_hash) REFERENCES pmmc_runtime.method_protocol_freezes (freeze_hash)
    """,
    "benchmark_selection_expansions": """
        selection_hash VARCHAR(64) NOT NULL,
        campaign_id VARCHAR(128) NOT NULL,
        selection VARCHAR(32) NOT NULL,
        parent_selection_hash VARCHAR(64),
        payload JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_benchmark_selection_expansions PRIMARY KEY (selection_hash),
        CONSTRAINT fk_benchmark_selection_expansions_campaign_id_benchmark_a127
            FOREIGN KEY (campaign_id) REFERENCES pmmc_runtime.benchmark_campaigns (campaign_id),
        CONSTRAINT fk_benchmark_selection_expansions_parent_selection_hash_b7c9
            FOREIGN KEY (parent_selection_hash)
            REFERENCES pmmc_runtime.benchmark_selection_expansions (selection_hash)
    """,
    "benchmark_experiment_cells": """
        cell_id VARCHAR(128) NOT NULL,
        protocol_hash VARCHAR(64) NOT NULL,
        dataset VARCHAR(32) NOT NULL,
        method VARCHAR(64) NOT NULL,
        answer_model VARCHAR(128) NOT NULL,
        payload JSONB NOT NULL,
        CONSTRAINT pk_benchmark_experiment_cells PRIMARY KEY (cell_id)
    """,
    "benchmark_cell_assignments": """
        assignment_id VARCHAR(64) NOT NULL,
        campaign_id VARCHAR(128) NOT NULL,
        cell_id VARCHAR(128) NOT NULL,
        query_id VARCHAR(256) NOT NULL,
        first_selection_hash VARCHAR(64) NOT NULL,
        CONSTRAINT pk_benchmark_cell_assignments PRIMARY KEY (assignment_id),
        CONSTRAINT uq_benchmark_cell_assignments_campaign_id_cell_id_query_id
            UNIQUE (campaign_id, cell_id, query_id),
        CONSTRAINT fk_benchmark_cell_assignments_campaign_id_benchmark_campaigns
            FOREIGN KEY (campaign_id) REFERENCES pmmc_runtime.benchmark_campaigns (campaign_id),
        CONSTRAINT fk_benchmark_cell_assignments_cell_id_benchmark_experim_3eaf
            FOREIGN KEY (cell_id) REFERENCES pmmc_runtime.benchmark_experiment_cells (cell_id),
        CONSTRAINT fk_benchmark_cell_assignments_first_selection_hash_benc_7546
            FOREIGN KEY (first_selection_hash)
            REFERENCES pmmc_runtime.benchmark_selection_expansions (selection_hash)
    """,
    "benchmark_execution_attempts": """
        attempt_id VARCHAR(128) NOT NULL,
        campaign_id VARCHAR(128) NOT NULL,
        execution_image TEXT NOT NULL,
        resource_class VARCHAR(32) NOT NULL,
        status VARCHAR(32) NOT NULL,
        started_at TIMESTAMP WITH TIME ZONE NOT NULL,
        completed_at TIMESTAMP WITH TIME ZONE,
        receipt_hash VARCHAR(64),
        CONSTRAINT pk_benchmark_execution_attempts PRIMARY KEY (attempt_id),
        CONSTRAINT ck_benchmark_execution_attempts_status_values
            CHECK (status IN ('running','completed','failed','interrupted','budget_exhausted')),
        CONSTRAINT fk_benchmark_execution_attempts_campaign_id_benchmark_campaigns
            FOREIGN KEY (campaign_id) REFERENCES pmmc_runtime.benchmark_campaigns (campaign_id)
    """,
    "benchmark_stage_jobs": """
        job_id VARCHAR(64) NOT NULL,
        campaign_id VARCHAR(128) NOT NULL,
        component_key VARCHAR(128) NOT NULL,
        stage VARCHAR(64) NOT NULL,
        resource_class VARCHAR(32) NOT NULL,
        scientific_input_hash VARCHAR(64) NOT NULL,
        source_fingerprint VARCHAR(64) NOT NULL,
        dependency_ids VARCHAR(64)[] NOT NULL,
        payload JSONB NOT NULL,
        status VARCHAR(16) NOT NULL,
        attempt_count INTEGER NOT NULL,
        lease_owner VARCHAR(128),
        lease_expires_at TIMESTAMP WITH TIME ZONE,
        receipt_hash VARCHAR(64),
        error_code VARCHAR(128),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        completed_at TIMESTAMP WITH TIME ZONE,
        CONSTRAINT pk_benchmark_stage_jobs PRIMARY KEY (job_id),
        CONSTRAINT uq_benchmark_stage_jobs_campaign_id_component_key
            UNIQUE (campaign_id, component_key),
        CONSTRAINT ck_benchmark_stage_jobs_status_values
            CHECK (status IN ('pending','running','completed','failed','unknown')),
        CONSTRAINT ck_benchmark_stage_jobs_attempt_count_nonnegative CHECK (attempt_count >= 0),
        CONSTRAINT ck_benchmark_stage_jobs_completed_receipt
            CHECK (status <> 'completed' OR receipt_hash IS NOT NULL),
        CONSTRAINT fk_benchmark_stage_jobs_campaign_id_benchmark_campaigns
            FOREIGN KEY (campaign_id) REFERENCES pmmc_runtime.benchmark_campaigns (campaign_id)
    """,
    "benchmark_component_receipts": """
        receipt_hash VARCHAR(64) NOT NULL,
        component_identity VARCHAR(64) NOT NULL,
        job_id VARCHAR(64) NOT NULL,
        payload JSONB NOT NULL,
        artifact_hashes VARCHAR(64)[] NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_benchmark_component_receipts PRIMARY KEY (receipt_hash),
        CONSTRAINT fk_benchmark_component_receipts_job_id_benchmark_stage_jobs
            FOREIGN KEY (job_id) REFERENCES pmmc_runtime.benchmark_stage_jobs (job_id)
    """,
    "benchmark_budget_approvals": """
        approval_id VARCHAR(64) NOT NULL,
        campaign_id VARCHAR(128) NOT NULL,
        selection_hash VARCHAR(64) NOT NULL,
        approver_id VARCHAR(256) NOT NULL,
        rate_card_hash VARCHAR(64) NOT NULL,
        limit_usd NUMERIC(20, 8) NOT NULL,
        settled_usd NUMERIC(20, 8) NOT NULL,
        reserved_usd NUMERIC(20, 8) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_benchmark_budget_approvals PRIMARY KEY (approval_id),
        CONSTRAINT uq_benchmark_budget_approvals_campaign_id_selection_hash
            UNIQUE (campaign_id, selection_hash),
        CONSTRAINT ck_benchmark_budget_approvals_positive_limit CHECK (limit_usd > 0),
        CONSTRAINT ck_benchmark_budget_approvals_spend_nonnegative
            CHECK (settled_usd >= 0 AND reserved_usd >= 0),
        CONSTRAINT fk_benchmark_budget_approvals_campaign_id_benchmark_campaigns
            FOREIGN KEY (campaign_id) REFERENCES pmmc_runtime.benchmark_campaigns (campaign_id),
        CONSTRAINT fk_benchmark_budget_approvals_selection_hash_benchmark__c95c
            FOREIGN KEY (selection_hash)
            REFERENCES pmmc_runtime.benchmark_selection_expansions (selection_hash)
    """,
    "benchmark_spend_reservations": """
        request_id VARCHAR(128) NOT NULL,
        approval_id VARCHAR(64) NOT NULL,
        request_hash VARCHAR(64) NOT NULL,
        provider VARCHAR(64) NOT NULL,
        reserved_usd NUMERIC(20, 8) NOT NULL,
        actual_usd NUMERIC(20, 8),
        state VARCHAR(16) NOT NULL,
        response_hash VARCHAR(64),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_benchmark_spend_reservations PRIMARY KEY (request_id),
        CONSTRAINT ck_benchmark_spend_reservations_reservation_nonnegative
            CHECK (reserved_usd >= 0),
        CONSTRAINT ck_benchmark_spend_reservations_state_values
            CHECK (state IN ('reserved','settled','unknown','cancelled')),
        CONSTRAINT fk_benchmark_spend_reservations_approval_id_benchmark_b_1d4a
            FOREIGN KEY (approval_id)
            REFERENCES pmmc_runtime.benchmark_budget_approvals (approval_id)
    """,
    "query_runtime_embedding_cache": """
        cache_id VARCHAR(128) NOT NULL,
        snapshot_id VARCHAR(128) NOT NULL,
        query_id VARCHAR(256) NOT NULL,
        channel VARCHAR(16) NOT NULL,
        model_entry_hash VARCHAR(128) NOT NULL,
        input_hash VARCHAR(128) NOT NULL,
        embedding_id VARCHAR(128) NOT NULL,
        collection_id VARCHAR(128) NOT NULL,
        model_call_id VARCHAR(128) NOT NULL,
        content_hash VARCHAR(128) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_query_runtime_embedding_cache PRIMARY KEY (cache_id),
        CONSTRAINT ck_query_runtime_embedding_cache_channel_values
            CHECK (channel IN ('text','multimodal')),
        CONSTRAINT fk_query_runtime_embedding_cache_embedding_id_embeddings
            FOREIGN KEY (embedding_id) REFERENCES pmmc_runtime.embeddings (embedding_id)
    """,
    "query_runtime_phase_states": """
        phase_state_id VARCHAR(128) NOT NULL,
        query_executor_run_id VARCHAR(128) NOT NULL,
        query_id VARCHAR(256) NOT NULL,
        phase VARCHAR(16) NOT NULL,
        payload JSONB NOT NULL,
        content_hash VARCHAR(128) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_query_runtime_phase_states PRIMARY KEY (phase_state_id),
        CONSTRAINT uq_query_runtime_phase_states_query_executor_run_id_que_471b
            UNIQUE (query_executor_run_id, query_id, phase),
        CONSTRAINT ck_query_runtime_phase_states_phase_values CHECK (phase IN ('match','answer')),
        CONSTRAINT fk_query_runtime_phase_states_query_executor_run_id_que_aa2c
            FOREIGN KEY (query_executor_run_id)
            REFERENCES pmmc_runtime.query_executor_runs (query_executor_run_id)
    """,
    "query_runtime_typed_answers": """
        typed_answer_id VARCHAR(128) NOT NULL,
        query_executor_run_id VARCHAR(128) NOT NULL,
        query_id VARCHAR(256) NOT NULL,
        answer_identity_hash VARCHAR(128) NOT NULL,
        answer_model_call_id VARCHAR(128) NOT NULL,
        terminal_state VARCHAR(32) NOT NULL,
        payload JSONB NOT NULL,
        content_hash VARCHAR(128) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_query_runtime_typed_answers PRIMARY KEY (typed_answer_id),
        CONSTRAINT uq_query_runtime_typed_answers_query_executor_run_id_qu_31c1
            UNIQUE (query_executor_run_id, query_id, answer_identity_hash),
        CONSTRAINT ck_query_runtime_typed_answers_terminal_state_values
            CHECK (terminal_state IN ('answered','abstained','fallback_required')),
        CONSTRAINT fk_query_runtime_typed_answers_query_executor_run_id_ru_8f00
            FOREIGN KEY (query_executor_run_id) REFERENCES pmmc_runtime.run_manifests (run_id)
    """,
    "query_experiment_cell_results": """
        cell_result_id VARCHAR(128) NOT NULL,
        query_executor_run_id VARCHAR(128) NOT NULL,
        query_id VARCHAR(256) NOT NULL,
        cell_id VARCHAR(128) NOT NULL,
        treatment VARCHAR(64) NOT NULL,
        question_budget INTEGER NOT NULL,
        revision_depth INTEGER NOT NULL,
        terminal_state VARCHAR(32) NOT NULL,
        selected_answer_id VARCHAR(128),
        payload JSONB NOT NULL,
        content_hash VARCHAR(128) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_query_experiment_cell_results PRIMARY KEY (cell_result_id),
        CONSTRAINT uq_query_experiment_cell_results_query_executor_run_id__7e68
            UNIQUE (query_executor_run_id, query_id, cell_id),
        CONSTRAINT ck_query_experiment_cell_results_cell_grid_values
            CHECK (question_budget IN (1,2,4,8) AND revision_depth BETWEEN 0 AND 3),
        CONSTRAINT ck_query_experiment_cell_results_terminal_state_values
            CHECK (terminal_state IN ('answered','abstained','no_answer','technical_failure')),
        CONSTRAINT fk_query_experiment_cell_results_query_executor_run_id__c30c
            FOREIGN KEY (query_executor_run_id)
            REFERENCES pmmc_runtime.query_executor_runs (query_executor_run_id),
        CONSTRAINT fk_query_experiment_cell_results_selected_answer_id_que_778d
            FOREIGN KEY (selected_answer_id)
            REFERENCES pmmc_runtime.query_runtime_typed_answers (typed_answer_id)
    """,
}

_INDEXES = (
    ("method_protocol_freezes", "protocol_hash"),
    ("benchmark_campaigns", "protocol_hash"),
    ("benchmark_selection_expansions", "campaign_id"),
    ("benchmark_experiment_cells", "protocol_hash"),
    ("benchmark_execution_attempts", "campaign_id"),
    ("benchmark_stage_jobs", "campaign_id"),
    ("benchmark_stage_jobs", "resource_class"),
    ("benchmark_stage_jobs", "status"),
    ("benchmark_component_receipts", "component_identity"),
)

_APPEND_ONLY = (
    "method_protocol_freezes",
    "benchmark_selection_expansions",
    "benchmark_experiment_cells",
    "benchmark_cell_assignments",
    "benchmark_component_receipts",
    "query_runtime_embedding_cache",
    "query_runtime_phase_states",
    "query_runtime_typed_answers",
    "query_experiment_cell_results",
)

_MUTABLE_COLUMNS = {
    "benchmark_campaigns": "status",
    "benchmark_execution_attempts": "status, completed_at, receipt_hash",
    "benchmark_stage_jobs": (
        "status, attempt_count, lease_owner, lease_expires_at, "
        "receipt_hash, error_code, completed_at"
    ),
    "benchmark_budget_approvals": "settled_usd, reserved_usd",
    "benchmark_spend_reservations": "state, actual_usd, response_hash",
}

# Do not grant compiler questions/programs, model-call parsed payloads, raw artifact
# locators, or evaluation tables. More capabilities need separately reviewed views.
_ONLINE_READ_TABLES = (
    "benchmark_sessions",
    "memory_units",
    "memory_unit_images",
    "images",
    "benchmark_queries",
    "query_images",
    "run_snapshot_links",
    "run_queries",
    "run_states",
    "compiler_units",
    "compiler_unit_memories",
    "memory_operator_index_manifests",
    "embedding_collections",
    "embeddings",
    "qbank_lexical_term_facts",
    "retrieval_corpora",
    "memory_windows",
    "memory_window_units",
    "retrieval_documents",
    "retrieval_document_images",
    "context_assemblies",
    "context_assembly_memory_units",
    "context_assembly_images",
    "context_assembly_retrieval_candidates",
    "retrieval_batches",
    "retrieval_traces",
    "retrieval_batch_query_embeddings",
    "retrieval_trace_components",
    "final_answers",
    "final_answer_evidence_ids",
    "final_answer_image_ids",
    "query_executor_runs",
    "query_executor_stage_jobs",
    "query_embedding_batches",
    "query_embeddings",
    "qbank_match_batches",
    "qbank_match_channel_candidates",
    "qbank_match_fused_candidates",
    "qbank_reranker_assessments",
    "query_route_decisions",
    "query_program_bindings",
    "query_evidence_sufficiency_assessments",
    "query_evidence_relevance_assessments",
    "query_evidence_relevance_items",
    "query_fallback_decisions",
    "query_answer_selections",
    "pmmc_fallback_run_links",
    "pmmc_fallback_query_links",
    "structured_answer_outputs",
    "answer_format_diagnostics",
    "answer_reuse_links",
    "query_execution_reuse_links",
    "query_runtime_phase_states",
    "query_runtime_embedding_cache",
    "query_runtime_typed_answers",
    "query_experiment_cell_results",
)
_ONLINE_INSERT_TABLES: tuple[str, ...] = ()
_ONLINE_VIEWS = (
    "query_runtime_programs_v3",
    "verified_qbank_routing_v4",
    "query_runtime_projections_v3",
    "query_runtime_program_vectors_v3",
    "qbank_routing_program_steps",
)

_ROUTING_COLUMNS = (
    "routing_projection_id",
    "compiler_run_id",
    "snapshot_id",
    "dataset",
    "conversation_id",
    "available_after_session",
    "cutoff_session_id",
    "qbank_id",
    "generation_rank",
    "route_class",
    "question_type",
    "canonical_question",
    "paraphrases",
    "required_modalities",
    "accepted_round",
    "plan_id",
    "program_hash",
    "operator_set_version",
    "requirement_contract_id",
    "requirement_contract_hash",
    "contract_required_modalities",
    "required_capabilities",
    "allowed_capabilities",
    "minimum_evidence_groups",
    "fallback_required",
    "refusal_only",
    "lexical_collection_id",
    "text_collection_id",
    "multimodal_collection_id",
    "active_round",
    "decision_hash",
    "ordinal",
    "projection_round",
)


def _create_online_role() -> None:
    op.execute("""DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pmmc_online') THEN
            CREATE ROLE pmmc_online NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB
                NOCREATEROLE NOREPLICATION NOBYPASSRLS;
        ELSE
            IF EXISTS (
                SELECT 1 FROM pg_roles r
                WHERE r.rolname = 'pmmc_online'
                  AND (r.rolsuper OR r.rolcreatedb OR r.rolcreaterole
                       OR r.rolreplication OR r.rolbypassrls)
            ) OR EXISTS (
                SELECT 1 FROM pg_auth_members m
                JOIN pg_roles r ON r.oid = m.member WHERE r.rolname = 'pmmc_online'
            ) THEN
                RAISE EXCEPTION 'step8_online_role_has_unsafe_privileges' USING ERRCODE = '42501';
            END IF;
            ALTER ROLE pmmc_online NOLOGIN NOINHERIT;
        END IF;
    END $$""")
    op.execute(f"REVOKE ALL ON SCHEMA {RUNTIME}, pmmc_evaluation FROM pmmc_online")
    op.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA {RUNTIME}, pmmc_evaluation FROM pmmc_online")
    op.execute(f"GRANT USAGE ON SCHEMA {RUNTIME} TO pmmc_online")


def _create_append_only_guards() -> None:
    op.execute(f"""CREATE FUNCTION {RUNTIME}.reject_step8_fact_mutation()
        RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $$
        BEGIN
            RAISE EXCEPTION 'step8_append_only_fact' USING ERRCODE = '55000';
        END $$""")
    op.execute(f"REVOKE ALL ON FUNCTION {RUNTIME}.reject_step8_fact_mutation() FROM PUBLIC")
    for table in _APPEND_ONLY:
        op.execute(
            f"CREATE TRIGGER step8_append_only BEFORE UPDATE OR DELETE ON {RUNTIME}.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION {RUNTIME}.reject_step8_fact_mutation()"
        )
        op.execute(
            f"CREATE TRIGGER step8_no_truncate BEFORE TRUNCATE ON {RUNTIME}.{table} "
            f"FOR EACH STATEMENT EXECUTE FUNCTION {RUNTIME}.reject_step8_fact_mutation()"
        )


def _create_online_views() -> None:
    op.execute(f"""CREATE VIEW {RUNTIME}.query_runtime_programs_v3
        WITH (security_barrier = true) AS
        SELECT plan_id, qbank_id, run_id, snapshot_id, compiler_unit_id, operator_set_version,
               round_id, fallback_operator, required_modalities, planner_confidence,
               program_hash, created_at
          FROM {RUNTIME}.memory_programs""")
    columns = ", ".join(f"routing.{column}" for column in _ROUTING_COLUMNS)
    op.execute(f"""CREATE VIEW {RUNTIME}.verified_qbank_routing_v4
        WITH (security_barrier = true) AS
        SELECT {columns}, questions.answer_type, programs.round_id AS program_round
          FROM {RUNTIME}.verified_qbank_routing_v2 routing
          JOIN {RUNTIME}.qbank_questions questions
            ON questions.qbank_id = routing.qbank_id
           AND questions.run_id = routing.compiler_run_id
          JOIN {RUNTIME}.compiler_runs compiler
            ON compiler.run_id = routing.compiler_run_id
          JOIN {RUNTIME}.memory_programs programs
            ON programs.plan_id = routing.plan_id
           AND programs.run_id = routing.compiler_run_id
           AND programs.qbank_id = routing.qbank_id
           AND programs.snapshot_id = routing.snapshot_id
         WHERE compiler.compiler_schema_version = '0.9.0'
           AND routing.operator_set_version = 'pmmc-operators-v7'
           AND questions.answer_type IS NOT NULL
           AND programs.round_id <= routing.projection_round
           AND EXISTS (
               SELECT 1 FROM {RUNTIME}.program_dry_run_assessments dry
                WHERE dry.run_id=programs.run_id AND dry.plan_id=programs.plan_id
                  AND dry.qbank_id=programs.qbank_id AND dry.execution_success
                  AND CASE WHEN routing.route_class='control' THEN dry.abstain_triggered
                      ELSE dry.nonempty_evidence AND dry.required_modality_sufficient
                           AND dry.pre_control_sufficient END)
           AND (routing.projection_round=0 AND programs.round_id=0 OR
                routing.projection_round>0 AND EXISTS (
                    SELECT 1 FROM {RUNTIME}.doubter_assessments d
                     WHERE d.run_id=programs.run_id AND d.plan_id=programs.plan_id
                       AND d.qbank_id=programs.qbank_id AND d.round_id=programs.round_id
                       AND d.action='pass' AND d.provider_schema_conformant))""")
    op.execute(f"""CREATE VIEW {RUNTIME}.query_runtime_projections_v3
        WITH (security_barrier = true) AS
        SELECT r.routing_projection_id, r.run_id, r.snapshot_id,
               r.active_round, r.max_round, r.content_hash
          FROM {RUNTIME}.routing_projection_manifests r
          JOIN {RUNTIME}.compiler_runs c ON c.run_id=r.run_id
          JOIN {RUNTIME}.run_states s ON s.run_id=r.run_id
         WHERE c.compiler_schema_version='0.9.0' AND s.status='completed'""")
    op.execute(f"""CREATE VIEW {RUNTIME}.query_runtime_program_vectors_v3
        WITH (security_barrier = true) AS
        SELECT refs.run_id, refs.qbank_id, refs.channel, refs.embedding_id
          FROM {RUNTIME}.program_query_embeddings refs
         WHERE refs.channel IN ('question_text_query', 'question_joint_query')
           AND EXISTS (SELECT 1 FROM {RUNTIME}.verified_qbank_routing_v4 routing
                       WHERE routing.compiler_run_id=refs.run_id
                         AND routing.qbank_id=refs.qbank_id)""")
    for view in _ONLINE_VIEWS[:-1]:
        op.execute(f"ALTER VIEW {RUNTIME}.{view} OWNER TO pmmc_owner")
        op.execute(f"REVOKE ALL ON {RUNTIME}.{view} FROM PUBLIC")
        op.execute(f"GRANT SELECT ON {RUNTIME}.{view} TO pmmc_runtime, pmmc_evaluator")


def _grant_permissions() -> None:
    for table in _TABLE_DDL:
        op.execute(f"REVOKE ALL ON {RUNTIME}.{table} FROM PUBLIC")
        op.execute(f"GRANT SELECT, INSERT ON {RUNTIME}.{table} TO pmmc_runtime")
        op.execute(f"GRANT SELECT ON {RUNTIME}.{table} TO pmmc_evaluator")
    for table, columns in _MUTABLE_COLUMNS.items():
        op.execute(f"GRANT UPDATE ({columns}) ON {RUNTIME}.{table} TO pmmc_runtime")
    for table in (*_ONLINE_READ_TABLES, *_ONLINE_VIEWS):
        op.execute(f"GRANT SELECT ON {RUNTIME}.{table} TO pmmc_online")
    for table in _ONLINE_INSERT_TABLES:
        op.execute(f"GRANT INSERT ON {RUNTIME}.{table} TO pmmc_online")


def _verify_online_boundary() -> None:
    allowed = ", ".join(f"'{name}'" for name in (*_ONLINE_READ_TABLES, *_ONLINE_VIEWS))
    # REVOKE at table level does not remove old column grants or PUBLIC grants.
    # Fail closed if a pre-existing role still has any private read capability.
    op.execute(f"""DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname IN ('pmmc_runtime', 'pmmc_evaluation')
              AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
              AND (n.nspname <> 'pmmc_runtime' OR c.relname NOT IN ({allowed}))
              AND (has_table_privilege('pmmc_online', c.oid, 'SELECT')
                   OR has_any_column_privilege('pmmc_online', c.oid, 'SELECT'))
        ) THEN
            RAISE EXCEPTION 'step8_online_private_read_privilege' USING ERRCODE = '42501';
        END IF;
    END $$""")


def upgrade() -> None:
    _create_online_role()
    for table, definition in _TABLE_DDL.items():
        op.execute(f"CREATE TABLE {RUNTIME}.{table} ({definition})")
    for table, column in _INDEXES:
        op.execute(f"CREATE INDEX ix_{table}_{column} ON {RUNTIME}.{table} ({column})")
    _create_append_only_guards()
    _create_online_views()
    _grant_permissions()
    _verify_online_boundary()


def downgrade() -> None:
    for table in _ONLINE_INSERT_TABLES:
        op.execute(f"REVOKE INSERT ON {RUNTIME}.{table} FROM pmmc_online")
    for table in (*_ONLINE_READ_TABLES, *_ONLINE_VIEWS):
        op.execute(f"REVOKE SELECT ON {RUNTIME}.{table} FROM pmmc_online")
    op.execute(f"REVOKE USAGE ON SCHEMA {RUNTIME} FROM pmmc_online")
    for view in reversed(_ONLINE_VIEWS[:-1]):
        op.execute(f"DROP VIEW {RUNTIME}.{view}")
    for table in reversed(_TABLE_DDL):
        op.execute(f"DROP TABLE {RUNTIME}.{table}")
    op.execute(f"DROP FUNCTION {RUNTIME}.reject_step8_fact_mutation()")
