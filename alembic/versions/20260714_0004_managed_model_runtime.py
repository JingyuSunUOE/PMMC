# ruff: noqa: E501
"""Add managed local model runtime provenance and model-call lineage.

Revision ID: 20260714_0004
Revises: 20260714_0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260714_0004"
down_revision: str | None = "20260714_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUNTIME = "pmmc_runtime"
EVALUATION = "pmmc_evaluation"


def upgrade() -> None:
    for statement in _table_statements():
        op.execute(statement)
    for schema in (RUNTIME, EVALUATION):
        op.add_column("model_calls", sa.Column("runtime_id", sa.String(128)), schema=schema)
        op.add_column(
            "model_calls",
            sa.Column("runtime_service_id", sa.String(256)),
            schema=schema,
        )
        prefix = "runtime" if schema == RUNTIME else "evaluation"
        op.create_foreign_key(
            f"fk_{prefix}_model_calls_runtime_session",
            "model_calls",
            "managed_runtime_sessions",
            ["runtime_id"],
            ["runtime_id"],
            source_schema=schema,
            referent_schema=RUNTIME,
        )
        op.create_foreign_key(
            f"fk_{prefix}_model_calls_runtime_service",
            "model_calls",
            "managed_runtime_services",
            ["runtime_service_id", "runtime_id"],
            ["runtime_service_id", "runtime_id"],
            source_schema=schema,
            referent_schema=RUNTIME,
        )
        op.create_check_constraint(
            f"ck_{prefix}_model_calls_runtime_service_runtime_present",
            "model_calls",
            "runtime_service_id IS NULL OR runtime_id IS NOT NULL",
            schema=schema,
        )
        op.create_index(
            f"ix_{prefix}_model_calls_runtime_service",
            "model_calls",
            ["runtime_id", "runtime_service_id"],
            schema=schema,
        )
    for statement in _grant_statements():
        op.execute(statement)


def downgrade() -> None:
    for schema in (EVALUATION, RUNTIME):
        prefix = "runtime" if schema == RUNTIME else "evaluation"
        op.drop_index(
            f"ix_{prefix}_model_calls_runtime_service",
            table_name="model_calls",
            schema=schema,
        )
        op.drop_constraint(
            f"ck_{prefix}_model_calls_runtime_service_runtime_present",
            "model_calls",
            schema=schema,
            type_="check",
        )
        op.drop_constraint(
            f"fk_{prefix}_model_calls_runtime_service",
            "model_calls",
            schema=schema,
            type_="foreignkey",
        )
        op.drop_constraint(
            f"fk_{prefix}_model_calls_runtime_session",
            "model_calls",
            schema=schema,
            type_="foreignkey",
        )
        op.drop_column("model_calls", "runtime_service_id", schema=schema)
        op.drop_column("model_calls", "runtime_id", schema=schema)
    for table in (
        "run_runtime_links",
        "runtime_gpu_allocations",
        "runtime_calibrations",
        "managed_runtime_services",
        "managed_runtime_sessions",
    ):
        op.execute(f"DROP TABLE IF EXISTS {RUNTIME}.{table} CASCADE")


def _table_statements() -> tuple[str, ...]:
    return (
        f"""CREATE TABLE {RUNTIME}.managed_runtime_sessions (
            runtime_id varchar(128) PRIMARY KEY,
            profile_name varchar(128) NOT NULL,
            profile_hash varchar(128) NOT NULL,
            hardware_fingerprint varchar(128) NOT NULL,
            status varchar(16) NOT NULL CHECK
                (status IN ('planned','starting','running','completed','failed','cancelled')),
            vllm_version varchar(64), litellm_version varchar(64),
            plan_artifact_id varchar(128), summary_artifact_id varchar(128),
            started_at timestamptz NOT NULL, completed_at timestamptz,
            error_type varchar(256), metadata jsonb NOT NULL,
            CHECK (status NOT IN ('completed','failed','cancelled') OR completed_at IS NOT NULL),
            CHECK (status != 'failed' OR error_type IS NOT NULL)
        )""",
        f"""CREATE TABLE {RUNTIME}.managed_runtime_services (
            runtime_service_id varchar(256) PRIMARY KEY,
            runtime_id varchar(128) NOT NULL
                REFERENCES {RUNTIME}.managed_runtime_sessions(runtime_id),
            service_id varchar(128) NOT NULL, stage_id varchar(128) NOT NULL,
            role varchar(64) NOT NULL CHECK
                (role IN ('answer','compiler','judge','text_embedding',
                    'multimodal_embedding','contrastive_embedding','reranker')),
            backend varchar(64) NOT NULL CHECK
                (backend IN ('vllm_chat','vllm_embedding',
                    'vllm_multimodal_embedding','litellm_proxy')),
            model_alias varchar(256) NOT NULL,
            status varchar(16) NOT NULL CHECK
                (status IN ('planned','starting','ready','stopping','stopped','failed')),
            allocation_json jsonb NOT NULL, process_id integer,
            observed_peak_memory_bytes bigint,
            started_at timestamptz, ready_at timestamptz, stopped_at timestamptz,
            exit_code integer, error_type varchar(256), metadata jsonb NOT NULL,
            UNIQUE (runtime_service_id, runtime_id),
            UNIQUE (runtime_id, service_id, stage_id),
            CHECK (observed_peak_memory_bytes IS NULL OR observed_peak_memory_bytes >= 0),
            CHECK (status != 'failed' OR error_type IS NOT NULL)
        )""",
        f"""CREATE TABLE {RUNTIME}.runtime_gpu_allocations (
            runtime_service_id varchar(256) NOT NULL,
            runtime_id varchar(128) NOT NULL, stage_id varchar(128) NOT NULL,
            device_ordinal integer NOT NULL CHECK (device_ordinal >= 0),
            anonymized_device_id varchar(64) NOT NULL,
            visible_index integer NOT NULL CHECK (visible_index >= 0),
            gpu_name varchar(256) NOT NULL,
            total_memory_bytes bigint NOT NULL,
            tensor_parallel_size integer NOT NULL CHECK (tensor_parallel_size > 0),
            memory_budget_bytes bigint NOT NULL CHECK (memory_budget_bytes > 0),
            topology_score double precision, metadata jsonb NOT NULL,
            PRIMARY KEY (runtime_service_id, device_ordinal),
            FOREIGN KEY (runtime_service_id, runtime_id)
                REFERENCES {RUNTIME}.managed_runtime_services(runtime_service_id, runtime_id)
        )""",
        f"""CREATE TABLE {RUNTIME}.runtime_calibrations (
            calibration_id varchar(128) PRIMARY KEY,
            service_id varchar(128) NOT NULL, model_alias varchar(256) NOT NULL,
            model_commit varchar(64) NOT NULL, profile_hash varchar(128) NOT NULL,
            hardware_fingerprint varchar(128) NOT NULL, vllm_version varchar(64) NOT NULL,
            status varchar(16) NOT NULL CHECK (status IN ('ready','failed')),
            allocation_json jsonb NOT NULL,
            observed_peak_memory_bytes bigint NOT NULL,
            observed_idle_memory_bytes bigint NOT NULL,
            server_config jsonb NOT NULL, error_type varchar(256),
            created_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            CHECK (observed_peak_memory_bytes >= 0 AND observed_idle_memory_bytes >= 0),
            CHECK (status != 'failed' OR error_type IS NOT NULL)
        )""",
        f"""CREATE TABLE {RUNTIME}.run_runtime_links (
            run_id varchar(128) NOT NULL REFERENCES {RUNTIME}.run_manifests(run_id),
            runtime_id varchar(128) NOT NULL
                REFERENCES {RUNTIME}.managed_runtime_sessions(runtime_id),
            linked_at timestamptz NOT NULL, metadata jsonb NOT NULL,
            PRIMARY KEY (run_id, runtime_id)
        )""",
        f"CREATE INDEX ix_managed_runtime_sessions_status ON {RUNTIME}.managed_runtime_sessions(status, started_at)",
        f"CREATE INDEX ix_managed_runtime_services_runtime_status ON {RUNTIME}.managed_runtime_services(runtime_id, status)",
        f"CREATE INDEX ix_runtime_calibrations_lookup ON {RUNTIME}.runtime_calibrations(model_commit, profile_hash, hardware_fingerprint, vllm_version)",
    )


def _grant_statements() -> tuple[str, ...]:
    tables = (
        f"{RUNTIME}.managed_runtime_sessions, "
        f"{RUNTIME}.managed_runtime_services, "
        f"{RUNTIME}.runtime_gpu_allocations, "
        f"{RUNTIME}.runtime_calibrations, "
        f"{RUNTIME}.run_runtime_links"
    )
    return (
        f"GRANT SELECT ON {tables} TO pmmc_runtime, pmmc_evaluator",
        f"GRANT INSERT ON {tables} TO pmmc_runtime",
        f"GRANT UPDATE ON {RUNTIME}.managed_runtime_sessions, {RUNTIME}.managed_runtime_services TO pmmc_runtime",
        f"REVOKE DELETE ON {tables} FROM pmmc_runtime, pmmc_evaluator",
    )
