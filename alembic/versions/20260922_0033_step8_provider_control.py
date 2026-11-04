"""Shared provider admission and immutable, evidence-bound reconciliation."""

from alembic import op

revision = "20260922_0033"
down_revision = "20260920_0032"
branch_labels = None
depends_on = None

RUNTIME = "pmmc_runtime"
_TABLE_DDL = {
    "benchmark_provider_rate_scopes": """
        scope_hash VARCHAR(64) NOT NULL,
        policy JSONB NOT NULL,
        blocked_until TIMESTAMP WITH TIME ZONE,
        CONSTRAINT pk_benchmark_provider_rate_scopes PRIMARY KEY (scope_hash),
        CONSTRAINT provider_scope_hash_valid CHECK (scope_hash ~ '^[a-f0-9]{64}$'),
        CONSTRAINT provider_scope_policy_object CHECK (jsonb_typeof(policy)='object')
    """,
    "benchmark_provider_rate_admissions": """
        dispatch_id VARCHAR(64) NOT NULL,
        request_hash VARCHAR(64) NOT NULL,
        scope_hash VARCHAR(64) NOT NULL,
        token_units INTEGER NOT NULL,
        admitted_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_benchmark_provider_rate_admissions PRIMARY KEY (dispatch_id),
        CONSTRAINT ck_benchmark_provider_rate_admissions_tokens_positive CHECK (token_units>0),
        CONSTRAINT provider_admission_hash_valid CHECK (
            dispatch_id ~ '^[a-f0-9]{64}$' AND request_hash ~ '^[a-f0-9]{64}$'),
        CONSTRAINT provider_admission_scope_fk FOREIGN KEY (scope_hash)
            REFERENCES pmmc_runtime.benchmark_provider_rate_scopes (scope_hash)
    """,
    "benchmark_provider_reconciliations": """
        request_id VARCHAR(128) NOT NULL,
        reconciliation_hash VARCHAR(64) NOT NULL,
        request_hash VARCHAR(64) NOT NULL,
        disposition VARCHAR(32) NOT NULL,
        payload JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_benchmark_provider_reconciliations PRIMARY KEY (request_id),
        CONSTRAINT provider_reconciliation_hash_unique UNIQUE (reconciliation_hash),
        CONSTRAINT provider_reconciliation_disposition CHECK (
            disposition IN ('completed','not_executed')),
        CONSTRAINT provider_reconciliation_hash_valid CHECK (
            reconciliation_hash ~ '^[a-f0-9]{64}$' AND request_hash ~ '^[a-f0-9]{64}$'),
        CONSTRAINT provider_reconciliation_request_fk FOREIGN KEY (request_id)
            REFERENCES pmmc_runtime.benchmark_spend_reservations (request_id)
    """,
}


def upgrade() -> None:
    for table, definition in _TABLE_DDL.items():
        op.execute(f"CREATE TABLE {RUNTIME}.{table} ({definition})")
        op.execute(f"ALTER TABLE {RUNTIME}.{table} OWNER TO pmmc_owner")
        op.execute(
            f"REVOKE ALL ON {RUNTIME}.{table} FROM PUBLIC,pmmc_runtime,pmmc_evaluator,pmmc_online"
        )
        op.execute(f"GRANT SELECT ON {RUNTIME}.{table} TO pmmc_runtime,pmmc_evaluator")
    for table in ("benchmark_provider_rate_scopes", "benchmark_provider_rate_admissions"):
        op.execute(f"GRANT INSERT ON {RUNTIME}.{table} TO pmmc_runtime")
    op.execute(
        f"GRANT UPDATE (blocked_until) ON {RUNTIME}.benchmark_provider_rate_scopes TO pmmc_runtime"
    )
    op.execute(f"""CREATE INDEX ix_provider_rate_scope_time ON
      {RUNTIME}.benchmark_provider_rate_admissions (scope_hash,admitted_at)""")
    op.execute(f"""CREATE INDEX ix_provider_rate_request_hash ON
      {RUNTIME}.benchmark_provider_rate_admissions (request_hash)""")
    op.execute("""CREATE FUNCTION pmmc_runtime.freeze_step8_provider_control()
    RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      RAISE EXCEPTION 'step8_provider_control_immutable' USING ERRCODE='55000';
    END $$""")
    for table in ("benchmark_provider_rate_admissions", "benchmark_provider_reconciliations"):
        op.execute(f"""CREATE TRIGGER step8_provider_control_immutable BEFORE UPDATE OR DELETE
          ON {RUNTIME}.{table} FOR EACH ROW
          EXECUTE FUNCTION pmmc_runtime.freeze_step8_provider_control()""")
    op.execute("""CREATE FUNCTION pmmc_runtime.enforce_step8_provider_scope()
    RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP='DELETE' OR NEW.scope_hash IS DISTINCT FROM OLD.scope_hash
         OR NEW.policy IS DISTINCT FROM OLD.policy
         OR (OLD.blocked_until IS NOT NULL AND
             (NEW.blocked_until IS NULL OR NEW.blocked_until<OLD.blocked_until)) THEN
        RAISE EXCEPTION 'step8_provider_scope_policy_immutable' USING ERRCODE='55000';
      END IF;
      RETURN NEW;
    END $$""")
    op.execute("""CREATE TRIGGER step8_provider_scope_policy BEFORE UPDATE OR DELETE
      ON pmmc_runtime.benchmark_provider_rate_scopes FOR EACH ROW
      EXECUTE FUNCTION pmmc_runtime.enforce_step8_provider_scope()""")
    op.execute("""CREATE FUNCTION pmmc_runtime.enforce_step8_provider_reconciliation()
    RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
    SET search_path=pg_catalog,pmmc_runtime AS $$
    DECLARE reservation pmmc_runtime.benchmark_spend_reservations%ROWTYPE;
    BEGIN
      SELECT * INTO reservation FROM pmmc_runtime.benchmark_spend_reservations
        WHERE request_id=NEW.request_id;
      IF NOT FOUND OR reservation.request_hash IS DISTINCT FROM NEW.request_hash
         OR jsonb_typeof(NEW.payload) IS DISTINCT FROM 'object'
         OR NEW.payload->>'protocol' IS DISTINCT FROM 'step8-provider-reconciliation-v1'
         OR NEW.payload->>'request_id' IS DISTINCT FROM NEW.request_id
         OR NEW.payload->>'request_hash' IS DISTINCT FROM NEW.request_hash
         OR NEW.payload->>'reconciliation_hash' IS DISTINCT FROM NEW.reconciliation_hash
         OR NEW.payload->>'approval_id' IS DISTINCT FROM reservation.approval_id
         OR NEW.payload->>'disposition' IS DISTINCT FROM NEW.disposition
         OR COALESCE(NEW.payload->>'evidence_sha256','') !~ '^[a-f0-9]{64}$'
         OR COALESCE(NEW.payload->>'receipt_sha256','') !~ '^[a-f0-9]{64}$'
         OR COALESCE(NEW.payload->>'reviewer_id','')=''
         OR COALESCE(NEW.payload->>'verifier_id','')=''
         OR COALESCE(NEW.payload->>'provider_request_id','')=''
         OR (NEW.payload->>'actual_usd')::numeric IS DISTINCT FROM reservation.actual_usd
         OR NEW.payload->>'response_sha256' IS DISTINCT FROM reservation.response_hash
         OR reservation.actual_usd IS NULL OR reservation.actual_usd>reservation.reserved_usd
         OR (NEW.disposition='completed' AND
             (reservation.state!='settled' OR reservation.response_hash IS NULL))
         OR (NEW.disposition='not_executed' AND
             (reservation.state!='cancelled' OR reservation.actual_usd!=0
              OR reservation.response_hash IS NOT NULL)) THEN
        RAISE EXCEPTION 'step8_provider_reconciliation_lineage_invalid' USING ERRCODE='23514';
      END IF;
      RETURN NEW;
    END $$""")
    op.execute("""CREATE TRIGGER step8_provider_reconciliation_lineage BEFORE INSERT
      ON pmmc_runtime.benchmark_provider_reconciliations FOR EACH ROW
      EXECUTE FUNCTION pmmc_runtime.enforce_step8_provider_reconciliation()""")
    for function in (
        "freeze_step8_provider_control",
        "enforce_step8_provider_scope",
        "enforce_step8_provider_reconciliation",
    ):
        op.execute(f"ALTER FUNCTION {RUNTIME}.{function}() OWNER TO pmmc_owner")
        op.execute(f"REVOKE ALL ON FUNCTION {RUNTIME}.{function}() FROM PUBLIC")


def downgrade() -> None:
    for table in reversed(_TABLE_DDL):
        op.execute(f"""DO $$ BEGIN IF EXISTS (SELECT 1 FROM {RUNTIME}.{table}) THEN
          RAISE EXCEPTION 'cannot_downgrade_step8_provider_control_with_facts';
          END IF; END $$""")
    for table in reversed(_TABLE_DDL):
        op.execute(f"DROP TABLE {RUNTIME}.{table}")
    for function in (
        "enforce_step8_provider_reconciliation",
        "enforce_step8_provider_scope",
        "freeze_step8_provider_control",
    ):
        op.execute(f"DROP FUNCTION {RUNTIME}.{function}()")
