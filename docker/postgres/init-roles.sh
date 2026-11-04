#!/bin/sh
set -eu

# Release bootstrap provisions and verifies all three least-privilege logins
# after PostgreSQL is ready. Do not require duplicate passwords at initdb time.
if [ "${PMMC_RELEASE_BOOTSTRAP_ROLES:-}" = 1 ]; then
    exit 0
fi

psql --set ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  --set=runtime_password="$PMMC_POSTGRES_RUNTIME_PASSWORD" \
  --set=evaluator_password="$PMMC_POSTGRES_EVALUATOR_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE pmmc_runtime LOGIN PASSWORD %L', :'runtime_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pmmc_runtime') \gexec
SELECT format('ALTER ROLE pmmc_runtime PASSWORD %L', :'runtime_password') \gexec

SELECT format('CREATE ROLE pmmc_evaluator LOGIN PASSWORD %L', :'evaluator_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pmmc_evaluator') \gexec
SELECT format('ALTER ROLE pmmc_evaluator PASSWORD %L', :'evaluator_password') \gexec

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
SQL
