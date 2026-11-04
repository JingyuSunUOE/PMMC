# syntax=docker/dockerfile:1.7

FROM pgvector/pgvector:pg16

COPY --chmod=0555 init-roles.sh \
    /docker-entrypoint-initdb.d/10-pmmc-roles.sh
