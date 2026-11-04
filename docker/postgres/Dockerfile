# syntax=docker/dockerfile:1.7

FROM docker.io/pgvector/pgvector@sha256:0dd3149a06e4579a0d70d8fef9945636cd514005e7237deb27f9b67886f8dc4f

COPY --chmod=0555 init-roles.sh \
    /docker-entrypoint-initdb.d/10-pmmc-roles.sh
