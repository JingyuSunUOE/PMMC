# syntax=docker/dockerfile:1.7

FROM docker.io/l33tlamer/minio-backup@sha256:a1ea29fa28355559ef137d71fc570e508a214ec84ff8083e39bc5428980b015e

COPY --chmod=0555 init.sh /init.sh

ENTRYPOINT ["/bin/sh", "/init.sh"]
