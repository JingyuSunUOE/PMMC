# syntax=docker/dockerfile:1.7

FROM quay.io/minio/mc:RELEASE.2025-04-16T18-13-26Z@sha256:aead63c77f9db9107f1696fb08ecb0faeda23729cde94b0f663edf4fe09728e3

COPY --chmod=0555 init.sh /init.sh

ENTRYPOINT ["/bin/sh", "/init.sh"]
