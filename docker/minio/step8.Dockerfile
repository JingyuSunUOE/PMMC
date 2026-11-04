# syntax=docker/dockerfile:1.7
# Build context: repository root. Supply the verified Step8 application image.
ARG PMMC_IMAGE
ARG PMMC_MC_IMAGE=quay.io/minio/mc:RELEASE.2025-04-16T18-13-26Z@sha256:aead63c77f9db9107f1696fb08ecb0faeda23729cde94b0f663edf4fe09728e3
FROM ${PMMC_MC_IMAGE} AS mc
FROM ${PMMC_IMAGE} AS iam-tools-readable
ARG PMMC_MC_SOURCE=/usr/bin/mc
COPY --chmod=0555 --from=mc ${PMMC_MC_SOURCE} /usr/local/bin/mc

FROM iam-tools-readable AS iam
ARG PMMC_MC_SHA256
COPY --chmod=0555 docker/minio/step8_iam.py /opt/pmmc-step8/step8_iam.py
RUN if [ -n "${PMMC_MC_SHA256}" ]; then \
        printf '%s  /usr/local/bin/mc\n' "${PMMC_MC_SHA256}" | sha256sum -c -; \
    fi \
    && python /workspace/docker/verify_source_contracts.py /workspace/src \
    && python /workspace/docker/verify_step8_source_contracts.py /workspace --runtime \
    && python -c "import runpy; runpy.run_path('/opt/pmmc-step8/step8_iam.py', run_name='build_check')" \
    && mc --version
ENTRYPOINT ["python", "/opt/pmmc-step8/step8_iam.py"]
CMD []
