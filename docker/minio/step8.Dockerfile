# syntax=docker/dockerfile:1.7
# Independent context: docker/minio. Supply the candidate application digest.
ARG PMMC_IMAGE
ARG PMMC_MC_IMAGE=docker.io/l33tlamer/minio-backup@sha256:a1ea29fa28355559ef137d71fc570e508a214ec84ff8083e39bc5428980b015e
FROM ${PMMC_MC_IMAGE} AS mc
FROM ${PMMC_IMAGE} AS iam-tools-readable
ARG PMMC_MC_SOURCE=/usr/bin/mc
COPY --chmod=0555 --from=mc ${PMMC_MC_SOURCE} /usr/local/bin/mc

FROM iam-tools-readable AS iam
COPY --chmod=0555 step8_iam.py /opt/pmmc-step8/step8_iam.py
RUN printf '%s  /usr/local/bin/mc\n' ac90da87a35641be5a0ac75d49de5161ddb47d629b5ba01261b0ae9e00aea15f | sha256sum -c - \
    && cmp /opt/pmmc-step8/step8_iam.py /workspace/docker/minio/step8_iam.py \
    && python /workspace/docker/verify_source_contracts.py /workspace/src \
    && python /workspace/docker/verify_step8_source_contracts.py /workspace --runtime \
    && python -c "import runpy; runpy.run_path('/opt/pmmc-step8/step8_iam.py', run_name='build_check')" \
    && mc --version
ENTRYPOINT ["python", "/opt/pmmc-step8/step8_iam.py"]
CMD []
