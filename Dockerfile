# syntax=docker/dockerfile:1.7

ARG VLLM_IMAGE=vllm/vllm-openai:v0.24.0@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
FROM ${VLLM_IMAGE}

LABEL org.opencontainers.image.title="PMMC"
LABEL org.opencontainers.image.description="Prospective Multimodal Memory Compilation research platform"
LABEL org.opencontainers.image.authors="PMMC Contributors"
LABEL io.pmmc.source-license="Apache-2.0"

# The upstream image starts a vLLM server. PMMC containers must also support
# CLI, Alembic, and pytest commands directly.
ENTRYPOINT []
USER root

ARG USER_ID=10001
ARG GROUP_ID=10001
ARG VLLM_NIXL_VERSION=1.2.0
ARG POSTGRES_CLIENT_MAJOR=16
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PMMC_PROJECT_ROOT=/workspace \
    HOME=/opt/pmmc-home \
    HF_HOME=/opt/pmmc-home/.cache/huggingface \
    HF_DATASETS_CACHE=/opt/pmmc-home/.cache/huggingface/datasets \
    TORCH_HOME=/opt/pmmc-home/.cache/torch

RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
            /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
            /etc/apt/sources.list.d/ubuntu.sources; \
    fi; \
    apt-get \
        -o Acquire::Retries=5 \
        -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 \
        -o APT::Update::Error-Mode=any \
        update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsndfile1 \
        tini \
    && install -d -m 0755 /usr/share/postgresql-common/pgdg \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get \
        -o Acquire::Retries=5 \
        -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 \
        -o APT::Update::Error-Mode=any \
        update \
    && apt-get install -y --no-install-recommends \
        "postgresql-client-${POSTGRES_CLIENT_MAJOR}" \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/python \
    && python3 --version \
    && python --version

# Set these build args to match the operator's storage, if necessary.
RUN groupadd --gid "${GROUP_ID}" usergroup \
    && useradd --no-log-init --create-home \
        --home-dir /opt/pmmc-home --uid "${USER_ID}" --gid "${GROUP_ID}" pmmc

WORKDIR /workspace

# Install every declared dependency through the project metadata. The pinned
# vLLM base already supplies its matching CUDA, PyTorch, and Transformers stack.
# Its optional KV-connector layer can contain the 1.2.0 NIXL metapackage with
# an unpinned, newer CUDA companion wheel, so align that ABI pair explicitly.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
COPY docker/lock_constraints.py ./docker/lock_constraints.py
COPY docker/verify_pip_check.py ./docker/verify_pip_check.py
COPY docker/verify_runtime_versions.py ./docker/verify_runtime_versions.py
COPY docker/verify_source_contracts.py ./docker/verify_source_contracts.py
RUN python3 docker/lock_constraints.py uv.lock > /tmp/pmmc-constraints.txt \
    && python3 -m pip install -c /tmp/pmmc-constraints.txt . \
    && python3 -m pip install --force-reinstall --no-deps \
        "nixl-cu13==${VLLM_NIXL_VERSION}" \
    && (python3 -m pip check > /tmp/pmmc-pip-check.txt 2>&1 || true) \
    && python3 docker/verify_pip_check.py /tmp/pmmc-pip-check.txt \
    && rm /tmp/pmmc-pip-check.txt \
    && python3 -c "import alembic, boto3, pmmc, psycopg, pydantic, sqlalchemy" \
    && python3 -c "from pmmc.models.protocols import EmbeddingGateway, GenerationGateway, ModelCallWriter, TokenCounter, TraceSink" \
    && python3 -c "from pmmc.evaluation.workflow import EvaluationRunPhase" \
    && python3 -c "from pmmc.experiments.checkpoint import verify_postgres_client_major; verify_postgres_client_major(16)" \
    && python3 -m pmmc experiment --help >/dev/null \
    && python3 -c "from inspect import signature; from pmmc.experiments.kubernetes import render_command; assert 'output' in signature(render_command).parameters" \
    && python3 docker/verify_runtime_versions.py \
        "vllm==0.24.0" \
        "torch==2.11.0" \
        "torchvision==0.26.0" \
        "litellm==1.92.0" \
        "nixl==${VLLM_NIXL_VERSION}" \
        "nixl-cu13==${VLLM_NIXL_VERSION}" \
        "transformers>=5.5.3,<6" \
        "huggingface-hub>=1.21,<1.22"

# Tests require repository-relative Alembic files and fixtures. Sensitive and
# third-party payloads are excluded by .dockerignore and mounted at runtime.
COPY --chown=${USER_ID}:${GROUP_ID} . .

RUN python3 docker/verify_source_contracts.py /workspace/src \
    && python3 docker/verify_step8_source_contracts.py /workspace --runtime \
    && mkdir -p artifacts outputs runs \
    && chown "${USER_ID}:${GROUP_ID}" /workspace artifacts outputs runs

USER pmmc

ENTRYPOINT ["tini", "--"]
CMD ["python3", "-m", "pmmc", "--help"]
