"""Application settings for the PMMC research platform."""

from __future__ import annotations

from enum import Enum, StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from pmmc.schemas.runtime import RuntimeMode


class AppMode(StrEnum):
    """Supported PMMC runtime modes."""

    DEV = "dev"
    PAPER = "paper"
    OFFLINE_ANALYSIS = "offline_analysis"


def default_project_root() -> Path:
    """Return the repository root for a source-layout checkout."""

    return Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """PMMC settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="PMMC_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    mode: AppMode = AppMode.DEV
    project_root: Path = Field(default_factory=default_project_root)
    benchmark_asset_root: Path | None = None
    memlens_root: Path = Path("benchmarks/memlens")
    memgallery_root: Path = Path("benchmarks/memgallery")
    artifacts_dir: Path = Path("artifacts")
    outputs_dir: Path = Path("outputs")
    runs_dir: Path = Path("runs")
    local_scratch_dir: Path | None = None
    local_asset_cache_enabled: bool = False

    database_url: SecretStr | None = None
    database_evaluator_url: SecretStr | None = None
    database_admin_url: SecretStr | None = None
    database_online_url: SecretStr | None = None
    database_pool_size: int = Field(default=10, ge=1)
    database_max_overflow: int = Field(default=20, ge=0)
    database_pool_timeout_seconds: float = Field(default=30.0, gt=0)
    database_echo: bool = False
    database_import_batch_size: int = Field(default=2000, ge=1, le=10_000)
    experiment_write_batch_size: int = Field(default=1000, ge=1, le=10_000)
    experiment_export_batch_size: int = Field(default=5000, ge=1, le=50_000)
    experiment_job_lease_seconds: int = Field(default=900, ge=1)
    experiment_artifact_prefix: str = "runs"
    baseline_corpora_dir: Path = Path("artifacts/baselines/corpora")
    full_context_config_path: Path = Path("configs/baselines/full_context.yaml")
    rag_config_path: Path = Path("configs/baselines/rag.yaml")
    evaluation_config_path: Path = Path("configs/evaluation/baseline.yaml")
    compiler_config_path: Path = Path("configs/compiler/pmmc-v1.yaml")
    questioner_config_path: Path = Path("configs/compiler/questioner-v2.yaml")

    runtime_mode: RuntimeMode = RuntimeMode.EXTERNAL
    runtime_profile_path: Path = Path("configs/runtime/h100-paper.yaml")
    model_lock_path: Path = Path("artifacts/models/paper.lock.json")
    runtime_model_lock_path: Path | None = None
    model_cache_dir: Path = Field(
        default=Path(".cache/huggingface"),
        validation_alias=AliasChoices("PMMC_MODEL_CACHE_DIR", "HF_HOME", "model_cache_dir"),
    )
    model_lock_dir: Path = Path("artifacts/models")
    runtime_calibration_dir: Path = Path("artifacts/runtime/calibrations")
    runtime_port_start: int = Field(default=8100, ge=1024, le=65535)
    runtime_port_end: int = Field(default=8199, ge=1024, le=65535)
    runtime_startup_timeout_seconds: int = Field(default=900, ge=1)
    runtime_shutdown_timeout_seconds: int = Field(default=60, ge=1)
    runtime_telemetry_interval_seconds: float = Field(default=2.0, gt=0)
    gpu_memory_accounting: Literal["process", "allocated_device"] = "process"
    runtime_require_calibration_paper: bool = True
    active_runtime_id: str | None = Field(
        default=None,
        validation_alias="PMMC_ACTIVE_RUNTIME_ID",
    )
    active_runtime_service_map: dict[str, str] = Field(
        default_factory=dict,
        validation_alias="PMMC_ACTIVE_RUNTIME_SERVICE_MAP",
    )
    active_runtime_endpoint_pools: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        validation_alias="PMMC_ACTIVE_RUNTIME_ENDPOINT_POOLS",
    )
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")

    object_store_endpoint: str | None = None
    object_store_region: str = "us-east-1"
    object_store_secure: bool = False
    object_store_path_style: bool = True
    object_store_model_endpoint: str | None = None
    object_store_presign_ttl_seconds: int = Field(default=3600, ge=60, le=86_400)
    object_store_bucket: str = "pmmc-runtime"
    object_store_access_key: SecretStr | None = None
    object_store_secret_key: SecretStr | None = None
    object_store_evaluation_bucket: str = "pmmc-evaluation"
    object_store_evaluator_access_key: SecretStr | None = None
    object_store_evaluator_secret_key: SecretStr | None = None
    object_store_admin_access_key: SecretStr | None = None
    object_store_admin_secret_key: SecretStr | None = None

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )
    step8_budget_approval_id: str | None = None
    step8_rate_card_path: Path | None = None
    step8_output_root: Path | None = None
    step8_attempt_id: str | None = None

    litellm_base_url: str | None = Field(default=None, validation_alias="LITELLM_BASE_URL")
    litellm_api_key: SecretStr | None = Field(default=None, validation_alias="LITELLM_API_KEY")

    vllm_chat_base_url: str | None = Field(
        default=None,
        validation_alias="VLLM_CHAT_BASE_URL",
    )
    vllm_embedding_base_url: str | None = Field(
        default=None,
        validation_alias="VLLM_EMBEDDING_BASE_URL",
    )
    vllm_text_embedding_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "VLLM_TEXT_EMBEDDING_BASE_URL",
            "VLLM_EMBEDDING_BASE_URL",
        ),
    )
    vllm_multimodal_embedding_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "VLLM_MULTIMODAL_EMBEDDING_BASE_URL",
            "VLLM_EMBEDDING_BASE_URL",
        ),
    )
    contrastive_embedding_base_url: str | None = Field(
        default=None,
        validation_alias="PMMC_CONTRASTIVE_EMBEDDING_BASE_URL",
    )
    vllm_api_key: SecretStr | None = Field(default=None, validation_alias="VLLM_API_KEY")

    model_registry_path: Path | None = None
    model_request_timeout_seconds: float = Field(default=300.0, gt=0)
    model_max_concurrency: int = Field(default=8, ge=1)
    compiler_cpu_workers: int | None = Field(default=None, ge=1, le=64)
    model_dev_max_attempts: int = Field(default=3, ge=1)
    model_paper_max_attempts: int = Field(default=2, ge=1)
    token_counter_backend: str = "processor-exact-v1"
    token_counter_cache_dir: Path = Path("runs/compiler-token-cache")
    token_counter_conformance_samples: int = Field(default=8, ge=1, le=64)
    qbank_text_query_batch_size: int = Field(default=32, ge=1)
    qbank_multimodal_query_batch_size: int = Field(default=8, ge=1)
    qbank_embedding_concurrency: int = Field(default=4, ge=1)
    qbank_score_concurrency: int = Field(default=8, ge=1)

    langfuse_public_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGFUSE_PUBLIC_KEY",
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGFUSE_SECRET_KEY",
    )
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")
    langfuse_enabled: bool = Field(default=False, validation_alias="LANGFUSE_ENABLED")
    langfuse_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
    )

    def resolve_path(self, value: Path) -> Path:
        """Resolve a path relative to the project root when needed."""

        if value.is_absolute():
            return value
        return self.project_root / value

    @property
    def local_asset_cache_path(self) -> Path | None:
        if not self.local_asset_cache_enabled or self.local_scratch_dir is None:
            return None
        return self.resolve_path(self.local_scratch_dir) / "benchmark-assets"

    @property
    def benchmark_asset_path(self) -> Path:
        """Return the root used to resolve canonical benchmark asset paths."""

        configured = self.benchmark_asset_root
        if configured is None:
            return self.project_root
        return configured if configured.is_absolute() else self.project_root / configured

    def resolve_benchmark_path(self, value: Path) -> Path:
        """Resolve benchmark payload paths against the independent asset root."""

        if value.is_absolute():
            return value
        return self.benchmark_asset_path / value

    @property
    def memlens_path(self) -> Path:
        return self.resolve_benchmark_path(self.memlens_root)

    @property
    def memgallery_path(self) -> Path:
        return self.resolve_benchmark_path(self.memgallery_root)

    @property
    def artifacts_path(self) -> Path:
        return self.resolve_path(self.artifacts_dir)

    @property
    def benchmark_artifacts_path(self) -> Path:
        return self.artifacts_path / "benchmarks"

    @property
    def runs_path(self) -> Path:
        return self.resolve_path(self.runs_dir)

    @property
    def outputs_path(self) -> Path:
        return self.resolve_path(self.outputs_dir)

    @property
    def experiment_outputs_path(self) -> Path:
        return self.outputs_path / "experiments"

    @property
    def baseline_corpora_path(self) -> Path:
        return self.resolve_path(self.baseline_corpora_dir)

    @property
    def resolved_full_context_config_path(self) -> Path:
        return self.resolve_path(self.full_context_config_path)

    @property
    def resolved_rag_config_path(self) -> Path:
        return self.resolve_path(self.rag_config_path)

    @property
    def resolved_evaluation_config_path(self) -> Path:
        return self.resolve_path(self.evaluation_config_path)

    @property
    def resolved_compiler_config_path(self) -> Path:
        return self.resolve_path(self.compiler_config_path)

    @property
    def resolved_questioner_config_path(self) -> Path:
        return self.resolve_path(self.questioner_config_path)

    @property
    def resolved_model_registry_path(self) -> Path:
        configured = self.model_registry_path
        if configured is None:
            mode = "paper" if self.mode is AppMode.PAPER else "dev"
            configured = Path("configs/models") / f"{mode}.yaml"
        return self.resolve_path(configured)

    @property
    def resolved_runtime_profile_path(self) -> Path:
        return self.resolve_path(self.runtime_profile_path)

    @property
    def resolved_model_lock_path(self) -> Path:
        return self.resolve_path(self.model_lock_path)

    @property
    def resolved_runtime_model_lock_path(self) -> Path | None:
        if self.runtime_model_lock_path is None:
            return None
        return self.resolve_path(self.runtime_model_lock_path)

    @property
    def model_cache_path(self) -> Path:
        return self.resolve_path(self.model_cache_dir)

    @property
    def model_locks_path(self) -> Path:
        return self.resolve_path(self.model_lock_dir)

    @property
    def runtime_calibrations_path(self) -> Path:
        return self.resolve_path(self.runtime_calibration_dir)

    @property
    def token_counter_cache_path(self) -> Path:
        return self.resolve_path(self.token_counter_cache_dir)

    @property
    def model_max_attempts(self) -> int:
        if self.mode is AppMode.PAPER:
            return self.model_paper_max_attempts
        return self.model_dev_max_attempts

    @property
    def env_file_path(self) -> Path:
        return self.project_root / ".env"

    def safe_dump(self) -> dict[str, Any]:
        """Return a JSON-serializable settings snapshot with secrets masked."""

        raw = self.model_dump()
        safe: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, SecretStr):
                safe[key] = "**********" if value.get_secret_value() else None
            elif isinstance(value, Path):
                safe[key] = str(value)
            elif isinstance(value, Enum):
                safe[key] = value.value
            else:
                safe[key] = value

        safe["memlens_path"] = str(self.memlens_path)
        safe["memgallery_path"] = str(self.memgallery_path)
        safe["artifacts_path"] = str(self.artifacts_path)
        safe["benchmark_artifacts_path"] = str(self.benchmark_artifacts_path)
        safe["runs_path"] = str(self.runs_path)
        safe["outputs_path"] = str(self.outputs_path)
        safe["experiment_outputs_path"] = str(self.experiment_outputs_path)
        safe["baseline_corpora_path"] = str(self.baseline_corpora_path)
        safe["resolved_full_context_config_path"] = str(
            self.resolved_full_context_config_path
        )
        safe["resolved_rag_config_path"] = str(self.resolved_rag_config_path)
        safe["resolved_evaluation_config_path"] = str(
            self.resolved_evaluation_config_path
        )
        safe["resolved_compiler_config_path"] = str(self.resolved_compiler_config_path)
        safe["resolved_questioner_config_path"] = str(
            self.resolved_questioner_config_path
        )
        safe["resolved_model_registry_path"] = str(self.resolved_model_registry_path)
        safe["resolved_runtime_profile_path"] = str(self.resolved_runtime_profile_path)
        safe["resolved_model_lock_path"] = str(self.resolved_model_lock_path)
        safe["resolved_runtime_model_lock_path"] = (
            str(self.resolved_runtime_model_lock_path)
            if self.resolved_runtime_model_lock_path is not None
            else None
        )
        safe["model_cache_path"] = str(self.model_cache_path)
        safe["model_locks_path"] = str(self.model_locks_path)
        safe["runtime_calibrations_path"] = str(self.runtime_calibrations_path)
        safe["model_max_attempts"] = self.model_max_attempts
        safe["env_file_present"] = self.env_file_path.is_file()
        return safe


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
