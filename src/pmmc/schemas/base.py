"""Shared schema primitives for PMMC."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, TypeVar

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

SCHEMA_VERSION = "0.3.0"
EXPERIMENT_SCHEMA_VERSION = "0.14.0"
RUNTIME_SCHEMA_VERSION = "0.3.0"
LAUNCHER_SCHEMA_VERSION = "0.10.0"
SUPPORTED_LAUNCHER_SCHEMA_VERSIONS = frozenset(
    {
        "0.1.0",
        "0.2.0",
        "0.3.0",
        "0.4.0",
        "0.5.0",
        "0.6.0",
        "0.7.0",
        "0.8.0",
        "0.9.0",
        "0.11.0",
        LAUNCHER_SCHEMA_VERSION,
    }
)
EXECUTION_CACHE_SCHEMA_VERSION = "0.8.0"
COMPILER_SCHEMA_VERSION = "0.8.0"
QUESTIONER_PILOT_SCHEMA_VERSION = "0.8.0"
PROGRAM_PILOT_SCHEMA_VERSION = "0.5.0"
QBANK_SEMANTIC_SCHEMA_VERSION = "0.4.0"
DOUBTER_SCHEMA_VERSION = "0.3.0"
VERIFIED_QBANK_SCHEMA_VERSION = "0.3.0"
STAGE_COMPONENT_SCHEMA_VERSION = "0.3.0"
QUERY_EXECUTOR_SCHEMA_VERSION = "0.2.0"
STEP7_PILOT_SCHEMA_VERSION = "0.1.0"
ANSWER_OUTPUT_SCHEMA_VERSION = "0.1.0"
STEP7B_PILOT_SCHEMA_VERSION = "0.1.0"
MODEL_VERIFICATION_SCHEMA_VERSION = "0.1.0"
RUNTIME_REPLICA_SCHEMA_VERSION = "0.1.0"
BASELINE_COMPONENT_SCHEMA_VERSION = "0.1.0"
SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS = frozenset(
    {
        "0.1.0",
        "0.2.0",
        "0.3.0",
        "0.4.0",
        "0.5.0",
        "0.6.0",
        "0.7.0",
        "0.8.0",
        "0.9.0",
        "0.10.0",
        "0.11.0",
        "0.12.0",
        "0.13.0",
        "0.15.0",
        EXPERIMENT_SCHEMA_VERSION,
    }
)
SUPPORTED_COMPILER_SCHEMA_VERSIONS = frozenset(
    {
        "0.1.0",
        "0.2.0",
        "0.3.0",
        "0.4.0",
        "0.5.0",
        "0.6.0",
        "0.7.0",
        "0.9.0",
        COMPILER_SCHEMA_VERSION,
    }
)

JsonDict = dict[str, Any]
SchemaT = TypeVar("SchemaT", bound="BaseSchema")

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


def normalize_to_utc(value: datetime) -> datetime:
    """Normalize an aware datetime to UTC."""

    return value.astimezone(UTC)


UtcDateTime = Annotated[AwareDatetime, AfterValidator(normalize_to_utc)]


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(UTC)


class BaseSchema(BaseModel):
    """Base model for PMMC's versioned Pydantic v2 contracts."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    def to_dict(self) -> JsonDict:
        """Return a JSON-compatible dictionary."""

        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Return a JSON string."""

        return self.model_dump_json()

    @classmethod
    def from_dict(cls: type[SchemaT], data: JsonDict) -> SchemaT:
        """Build a schema from a dictionary."""

        return cls.model_validate(data)

    @classmethod
    def from_json(cls: type[SchemaT], data: str) -> SchemaT:
        """Build a schema from a JSON string."""

        return cls.model_validate_json(data)


class DatasetName(StrEnum):
    """Datasets supported by the PMMC schema layer."""

    MEMLENS = "memlens"
    MEMGALLERY = "memgallery"
    UNKNOWN = "unknown"


class SpeakerRole(StrEnum):
    """Speaker roles in multimodal interaction histories."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class Modality(StrEnum):
    """Modalities and modality-derived representations used by PMMC."""

    TEXT = "text"
    CAPTION = "caption"
    IMAGE = "image"
    RAW_IMAGE = "raw_image"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_EMBEDDING = "image_embedding"
    JOINT_MM_EMBEDDING = "joint_mm_embedding"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"


class QuestionType(StrEnum):
    """Internal PMMC question taxonomy."""

    TEXTUAL_FACT = "textual_fact"
    VISUAL_DETAIL = "visual_detail"
    IMAGE_TEXT_BINDING = "image_text_binding"
    VISUAL_SEARCH = "visual_search"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    KNOWLEDGE_UPDATE = "knowledge_update"
    CONFLICT_DETECTION = "conflict_detection"
    ANSWER_REFUSAL = "answer_refusal"
    PERSONAL_PREFERENCE = "personal_preference"
    TEST_TIME_LEARNING = "test_time_learning"
    VISUAL_REASONING = "visual_reasoning"
    MULTI_ENTITY_REASONING = "multi_entity_reasoning"
    UNKNOWN = "unknown"


class EvidenceType(StrEnum):
    """Evidence entity categories."""

    MEMORY = "memory"
    TEXT = "text"
    CAPTION = "caption"
    IMAGE = "image"
    RAW_IMAGE = "raw_image"
    IMAGE_TEXT_PAIR = "image_text_pair"
    SESSION = "session"
    QUERY = "query"
    UNKNOWN = "unknown"


class AppStage(StrEnum):
    """Pipeline stages used in traces and model-call records."""

    COMPILE = "compile"
    QUESTIONER = "questioner"
    PLANNER = "planner"
    DOUBTER = "doubter"
    MATCH = "match"
    RETRIEVE = "retrieve"
    EXECUTE = "execute"
    ANSWER = "answer"
    JUDGE = "judge"
    ANALYZE = "analyze"
    BASELINE = "baseline"
    EMBED = "embed"


class EvidenceSufficiency(StrEnum):
    """Doubter and executor evidence sufficiency states."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class AssetType(StrEnum):
    """Large artifact types tracked outside the core database."""

    RAW_IMAGE = "raw_image"
    RESIZED_IMAGE = "resized_image"
    PROMPT = "prompt"
    MODEL_RESPONSE = "model_response"
    TRACE_EXPORT = "trace_export"
    EMBEDDING_DUMP = "embedding_dump"
    DATASET_SNAPSHOT = "dataset_snapshot"
    OTHER = "other"


class TraceStatus(StrEnum):
    """Execution status for traces."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


def ensure_unique_strings(values: list[str], field_name: str) -> None:
    """Reject duplicate strings in identifier-like lists."""

    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate values")
