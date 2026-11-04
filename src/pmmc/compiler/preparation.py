"""Content-addressed, CPU-only preparation of Questioner compilation units."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import zstandard
from pydantic import Field, model_validator

from pmmc.compiler.ids import (
    COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1,
    canonical_content_hash,
    compilation_unit_content_id,
    compilation_unit_instance_id,
    questioner_model_call_id,
    stable_compiler_id,
)
from pmmc.compiler.questioner_config import QuestionerConfig, questioner_config_hash
from pmmc.compiler.questioner_prompt import questioner_request_contract_hash
from pmmc.compiler.units import (
    BuiltCompilationUnit,
    SessionAnchoredUnitBuilder,
    create_compilation_unit_builder,
)
from pmmc.experiments.execution import UnitPreparationCachePolicy
from pmmc.infrastructure.settings import Settings
from pmmc.models.factory import create_token_counter
from pmmc.models.registry import ModelRegistry
from pmmc.models.routing import stable_replica_index
from pmmc.schemas import (
    EXECUTION_CACHE_SCHEMA_VERSION,
    CompilationUnit,
    CompilationUnitLink,
    CompilationUnitMemory,
    DatasetName,
    MemoryUnit,
    SessionRecord,
)
from pmmc.schemas.base import (
    BaseSchema,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    UtcDateTime,
    utc_now,
)

PreparationProgress = Callable[[int, int, int, int, JsonDict], None]
LEGACY_REQUEST_CONTRACT_HASH = "legacy-unversioned-questioner-request-contract"


class TokenCounterAudit(BaseSchema):
    backend: NonEmptyStr
    revision: NonEmptyStr
    processor_calls: NonNegativeInt = 0
    request_cache_hits: NonNegativeInt = 0
    image_cache_hits: NonNegativeInt = 0
    estimated: bool = False
    metadata: JsonDict = Field(default_factory=dict)


class PreparedCompilationUnitDescriptor(BaseSchema):
    compiler_unit_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    conversation_id: NonEmptyStr
    available_after_session: NonEmptyStr
    local_ordinal: NonNegativeInt
    builder_version: NonEmptyStr
    input_hash: NonEmptyStr
    memories: list[CompilationUnitMemory]
    links: list[CompilationUnitLink]
    input_tokens: PositiveInt
    request_sha256: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_members(self) -> PreparedCompilationUnitDescriptor:
        if not self.memories:
            raise ValueError("prepared compilation unit requires memories")
        if any(item.snapshot_id != self.snapshot_id for item in self.memories):
            raise ValueError("prepared memories must use the descriptor snapshot")
        if any(item.conversation_id != self.conversation_id for item in self.memories):
            raise ValueError("prepared memories must use the descriptor conversation")
        if any(item.compiler_unit_id != self.compiler_unit_id for item in self.links):
            raise ValueError("prepared links must use the descriptor unit")
        return self

    def to_unit(
        self,
        *,
        run_id: str,
        ordinal: int,
        request_contract_hash: str | None = None,
        run_scoped_identity: bool = False,
    ) -> CompilationUnit:
        compiler_unit_id = (
            compilation_unit_instance_id(run_id=run_id, input_hash=self.input_hash)
            if run_scoped_identity
            else self.compiler_unit_id
        )
        namespace_rebound = compiler_unit_id != self.compiler_unit_id
        prepared_metadata = {
            **self.metadata,
            **(
                {
                    "compiler_unit_content_id": compilation_unit_content_id(
                        self.input_hash
                    ),
                    "compiler_unit_identity_protocol": (
                        COMPILER_UNIT_RUN_SCOPED_IDENTITY_V1
                    ),
                }
                if run_scoped_identity
                else {}
            ),
            **(
                {
                    "prepared_request_namespace_rebound": True,
                    "prepared_request_source_sha256": self.request_sha256,
                }
                if namespace_rebound
                else {"prepared_request_sha256": self.request_sha256}
            ),
            "prepared_input_tokens": self.input_tokens,
            **(
                {"prepared_request_contract_hash": request_contract_hash}
                if request_contract_hash is not None
                else {}
            ),
        }
        return CompilationUnit(
            compiler_unit_id=compiler_unit_id,
            run_id=run_id,
            snapshot_id=self.snapshot_id,
            conversation_id=self.conversation_id,
            available_after_session=self.available_after_session,
            ordinal=ordinal,
            builder_version=self.builder_version,
            input_hash=self.input_hash,
            memories=self.memories,
            metadata=prepared_metadata,
        )

    def rebound_links(
        self,
        *,
        run_id: str,
        run_scoped_identity: bool = False,
    ) -> list[CompilationUnitLink]:
        compiler_unit_id = (
            compilation_unit_instance_id(run_id=run_id, input_hash=self.input_hash)
            if run_scoped_identity
            else self.compiler_unit_id
        )
        values: list[CompilationUnitLink] = []
        for item in self.links:
            payload = [
                compiler_unit_id,
                item.anchor_memory_id,
                item.support_memory_id,
                item.relation_type.value,
            ]
            values.append(
                item.model_copy(
                    update={
                        "compiler_unit_id": compiler_unit_id,
                        "link_id": stable_compiler_id("compiler-link", payload),
                        "relation_key_hash": canonical_content_hash(payload),
                        "run_id": run_id,
                    }
                )
            )
        return values


class PreparedConversationUnits(BaseSchema):
    conversation_id: NonEmptyStr
    source_hash: NonEmptyStr
    request_contract_hash: NonEmptyStr = LEGACY_REQUEST_CONTRACT_HASH
    units: list[PreparedCompilationUnitDescriptor]
    token_counter_audit: TokenCounterAudit
    session_count: PositiveInt
    memory_count: PositiveInt


class CompilationUnitPreparationManifest(BaseSchema):
    execution_cache_schema_version: NonEmptyStr = EXECUTION_CACHE_SCHEMA_VERSION
    preparation_id: NonEmptyStr
    run_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    questioner_config_hash: NonEmptyStr
    request_contract_hash: NonEmptyStr = LEGACY_REQUEST_CONTRACT_HASH
    processor_revision: NonEmptyStr
    unit_builder_version: NonEmptyStr
    conversation_ids: list[NonEmptyStr]
    conversation_count: PositiveInt
    unit_count: PositiveInt
    index_sha256: NonEmptyStr
    cache_policy: UnitPreparationCachePolicy
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


def unit_preparation_identity(
    *,
    snapshot_id: str,
    conversation_ids: Sequence[str],
    config: QuestionerConfig,
    processor_revision: str,
    dataset: DatasetName | None = None,
) -> str:
    request_contract_hash = questioner_request_contract_hash(
        prompt_version=config.prompt_version,
        renderer_version=config.renderer_version,
        opportunity_builder_version=(
            config.opportunity_builder_version if config.is_v3 else None
        ),
    )
    payload = {
        "schema": EXECUTION_CACHE_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "conversation_ids": sorted(conversation_ids),
        "questioner_config_hash": questioner_config_hash(config),
        "request_contract_hash": request_contract_hash,
        "processor_revision": processor_revision,
        "unit_builder_version": config.unit_builder_for(dataset or DatasetName.UNKNOWN),
    }
    return "unit-preparation-" + canonical_content_hash(payload)[:24]


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _audit_totals(
    values: Sequence[PreparedConversationUnits],
    *,
    reused_conversations: int,
    prepared_conversation_ids: set[str],
) -> JsonDict:
    prepared = [
        item for item in values if item.conversation_id in prepared_conversation_ids
    ]
    cached = [
        item for item in values if item.conversation_id not in prepared_conversation_ids
    ]
    return {
        "reused_conversations": reused_conversations,
        "processor_calls": sum(
            item.token_counter_audit.processor_calls for item in prepared
        ),
        "request_cache_hits": sum(
            item.token_counter_audit.request_cache_hits for item in prepared
        ),
        "image_cache_hits": sum(
            item.token_counter_audit.image_cache_hits for item in prepared
        ),
        "cached_processor_calls": sum(
            item.token_counter_audit.processor_calls for item in cached
        ),
        "cached_request_cache_hits": sum(
            item.token_counter_audit.request_cache_hits for item in cached
        ),
        "cached_image_cache_hits": sum(
            item.token_counter_audit.image_cache_hits for item in cached
        ),
    }


def _append_event(path: Path, value: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("ab") as stream:
        stream.write(_canonical_bytes(value))
    os.chmod(path, 0o600)


@contextmanager
def _exclusive_preparation_lock(path: Path) -> Iterator[None]:
    """Serialize writers; an abandoned lock file is harmless after process exit."""

    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    with path.open("a+", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _source_hash(
    sessions: Sequence[SessionRecord], memories: Sequence[MemoryUnit]
) -> str:
    return canonical_content_hash(
        {
            "sessions": [item.model_dump(mode="json") for item in sessions],
            "memories": [item.model_dump(mode="json") for item in memories],
        }
    )


def _descriptor_from_built(value: BuiltCompilationUnit) -> PreparedCompilationUnitDescriptor:
    request_payload = value.request.to_json().encode("utf-8")
    return PreparedCompilationUnitDescriptor(
        compiler_unit_id=value.unit.compiler_unit_id,
        snapshot_id=value.unit.snapshot_id,
        conversation_id=value.unit.conversation_id,
        available_after_session=value.unit.available_after_session,
        local_ordinal=value.unit.ordinal,
        builder_version=value.unit.builder_version,
        input_hash=value.unit.input_hash,
        memories=value.unit.memories,
        links=list(value.links),
        input_tokens=value.input_tokens,
        request_sha256=hashlib.sha256(request_payload).hexdigest(),
        metadata=dict(value.unit.metadata),
    )


_WORKER_CONTEXT: dict[str, Any] = {}


def _initialize_worker(
    settings_payload: dict[str, Any],
    registry_payload: dict[str, Any],
    config_payload: dict[str, Any],
) -> None:
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    settings = Settings.model_validate(settings_payload)
    registry = ModelRegistry.model_validate(registry_payload)
    config = QuestionerConfig.model_validate(config_payload)
    fts_relation_provider = None
    if config.is_v3:
        if settings.database_url is None:
            if settings.mode.value == "paper":
                raise ValueError(
                    "Questioner v3 paper unit preparation requires PostgreSQL FTS"
                )
        else:
            from pmmc.infrastructure.database.questioner_fts import (
                PostgresSimpleFtsRelationProvider,
            )

            fts_relation_provider = PostgresSimpleFtsRelationProvider(settings)
    counter = create_token_counter(
        settings,
        registry,
        config.compiler_model_alias,
        allow_estimated=settings.mode.value != "paper",
    )
    _WORKER_CONTEXT.update(
        {
            "config": config,
            "counter": counter,
            "fts_relation_provider": fts_relation_provider,
            "last_audit": {
                "processor_calls": 0,
                "request_cache_hits": 0,
                "image_cache_hits": 0,
            },
        }
    )


def _prepare_worker(payload: dict[str, Any]) -> dict[str, Any]:
    config: QuestionerConfig = _WORKER_CONTEXT["config"]
    counter = _WORKER_CONTEXT["counter"]
    sessions = [SessionRecord.model_validate(item) for item in payload["sessions"]]
    memories = [MemoryUnit.model_validate(item) for item in payload["memories"]]
    datasets = {item.dataset for item in memories}
    if len(datasets) != 1:
        raise ValueError("one prepared conversation cannot mix datasets")
    dataset = next(iter(datasets))
    builder: SessionAnchoredUnitBuilder = create_compilation_unit_builder(
        config=config,
        token_counter=counter,
        dataset=dataset,
        fts_relation_provider=_WORKER_CONTEXT.get("fts_relation_provider"),
    )
    built = builder.build(
        run_id=payload["run_id"],
        snapshot_id=payload["snapshot_id"],
        sessions=sessions,
        memories=memories,
    )
    raw_audit = getattr(counter, "audit", {})
    previous_audit: dict[str, int] = _WORKER_CONTEXT["last_audit"]
    current_audit = {
        name: int(raw_audit.get(name, 0))
        for name in (
            "processor_calls",
            "request_cache_hits",
            "image_cache_hits",
        )
    }
    audit_delta = {
        name: max(0, current_audit[name] - previous_audit.get(name, 0))
        for name in current_audit
    }
    _WORKER_CONTEXT["last_audit"] = current_audit
    audit = TokenCounterAudit(
        backend=str(raw_audit.get("backend", type(counter).__qualname__)),
        revision=counter.revision,
        processor_calls=audit_delta["processor_calls"],
        request_cache_hits=audit_delta["request_cache_hits"],
        image_cache_hits=audit_delta["image_cache_hits"],
        estimated=bool(counter.estimated),
        metadata={
            "protocol": config.unit_builder_for(dataset),
            **(
                {
                    "relation_selector_version": (
                        config.relation_selector_version
                        if _WORKER_CONTEXT.get("fts_relation_provider") is not None
                        else "deterministic-dev-fallback-v1"
                    ),
                    "fts_recall_selector_version": (
                        _WORKER_CONTEXT["fts_relation_provider"].selector_version
                        if _WORKER_CONTEXT.get("fts_relation_provider") is not None
                        else "deterministic-dev-fallback-v1"
                    ),
                }
                if config.is_v3
                else {}
            ),
        },
    )
    result = PreparedConversationUnits(
        conversation_id=payload["conversation_id"],
        source_hash=payload["source_hash"],
        request_contract_hash=payload["request_contract_hash"],
        units=[_descriptor_from_built(item) for item in built],
        token_counter_audit=audit,
        session_count=len(sessions),
        memory_count=len(memories),
    )
    return result.model_dump(mode="json")


class CompilationUnitPreparationService:
    """Prepare and verify per-conversation unit caches before run registration."""

    def __init__(
        self,
        *,
        settings: Settings,
        registry: ModelRegistry,
        config: QuestionerConfig,
        workers: int = 8,
        max_tasks_per_child: int = 16,
        cache_policy: UnitPreparationCachePolicy = UnitPreparationCachePolicy.READ_WRITE,
    ) -> None:
        if workers <= 0 or max_tasks_per_child <= 0:
            raise ValueError("unit preparation worker settings must be positive")
        self.settings = settings
        self.registry = registry
        self.config = config
        self.workers = workers
        self.max_tasks_per_child = max_tasks_per_child
        self.cache_policy = cache_policy

    @staticmethod
    def _conversation_path(root: Path, ordinal: int, conversation_id: str) -> Path:
        digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:16]
        return root / "conversations" / f"{ordinal:06d}-{digest}.json.zst"

    @staticmethod
    def _write_conversation(path: Path, value: PreparedConversationUnits) -> str:
        payload = _canonical_bytes(value.model_dump(mode="json"))
        compressed = zstandard.ZstdCompressor(level=9).compress(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(compressed)
        os.chmod(temporary, 0o600)
        temporary.replace(path)
        return hashlib.sha256(compressed).hexdigest()

    @staticmethod
    def _read_conversation(path: Path) -> PreparedConversationUnits:
        payload = zstandard.ZstdDecompressor().decompress(path.read_bytes())
        raw = json.loads(payload)
        return PreparedConversationUnits.model_validate(raw)

    def prepare(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        sessions: Sequence[SessionRecord],
        memories: Sequence[MemoryUnit],
        on_progress: PreparationProgress | None = None,
    ) -> tuple[CompilationUnitPreparationManifest, list[PreparedCompilationUnitDescriptor], Path]:
        conversation_ids = sorted({item.conversation_id for item in sessions})
        if not conversation_ids:
            raise ValueError("unit preparation conversation denominator is empty")
        datasets = {item.dataset for item in memories}
        if len(datasets) != 1:
            raise ValueError("unit preparation must contain exactly one dataset")
        dataset = next(iter(datasets))
        model_entry = self.registry.get(self.config.compiler_model_alias)
        processor_revision = (
            f"{model_entry.tokenizer_revision}+{model_entry.processor_revision}"
            if model_entry.processor_revision is not None
            else model_entry.tokenizer_revision
        )
        preparation_id = unit_preparation_identity(
            snapshot_id=snapshot_id,
            conversation_ids=conversation_ids,
            config=self.config,
            processor_revision=processor_revision,
            dataset=dataset,
        )
        lock_path = (
            self.settings.runs_path
            / "compiler-unit-preparations"
            / ".locks"
            / f"{preparation_id}.lock"
        )
        with _exclusive_preparation_lock(lock_path):
            return self._prepare_unlocked(
                run_id=run_id,
                snapshot_id=snapshot_id,
                sessions=sessions,
                memories=memories,
                on_progress=on_progress,
            )

    def _prepare_unlocked(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        sessions: Sequence[SessionRecord],
        memories: Sequence[MemoryUnit],
        on_progress: PreparationProgress | None = None,
    ) -> tuple[CompilationUnitPreparationManifest, list[PreparedCompilationUnitDescriptor], Path]:
        sessions_by_conversation: dict[str, list[SessionRecord]] = defaultdict(list)
        memories_by_conversation: dict[str, list[MemoryUnit]] = defaultdict(list)
        for session_record in sessions:
            sessions_by_conversation[session_record.conversation_id].append(
                session_record
            )
        for memory_record in memories:
            memories_by_conversation[memory_record.conversation_id].append(memory_record)
        conversation_ids = sorted(sessions_by_conversation)
        if not conversation_ids:
            raise ValueError("unit preparation conversation denominator is empty")
        if any(not memories_by_conversation[item] for item in conversation_ids):
            raise ValueError("unit preparation requires memory for every conversation")
        datasets = {item.dataset for item in memories}
        if len(datasets) != 1:
            raise ValueError("unit preparation must contain exactly one dataset")
        dataset = next(iter(datasets))

        model_entry = self.registry.get(self.config.compiler_model_alias)
        processor_revision = (
            f"{model_entry.tokenizer_revision}+{model_entry.processor_revision}"
            if model_entry.processor_revision is not None
            else model_entry.tokenizer_revision
        )
        preparation_id = unit_preparation_identity(
            snapshot_id=snapshot_id,
            conversation_ids=conversation_ids,
            config=self.config,
            processor_revision=processor_revision,
            dataset=dataset,
        )
        request_contract_hash = questioner_request_contract_hash(
            prompt_version=self.config.prompt_version,
            renderer_version=self.config.renderer_version,
            opportunity_builder_version=(
                self.config.opportunity_builder_version
                if self.config.is_v3
                else None
            ),
        )
        root = self.settings.runs_path / "compiler-unit-preparations" / preparation_id
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        events_path = root / "events.jsonl"
        results: dict[str, PreparedConversationUnits] = {}
        pending: list[dict[str, Any]] = []
        prepared_conversation_ids: set[str] = set()
        reused = 0
        for ordinal, conversation_id in enumerate(conversation_ids):
            conversation_sessions = sorted(
                sessions_by_conversation[conversation_id],
                key=lambda item: (item.session_order, item.session_id),
            )
            conversation_memories = sorted(
                memories_by_conversation[conversation_id],
                key=lambda item: (item.session_id, item.timestamp_order, item.memory_id),
            )
            source_hash = _source_hash(conversation_sessions, conversation_memories)
            path = self._conversation_path(root, ordinal, conversation_id)
            if (
                self.cache_policy
                in {UnitPreparationCachePolicy.READ_ONLY, UnitPreparationCachePolicy.READ_WRITE}
                and path.is_file()
            ):
                try:
                    cached = self._read_conversation(path)
                    if (
                        cached.conversation_id == conversation_id
                        and cached.source_hash == source_hash
                        and cached.request_contract_hash == request_contract_hash
                    ):
                        results[conversation_id] = cached
                        reused += 1
                        if on_progress is not None:
                            audit = _audit_totals(
                                list(results.values()),
                                reused_conversations=reused,
                                prepared_conversation_ids=prepared_conversation_ids,
                            )
                            on_progress(
                                len(results), len(conversation_ids),
                                sum(item.session_count for item in results.values()),
                                sum(len(item.units) for item in results.values()),
                                audit,
                            )
                        _append_event(
                            events_path,
                            {
                                "conversation_ordinal": ordinal,
                                "state": "reused",
                                "unit_count": len(cached.units),
                            },
                        )
                        continue
                except (ValueError, json.JSONDecodeError, zstandard.ZstdError):
                    if self.cache_policy is UnitPreparationCachePolicy.READ_ONLY:
                        raise ValueError("unit preparation cache is corrupt") from None
            if self.cache_policy is UnitPreparationCachePolicy.READ_ONLY:
                raise FileNotFoundError(
                    f"unit preparation cache miss for conversation {conversation_id}"
                )
            pending.append(
                {
                    "run_id": run_id,
                    "snapshot_id": snapshot_id,
                    "conversation_id": conversation_id,
                    "source_hash": source_hash,
                    "request_contract_hash": request_contract_hash,
                    "ordinal": ordinal,
                    "sessions": [item.model_dump(mode="json") for item in conversation_sessions],
                    "memories": [item.model_dump(mode="json") for item in conversation_memories],
                }
            )

        settings_payload = self.settings.model_dump(mode="python")
        registry_payload = self.registry.model_dump(mode="python")
        config_payload = self.config.model_dump(mode="python")
        if pending and self.workers == 1:
            _WORKER_CONTEXT.clear()
            _initialize_worker(settings_payload, registry_payload, config_payload)
            for payload in pending:
                result = PreparedConversationUnits.model_validate(_prepare_worker(payload))
                results[result.conversation_id] = result
                prepared_conversation_ids.add(result.conversation_id)
                ordinal = conversation_ids.index(result.conversation_id)
                self._write_conversation(
                    self._conversation_path(root, ordinal, result.conversation_id), result
                )
                if on_progress is not None:
                    audit = _audit_totals(
                        list(results.values()),
                        reused_conversations=reused,
                        prepared_conversation_ids=prepared_conversation_ids,
                    )
                    on_progress(
                        len(results), len(conversation_ids),
                        sum(item.session_count for item in results.values()),
                        sum(len(item.units) for item in results.values()),
                        audit,
                    )
                _append_event(
                    events_path,
                    {
                        "conversation_ordinal": ordinal,
                        "state": "prepared",
                        "unit_count": len(result.units),
                    },
                )
        elif pending:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(self.workers, len(pending)),
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(settings_payload, registry_payload, config_payload),
                max_tasks_per_child=self.max_tasks_per_child,
            ) as executor:
                futures = {executor.submit(_prepare_worker, item): item for item in pending}
                for future in as_completed(futures):
                    result = PreparedConversationUnits.model_validate(future.result())
                    results[result.conversation_id] = result
                    prepared_conversation_ids.add(result.conversation_id)
                    ordinal = conversation_ids.index(result.conversation_id)
                    self._write_conversation(
                        self._conversation_path(root, ordinal, result.conversation_id), result
                    )
                    if on_progress is not None:
                        audit = _audit_totals(
                            list(results.values()),
                            reused_conversations=reused,
                            prepared_conversation_ids=prepared_conversation_ids,
                        )
                        on_progress(
                            len(results), len(conversation_ids),
                            sum(item.session_count for item in results.values()),
                            sum(len(item.units) for item in results.values()),
                            audit,
                        )
                    _append_event(
                        events_path,
                        {
                            "conversation_ordinal": ordinal,
                            "state": "prepared",
                            "unit_count": len(result.units),
                        },
                    )

        descriptors: list[PreparedCompilationUnitDescriptor] = []
        index_rows: list[dict[str, Any]] = []
        global_ordinal = 0
        for conversation_ordinal, conversation_id in enumerate(conversation_ids):
            value = results[conversation_id]
            path = self._conversation_path(root, conversation_ordinal, conversation_id)
            compressed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            for descriptor in sorted(value.units, key=lambda item: item.local_ordinal):
                descriptors.append(
                    descriptor.model_copy(update={"local_ordinal": global_ordinal})
                )
                global_ordinal += 1
            index_rows.append(
                {
                    "conversation_id": conversation_id,
                    "ordinal": conversation_ordinal,
                    "path": str(path.relative_to(root)),
                    "sha256": compressed_sha,
                    "source_hash": value.source_hash,
                    "unit_count": len(value.units),
                }
            )
        if not descriptors:
            raise ValueError("unit preparation produced an empty denominator")
        index_bytes = b"".join(_canonical_bytes(item) for item in index_rows)
        index_path = root / "index.jsonl"
        temporary_index = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
        temporary_index.write_bytes(index_bytes)
        os.chmod(temporary_index, 0o600)
        temporary_index.replace(index_path)
        manifest = CompilationUnitPreparationManifest(
            preparation_id=preparation_id,
            run_id=run_id,
            snapshot_id=snapshot_id,
            questioner_config_hash=questioner_config_hash(self.config),
            request_contract_hash=request_contract_hash,
            processor_revision=processor_revision,
            unit_builder_version=self.config.unit_builder_for(dataset),
            conversation_ids=conversation_ids,
            conversation_count=len(conversation_ids),
            unit_count=len(descriptors),
            index_sha256=hashlib.sha256(index_bytes).hexdigest(),
            cache_policy=self.cache_policy,
            metadata={
                **_audit_totals(
                    list(results.values()),
                    reused_conversations=reused,
                    prepared_conversation_ids=prepared_conversation_ids,
                ),
                **(
                    {
                        "relation_selector_version": (
                            self.config.relation_selector_version
                            if self.settings.database_url is not None
                            else "deterministic-dev-fallback-v1"
                        )
                    }
                    if self.config.is_v3
                    else {}
                ),
            },
        )
        manifest_path = root / "manifest.json"
        temporary_manifest = manifest_path.with_name(
            f".{manifest_path.name}.{os.getpid()}.tmp"
        )
        temporary_manifest.write_bytes(_canonical_bytes(manifest.model_dump(mode="json")))
        os.chmod(temporary_manifest, 0o600)
        temporary_manifest.replace(manifest_path)
        self.verify(root)
        return manifest, descriptors, root

    @classmethod
    def verify(
        cls, root: Path
    ) -> tuple[CompilationUnitPreparationManifest, list[PreparedCompilationUnitDescriptor]]:
        manifest = CompilationUnitPreparationManifest.from_json(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        index_bytes = (root / "index.jsonl").read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != manifest.index_sha256:
            raise ValueError("unit preparation index checksum mismatch")
        rows = [json.loads(line) for line in index_bytes.splitlines() if line]
        if len(rows) != manifest.conversation_count:
            raise ValueError("unit preparation conversation count mismatch")
        descriptors: list[PreparedCompilationUnitDescriptor] = []
        for row in rows:
            path = root / str(row["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("unit preparation conversation checksum mismatch")
            value = cls._read_conversation(path)
            if (
                value.conversation_id != row["conversation_id"]
                or value.source_hash != row["source_hash"]
                or len(value.units) != row["unit_count"]
            ):
                raise ValueError("unit preparation conversation identity mismatch")
            if value.request_contract_hash != manifest.request_contract_hash:
                raise ValueError("unit preparation request contract mismatch")
            descriptors.extend(value.units)
        ordered = [
            item.model_copy(update={"local_ordinal": ordinal})
            for ordinal, item in enumerate(
                sorted(
                    descriptors,
                    key=lambda item: (
                        manifest.conversation_ids.index(item.conversation_id),
                        item.local_ordinal,
                    ),
                )
            )
        ]
        if len(ordered) != manifest.unit_count:
            raise ValueError("unit preparation unit count mismatch")
        selector = manifest.metadata.get("relation_selector_version")
        if selector is not None and any(
            item.metadata.get("relation_selector_version") != selector
            for item in ordered
        ):
            raise ValueError("unit preparation relation selector lineage mismatch")
        verification = {
            "preparation_id": manifest.preparation_id,
            "conversation_count": manifest.conversation_count,
            "unit_count": manifest.unit_count,
            "status": "ready",
        }
        verification_path = root / "verification.json"
        verification_path.write_bytes(_canonical_bytes(verification))
        os.chmod(verification_path, 0o600)
        return manifest, ordered


def slice_unit_preparation(
    root: Path,
    *,
    max_units: int,
    replica_count: int = 1,
) -> tuple[CompilationUnitPreparationManifest, Path]:
    """Create a signed execution-only subset without mutating the source cache."""

    if max_units <= 0:
        raise ValueError("max_units must be positive")
    if replica_count <= 0:
        raise ValueError("replica_count must be positive")
    if max_units < replica_count:
        raise ValueError("unit slice cannot cover every requested replica")
    manifest, descriptors = CompilationUnitPreparationService.verify(root)
    resolved_run_id, routing_salt = resolve_replica_covering_run_id(
        manifest.run_id,
        descriptors,
        replica_count=replica_count,
    )
    selected_ids: set[str] = set()
    if replica_count > 1:
        for replica_index in range(replica_count):
            selected_descriptor = next(
                (
                    item
                    for item in descriptors
                    if stable_replica_index(
                        questioner_model_call_id(
                            run_id=resolved_run_id,
                            compiler_unit_id=item.compiler_unit_id,
                            attempt_number=1,
                        ),
                        replica_count,
                    )
                    == replica_index
                ),
                None,
            )
            assert selected_descriptor is not None
            selected_ids.add(selected_descriptor.compiler_unit_id)
    for descriptor in descriptors:
        if len(selected_ids) >= min(max_units, len(descriptors)):
            break
        selected_ids.add(descriptor.compiler_unit_id)
    selected_descriptors = [
        item for item in descriptors if item.compiler_unit_id in selected_ids
    ]
    if len(selected_descriptors) == len(descriptors) and resolved_run_id == manifest.run_id:
        return manifest, root
    identity = canonical_content_hash(
        {
            "source_preparation_id": manifest.preparation_id,
            "source_run_id": manifest.run_id,
            "resolved_run_id": resolved_run_id,
            "selected_unit_ids": [
                item.compiler_unit_id for item in selected_descriptors
            ],
            "replica_count": replica_count,
            "protocol": "unit-preparation-slice-v3",
        }
    )
    derived_root = root.parent / f"{manifest.preparation_id}-slice-{identity[:12]}"
    derived_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(derived_root, 0o700)
    source_rows = [
        json.loads(line)
        for line in (root / "index.jsonl").read_bytes().splitlines()
        if line
    ]
    index_rows: list[dict[str, Any]] = []
    for row in source_rows:
        value = CompilationUnitPreparationService._read_conversation(
            root / str(row["path"])
        )
        sliced = value.model_copy(
            update={
                "units": [
                    item for item in value.units if item.compiler_unit_id in selected_ids
                ]
            }
        )
        ordinal = int(row["ordinal"])
        target = CompilationUnitPreparationService._conversation_path(
            derived_root,
            ordinal,
            value.conversation_id,
        )
        compressed_sha = CompilationUnitPreparationService._write_conversation(
            target,
            sliced,
        )
        index_rows.append(
            {
                **row,
                "path": str(target.relative_to(derived_root)),
                "sha256": compressed_sha,
                "unit_count": len(sliced.units),
            }
        )
    index_bytes = b"".join(_canonical_bytes(item) for item in index_rows)
    (derived_root / "index.jsonl").write_bytes(index_bytes)
    os.chmod(derived_root / "index.jsonl", 0o600)
    derived = manifest.model_copy(
        update={
            "preparation_id": f"unit-preparation-slice-{identity[:24]}",
            "run_id": resolved_run_id,
            "unit_count": len(selected_descriptors),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "created_at": utc_now(),
            "metadata": {
                **manifest.metadata,
                "execution_only_slice": True,
                "source_preparation_id": manifest.preparation_id,
                "source_run_id": manifest.run_id,
                "source_unit_count": manifest.unit_count,
                "max_units": max_units,
                "replica_count": replica_count,
                "replica_routing_salt": routing_salt,
                "covered_replica_indices": list(range(replica_count)),
            },
        }
    )
    (derived_root / "manifest.json").write_bytes(
        _canonical_bytes(derived.model_dump(mode="json"))
    )
    os.chmod(derived_root / "manifest.json", 0o600)
    CompilationUnitPreparationService.verify(derived_root)
    return derived, derived_root


def resolve_replica_covering_run_id(
    base_run_id: str,
    descriptors: Sequence[PreparedCompilationUnitDescriptor],
    *,
    replica_count: int,
    max_salts: int = 65_536,
) -> tuple[str, int]:
    """Resolve a deterministic smoke-only run ID whose requests cover the pool."""

    if not base_run_id:
        raise ValueError("replica routing requires a base run ID")
    if replica_count <= 0 or max_salts <= 0:
        raise ValueError("replica routing limits must be positive")
    if len(descriptors) < replica_count:
        raise ValueError("unit preparation cannot exercise every compiler replica")
    expected = set(range(replica_count))
    for salt in range(max_salts):
        candidate = base_run_id if salt == 0 else f"{base_run_id}-route-{salt:04x}"
        covered = {
            stable_replica_index(
                questioner_model_call_id(
                    run_id=candidate,
                    compiler_unit_id=item.compiler_unit_id,
                    attempt_number=1,
                ),
                replica_count,
            )
            for item in descriptors
        }
        if covered == expected:
            return candidate, salt
    raise ValueError("unable to resolve complete compiler replica routing")
