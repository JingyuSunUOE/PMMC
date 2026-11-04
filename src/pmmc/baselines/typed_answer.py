"""Source-only v3 request adapter shared by all five baseline context builders."""

from __future__ import annotations

from pmmc.baselines.rendering import render_memory_text, render_session_header
from pmmc.runtime.answering import prepare_typed_answer_request
from pmmc.runtime.grounding import GROUNDING_MODEL_ALIAS, load_step8_prompt
from pmmc.schemas import (
    BaselineMethod,
    GenerationRequest,
    ImageRef,
    MemoryUnit,
    QueryInput,
    SessionRecord,
)
from pmmc.schemas.answer_output import (
    GROUNDING_POLICY_VERSION,
    TYPED_ANSWER_CONTRACT_VERSION,
    AnswerEvidenceSource,
    AnswerSourceImage,
    TypedAnswerCandidate,
    TypedAnswerResult,
)


def baseline_answer_contract(config, metadata=None) -> str | None:
    configured = config.answer_contract_version
    declared = (metadata or {}).get("answer_contract_version")
    if configured and declared and configured != declared:
        raise ValueError("baseline config and manifest answer contract disagree")
    return declared or configured


def typed_baseline_metadata(config, metadata=None) -> dict:
    result = dict(metadata or {})
    memory_policy = getattr(config, "memory_raw_image_policy", "allow")
    if hasattr(config, "memory_raw_image_policy") and (
        result.get("memory_raw_image_policy", memory_policy) != memory_policy
        or result.get("allow_memory_raw_images", memory_policy == "allow")
        is not (memory_policy == "allow")
    ):
        raise ValueError("baseline memory pixel policy differs from its manifest")
    if memory_policy == "forbid":
        result["memory_raw_image_policy"] = "forbid"
        result["allow_memory_raw_images"] = False
    if baseline_answer_contract(config, metadata) == TYPED_ANSWER_CONTRACT_VERSION:
        from pmmc.baselines.config import prompt_sha256

        frozen = {
            "answer_contract_version": TYPED_ANSWER_CONTRACT_VERSION,
            "answer_prompt_version": "pmmc-answer-contract-v3",
            "grounding_policy_version": GROUNDING_POLICY_VERSION,
            "grounding_model_alias": GROUNDING_MODEL_ALIAS,
            "grounding_prompt_hash": prompt_sha256(load_step8_prompt("answer_grounding_v1.txt")),
        }
        if any(key in result and result[key] != value for key, value in frozen.items()):
            raise ValueError("typed baseline metadata contradicts its frozen policy")
        result.update(frozen)
    return result


def baseline_model_aliases(config, metadata, aliases: list[str]) -> list[str]:
    return (
        [*aliases, GROUNDING_MODEL_ALIAS]
        if (baseline_answer_contract(config, metadata) == TYPED_ANSWER_CONTRACT_VERSION)
        else aliases
    )


def validate_baseline_typed_terminal(result: TypedAnswerResult, *, job, method) -> None:
    if (
        result.run_id != job.run_id
        or result.snapshot_id != job.snapshot_id
        or result.query_id != job.query_id
        or result.method != method.value
        or result.answer_variant != "primary"
        or result.terminal_state not in {"answered", "abstained"}
    ):
        raise ValueError("typed baseline terminal has incompatible lineage")


async def successful_typed_answer_trace(uow, job):
    """A grounding call also has ANSWER stage; recover only the answer-role journal."""
    from sqlalchemy import select

    from pmmc.infrastructure.database.experiment_models import RuntimeModelCallModel

    model = RuntimeModelCallModel
    call_id = await uow.session.scalar(
        select(model.model_call_id)
        .where(
            model.run_id == job.run_id,
            model.query_id == job.query_id,
            model.status == "success",
            model.stage == "answer",
            model.request_config["request_metadata"]["answer_contract_version"].astext
            == TYPED_ANSWER_CONTRACT_VERSION,
            model.request_config["request_metadata"]["model_role"].astext.in_(
                ["answer", "answer_format_retry"],
            ),
        )
        .order_by(model.created_at.desc(), model.model_call_id.desc())
        .limit(1)
    )
    return await uow.model_calls.get(call_id) if call_id else None


async def typed_baseline_inventory(repository, *, run_id, selected, method) -> dict[str, int]:
    counts = {"answered": 0, "abstained": 0, "missing": 0}
    for record in selected:
        result = await repository.get_typed_answer_for_variant(run_id, record.query.query_id)
        if result is None:
            counts["missing"] += 1
        elif (
            result.run_id != run_id
            or result.query_id != record.query.query_id
            or result.method != method.value
            or result.terminal_state not in {"answered", "abstained"}
        ):
            raise ValueError("typed baseline inventory has incompatible terminals")
        else:
            counts[result.terminal_state] += 1
    return counts


async def process_typed_baseline_answer(
    *,
    prepared,
    job,
    worker_id,
    method,
    answer_service,
    uow_factory,
    phase="answer",
) -> TypedAnswerResult | TypedAnswerCandidate:
    async with uow_factory() as uow:
        if await uow.answers.get_for_query(job.run_id, job.query_id) is not None:
            raise ValueError("typed baseline cannot overlap a legacy final answer")
        result = await uow.query_executor.get_typed_answer_for_variant(job.run_id, job.query_id)
    if result is None:
        async with uow_factory() as uow:
            recovered = await successful_typed_answer_trace(uow, job)
        result = await answer_service.answer(
            prepared,
            job,
            recovered_trace=recovered,
            **({"phase": phase} if phase != "answer" else {}),
        )
        if phase == "generate":
            if not isinstance(result, TypedAnswerCandidate):
                raise ValueError("generation phase requires a nonterminal answer candidate")
            async with uow_factory() as uow:
                await uow.jobs.finish(
                    job.job_id, worker_id, success=True,
                    expected_attempt_count=job.attempt_count,
                )
            return result
        if not isinstance(result, TypedAnswerResult):
            raise ValueError("typed baseline service returned a legacy answer")
        validate_baseline_typed_terminal(result, job=job, method=method)
        async with uow_factory() as uow:
            # Lock and fence before publishing a terminal; either both facts commit or neither.
            await uow.jobs.finish(
                job.job_id, worker_id, success=True,
                expected_attempt_count=job.attempt_count,
            )
            await uow.query_executor.add_native_typed_answer(result)
    else:
        validate_baseline_typed_terminal(result, job=job, method=method)
        async with uow_factory() as uow:
            await uow.jobs.finish(
                job.job_id, worker_id, success=True,
                expected_attempt_count=job.attempt_count,
            )
    return result


def typed_generation_job_id(run_id, query_id, method) -> str:
    from pmmc.benchmarks.ids import stable_hash

    return "job-typed-generation-" + stable_hash([run_id, query_id, method.value])[:24]


async def typed_context_job(uow, job, method):
    from pmmc.schemas import AppStage

    generation = await uow.jobs.get(typed_generation_job_id(job.run_id, job.query_id, method))
    return generation if generation is not None and generation.stage is AppStage.BASELINE else job


def adapt_baseline_typed_request(
    request: GenerationRequest,
    *,
    query: QueryInput,
    memories: list[MemoryUnit],
    sessions: list[SessionRecord],
    method: BaselineMethod,
    memory_raw_image_policy: str = "allow",
) -> GenerationRequest:
    caption_only = memory_raw_image_policy == "forbid" or method in {
        BaselineMethod.FULL_CONTEXT_CAPTIONS,
        BaselineMethod.CAPTION_TEXT_RAG,
    }
    session_map = {s.session_id: s for s in sessions}
    sources = []
    for ordinal, memory in enumerate(memories, 1):
        lines = []
        if memory.session_id in session_map:
            lines.append(render_session_header(session_map[memory.session_id]))
        lines.append(render_memory_text(memory))
        refs = {i.image_id: i for i in memory.image_refs}
        image_ids = list(dict.fromkeys([*memory.image_ids, *refs]))
        images = []
        if caption_only and memory.caption:
            lines.append(f"[IMAGE CAPTION] {memory.caption}")
        for image_id in image_ids:
            image = refs.get(image_id, ImageRef(image_id=image_id))
            caption = image.caption or (memory.caption if len(image_ids) == 1 else None)
            if caption_only:
                if caption:
                    images.append(AnswerSourceImage(image_id=image_id, caption=caption))
            else:
                images.append(AnswerSourceImage(image_id=image_id, image=image))
        sources.append(
            AnswerEvidenceSource(
                evidence_alias=f"E{ordinal:02d}",
                evidence_id=memory.memory_id,
                text="\n".join(lines),
                images=images,
            )
        )
    result = prepare_typed_answer_request(request, query=query, evidence_sources=sources)
    if memory_raw_image_policy == "forbid":
        result = result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "memory_raw_image_policy": "forbid",
                }
            }
        )
    return result
