"""Versioned, deterministic baseline prompt and MemoryUnit renderers."""

from __future__ import annotations

from pmmc.schemas import (
    BaselineMethod,
    ImageRef,
    MemoryUnit,
    ModelContentKind,
    ModelContentPart,
    ModelMessage,
    QueryInput,
    SessionRecord,
    SpeakerRole,
)

RENDERER_VERSION = "baseline-renderer-v1"
FULL_CONTEXT_RENDERER_VERSION = "full-context-renderer-v1"
RAG_RENDERER_VERSION = "rag-renderer-v1"


def render_memory_text(memory: MemoryUnit) -> str:
    """Render dialogue text without benchmark captions or image payloads."""

    speaker = memory.speaker.value.upper()
    body = memory.text or ""
    return f"[{speaker}] {body}".rstrip()


def render_memory_text_caption(memory: MemoryUnit) -> str:
    """Render dialogue text and benchmark-provided captions for caption baselines."""

    parts = [render_memory_text(memory)]
    if memory.caption:
        parts.append(f"[IMAGE CAPTION] {memory.caption}")
    for image in memory.image_refs:
        if image.caption and image.caption != memory.caption:
            parts.append(f"[IMAGE CAPTION] {image.caption}")
    return "\n".join(parts)


def render_session_header(session: SessionRecord) -> str:
    """Render a canonical session boundary without inferring time from metadata."""

    session_type = "profile" if session.is_synthetic else "dialogue"
    date = f" date={session.source_date_iso}" if session.source_date_iso else ""
    return f"[SESSION order={session.session_order}{date} type={session_type}]"


def render_query_text(query: QueryInput) -> str:
    """Render runtime-safe query time when the adapter supplied one."""

    query_date = query.metadata.get("question_date_iso")
    date = f" date={query_date}" if isinstance(query_date, str) and query_date else ""
    return f"[QUERY{date}]\n{query.raw_query}"


def render_full_context_messages(
    *,
    instruction: str,
    query: QueryInput,
    memories: list[MemoryUnit],
    method: BaselineMethod,
    sessions: list[SessionRecord] | None = None,
) -> list[ModelMessage]:
    """Render the exact provider-neutral message sequence used by full-context baselines."""

    if method not in {
        BaselineMethod.FULL_CONTEXT_CAPTIONS,
        BaselineMethod.FULL_CONTEXT_RAW_IMAGES,
    }:
        raise ValueError("full-context message rendering requires a full-context method")

    def memory_image_refs(memory: MemoryUnit) -> list[ImageRef]:
        refs = {image.image_id: image for image in memory.image_refs}
        image_ids = list(
            dict.fromkeys([*memory.image_ids, *(image.image_id for image in memory.image_refs)])
        )
        return [refs.get(image_id, ImageRef(image_id=image_id)) for image_id in image_ids]

    session_map = {session.session_id: session for session in sessions or []}
    if len(session_map) != len(sessions or []):
        raise ValueError("full-context sessions cannot contain duplicate IDs")
    if sessions is not None:
        if any(session.conversation_id != query.conversation_id for session in sessions):
            raise ValueError("full-context sessions must belong to the query conversation")
        missing = sorted({memory.session_id for memory in memories} - set(session_map))
        if missing:
            raise ValueError(f"missing canonical session records: {', '.join(missing)}")

    user_parts: list[ModelContentPart] = []
    if sessions is not None:
        user_parts.append(ModelContentPart(kind=ModelContentKind.TEXT, text="[MEMORY]"))
    previous_session_id: str | None = None
    for memory in memories:
        if sessions is not None and memory.session_id != previous_session_id:
            user_parts.append(
                ModelContentPart(
                    kind=ModelContentKind.TEXT,
                    text=render_session_header(session_map[memory.session_id]),
                )
            )
            previous_session_id = memory.session_id
        rendered = (
            render_memory_text_caption(memory)
            if method is BaselineMethod.FULL_CONTEXT_CAPTIONS
            else render_memory_text(memory)
        )
        user_parts.append(ModelContentPart(kind=ModelContentKind.TEXT, text=rendered))
        if method is BaselineMethod.FULL_CONTEXT_RAW_IMAGES:
            user_parts.extend(
                ModelContentPart(kind=ModelContentKind.IMAGE, image=image)
                for image in memory_image_refs(memory)
            )

    if sessions is not None:
        user_parts.append(ModelContentPart(kind=ModelContentKind.TEXT, text="[/MEMORY]"))
    user_parts.append(
        ModelContentPart(
            kind=ModelContentKind.TEXT,
            text=(
                render_query_text(query)
                if sessions is not None
                else f"[QUERY] {query.raw_query}"
            ),
        )
    )
    user_parts.extend(
        ModelContentPart(kind=ModelContentKind.IMAGE, image=image)
        for image in query.query_image_refs
    )
    return [
        ModelMessage(
            role=SpeakerRole.SYSTEM,
            parts=[ModelContentPart(kind=ModelContentKind.TEXT, text=instruction)],
        ),
        ModelMessage(role=SpeakerRole.USER, parts=user_parts),
    ]


def render_rag_messages(
    *,
    instruction: str,
    query: QueryInput,
    memories: list[MemoryUnit],
    method: BaselineMethod,
    sessions: list[SessionRecord],
    evidence_aliases: list[str] | None = None,
) -> list[ModelMessage]:
    """Render ranked RAG evidence without exposing scores or internal identifiers."""

    rag_methods = {
        BaselineMethod.CAPTION_TEXT_RAG,
        BaselineMethod.CONTRASTIVE_MULTIMODAL_RAG,
        BaselineMethod.NATIVE_MULTIMODAL_RAG,
    }
    if method not in rag_methods:
        raise ValueError("RAG message rendering requires a RAG baseline method")
    session_map = {session.session_id: session for session in sessions}
    if len(session_map) != len(sessions):
        raise ValueError("RAG sessions cannot contain duplicate IDs")
    missing = sorted({memory.session_id for memory in memories} - set(session_map))
    if missing:
        raise ValueError(f"missing canonical session records: {', '.join(missing)}")
    if evidence_aliases is not None and len(evidence_aliases) != len(memories):
        raise ValueError("strict RAG evidence aliases must match the memory denominator")

    user_parts = [ModelContentPart(kind=ModelContentKind.TEXT, text="[MEMORY]")]
    previous_session_id: str | None = None
    for ordinal, memory in enumerate(memories):
        if memory.session_id != previous_session_id:
            user_parts.append(
                ModelContentPart(
                    kind=ModelContentKind.TEXT,
                    text=render_session_header(session_map[memory.session_id]),
                )
            )
            previous_session_id = memory.session_id
        rendered = (
            render_memory_text_caption(memory)
            if method is BaselineMethod.CAPTION_TEXT_RAG
            else render_memory_text(memory)
        )
        if evidence_aliases is not None:
            rendered = f"[{evidence_aliases[ordinal]}]\n{rendered}"
        user_parts.append(ModelContentPart(kind=ModelContentKind.TEXT, text=rendered))
        if method is not BaselineMethod.CAPTION_TEXT_RAG:
            refs = {image.image_id: image for image in memory.image_refs}
            ordered_ids = list(
                dict.fromkeys(
                    [*memory.image_ids, *(image.image_id for image in memory.image_refs)]
                )
            )
            user_parts.extend(
                ModelContentPart(
                    kind=ModelContentKind.IMAGE,
                    image=refs.get(image_id, ImageRef(image_id=image_id)),
                )
                for image_id in ordered_ids
            )
    user_parts.append(ModelContentPart(kind=ModelContentKind.TEXT, text="[/MEMORY]"))
    user_parts.append(
        ModelContentPart(kind=ModelContentKind.TEXT, text=render_query_text(query))
    )
    user_parts.extend(
        ModelContentPart(kind=ModelContentKind.IMAGE, image=image)
        for image in query.query_image_refs
    )
    return [
        ModelMessage(
            role=SpeakerRole.SYSTEM,
            parts=[ModelContentPart(kind=ModelContentKind.TEXT, text=instruction)],
        ),
        ModelMessage(role=SpeakerRole.USER, parts=user_parts),
    ]
