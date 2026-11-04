from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from pmmc.benchmarks.adapters.memgallery import MemGalleryAdapter
from pmmc.benchmarks.ids import canonical_id
from pmmc.compiler.ids import canonical_content_hash
from pmmc.evaluation.paper_statistics import (
    MethodVariant,
    ScoreObservation,
    SelectedQuery,
    TerminalStatus,
    TypedAnswer,
    summarize_method,
)
from pmmc.evaluation.step8_typed_gold import (
    REFERENCE_ANNOTATION_VERSION,
    RESOLVER_VERSION,
    UNRESOLVED,
    resolve_typed_gold,
    score_resolved_typed_answer,
    text_judge_reference,
)
from pmmc.schemas import DatasetName, GoldEvidence, ImageRef, MemoryUnit, QueryGold, QueryInput

DATASET = DatasetName.MEMGALLERY
SCENARIO = "synthetic-scenario"
FIRST = "D1_IMG_001"
SECOND = "D2_IMG_002"
CANONICAL_FIRST = canonical_id(DATASET, "image", SCENARIO, FIRST)
CANONICAL_SECOND = canonical_id(DATASET, "image", SCENARIO, SECOND)


@pytest.fixture
def sources():
    conversation = canonical_id(DATASET, "conversation", SCENARIO)
    session = canonical_id(DATASET, "session", SCENARIO, "D2")
    query = QueryInput(
        query_id=canonical_id(DATASET, "query", SCENARIO, 0),
        dataset=DATASET,
        conversation_id=conversation,
        available_after_session=session,
        raw_query="Synthetic fixture only; no benchmark answers loaded.",
        metadata={"scenario_id": SCENARIO},
    )
    memories = tuple(
        MemoryUnit(
            memory_id=f"synthetic-memory-{i}",
            dataset=DATASET,
            conversation_id=conversation,
            session_id=session,
            turn_id=f"synthetic-turn-{i}",
            timestamp_order=i,
            available_after_session=session,
            text="This prose must never be interpreted as typed gold.",
            image_refs=[
                ImageRef(
                    image_id=canonical,
                    caption="a red hat",
                    local_path=f"not-read/scenario/image-{i}.png",
                    metadata={
                        "source_image_id": source,
                        "dataset_relative_path": f"images/{SCENARIO}/{source}.png",
                    },
                )
            ],
        )
        for i, (source, canonical) in enumerate(
            [(FIRST, CANONICAL_FIRST), (SECOND, CANONICAL_SECOND)]
        )
    )
    return query, memories


def resolve(sources, answer, *, kind=None, gold=None):
    query, memories = sources
    return resolve_typed_gold(
        query,
        gold or QueryGold(answer=answer),
        source_memories=memories,
        expected_kind=kind,
    )


@pytest.mark.parametrize(
    "answer",
    [
        FIRST,
        CANONICAL_FIRST,
        f"images/{SCENARIO}/{FIRST}.png",
        f"Image: {FIRST}",
    ],
)
def test_exact_source_image_names_map_to_existing_canonical_refs(sources, answer):
    resolved = resolve(sources, answer)
    assert resolved.status == "resolved"
    assert resolved.reference == TypedAnswer("image", image_ids=(CANONICAL_FIRST,))
    assert score_resolved_typed_answer(resolved.reference, resolved).value == 1
    wrong = TypedAnswer("image", image_ids=(CANONICAL_SECOND,))
    assert score_resolved_typed_answer(wrong, resolved).value == 0


def test_native_colon_image_id_requires_exact_unique_source(sources):
    query, memories = sources
    image = memories[0].image_refs[0].model_copy(update={
        "metadata": {"source_image_id": "D1:IMG_001"},
    })
    memory = memories[0].model_copy(update={"image_refs": [image]})
    gold = QueryGold(answer="D1:IMG_001")
    result = resolve_typed_gold(query, gold, source_memories=(memory,))
    assert result.reference == TypedAnswer("image", image_ids=(CANONICAL_FIRST,))
    assert result.derivation == "source_image_refs"
    assert result.detail is None
    # The separator is recognized, not normalized to an unrelated source alias.
    for answer in ("D1_IMG_001", "D9:IMG_999"):
        assert resolve_typed_gold(query, QueryGold(answer=answer),
                                  source_memories=(memory,)).reference is None
    assert resolve_typed_gold(query, QueryGold(answer="d1:img_001"),
                              source_memories=(memory,)).status == "text_only"
    other = memories[1].model_copy(update={"image_refs": [
        image.model_copy(update={"image_id": CANONICAL_SECOND})]})
    assert resolve_typed_gold(query, gold, source_memories=(memory, other)).reference is None
    assert resolve_typed_gold(query, gold, source_memories=()).reference is None


def test_explicit_source_sets_are_not_ordered_by_memory_or_canonical_id(sources):
    resolved = resolve(sources, f"Image set: {SECOND} and {FIRST}")
    assert resolved.reference == TypedAnswer(
        "image_set", image_ids=(CANONICAL_SECOND, CANONICAL_FIRST)
    )
    swapped = TypedAnswer("image_set", image_ids=(CANONICAL_FIRST, CANONICAL_SECOND))
    assert score_resolved_typed_answer(swapped, resolved).value == 1


@pytest.mark.parametrize(
    "answer",
    [
        f"{SECOND} -> {FIRST}",
        f"{SECOND} then {FIRST}",
        f"Order: {SECOND}, {FIRST}",
        f"Ordered images: {json.dumps([SECOND, FIRST])}",
    ],
)
def test_source_explicit_order_is_preserved(sources, answer):
    resolved = resolve(sources, answer)
    assert resolved.reference == TypedAnswer(
        "ordered_image_list", image_ids=(CANONICAL_SECOND, CANONICAL_FIRST)
    )
    swapped = TypedAnswer("ordered_image_list", image_ids=(CANONICAL_FIRST, CANONICAL_SECOND))
    assert score_resolved_typed_answer(swapped, resolved).value == 0


@pytest.mark.parametrize(
    "answer",
    [
        f"{SECOND}, {FIRST}",
        json.dumps([SECOND, FIRST]),
        f"Images: {SECOND}, {FIRST}",
    ],
)
def test_plain_collections_require_a_frozen_kind_not_a_guess(sources, answer):
    ambiguous = resolve(sources, answer)
    assert ambiguous.status == "unresolved"
    assert ambiguous.detail == "collection_kind_requires_protocol"
    for kind in ("image_set", "ordered_image_list"):
        typed = resolve(sources, answer, kind=kind)
        assert typed.reference.kind == kind
        assert typed.reference.image_ids == (CANONICAL_SECOND, CANONICAL_FIRST)


@pytest.mark.parametrize(
    "answer,kind",
    [
        ("a red hat", "image"),
        ("the second image", "image"),
        ("the photo from D2", "image"),
        ("[image]", "image"),
        ("I01", "image"),
        ("D1_IMG_001.png", "image"),
        ("d1_img_001", "image"),
        (f"Not {FIRST}; choose {SECOND}", "image"),
        (f"{FIRST} or {SECOND}", "image_set"),
        (f"{FIRST} because it shows the right object", "image"),
        (f"Images: {FIRST} -> {SECOND}", "image_set"),
        (f"Image: {FIRST}, {SECOND}", "image"),
        (f"{FIRST}, {FIRST}", "image_set"),
        (json.dumps([FIRST, CANONICAL_FIRST]), "ordered_image_list"),
        (f"{FIRST}, D9_IMG_999", "image_set"),
        (f"{FIRST},", "image_set"),
        (f"Order: {FIRST}, {SECOND}", "image_set"),
        (f"{FIRST}, {SECOND}", "number"),
    ],
)
def test_unresolved_never_infers_from_evidence_caption_position_or_semantics(sources, answer, kind):
    resolved = resolve(sources, answer, kind=kind)
    assert resolved.status == "unresolved"
    assert resolved.reference is None
    assert resolved.unavailable_reason == UNRESOLVED
    candidate = TypedAnswer("image", image_ids=(CANONICAL_FIRST,))
    score = score_resolved_typed_answer(candidate, resolved)
    assert score.value is None and score.applicable is False
    assert score.unavailable_reason == UNRESOLVED


def test_evidence_only_gold_is_not_an_answer_even_if_there_is_one_image(sources):
    query, memories = sources
    gold = QueryGold(
        evidence=GoldEvidence(image_ids=[CANONICAL_FIRST], memory_ids=[memories[0].memory_id])
    )
    result = resolve_typed_gold(query, gold, source_memories=memories[:1])
    assert result.status == "unresolved" and result.detail == "source_answer_missing"
    assert result.reference is None and result.text_reference is None
    assert (
        score_resolved_typed_answer(
            TypedAnswer("image", image_ids=(CANONICAL_FIRST,)), result
        ).value
        is None
    )


def test_gold_clues_and_query_wording_do_not_control_image_selection(sources):
    query, memories = sources
    gold = QueryGold(
        answer=FIRST,
        evidence=GoldEvidence(image_ids=[CANONICAL_SECOND]),
        metadata={"source_clue_ids": ["D2:1"], "source_session_ids": ["D2"]},
    )
    changed = gold.model_copy(
        update={
            "evidence": GoldEvidence(image_ids=[CANONICAL_FIRST]),
            "metadata": {"source_clue_ids": ["D1:1"]},
        }
    )
    one = resolve_typed_gold(query, gold, source_memories=memories)
    two = resolve_typed_gold(query, changed, source_memories=memories)
    assert one.reference == two.reference
    assert one.gold_hash != two.gold_hash  # Evidence changes lineage, not the answer.
    other_query = query.model_copy(update={"raw_query": f"Return {SECOND}, not {FIRST}"})
    assert (
        resolve_typed_gold(other_query, gold, source_memories=memories).reference == one.reference
    )


@pytest.mark.parametrize(
    "answer,kind,value",
    [
        ("3", "number", Decimal(3)),
        ("0 images", "number", Decimal(0)),
        ("12 photos", "number", Decimal(12)),
        ("1 picture", "number", Decimal(1)),
        ("-1.25e2", "number", Decimal(-125)),
        ("true", "boolean", True),
        ("False", "boolean", False),
        ("Yes", "boolean", True),
        ("Yes.", "boolean", True),
        ("YES.", "boolean", True),
        ("No", "boolean", False),
        ("no.", "boolean", False),
        ("2024-02-29", "date", date(2024, 2, 29)),
    ],
)
def test_whole_source_scalar_literals_are_lossless(sources, answer, kind, value):
    result = resolve(sources, answer)
    assert result.reference.kind == kind
    assert getattr(result.reference, "date_value" if kind == "date" else kind) == value


@pytest.mark.parametrize(
    "answer,kind",
    [
        ("about 3", "number"),
        ("three matching images", "number"),
        ("2-3", "number"),
        ("3 photos, but 2 were usable", "number"),
        ("03", "number"),
        ("3/4", "number"),
        ("NaN", "number"),
        ("Infinity", "number"),
        ("Yes, but not always.", "boolean"),
        ("No evidence.", "boolean"),
        ("Yes/No", "boolean"),
        ("Yes..", "boolean"),
        ("Yes!", "boolean"),
        ("03/04/2026", "date"),
        ("2023-02-29", "date"),
        (f"{FIRST}, {SECOND}", "number"),
    ],
)
def test_ambiguous_scalar_or_count_is_not_computed_from_images(sources, answer, kind):
    result = resolve(sources, answer, kind=kind)
    assert result.status == "unresolved" and result.reference is None
    assert result.text_reference.text == answer


def test_plain_text_keeps_text_judge_and_does_not_zero_typed_images(sources):
    result = resolve(sources, "a red hat")
    text = TypedAnswer("text", text="a red hat")
    image = TypedAnswer("image", image_ids=(CANONICAL_FIRST,))
    assert result.status == "text_only"
    assert score_resolved_typed_answer(text, result).value == 1
    unavailable = score_resolved_typed_answer(image, result)
    assert unavailable.value is None and unavailable.unavailable_reason == UNRESOLVED
    assert text_judge_reference(text, result).text == "a red hat"
    assert text_judge_reference(image, result) is None


@pytest.mark.parametrize("source,value", [("Yes.", True), ("No.", False)])
def test_source_boolean_literal_keeps_gold_and_text_judge_identity(sources, source, value):
    query, memories = sources
    gold = QueryGold(answer=source)
    before = gold.to_json()
    result = resolve_typed_gold(query, gold, source_memories=memories)
    assert result.status == "resolved" and result.derivation == "source_scalar"
    assert result.reference == TypedAnswer("boolean", boolean=value)
    assert score_resolved_typed_answer(TypedAnswer("boolean", boolean=value), result).value == 1
    assert score_resolved_typed_answer(TypedAnswer("boolean", boolean=not value), result).value == 0
    assert text_judge_reference(TypedAnswer("text", text=source), result).text == source
    assert gold.to_json() == before
    # Resolution is source-only: it does not coerce a text candidate into a boolean.
    assert score_resolved_typed_answer(TypedAnswer("text", text=source), result).value == 0


def test_text_judge_uses_original_source_not_typed_image_placeholders(sources):
    result = resolve(sources, "There were three matching photos.", kind="number")
    text = TypedAnswer("text", text="Three photos.")
    assert text_judge_reference(text, result).text == "There were three matching photos."
    assert result.status == "unresolved"
    known = resolve(sources, "3 photos")
    assert text_judge_reference(text, known).text == "3 photos"


def test_explicit_metadata_is_validated_and_source_images_must_exist(sources):
    gold = QueryGold(
        answer="fixture",
        metadata={
            "typed_answer": {"kind": "image", "image_ids": [CANONICAL_FIRST]},
        },
    )
    result = resolve(sources, None, gold=gold)
    assert result.reference.image_ids == (CANONICAL_FIRST,)
    assert result.derivation == "explicit_typed_annotation"
    absent = resolve_typed_gold(sources[0], gold)
    assert absent.reference is None and absent.status == "unresolved"
    assert absent.detail == "annotation_image_not_in_source_refs"


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "number", "number": "2.00"},
        {"kind": "boolean", "boolean": True},
        {"kind": "date", "date_value": "2026-09-20"},
        {"kind": "text", "text": "explicit text"},
        {"kind": "image_set", "image_ids": []},
    ],
)
def test_existing_explicit_typed_gold_is_preserved(sources, payload):
    result = resolve(
        sources, None, gold=QueryGold(answer="fixture", metadata={"typed_answer": payload})
    )
    assert result.status == "resolved" and result.reference.kind == payload["kind"]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"kind": "image", "image_ids": CANONICAL_FIRST},
        {"kind": "image", "image_ids": []},
        {"kind": "image_set", "image_ids": [CANONICAL_FIRST, CANONICAL_FIRST]},
        {"kind": "boolean", "boolean": "true"},
        {"kind": "number", "number": float("nan")},
        {"kind": "number", "number": "NaN"},
        {"kind": "text", "text": "yes", "image_ids": [CANONICAL_FIRST]},
        {"kind": "text", "text": "yes", "unexpected": True},
    ],
)
def test_invalid_annotations_fail_closed_without_legacy_fallback(sources, payload):
    with pytest.raises(ValueError, match="step8_typed_gold_invalid"):
        resolve(sources, None, gold=QueryGold(answer=FIRST, metadata={"typed_answer": payload}))


def test_scope_mismatch_and_conflicting_canonical_metadata_fail_closed(sources):
    query, memories = sources
    wrong = memories[0].model_copy(update={"conversation_id": "different-scenario"})
    with pytest.raises(ValueError, match="source_scope_mismatch"):
        resolve_typed_gold(query, QueryGold(answer=FIRST), source_memories=[wrong])
    image = memories[0].image_refs[0].model_copy(update={"metadata": {"source_image_id": "other"}})
    conflicting = memories[0].model_copy(update={"image_refs": [image]})
    with pytest.raises(ValueError, match="source_ref_conflict"):
        resolve_typed_gold(
            query, QueryGold(answer=FIRST), source_memories=[memories[0], conflicting]
        )


def test_ambiguous_source_alias_cannot_pick_the_first_canonical_image(sources):
    query, memories = sources
    extra = memories[1].model_copy(deep=True)
    extra.image_refs[0].metadata["source_image_id"] = FIRST
    result = resolve_typed_gold(
        query, QueryGold(answer=FIRST), source_memories=[memories[0], extra]
    )
    assert result.status == "unresolved" and result.detail == "ambiguous_source_image_ref"


def test_scalar_looking_source_image_id_requires_an_explicit_kind(sources):
    query, memories = sources
    memory = memories[0].model_copy(deep=True)
    memory.image_refs[0].metadata["source_image_id"] = "2"
    gold = QueryGold(answer="2")
    ambiguous = resolve_typed_gold(query, gold, source_memories=[memory])
    assert ambiguous.status == "unresolved"
    numeric = resolve_typed_gold(query, gold, source_memories=[memory], expected_kind="number")
    assert numeric.reference.number == Decimal(2)
    image = resolve_typed_gold(query, gold, source_memories=[memory], expected_kind="image")
    assert image.reference.image_ids == (CANONICAL_FIRST,)


def test_resolution_and_cache_identity_are_method_independent_and_private(sources):
    query, memories = sources
    gold = QueryGold(answer=f"Image set: {SECOND}, {FIRST}")
    before = [memory.to_json() for memory in memories], gold.to_json(), query.to_json()
    result = resolve_typed_gold(query, gold, source_memories=memories)
    repeated = resolve_typed_gold(query, gold, source_memories=[*reversed(memories), memories[0]])
    assert result == repeated
    assert ([memory.to_json() for memory in memories], gold.to_json(), query.to_json()) == before
    audit = json.dumps(result.audit_metadata())
    assert RESOLVER_VERSION in audit
    assert FIRST not in audit and SECOND not in audit and CANONICAL_FIRST not in repr(result)
    changed = memories[0].model_copy(deep=True)
    changed.image_refs[0].metadata["dataset_relative_path"] = "images/changed.png"
    modified = resolve_typed_gold(query, gold, source_memories=[changed, memories[1]])
    assert modified.source_refs_hash != result.source_refs_hash


def test_unresolved_statistical_hook_is_not_applicable_not_zero(sources):
    result = resolve(sources, "a red hat", kind="image")
    score = score_resolved_typed_answer(TypedAnswer("image", image_ids=(CANONICAL_FIRST,)), result)
    method = MethodVariant("synthetic-method", "synthetic-cell")
    selected = SelectedQuery(
        "query", "memgallery", SCENARIO, "scenario", applicable=score.applicable
    )
    observation = ScoreObservation("query", method, TerminalStatus.ANSWERED, score)
    summary = summarize_method((selected,), (observation,), method)
    assert summary.status == "not_applicable" and summary.micro is None
    assert summary.engineering_missing_count == 0
    abstention = replace(observation, status=TerminalStatus.ABSTENTION, score=None)
    assert summarize_method((selected,), (abstention,), method).micro is None


def test_absent_gold_is_distinct_from_unresolved_typed_reference(sources):
    result = resolve_typed_gold(sources[0], None)
    assert result.status == "missing" and result.unavailable_reason == "gold_missing"
    score = score_resolved_typed_answer(TypedAnswer("text", text="anything"), result)
    assert score.value is None and score.unavailable_reason == "gold_missing"


def test_reference_annotation_binds_derived_image_without_rewriting_text_gold(sources):
    query, memories = sources
    gold = QueryGold(answer=FIRST)
    before = gold.to_json()
    result = resolve_typed_gold(query, gold, source_memories=memories)
    annotation = result.reference_annotation("synthetic-snapshot", resolver_source_hash="a" * 64)
    assert annotation["schema_version"] == REFERENCE_ANNOTATION_VERSION
    assert annotation["kind"] == "image" and annotation["status"] == "resolved"
    assert annotation["gold_hash"] == canonical_content_hash(gold.model_dump(mode="json"))
    assert annotation["query_hash"] == canonical_content_hash(query.model_dump(mode="json"))
    assert annotation["hash"] == canonical_content_hash(
        {key: value for key, value in annotation.items() if key != "hash"}
    )
    assert gold.to_json() == before and "typed_answer" not in gold.metadata
    assert FIRST not in json.dumps(annotation) and CANONICAL_FIRST not in json.dumps(annotation)
    # Same source strings with different image resolution must invalidate completion identity.
    other = replace(result, reference=TypedAnswer("image", image_ids=(CANONICAL_SECOND,)))
    assert (
        other.reference_annotation("synthetic-snapshot", resolver_source_hash="a" * 64)["hash"]
        != annotation["hash"]
    )


def test_reference_annotation_changes_with_scope_policy_and_resolver_code(sources):
    result = resolve(sources, FIRST)
    annotation = result.reference_annotation("snapshot-a", resolver_source_hash="a" * 64)
    variants = [
        result.reference_annotation("snapshot-b", resolver_source_hash="a" * 64),
        result.reference_annotation("snapshot-a", resolver_source_hash="b" * 64),
        replace(result, expected_kind="image").reference_annotation(
            "snapshot-a", resolver_source_hash="a" * 64
        ),
        replace(result, source_refs_hash="c" * 64).reference_annotation(
            "snapshot-a", resolver_source_hash="a" * 64
        ),
    ]
    assert all(item["hash"] != annotation["hash"] for item in variants)
    assert result.reference_annotation("snapshot-a", resolver_source_hash="a" * 64) == annotation


def test_unresolved_annotation_cannot_claim_a_typed_kind(sources):
    result = resolve(sources, "the second picture", kind="image")
    annotation = result.reference_annotation("synthetic-snapshot", resolver_source_hash="a" * 64)
    assert annotation["kind"] is None
    assert annotation["status"] == "unresolved"
    assert annotation["unavailable_reason"] == UNRESOLVED


@pytest.mark.parametrize(
    "snapshot,digest",
    [("", "a" * 64), (" padded", "a" * 64), ("snapshot", "not-a-sha256"), ("snapshot", "A" * 64)],
)
def test_reference_annotation_requires_explicit_scope_and_code_hash(sources, snapshot, digest):
    result = resolve(sources, FIRST)
    with pytest.raises(ValueError, match="reference_annotation"):
        result.reference_annotation(snapshot, resolver_source_hash=digest)


def test_real_adapter_contract_resolves_synthetic_string_gold_without_snapshot_edits(
    memgallery_fixture_root,
    tmp_path,
):
    # This path is created by conftest's synthetic fixture, never a benchmark checkout.
    source = memgallery_fixture_root / "data" / "dialogs" / "scenario_one.json"
    payload = json.loads(source.read_text())
    payload["human-annotated QAs"][0]["answer"] = FIRST
    source.write_text(json.dumps(payload))
    adapter = MemGalleryAdapter(memgallery_fixture_root, tmp_path)
    conversation = next(adapter.iter_conversations())
    record = conversation.queries[0].record
    gold_before = record.gold.to_json()
    result = resolve_typed_gold(
        record.query,
        record.gold,
        source_memories=conversation.memory_units,
    )
    actual_refs = [image for memory in conversation.memory_units for image in memory.image_refs]
    assert result.reference == TypedAnswer("image", image_ids=(actual_refs[0].image_id,))
    assert record.gold.to_json() == gold_before
    assert "typed_answer" not in record.gold.metadata


def test_canonical_id_without_a_source_ref_is_not_fabricated(sources):
    query, memories = sources
    memory = memories[0].model_copy(update={"image_refs": [], "image_ids": [CANONICAL_FIRST]})
    result = resolve_typed_gold(query, QueryGold(answer=FIRST), source_memories=[memory])
    assert result.status == "unresolved" and result.reference is None


def test_caption_changes_do_not_change_reference_mapping_or_reference_kind(sources):
    query, memories = sources
    changed = memories[0].model_copy(deep=True)
    changed.image_refs[0].caption = f"The answer is {SECOND}"
    gold = QueryGold(answer=FIRST)
    one = resolve_typed_gold(query, gold, source_memories=memories)
    two = resolve_typed_gold(query, gold, source_memories=[changed, memories[1]])
    assert one == two
