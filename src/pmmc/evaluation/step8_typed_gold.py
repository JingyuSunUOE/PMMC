"""Evaluator-only, lossless typed gold resolution. No I/O or runtime imports.

Supply canonical memories from the selected snapshot/conversation at the query
cutoff, not retrieved/evidence-selected images. Only references explicitly named
in the source answer are resolved. Captions, clues, query wording, image position
and evidence membership never establish the answer. Ambiguous legacy strings
remain available to the text Judge, not fabricated typed references.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal, cast

from pmmc.benchmarks.ids import stable_hash
from pmmc.evaluation.paper_statistics import AnswerKind, TypedAnswer, TypedScore, score_typed_answer
from pmmc.schemas import MemoryUnit, QueryGold, QueryInput

RESOLVER_VERSION = "step8-lossless-typed-gold-v1"
REFERENCE_ANNOTATION_VERSION = "step8-reference-annotation-v1"
UNRESOLVED = "typed_reference_unresolved"
_IMAGE_KINDS = {"image", "image_set", "ordered_image_list"}
_KINDS = {"text", "number", "boolean", "date", *_IMAGE_KINDS}
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_COUNT = re.compile(r"(0|[1-9][0-9]*) (?:images?|photos?|pictures?)", re.IGNORECASE)


@dataclass(frozen=True)
class TypedGoldResolution:
    status: Literal["resolved", "text_only", "unresolved", "missing"]
    reference: TypedAnswer | None = field(repr=False)
    text_reference: TypedAnswer | None = field(repr=False)
    expected_kind: AnswerKind | None
    derivation: str
    source_refs_hash: str
    query_id: str
    query_hash: str
    gold_hash: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "text_only", "unresolved", "missing"}:
            raise ValueError("invalid typed gold resolution status")
        if (self.reference is not None) != (self.status in {"resolved", "text_only"}):
            raise ValueError("typed gold resolution status/reference mismatch")
        if self.status == "text_only" and (self.reference is None or self.reference.kind != "text"):
            raise ValueError("text-only gold requires a text reference")

    @property
    def unavailable_reason(self) -> str | None:
        if self.status == "unresolved":
            return UNRESOLVED
        return "gold_missing" if self.status == "missing" else None

    def audit_metadata(self) -> dict[str, object]:
        """Cache/report identity without gold text, answer IDs or evidence IDs."""
        return {
            "resolver_version": RESOLVER_VERSION,
            "status": self.status,
            "expected_kind": self.expected_kind,
            "derivation": self.derivation,
            "source_refs_hash": self.source_refs_hash,
            "unavailable_reason": self.unavailable_reason,
            "detail": self.detail,
        }

    def reference_annotation(
        self,
        snapshot_id: str,
        *,
        resolver_source_hash: str,
    ) -> dict[str, object]:
        """Hash-bound evaluator attestation, not a DB-gold mutation or SQL permit.

        The parent computes resolver_source_hash from this module's bytes, then
        binds the same annotation to evaluation metadata and the completion report.
        No gold values or resolved image IDs are emitted in the annotation.
        """
        if not snapshot_id or snapshot_id != snapshot_id.strip():
            raise ValueError("step8_reference_annotation_snapshot_invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", resolver_source_hash):
            raise ValueError("step8_reference_annotation_resolver_hash_invalid")
        payload = {
            **self.audit_metadata(),
            "schema_version": REFERENCE_ANNOTATION_VERSION,
            "snapshot_id": snapshot_id,
            "query_id": self.query_id,
            "query_hash": self.query_hash,
            "gold_hash": self.gold_hash,
            "resolver_source_hash": resolver_source_hash,
            "kind": self.reference.kind if self.reference is not None else None,
            "reference_hash": stable_hash(_answer_payload(self.reference)),
            "text_judge_reference_hash": stable_hash(_answer_payload(self.text_reference)),
        }
        return {**payload, "hash": stable_hash(payload)}


def _answer_payload(answer: TypedAnswer | None) -> dict[str, object] | None:
    if answer is None:
        return None
    return {
        "kind": answer.kind,
        "text": answer.text,
        "number": str(answer.number) if answer.number is not None else None,
        "boolean": answer.boolean,
        "date_value": answer.date_value.isoformat() if answer.date_value is not None else None,
        "image_ids": list(answer.image_ids),
    }


@dataclass(frozen=True)
class _ImageIndex:
    aliases: dict[str, frozenset[str]]
    canonical_ids: frozenset[str]
    digest: str

    def resolve(self, token: str) -> tuple[str | None, str | None]:
        targets = self.aliases.get(token, frozenset())
        if len(targets) != 1:
            return None, "ambiguous_source_image_ref" if targets else "unknown_source_image_ref"
        return next(iter(targets)), None


def _image_index(query: QueryInput, memories: Iterable[MemoryUnit]) -> _ImageIndex:
    aliases: dict[str, set[str]] = {}
    identities: dict[str, dict[str, str]] = {}
    for memory in memories:
        if memory.dataset != query.dataset or memory.conversation_id != query.conversation_id:
            raise ValueError("step8_typed_gold_source_scope_mismatch")
        for image in memory.image_refs:
            identity = identities.setdefault(image.image_id, {})
            aliases.setdefault(image.image_id, set()).add(image.image_id)
            for key in ("source_image_id", "dataset_relative_path"):
                value = image.metadata.get(key)
                if value is None:
                    continue
                if not isinstance(value, str) or not value.strip() or value != value.strip():
                    raise ValueError("step8_typed_gold_source_ref_invalid")
                if key in identity and identity[key] != value:
                    raise ValueError("step8_typed_gold_source_ref_conflict")
                identity[key] = value
                aliases.setdefault(value, set()).add(image.image_id)
    # Do not use filenames/basenames, captions, runtime I01 aliases, or evidence.
    digest = stable_hash(
        {
            "dataset": query.dataset.value,
            "conversation_id": query.conversation_id,
            "images": identities,
        }
    )
    return _ImageIndex(
        {key: frozenset(value) for key, value in aliases.items()}, frozenset(identities), digest
    )


def _explicit_annotation(
    payload: object, index: _ImageIndex
) -> tuple[TypedAnswer | None, str | None]:
    if not isinstance(payload, dict):
        raise ValueError("step8_typed_gold_invalid")
    value = dict(payload)
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError("step8_typed_gold_invalid")
    try:
        if "image_ids" in value:
            if not isinstance(value["image_ids"], (list, tuple)):
                raise ValueError("invalid image IDs")
            value["image_ids"] = tuple(value["image_ids"])
        if value.get("number") is not None:
            number = value["number"]
            if type(number) not in (int, str, Decimal):
                raise ValueError("number must have an exact decimal representation")
            value["number"] = Decimal(str(number))
        if value.get("date_value") is not None:
            raw_date = value["date_value"]
            if not isinstance(raw_date, str) or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}", raw_date
            ):
                raise ValueError("date must be ISO formatted")
            value["date_value"] = date.fromisoformat(raw_date)
        reference = TypedAnswer(**value)
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError("step8_typed_gold_invalid") from exc
    if kind in _IMAGE_KINDS and not set(reference.image_ids) <= index.canonical_ids:
        return None, "annotation_image_not_in_source_refs"
    return reference, None


def _scalar(text: str) -> TypedAnswer | None:
    if _NUMBER.fullmatch(text):
        try:
            return TypedAnswer("number", number=Decimal(text))
        except (ValueError, InvalidOperation):
            return None
    count = _COUNT.fullmatch(text)
    if count:
        return TypedAnswer("number", number=Decimal(count[1]))
    if text.casefold() in {"true", "false"}:
        return TypedAnswer("boolean", boolean=text.casefold() == "true")
    # Whole source literals only; do not infer polarity from explanatory prose.
    if text.casefold() in {"yes", "yes.", "no", "no."}:
        return TypedAnswer("boolean", boolean=text.casefold() in {"yes", "yes."})
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
        try:
            return TypedAnswer("date", date_value=date.fromisoformat(text))
        except ValueError:
            return None
    return None


def _tokens(text: str) -> tuple[list[str] | None, bool]:
    """Parse a complete reference expression; never extract IDs from prose."""
    if text.startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None, False
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return None, False
        return value, False
    ordered = " -> " in text or " then " in text
    parts = (
        re.split(r"\s+->\s+|\s+then\s+", text)
        if ordered
        else re.split(
            r"\s*[,;]\s*|\s+and\s+",
            text,
        )
    )
    return parts if all(parts) else None, ordered


def _explicit_image_token(token: str) -> bool:
    return bool(re.fullmatch(r"D[0-9]+[:_]IMG_[A-Za-z0-9_-]+", token)) or (
        token.startswith(("memgallery:image:", "memlens:image:", "images/"))
    )


def _image_literal(
    text: str,
    expected_kind: AnswerKind | None,
    index: _ImageIndex,
) -> tuple[TypedAnswer | None, str | None]:
    declared: str | None = None
    collection_hint = False
    for prefix, prefix_kind in (
        ("Ordered images: ", "ordered_image_list"),
        ("Order: ", "ordered_image_list"),
        ("Unordered images: ", "image_set"),
        ("Image set: ", "image_set"),
        ("Images: ", None),
        ("Image: ", "image"),
    ):
        if text.startswith(prefix):
            text, declared = text[len(prefix) :], prefix_kind
            collection_hint = prefix == "Images: "
            break
    tokens: list[str] | None
    if text in index.aliases:
        tokens, ordered = [text], False
    else:
        tokens, ordered = _tokens(text)
    if tokens is None:
        return None, "source_answer_not_an_exact_reference_expression"
    if ordered:
        if declared is not None and declared != "ordered_image_list":
            return None, "reference_kind_conflict"
        declared = "ordered_image_list"
    if declared is not None and expected_kind is not None and declared != expected_kind:
        return None, "reference_kind_conflict"
    kind = expected_kind or declared
    canonical = []
    for token in tokens:
        identifier, error = index.resolve(token)
        if error:
            return None, error
        assert identifier is not None
        canonical.append(identifier)
    if len(canonical) != len(set(canonical)):
        return None, "duplicate_answer_image_ref"
    if kind is None:
        if len(canonical) != 1 or collection_hint:
            return None, "collection_kind_requires_protocol"
        if not _explicit_image_token(tokens[0]):
            return None, "reference_kind_requires_protocol"
        kind = "image"
    if kind not in _IMAGE_KINDS or (kind == "image" and len(canonical) != 1):
        return None, "reference_kind_conflict"
    return TypedAnswer(cast(AnswerKind, kind), image_ids=tuple(canonical)), None


def resolve_typed_gold(
    query: QueryInput,
    gold: QueryGold | None,
    *,
    source_memories: Iterable[MemoryUnit] = (),
    expected_kind: AnswerKind | None = None,
) -> TypedGoldResolution:
    """Resolve source literals without using candidate answers or evidence.

    expected_kind, if supplied, must come from a predeclared evaluator protocol,
    never from a candidate's chosen output kind. Without it, only explicit source
    type syntax (e.g. ``Image set: A and B`` or ``A -> B``) establishes a collection
    kind. A plain multi-image list requires that protocol; its order is not guessed.
    Canonical source refs must be complete for the snapshot/query cutoff, not
    filtered to the gold clues or to any method's retrieved evidence.
    """
    if expected_kind is not None and expected_kind not in _KINDS:
        raise ValueError("step8_typed_gold_expected_kind_invalid")
    index = _image_index(query, source_memories)
    raw = gold.answer if gold is not None else None
    text_reference = TypedAnswer("text", text=raw) if raw else None

    def result(status, reference, derivation, detail=None):
        return TypedGoldResolution(
            status=status,
            reference=reference,
            text_reference=text_reference,
            expected_kind=expected_kind,
            derivation=derivation,
            source_refs_hash=index.digest,
            query_id=query.query_id,
            query_hash=stable_hash(query.model_dump(mode="json")),
            gold_hash=stable_hash(gold.model_dump(mode="json") if gold is not None else None),
            detail=detail,
        )

    if gold is None:
        return result("missing", None, "none")
    if gold.metadata.get("typed_answer") is not None:
        reference, error = _explicit_annotation(gold.metadata["typed_answer"], index)
        if error:
            return result("unresolved", None, "explicit_typed_annotation", error)
        assert reference is not None
        if expected_kind is not None and reference.kind != expected_kind:
            return result(
                "unresolved", None, "explicit_typed_annotation", "reference_kind_conflict"
            )
        if reference.kind == "text":
            text_reference = reference
        return result("resolved", reference, "explicit_typed_annotation")
    if raw is None:
        return result("unresolved", None, "none", "source_answer_missing")
    if expected_kind == "text":
        return result("resolved", text_reference, "source_text")
    # A scalar-looking image ID such as "2" is ambiguous without a frozen kind.
    if expected_kind in _IMAGE_KINDS or (expected_kind is None and raw in index.aliases):
        reference, error = _image_literal(raw, expected_kind, index)
        return result("unresolved" if error else "resolved", reference, "source_image_refs", error)
    scalar = _scalar(raw)
    if scalar is not None:
        if expected_kind is not None and scalar.kind != expected_kind:
            return result("unresolved", None, "source_scalar", "reference_kind_conflict")
        return result("resolved", scalar, "source_scalar")
    if expected_kind is not None:
        return result("unresolved", None, "none", "source_scalar_not_losslessly_typed")
    reference, error = _image_literal(raw, None, index)
    if reference is not None:
        return result("resolved", reference, "source_image_refs")
    tokens, _ = _tokens(raw)
    if (
        error
        in {
            "collection_kind_requires_protocol",
            "reference_kind_requires_protocol",
            "duplicate_answer_image_ref",
            "ambiguous_source_image_ref",
            "reference_kind_conflict",
        }
        or raw.startswith(
            (
                "Image: ",
                "Images: ",
                "Ordered images: ",
                "Order: ",
                "Unordered images: ",
                "Image set: ",
            )
        )
        or (
            tokens
            and any(token in index.aliases or _explicit_image_token(token) for token in tokens)
        )
    ):
        return result("unresolved", None, "source_image_refs", error)
    return result("text_only", text_reference, "source_text")


def score_resolved_typed_answer(
    candidate: TypedAnswer, resolution: TypedGoldResolution
) -> TypedScore:
    """Unresolved typed references are unavailable, never kind-mismatch zeroes."""
    if resolution.status == "missing":
        return TypedScore(None, False, unavailable_reason="gold_missing")
    if resolution.status == "unresolved" or (
        resolution.status == "text_only" and candidate.kind != "text"
    ):
        return TypedScore(None, False, unavailable_reason=UNRESOLVED)
    return score_typed_answer(candidate, resolution.reference)


def text_judge_reference(
    candidate: TypedAnswer,
    resolution: TypedGoldResolution,
) -> TypedAnswer | None:
    """Keep original text evaluation available, but never stringify typed images."""
    return resolution.text_reference if candidate.kind == "text" else None
