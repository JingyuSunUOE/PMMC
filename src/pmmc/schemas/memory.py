"""Memory and asset schemas."""

from __future__ import annotations

from pydantic import Field, model_validator

from pmmc.schemas.base import (
    AssetType,
    BaseSchema,
    DatasetName,
    JsonDict,
    Modality,
    NonEmptyStr,
    NonNegativeInt,
    SpeakerRole,
    UtcDateTime,
    ensure_unique_strings,
    utc_now,
)


class AssetRef(BaseSchema):
    """Reference to a large artifact stored locally or in object storage."""

    asset_id: NonEmptyStr | None = None
    asset_type: AssetType = AssetType.OTHER
    storage_uri: NonEmptyStr | None = None
    local_path: NonEmptyStr | None = None
    remote_url: NonEmptyStr | None = None
    sha256: NonEmptyStr | None = None
    mime_type: NonEmptyStr | None = None
    size_bytes: NonNegativeInt | None = None
    width: NonNegativeInt | None = None
    height: NonNegativeInt | None = None
    source_id: NonEmptyStr | None = None
    run_id: NonEmptyStr | None = None
    created_at: UtcDateTime = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class ImageRef(BaseSchema):
    """Reference to an image without reading the image content."""

    image_id: NonEmptyStr
    asset_id: NonEmptyStr | None = None
    local_path: NonEmptyStr | None = None
    remote_url: NonEmptyStr | None = None
    storage_uri: NonEmptyStr | None = None
    caption: NonEmptyStr | None = None
    sha256: NonEmptyStr | None = None
    mime_type: NonEmptyStr | None = None
    width: NonNegativeInt | None = None
    height: NonNegativeInt | None = None
    metadata: JsonDict = Field(default_factory=dict)


class MemoryUnit(BaseSchema):
    """Unified long-term memory unit used by PMMC."""

    memory_id: NonEmptyStr
    dataset: DatasetName
    conversation_id: NonEmptyStr
    session_id: NonEmptyStr
    turn_id: NonEmptyStr
    timestamp_order: NonNegativeInt
    speaker: SpeakerRole = SpeakerRole.UNKNOWN
    text: NonEmptyStr | None = None
    caption: NonEmptyStr | None = None
    image_ids: list[NonEmptyStr] = Field(default_factory=list)
    image_refs: list[ImageRef] = Field(default_factory=list)
    entity_mentions: list[NonEmptyStr] = Field(default_factory=list)
    modality: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    available_after_session: NonEmptyStr
    token_count: NonNegativeInt | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> MemoryUnit:
        if not (self.text or self.caption or self.image_ids or self.image_refs):
            raise ValueError("memory unit must contain text, caption, or image content")
        if not self.modality:
            raise ValueError("modality must contain at least one item")

        ensure_unique_strings(self.image_ids, "image_ids")
        ensure_unique_strings(self.entity_mentions, "entity_mentions")
        image_ref_ids = [image_ref.image_id for image_ref in self.image_refs]
        ensure_unique_strings(image_ref_ids, "image_refs.image_id")
        if self.image_ids and self.image_refs and set(self.image_ids) != set(image_ref_ids):
            raise ValueError("image_ids and image_refs must reference the same image IDs")
        if len(self.modality) != len(set(self.modality)):
            raise ValueError("modality must not contain duplicate values")
        return self
