"""Read-only image metadata projection, separate from table-creation metadata."""

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class _ViewBase(DeclarativeBase):
    pass


class RuntimeImageAssetView(_ViewBase):
    __tablename__ = "runtime_image_assets_v1"
    __table_args__ = {"schema": "pmmc_runtime"}

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    local_path: Mapped[str | None] = mapped_column(Text)
    remote_url: Mapped[str | None] = mapped_column(Text)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    mime_type: Mapped[str | None] = mapped_column(String(128))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
