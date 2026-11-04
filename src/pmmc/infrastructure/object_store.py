"""Private S3-compatible object storage with checksum verification."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pmmc.infrastructure.settings import Settings

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ObjectStoreError(RuntimeError):
    """Object storage operation failed without exposing credentials."""


@dataclass(frozen=True)
class ObjectMetadata:
    key: str
    size_bytes: int
    sha256: str | None
    uri: str
    content_type: str | None = None


class ObjectStore(Protocol):
    async def check(self) -> None: ...

    async def put_file(
        self,
        source: Path,
        key: str,
        *,
        expected_sha256: str | None = None,
        content_type: str | None = None,
    ) -> ObjectMetadata: ...

    async def put_bytes(
        self,
        data: bytes,
        key: str,
        *,
        expected_sha256: str | None = None,
        content_type: str | None = None,
    ) -> ObjectMetadata: ...

    async def get_bytes(self, key: str) -> bytes: ...

    async def head(self, key: str) -> ObjectMetadata | None: ...

    async def presign_get(self, key: str, *, expires_seconds: int) -> str: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_object_key(key: str) -> str:
    path = PurePosixPath(key)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in key:
        raise ValueError("object key must be a safe relative POSIX path")
    return str(path)


def asset_object_key(dataset: str, sha256: str) -> str:
    normalized = sha256.lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("asset SHA256 must contain 64 lowercase hexadecimal characters")
    return f"benchmark-assets/{dataset}/sha256/{normalized[:2]}/{normalized}"


def snapshot_object_key(dataset: str, snapshot_id: str, file_name: str) -> str:
    return validate_object_key(f"snapshots/{dataset}/{snapshot_id}/{file_name}")


def corpus_object_key(
    dataset: str,
    snapshot_id: str,
    corpus_id: str,
    file_name: str,
) -> str:
    return validate_object_key(
        f"baseline-corpora/{dataset}/{snapshot_id}/{corpus_id}/{file_name}"
    )


def run_artifact_object_key(
    run_id: str,
    artifact_type: str,
    sha256: str,
    *,
    extension: str | None = None,
    prefix: str = "runs",
) -> str:
    normalized = sha256.lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("artifact SHA256 must contain 64 lowercase hexadecimal characters")
    suffix = ""
    if extension is not None:
        clean = extension.removeprefix(".")
        if not clean or not re.fullmatch(r"[a-z0-9]+", clean):
            raise ValueError("artifact extension must contain lowercase letters or digits")
        suffix = f".{clean}"
    return validate_object_key(
        f"{prefix}/{run_id}/{artifact_type}/sha256/{normalized[:2]}/{normalized}{suffix}"
    )


def runtime_artifact_object_key(
    runtime_id: str,
    artifact_type: str,
    sha256: str,
    *,
    extension: str | None = None,
) -> str:
    """Return a content-addressed key for secret-free managed-runtime artifacts."""

    return run_artifact_object_key(
        runtime_id,
        artifact_type,
        sha256,
        extension=extension,
        prefix="managed-runtimes",
    )


def resolve_local_asset(project_root: Path, local_path: str) -> Path:
    candidate = (project_root / local_path).resolve()
    root = project_root.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("asset local_path escapes the project root")
    if not candidate.is_file():
        raise FileNotFoundError(f"local asset does not exist: {local_path}")
    return candidate


class FileSystemObjectStore:
    """Deterministic test backend with the same key and checksum semantics."""

    def __init__(self, root: Path, bucket: str) -> None:
        self.root = root.resolve()
        self.bucket = bucket
        (self.root / bucket).mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        candidate = (self.root / self.bucket / validate_object_key(key)).resolve()
        bucket_root = (self.root / self.bucket).resolve()
        if not candidate.is_relative_to(bucket_root):
            raise ValueError("object key escapes the bucket root")
        return candidate

    async def check(self) -> None:
        if not (self.root / self.bucket).is_dir():
            raise ObjectStoreError("object store bucket is unavailable")

    async def put_file(
        self,
        source: Path,
        key: str,
        *,
        expected_sha256: str | None = None,
        content_type: str | None = None,
    ) -> ObjectMetadata:
        actual = await asyncio.to_thread(sha256_file, source)
        if expected_sha256 is not None and actual != expected_sha256.lower():
            raise ObjectStoreError("source checksum does not match expected SHA256")
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = await asyncio.to_thread(sha256_file, target)
            if existing != actual:
                raise ObjectStoreError("existing object has a different checksum")
        else:
            await asyncio.to_thread(shutil.copyfile, source, target)
        return ObjectMetadata(
            key=validate_object_key(key),
            size_bytes=source.stat().st_size,
            sha256=actual,
            uri=f"s3://{self.bucket}/{validate_object_key(key)}",
            content_type=content_type,
        )

    async def put_bytes(
        self,
        data: bytes,
        key: str,
        *,
        expected_sha256: str | None = None,
        content_type: str | None = None,
    ) -> ObjectMetadata:
        actual = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and actual != expected_sha256.lower():
            raise ObjectStoreError("content checksum does not match expected SHA256")
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = await asyncio.to_thread(sha256_file, target)
            if existing != actual:
                raise ObjectStoreError("existing object has a different checksum")
        else:
            await asyncio.to_thread(target.write_bytes, data)
        return ObjectMetadata(
            key=validate_object_key(key),
            size_bytes=len(data),
            sha256=actual,
            uri=f"s3://{self.bucket}/{validate_object_key(key)}",
            content_type=content_type,
        )

    async def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ObjectStoreError("object does not exist")
        return await asyncio.to_thread(path.read_bytes)

    async def head(self, key: str) -> ObjectMetadata | None:
        path = self._path(key)
        if not path.is_file():
            return None
        digest = await asyncio.to_thread(sha256_file, path)
        return ObjectMetadata(
            key=validate_object_key(key),
            size_bytes=path.stat().st_size,
            sha256=digest,
            uri=f"s3://{self.bucket}/{validate_object_key(key)}",
        )

    async def presign_get(self, key: str, *, expires_seconds: int) -> str:
        if expires_seconds <= 0:
            raise ValueError("presigned URL expiry must be positive")
        path = self._path(key)
        if not path.is_file():
            raise ObjectStoreError("object does not exist")
        return path.as_uri()


class S3ObjectStore:
    """Async facade over boto3 for AWS S3 and MinIO-compatible endpoints."""

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str,
        secure: bool,
        path_style: bool,
        model_endpoint: str | None = None,
    ) -> None:
        try:
            boto3 = importlib.import_module("boto3")
            botocore_config = importlib.import_module("botocore.config")
        except ModuleNotFoundError as exc:
            raise ObjectStoreError("boto3 is required for S3-compatible object storage") from exc
        config = botocore_config.Config(s3={"addressing_style": "path" if path_style else "auto"})
        self.bucket = bucket

        def client_for(client_endpoint: str) -> Any:
            return boto3.client(
                "s3",
                endpoint_url=client_endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
                use_ssl=secure,
                config=config,
            )

        self.client: Any = client_for(endpoint)
        self.presign_client: Any = (
            client_for(model_endpoint)
            if model_endpoint and model_endpoint != endpoint
            else self.client
        )

    async def check(self) -> None:
        try:
            await asyncio.to_thread(self.client.head_bucket, Bucket=self.bucket)
        except Exception as exc:
            raise ObjectStoreError("object store bucket is unavailable") from exc

    async def put_file(
        self,
        source: Path,
        key: str,
        *,
        expected_sha256: str | None = None,
        content_type: str | None = None,
    ) -> ObjectMetadata:
        safe_key = validate_object_key(key)
        actual = await asyncio.to_thread(sha256_file, source)
        if expected_sha256 is not None and actual != expected_sha256.lower():
            raise ObjectStoreError("source checksum does not match expected SHA256")
        existing = await self.head(safe_key)
        if existing is not None:
            if existing.sha256 != actual:
                raise ObjectStoreError("existing object has a different checksum")
            return existing
        try:
            extra_args: dict[str, Any] = {"Metadata": {"sha256": actual}}
            if content_type is not None:
                extra_args["ContentType"] = content_type
            await asyncio.to_thread(
                self.client.upload_file,
                str(source),
                self.bucket,
                safe_key,
                ExtraArgs=extra_args,
            )
        except Exception as exc:
            raise ObjectStoreError("failed to upload object") from exc
        return ObjectMetadata(
            key=safe_key,
            size_bytes=source.stat().st_size,
            sha256=actual,
            uri=f"s3://{self.bucket}/{safe_key}",
            content_type=content_type,
        )

    async def put_bytes(
        self,
        data: bytes,
        key: str,
        *,
        expected_sha256: str | None = None,
        content_type: str | None = None,
    ) -> ObjectMetadata:
        safe_key = validate_object_key(key)
        actual = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and actual != expected_sha256.lower():
            raise ObjectStoreError("content checksum does not match expected SHA256")
        existing = await self.head(safe_key)
        if existing is not None:
            if existing.sha256 != actual:
                raise ObjectStoreError("existing object has a different checksum")
            return existing
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": safe_key,
            "Body": data,
            "Metadata": {"sha256": actual},
        }
        if content_type is not None:
            kwargs["ContentType"] = content_type
        try:
            await asyncio.to_thread(self.client.put_object, **kwargs)
        except Exception as exc:
            raise ObjectStoreError("failed to upload object") from exc
        return ObjectMetadata(
            key=safe_key,
            size_bytes=len(data),
            sha256=actual,
            uri=f"s3://{self.bucket}/{safe_key}",
            content_type=content_type,
        )

    async def get_bytes(self, key: str) -> bytes:
        safe_key = validate_object_key(key)
        try:
            response = await asyncio.to_thread(
                self.client.get_object, Bucket=self.bucket, Key=safe_key
            )
            return await asyncio.to_thread(response["Body"].read)
        except Exception as exc:
            raise ObjectStoreError("failed to read object") from exc

    async def head(self, key: str) -> ObjectMetadata | None:
        safe_key = validate_object_key(key)
        try:
            response = await asyncio.to_thread(
                self.client.head_object, Bucket=self.bucket, Key=safe_key
            )
        except Exception as exc:
            response = getattr(exc, "response", {})
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404:
                return None
            raise ObjectStoreError("failed to inspect object") from exc
        return ObjectMetadata(
            key=safe_key,
            size_bytes=int(response["ContentLength"]),
            sha256=response.get("Metadata", {}).get("sha256"),
            uri=f"s3://{self.bucket}/{safe_key}",
            content_type=response.get("ContentType"),
        )

    async def presign_get(self, key: str, *, expires_seconds: int) -> str:
        safe_key = validate_object_key(key)
        if expires_seconds <= 0:
            raise ValueError("presigned URL expiry must be positive")
        try:
            return str(
                await asyncio.to_thread(
                    self.presign_client.generate_presigned_url,
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": safe_key},
                    ExpiresIn=expires_seconds,
                )
            )
        except Exception as exc:
            raise ObjectStoreError("failed to create presigned object URL") from exc


def _create_role_store(
    settings: Settings,
    *,
    bucket: str,
    access_key: str | None,
    secret_key: str | None,
) -> S3ObjectStore:
    missing = [
        name
        for name, value in {
            "endpoint": settings.object_store_endpoint,
            "bucket": bucket,
            "access key": access_key,
            "secret key": secret_key,
        }.items()
        if not value
    ]
    if missing:
        raise ObjectStoreError("missing object store configuration: " + ", ".join(missing))
    return S3ObjectStore(
        endpoint=str(settings.object_store_endpoint),
        bucket=bucket,
        access_key=str(access_key),
        secret_key=str(secret_key),
        region=settings.object_store_region,
        secure=settings.object_store_secure,
        path_style=settings.object_store_path_style,
        model_endpoint=settings.object_store_model_endpoint,
    )


def create_runtime_object_store(settings: Settings) -> S3ObjectStore:
    return _create_role_store(
        settings,
        bucket=settings.object_store_bucket,
        access_key=(
            settings.object_store_access_key.get_secret_value()
            if settings.object_store_access_key
            else None
        ),
        secret_key=(
            settings.object_store_secret_key.get_secret_value()
            if settings.object_store_secret_key
            else None
        ),
    )


def create_evaluator_object_stores(settings: Settings) -> tuple[S3ObjectStore, S3ObjectStore]:
    access_key = (
        settings.object_store_evaluator_access_key.get_secret_value()
        if settings.object_store_evaluator_access_key
        else None
    )
    secret_key = (
        settings.object_store_evaluator_secret_key.get_secret_value()
        if settings.object_store_evaluator_secret_key
        else None
    )
    return (
        _create_role_store(
            settings,
            bucket=settings.object_store_bucket,
            access_key=access_key,
            secret_key=secret_key,
        ),
        _create_role_store(
            settings,
            bucket=settings.object_store_evaluation_bucket,
            access_key=access_key,
            secret_key=secret_key,
        ),
    )


def create_admin_object_stores(
    settings: Settings,
) -> tuple[S3ObjectStore, S3ObjectStore]:
    """Create owner-scoped stores for explicit snapshot and asset synchronization."""

    required = {
        "endpoint": settings.object_store_endpoint,
        "runtime bucket": settings.object_store_bucket,
        "evaluation bucket": settings.object_store_evaluation_bucket,
        "admin access key": (
            settings.object_store_admin_access_key.get_secret_value()
            if settings.object_store_admin_access_key
            else None
        ),
        "admin secret key": (
            settings.object_store_admin_secret_key.get_secret_value()
            if settings.object_store_admin_secret_key
            else None
        ),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ObjectStoreError("missing object store configuration: " + ", ".join(missing))
    return (
        S3ObjectStore(
            endpoint=str(required["endpoint"]),
            bucket=str(required["runtime bucket"]),
            access_key=str(required["admin access key"]),
            secret_key=str(required["admin secret key"]),
            region=settings.object_store_region,
            secure=settings.object_store_secure,
            path_style=settings.object_store_path_style,
            model_endpoint=settings.object_store_model_endpoint,
        ),
        S3ObjectStore(
            endpoint=str(required["endpoint"]),
            bucket=str(required["evaluation bucket"]),
            access_key=str(required["admin access key"]),
            secret_key=str(required["admin secret key"]),
            region=settings.object_store_region,
            secure=settings.object_store_secure,
            path_style=settings.object_store_path_style,
            model_endpoint=settings.object_store_model_endpoint,
        ),
    )
