"""Verified model storage with conditional writes and atomic local publication."""

import hashlib
import inspect
import os
import re
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from .artifact_format import inspect_model_artifact
from .config import load_infrastructure_config, validate_infrastructure_config
from .config_validation import REQUIRED_MINIO_BUCKET
from .model_keys import build_model_object_key, validate_model_object_key
from .storage_errors import (
    StorageBucketNotFoundError,
    StorageCredentialError,
    StorageObjectNotFoundError,
    StoragePermissionError,
    StorageUnavailableError,
    storage_call,
)


def ensure_bucket_exists(client: object, *, bucket_name: str) -> bool:
    """Ensure the assignment bucket exists; return True only if newly created.

    The caller supplies an authenticated MinIO SDK client. We intentionally do
    not create objects, configure policies, or upload model artifacts here.
    Exceptions from the MinIO SDK propagate rather than being mistaken for a
    missing bucket. A second existence check confirms successful provisioning.
    """
    if bucket_name != REQUIRED_MINIO_BUCKET:
        raise ValueError(
            f"MinIO bucket must be exactly {REQUIRED_MINIO_BUCKET!r}."
        )

    if storage_call("bucket lookup", bucket_name, lambda: client.bucket_exists(bucket_name)):
        return False

    storage_call("bucket creation", bucket_name, lambda: client.make_bucket(bucket_name))
    if not storage_call("bucket lookup", bucket_name, lambda: client.bucket_exists(bucket_name)):
        raise RuntimeError(
            f"MinIO bucket {bucket_name!r} was not visible after creation."
        )
    return True


# A compact LeNet5 artifact is about 180 KiB. Restrict this API to single PUTs;
# do not silently fall back to multipart where conditional-write guarantees differ.
MAX_SINGLE_PUT_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class UploadedModelArtifact:
    """Confirmed bucket/key and complete serialized-file fingerprint."""

    bucket: str
    object_key: str
    size_bytes: int
    sha256: str



def _put_if_absent(client: object, *, bucket_name: str, object_key: str, payload: bytes) -> None:
    """Perform a signed, server-conditional, single-request S3 PUT.

    MinIO Python SDK versions differ: newer put_object() accepts extra_headers,
    but older releases do not. For older SDKs use their signed _execute()
    transport directly. In particular, do NOT use put_object(headers=...): that
    parameter normalizes unknown headers to x-amz-meta-* and can silently
    discard the If-None-Match precondition.

    If neither safe mechanism is supported, fail closed; never upload without
    server-side If-None-Match: * protection.
    """
    try:
        params = inspect.signature(client.put_object).parameters
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Cannot verify safe MinIO conditional-write support.") from exc

    has_extra_headers = "extra_headers" in params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    if has_extra_headers:
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_key,
            data=BytesIO(payload),
            length=len(payload),
            content_type="application/octet-stream",
            extra_headers={"If-None-Match": "*"},
        )
        return

    # The private method is used only as a compatibility fallback for older
    # minio-py versions. It signs the real S3 PUT, preserving these raw headers.
    # Keeping this isolated avoids a change to the project's storage clients.
    execute = getattr(client, "_execute", None)
    if not callable(execute):
        raise RuntimeError(
            "Installed MinIO SDK lacks a safe conditional-write API; "
            "refusing to upload without If-None-Match."
        )
    execute(
        method="PUT",
        bucket_name=bucket_name,
        object_name=object_key,
        body=payload,
        headers={
            "Content-Type": "application/octet-stream",
            "If-None-Match": "*",
        },
    )

def upload_model_artifact(
    client: object,
    *,
    artifact_path: str | Path,
    run_id: str,
    bucket_name: str,
) -> UploadedModelArtifact:
    """Create a new versioned model object without replacing an existing one.

    Validate and snapshot the local artifact, ensure the required bucket exists,
    upload with a *server-enforced* no-overwrite condition, then read the object
    back and compare its entire SHA-256 and length. A failed verification raises;
    it never deletes an object whose write might have succeeded.
    """
    if bucket_name != REQUIRED_MINIO_BUCKET:
        raise ValueError(f"Model uploads require bucket {REQUIRED_MINIO_BUCKET!r}.")

    object_key = build_model_object_key(run_id)
    info = inspect_model_artifact(artifact_path, expected_run_id=run_id)
    if not 0 < info.size_bytes <= MAX_SINGLE_PUT_BYTES:
        raise ValueError(
            f"Artifact must be 1–{MAX_SINGLE_PUT_BYTES} bytes for an atomic "
            "conditional single-request upload."
        )

    # Snapshot the exact validated bytes; refuse a changed file before sending.
    payload = info.path.read_bytes()
    if len(payload) != info.size_bytes or hashlib.sha256(payload).hexdigest() != info.sha256:
        raise RuntimeError("Artifact changed after storage-readiness inspection.")

    if not storage_call("bucket lookup", bucket_name, lambda: client.bucket_exists(bucket_name)):
        raise FileNotFoundError(f"Required MinIO bucket is missing: {bucket_name}")

    # Do not replace this with a stat_object()/put_object() check-then-write:
    # the condition MUST be evaluated by the storage server at write time.
    storage_call(
        "upload", bucket_name,
        lambda: _put_if_absent(
            client, bucket_name=bucket_name, object_key=object_key, payload=payload,
        ),
        object_key=object_key,
    )

    # HEAD metadata/ETags are not proof of full byte identity (and multipart
    # ETags are not MD5). Verify by streaming the actual newly stored bytes.
    response = storage_call(
        "upload verification", bucket_name,
        lambda: client.get_object(bucket_name, object_key), object_key=object_key,
    )
    hasher = hashlib.sha256()
    count = 0
    try:
        while chunk := storage_call(
            "upload verification", bucket_name,
            lambda: response.read(64 * 1024), object_key=object_key,
        ):
            count += len(chunk)
            hasher.update(chunk)
    finally:
        try:
            response.close()
        finally:
            response.release_conn()

    if count != len(payload) or hasher.hexdigest() != info.sha256:
        raise RuntimeError(
            f"Uploaded MinIO model failed byte-integrity verification: "
            f"{bucket_name}/{object_key}. Object was not removed."
        )

    return UploadedModelArtifact(
        bucket=bucket_name,
        object_key=object_key,
        size_bytes=count,
        sha256=info.sha256,
    )


def publish_model_artifact(
    artifact_path: str | Path,
    *,
    run_id: str,
) -> UploadedModelArtifact:
    """Use validated environment settings to publish a completed tracked run."""
    config = load_infrastructure_config()
    validate_infrastructure_config(config, require_minio_credentials=True)
    settings = config.minio

    from minio import Minio

    client = Minio(
        endpoint=settings.endpoint,
        access_key=settings.access_key,
        secret_key=settings.secret_key,
        secure=settings.secure,
    )
    return upload_model_artifact(
        client,
        artifact_path=artifact_path,
        run_id=run_id,
        bucket_name=settings.bucket,
    )


_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class DownloadedModelArtifact:
    """A published local file verified against the run's stored fingerprint."""

    bucket: str
    object_key: str
    path: Path
    size_bytes: int
    sha256: str


def download_model_artifact(
    client: object,
    *,
    bucket_name: str,
    object_key: str,
    destination: str | Path,
    expected_sha256: str,
    expected_size_bytes: int,
) -> DownloadedModelArtifact:
    """Stream a specific MinIO object into a new, verified local file.

    Writes to a temporary file beside the destination, validates its complete
    bytes, then publishes via a same-filesystem hard link. os.link() fails if
    another process created the destination first; os.replace() is deliberately
    NOT used because it would overwrite an existing artifact.
    """
    if bucket_name != REQUIRED_MINIO_BUCKET:
        raise ValueError(f"Model downloads require bucket {REQUIRED_MINIO_BUCKET!r}.")
    validate_model_object_key(object_key)
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise ValueError("Expected model SHA-256 must be lowercase 64-digit hexadecimal.")
    if (
        type(expected_size_bytes) is not int
        or not 0 < expected_size_bytes <= MAX_SINGLE_PUT_BYTES
    ):
        raise ValueError(
            f"Expected model size must be between 1 and {MAX_SINGLE_PUT_BYTES} bytes."
        )

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite downloaded model: {target}")
    if not storage_call("bucket lookup", bucket_name, lambda: client.bucket_exists(bucket_name)):
        raise FileNotFoundError(f"Required MinIO bucket is missing: {bucket_name}")

    # A missing S3 object or authentication failure must not create a final file.
    response = storage_call(
        "download", bucket_name,
        lambda: client.get_object(bucket_name, object_key), object_key=object_key,
    )
    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        count = 0
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".part",
            dir=target.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            while chunk := storage_call(
                "download", bucket_name,
                lambda: response.read(_CHUNK_BYTES), object_key=object_key,
            ):
                count += len(chunk)
                if count > expected_size_bytes:
                    raise ValueError("MinIO object exceeds its recorded size.")
                hasher.update(chunk)
                output.write(chunk)

        if count != expected_size_bytes or hasher.hexdigest() != expected_sha256:
            raise ValueError("Downloaded MinIO model failed size/SHA-256 integrity verification.")

        # Atomic, no-clobber publication on NTFS and POSIX filesystems.
        # Fail closed if the filesystem cannot support an atomic hard link.
        os.link(temporary, target)
        return DownloadedModelArtifact(
            bucket=bucket_name,
            object_key=object_key,
            path=target.resolve(),
            size_bytes=count,
            sha256=expected_sha256,
        )
    finally:
        try:
            response.close()
        finally:
            try:
                response.release_conn()
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def retrieve_model_artifact(
    destination: str | Path,
    *,
    object_key: str | None = None,
) -> DownloadedModelArtifact:
    """Retrieve the selected object with a read-only MLflow reference check.

    Uses MODEL_OBJECT_KEY when no explicit key is provided. The run ID is the
    version segment of that key; the corresponding FINISHED MLflow run must
    record matching verified MinIO metadata. No training or inference occurs.
    """
    config = load_infrastructure_config()
    validate_infrastructure_config(
        config,
        require_minio_credentials=True,
        require_model_object_key=object_key is None,
    )
    selected_key = config.model.object_key if object_key is None else object_key
    validate_model_object_key(selected_key)
    run_id = selected_key.split("/")[1]

    from mlflow.tracking import MlflowClient

    tracking = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
    run = tracking.get_run(run_id)
    tags = run.data.tags
    if run.info.run_id != run_id or run.info.status != "FINISHED":
        raise ValueError("Selected MLflow run must be FINISHED and match the object key.")
    if (
        tags.get("model.storage_status") != "minio-verified"
        or tags.get("model.minio_bucket") != config.minio.bucket
        or tags.get("model.minio_object_key") != selected_key
    ):
        raise ValueError("MLflow does not confirm this exact verified MinIO object.")
    try:
        recorded_size = int(tags["model.minio_size_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MLflow is missing a valid MinIO object size.") from exc
    recorded_digest = tags.get("model.minio_sha256", "")

    # Validate reference metadata before connecting to storage or creating dirs.
    if type(recorded_size) is not int or not 0 < recorded_size <= MAX_SINGLE_PUT_BYTES:
        raise ValueError("MLflow MinIO object size is out of bounds.")
    if not isinstance(recorded_digest, str) or not _SHA256.fullmatch(recorded_digest):
        raise ValueError("MLflow MinIO SHA-256 is invalid.")

    from minio import Minio

    minio = Minio(
        endpoint=config.minio.endpoint,
        access_key=config.minio.access_key,
        secret_key=config.minio.secret_key,
        secure=config.minio.secure,
    )
    return download_model_artifact(
        minio,
        bucket_name=config.minio.bucket,
        object_key=selected_key,
        destination=destination,
        expected_sha256=recorded_digest,
        expected_size_bytes=recorded_size,
    )
