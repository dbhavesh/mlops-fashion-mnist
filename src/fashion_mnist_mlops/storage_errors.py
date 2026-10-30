"""Consistent, non-secret MinIO error handling for the storage boundary.

S3 SDK errors expose a ``code`` attribute; transport failures usually come
from urllib3.  Do not stringify underlying SDK exceptions into user-facing
messages: response details and request context are not guaranteed secret-free.
Unknown exceptions are deliberately left unchanged for diagnosis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from urllib3.exceptions import HTTPError as Urllib3HTTPError


class StorageCredentialError(PermissionError):
    """MinIO rejected the configured access key, signature, or token."""


class StoragePermissionError(PermissionError):
    """The identity lacks permission for a requested S3 operation."""


class StorageUnavailableError(ConnectionError):
    """MinIO could not be reached, or the connection was interrupted."""


class StorageBucketNotFoundError(FileNotFoundError):
    """The required bucket is absent."""


class StorageObjectNotFoundError(FileNotFoundError):
    """A requested model object is absent."""


_CREDENTIAL_CODES = frozenset({
    "InvalidAccessKeyId", "SignatureDoesNotMatch", "InvalidToken",
    "ExpiredToken", "InvalidAccessKey", "TokenRefreshRequired",
})
_PERMISSION_CODES = frozenset({"AccessDenied", "AllAccessDisabled"})
_BUCKET_MISSING_CODES = frozenset({"NoSuchBucket"})
_OBJECT_MISSING_CODES = frozenset({"NoSuchKey", "NoSuchObject", "NotFound"})
_CONFLICT_CODES = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})
_T = TypeVar("_T")


def normalize_storage_error(
    error: Exception,
    *,
    operation: str,
    bucket: str,
    object_key: str | None = None,
) -> Exception:
    """Translate recognized SDK/transport failures into stable exception types.

    Preserve unknown exceptions rather than diagnosing an unrelated programming
    failure as an S3 outage.  No secrets or original exception text are copied.
    """
    code = getattr(error, "code", None)
    location = f"{bucket}/{object_key}" if object_key else bucket
    if code in _CREDENTIAL_CODES:
        return StorageCredentialError(
            f"MinIO credentials rejected during {operation}. "
            "Check MINIO_ACCESS_KEY and MINIO_SECRET_KEY."
        )
    if code in _PERMISSION_CODES:
        return StoragePermissionError(
            f"MinIO AccessDenied during {operation} for {location}. "
            "Check the identity's bucket/object permissions and credentials."
        )
    if code in _BUCKET_MISSING_CODES:
        return StorageBucketNotFoundError(
            f"Required MinIO bucket is missing during {operation}: {bucket}"
        )
    if code in _OBJECT_MISSING_CODES and object_key is not None:
        return StorageObjectNotFoundError(
            f"MinIO model object not found during {operation}: {location}"
        )
    if code in _CONFLICT_CODES and object_key is not None:
        return FileExistsError(
            f"MinIO model object already exists or conflicts with concurrent write: {location}"
        )
    # Native ConnectionError/PermissionError are already meaningful builtins.
    # Keep their original detail and identity for existing storage callers;
    # normalize the urllib3 transport errors actually emitted by minio-py.
    if isinstance(error, (TimeoutError, Urllib3HTTPError)):
        return StorageUnavailableError(
            f"MinIO {operation} unavailable. "
            "Check the MinIO service and MINIO_ENDPOINT, then retry safely."
        )
    return error


def storage_call(
    operation: str,
    bucket: str,
    invoke: Callable[[], _T],
    *,
    object_key: str | None = None,
) -> _T:
    """Invoke one MinIO operation and translate known failures without retrying.

    Never retry a conditional PUT automatically: an ambiguous upload failure
    may have published the object even when the client lost the response.
    """
    try:
        return invoke()
    except Exception as exc:
        translated = normalize_storage_error(
            exc, operation=operation, bucket=bucket, object_key=object_key,
        )
        if translated is exc:
            raise
        raise translated from exc
