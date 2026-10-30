"""Canonical, storage-independent naming contract for model artifact objects.

This module defines object identities without uploading or retrieving objects.
Both training/storage and inference configuration must use the same contract.
"""

import re


MODEL_OBJECT_PREFIX = "model/"
MODEL_OBJECT_FILENAME = "model.pt"

# MLflow run IDs are hexadecimal; safe alternative identifiers may contain
# alphanumeric characters, underscores or hyphens. Forbid any path syntax,
# whitespace, URL escaping and punctuation to keep keys portable and unambiguous.
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z", re.ASCII)
_KEY_FORMAT = "model/<run-id>/model.pt"


def build_model_object_key(run_id: str) -> str:
    """Build the deterministic S3 key for one MLflow run or unique identifier.

    This function does not guarantee that *uploads* cannot overwrite objects;
    exclusive-write semantics must be enforced by the uploader.
    """
    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1–128 ASCII characters, begin with a letter "
            "or digit, and contain only letters, digits, '_' or '-'."
        )
    return f"{MODEL_OBJECT_PREFIX}{run_id}/{MODEL_OBJECT_FILENAME}"


def validate_model_object_key(object_key: str) -> str:
    """Validate a configured object key, preserving it unchanged on success."""
    if not isinstance(object_key, str):
        raise ValueError(f"Object key must follow '{_KEY_FORMAT}'.")

    pieces = object_key.split("/")
    if len(pieces) != 3:
        raise ValueError(f"Object key must follow '{_KEY_FORMAT}'.")

    prefix, run_id, filename = pieces
    if prefix != MODEL_OBJECT_PREFIX.rstrip("/") or filename != MODEL_OBJECT_FILENAME:
        raise ValueError(f"Object key must follow '{_KEY_FORMAT}'.")

    try:
        expected = build_model_object_key(run_id)
    except ValueError as exc:
        raise ValueError(f"Object key must follow '{_KEY_FORMAT}' with a safe run ID.") from exc
    if object_key != expected:
        raise ValueError(f"Object key must follow '{_KEY_FORMAT}'.")
    return object_key
