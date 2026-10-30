"""Read-only acceptance checks for an observed webhook payload.

Checks the live MLflow run and streams the matching MinIO model bytes. This
module never writes to, uploads to, or changes either external service.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from fashion_mnist_mlops.config_validation import REQUIRED_MINIO_BUCKET
from .webhook_receiver import validate_received_payload

_CHUNK_BYTES = 64 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


@dataclass(frozen=True)
class VerifiedWebhookEvidence:
    mlflow_run_id: str
    minio_bucket: str
    minio_object_key: str
    validation_accuracy: float
    model_size_bytes: int
    model_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_webhook_evidence(
    payload: object,
    *,
    tracking_client: object,
    minio_client: object,
    experiment_name: str = "fashion-mnist-classifier",
) -> VerifiedWebhookEvidence:
    """Compare one *observed* receiver payload with MLflow and MinIO.

    The payload must be the JSON actually captured by the local receiver, not
    reconstructed from expected values. A completed MLflow run with verified
    storage is necessary but does not, alone, prove webhook delivery.

    Reads MinIO bytes in chunks and compares the complete SHA-256 and size to
    MLflow tags. Does not deserialize potentially untrusted artifact bytes.
    """
    observed = validate_received_payload(payload)
    _require(isinstance(experiment_name, str) and bool(experiment_name.strip()),
             "Expected MLflow experiment name is required.")
    run_id = observed["mlflow_run_id"]
    bucket = observed["minio_bucket"]
    key = observed["minio_object_key"]
    accuracy = float(observed["validation_accuracy"])

    experiment = tracking_client.get_experiment_by_name(experiment_name)
    _require(experiment is not None and experiment.lifecycle_stage == "active",
             "Expected active MLflow experiment was not found.")
    run = tracking_client.get_run(run_id)
    _require(run.info.run_id == run_id, "MLflow run identity does not match webhook.")
    _require(str(run.info.experiment_id) == str(experiment.experiment_id),
             "MLflow run belongs to another experiment.")
    _require(run.info.status == "FINISHED" and run.info.end_time is not None,
             "Webhook must reference a FINISHED MLflow run.")

    tags = run.data.tags
    _require(tags.get("model.storage_status") == "minio-verified",
             "MLflow does not confirm integrity-verified MinIO storage.")
    _require(tags.get("model.minio_bucket") == bucket == REQUIRED_MINIO_BUCKET,
             "MinIO bucket differs between webhook and MLflow.")
    _require(tags.get("model.minio_object_key") == key,
             "MinIO object key differs between webhook and MLflow.")

    try:
        recorded_size = int(tags["model.minio_size_bytes"])
        recorded_sha256 = tags["model.minio_sha256"]
        tracked_accuracy = float(run.data.metrics["val_accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MLflow is missing valid MinIO or validation metadata.") from exc

    _require(recorded_size > 0, "MLflow MinIO object size must be positive.")
    _require(isinstance(recorded_sha256, str) and bool(_SHA256.fullmatch(recorded_sha256)),
             "MLflow MinIO SHA-256 is invalid.")
    _require(math.isfinite(tracked_accuracy) and 0 <= tracked_accuracy <= 1,
             "MLflow validation accuracy is invalid.")
    _require(abs(tracked_accuracy - accuracy) <= 1e-6,
             "Webhook accuracy differs from final MLflow validation accuracy.")

    response = minio_client.get_object(bucket, key)
    hasher = hashlib.sha256()
    total = 0
    try:
        while chunk := response.read(_CHUNK_BYTES):
            total += len(chunk)
            _require(total <= recorded_size, "MinIO model is larger than recorded size.")
            hasher.update(chunk)
    finally:
        try:
            response.close()
        finally:
            response.release_conn()

    _require(total == recorded_size, "MinIO model size differs from MLflow.")
    _require(hasher.hexdigest() == recorded_sha256,
             "MinIO model SHA-256 differs from MLflow.")
    return VerifiedWebhookEvidence(
        mlflow_run_id=run_id,
        minio_bucket=bucket,
        minio_object_key=key,
        validation_accuracy=accuracy,
        model_size_bytes=total,
        model_sha256=recorded_sha256,
    )
