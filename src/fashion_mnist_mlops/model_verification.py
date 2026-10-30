"""verify independent model versions without changing remote storage.

Two *completed* MLflow runs must point to distinct, byte-verified MinIO keys.
Re-download and restore both objects from S3, without depending on the local
training artifacts or modifying anything in MinIO/MLflow.
"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from fashion_mnist_mlops.inference import (
    predict_preprocessed_image,
    restore_downloaded_model,
)
from fashion_mnist_mlops.model_keys import build_model_object_key
from fashion_mnist_mlops.config_validation import REQUIRED_MINIO_BUCKET
from fashion_mnist_mlops.tracking import minio_artifact_tags

from fashion_mnist_mlops.storage import download_model_artifact
from fashion_mnist_mlops.storage import UploadedModelArtifact


@dataclass(frozen=True)
class VerifiedModelVersion:
    run_id: str
    object_key: str
    sha256: str
    size_bytes: int
    predicted_class: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _get_reference(tracking_client: object, run_id: str, experiment_id: str):
    run = tracking_client.get_run(run_id)
    _require(run.info.run_id == run_id, f"MLflow run identity mismatch: {run_id}")
    _require(str(run.info.experiment_id) == experiment_id,
             f"Run {run_id} is not in the expected experiment")
    _require(run.info.status == "FINISHED" and run.info.end_time is not None,
             f"Run {run_id} is not FINISHED")
    tags = run.data.tags
    try:
        uploaded = UploadedModelArtifact(
            bucket=tags["model.minio_bucket"],
            object_key=tags["model.minio_object_key"],
            size_bytes=int(tags["model.minio_size_bytes"]),
            sha256=tags["model.minio_sha256"],
        )
        expected_tags = minio_artifact_tags(run_id, uploaded)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Run {run_id} has an invalid MinIO reference") from exc
    _require(all(tags.get(k) == v for k, v in expected_tags.items()),
             f"Run {run_id} has inconsistent MinIO tags")
    return run, uploaded


def verify_model_versions(
    tracking_client: object,
    minio_client: object,
    *,
    older_run_id: str,
    newer_run_id: str,
    experiment_name: str = "fashion-mnist-classifier",
    bucket_name: str = REQUIRED_MINIO_BUCKET,
) -> tuple[VerifiedModelVersion, VerifiedModelVersion]:
    """Verify an older and newer run still own separate intact S3 model objects.

    Reads the complete remote bytes, validates their SHA-256 and artifact
    identity, restores both LeNet5 models, and runs a small CPU prediction.
    Temporary local copies are deleted. No writes are made to MLflow or MinIO.
    """
    _require(bucket_name == REQUIRED_MINIO_BUCKET,
             f"Expected exact bucket {REQUIRED_MINIO_BUCKET!r}")
    older_key = build_model_object_key(older_run_id)
    newer_key = build_model_object_key(newer_run_id)
    _require(older_run_id != newer_run_id, "Two distinct MLflow run IDs are required")
    _require(bool(experiment_name.strip()), "MLflow experiment name is required")

    experiment = tracking_client.get_experiment_by_name(experiment_name)
    _require(experiment is not None and experiment.lifecycle_stage == "active",
             f"Active MLflow experiment not found: {experiment_name}")
    experiment_id = str(experiment.experiment_id)
    older_run, older_ref = _get_reference(tracking_client, older_run_id, experiment_id)
    newer_run, newer_ref = _get_reference(tracking_client, newer_run_id, experiment_id)
    _require(older_run.info.end_time <= newer_run.info.end_time,
             "Pass the earlier run as --older-run and the later run as --newer-run")
    _require(older_ref.bucket == newer_ref.bucket == bucket_name,
             "Runs do not reference the same required bucket")
    _require(older_ref.object_key == older_key and newer_ref.object_key == newer_key,
             "Run object-key identity mismatch")
    _require(older_ref.object_key != newer_ref.object_key,
             "Different runs must not share one MinIO object key")

    result = []
    # Verified download and restoration operations are intentionally reused rather than duplicating
    # download/validation code; the original local training files are not used.
    with TemporaryDirectory(prefix="fashion-minio-version-check-") as scratch:
        for run_id, ref in ((older_run_id, older_ref), (newer_run_id, newer_ref)):
            destination = Path(scratch) / run_id / "model.pt"
            fetched = download_model_artifact(
                minio_client,
                bucket_name=bucket_name,
                object_key=ref.object_key,
                destination=destination,
                expected_sha256=ref.sha256,
                expected_size_bytes=ref.size_bytes,
            )
            restored = restore_downloaded_model(fetched)
            _require(restored.run_id == run_id, "Restored model belongs to a different run")
            prediction = predict_preprocessed_image(
                restored, torch.zeros((1, 1, 28, 28), dtype=torch.float32),
            )
            result.append(VerifiedModelVersion(
                run_id=run_id,
                object_key=ref.object_key,
                sha256=fetched.sha256,
                size_bytes=fetched.size_bytes,
                predicted_class=prediction.class_name,
            ))
    return result[0], result[1]
