"""MLflow run lifecycle, effective configuration, metrics, and artifact references."""

import json
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from typing import Any

from mlflow.entities import Param, RunTag
from mlflow.tracking import MlflowClient

from .config import load_infrastructure_config
from .config_validation import REQUIRED_MINIO_BUCKET
from .labels import FASHION_MNIST_CLASSES, NUM_CLASSES
from .model_artifacts import ARCHITECTURE_NAME, ARTIFACT_FORMAT_VERSION
from .model_keys import build_model_object_key
from .storage import UploadedModelArtifact


def _report_failed_cleanup(
    *, run_id: str, status: str, cleanup_error: Exception, primary: BaseException
) -> None:
    """Preserve the primary error and make a failed MLflow cleanup visible."""
    detail = (
        f"MLflow run {run_id!r}: could not set status {status} after an "
        f"earlier failure ({type(cleanup_error).__name__}: {cleanup_error}). "
        "Its remote status is unconfirmed; inspect the tracking server."
    )
    primary.add_note(detail)
    print(f"WARNING: {detail}", file=sys.stderr)


@contextmanager
def start_tracking_run() -> Iterator[str]:
    """Create and finalize exactly one run in the configured experiment.

    A training or logging failure is never hidden by an additional cleanup
    failure. A failure to confirm FINISHED propagates and triggers a best-effort
    FAILED update; an already-published local artifact is not removed.
    """
    settings = load_infrastructure_config().mlflow
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)

    if experiment is None or experiment.lifecycle_stage != "active":
        raise RuntimeError(
            f"MLflow experiment '{settings.experiment_name}' does not exist "
            "or is not active. Create or reactivate the configured experiment before training."
        )

    run = client.create_run(experiment_id=experiment.experiment_id)
    run_id = run.info.run_id
    if not isinstance(run_id, str) or not run_id.strip():
        raise RuntimeError("MLflow did not return a valid run ID.")

    try:
        yield run_id
    except BaseException as primary:
        # Report the original training/logging failure even if the tracking
        # server is also unreachable during the FAILED status update.
        try:
            client.set_terminated(run_id, status="FAILED")
        except Exception as cleanup_error:
            _report_failed_cleanup(
                run_id=run_id,
                status="FAILED",
                cleanup_error=cleanup_error,
                primary=primary,
            )
        raise
    else:
        try:
            client.set_terminated(run_id, status="FINISHED")
        except Exception as finish_error:
            # Completion is not confirmed, so do not report a successful run.
            # A timeout may have applied FINISHED remotely; FAILED is best
            # effort, and the caller must inspect the actual server status.
            primary = RuntimeError(
                f"Could not confirm MLflow run {run_id!r} as FINISHED. "
                "Tracking finalization failed; a local artifact may already "
                "exist. Inspect the MLflow run before treating it as successful."
            )
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception as cleanup_error:
                _report_failed_cleanup(
                    run_id=run_id,
                    status="FAILED",
                    cleanup_error=cleanup_error,
                    primary=primary,
                )
            raise primary from finish_error


# These fields describe the values actually used by the training runner.
_TRAINING_KEYS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "loss",
    "seed",
)
_DATASET_KEYS = (
    "train_samples_used",
    "validation_samples_used",
    "max_train_samples",
    "max_validation_samples",
    "download_allowed",
)


def _as_mlflow_string(value: Any) -> str:
    """Represent booleans and optional values consistently in MLflow."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def effective_training_parameters(provenance: Mapping[str, Any]) -> dict[str, str]:
    """Select safe, effective settings from the runner's provenance snapshot."""
    training = provenance["training"]
    validation = provenance["validation"]
    dataset = provenance["dataset"]

    values = {key: training[key] for key in _TRAINING_KEYS}
    values["validation_split"] = validation["validation_split"]
    values.update({key: dataset[key] for key in _DATASET_KEYS})
    return {key: _as_mlflow_string(value) for key, value in values.items()}


def log_training_parameters(run_id: str, provenance: Mapping[str, Any]) -> None:
    """Log all effective settings in one MLflow batch, propagating failures."""
    settings = load_infrastructure_config().mlflow
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    params = effective_training_parameters(provenance)
    client.log_batch(
        run_id=run_id,
        params=[Param(key, value) for key, value in params.items()],
    )


def log_training_loss(run_id: str, *, epoch: int, loss: float) -> None:
    """Record one completed training epoch using a 1-based epoch index.

    Propagate invalid values and tracking errors rather than publishing a
    successful artifact with incomplete required metrics.
    """
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer.")
    if not isfinite(loss):
        raise ValueError("training loss must be finite.")

    settings = load_infrastructure_config().mlflow
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    client.log_metric(run_id, key="train_loss", value=loss, step=epoch)


def log_validation_loss(run_id: str, *, epoch: int, loss: float) -> None:
    """Log validation loss at the completed epoch's 1-based epoch index."""
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer.")
    if not isfinite(loss):
        raise ValueError("validation loss must be finite.")

    settings = load_infrastructure_config().mlflow
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    client.log_metric(run_id, key="val_loss", value=loss, step=epoch)


def log_final_validation_accuracy(
    run_id: str, *, epoch: int, accuracy: float
) -> None:
    """Log only the final validation accuracy, at the final epoch index."""
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer.")
    if not isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError("validation accuracy must be finite and within [0, 1].")

    settings = load_infrastructure_config().mlflow
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    client.log_metric(run_id, key="val_accuracy", value=accuracy, step=epoch)


PROVENANCE_ARTIFACT = "metadata/training_provenance.json"


def model_run_tags(
    provenance: Mapping[str, Any],
    *,
    artifact_path: Path,
    project_root: Path,
) -> dict[str, str]:
    """Describe a *published* local artifact without leaking host paths.

    For artifacts outside the project, only the basename is recorded. Such
    references are descriptive, not portable retrieval locations. MinIO object
    references are recorded separately after verified upload.
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Artifact not published: {artifact_path}")

    resolved = artifact_path.resolve()
    try:
        reference = resolved.relative_to(Path(project_root).resolve()).as_posix()
        scope = "project"
    except ValueError:
        reference = resolved.name
        scope = "external"

    source = provenance["source_config"]
    preprocessing = provenance["preprocessing"]

    return {
        "model.architecture": ARCHITECTURE_NAME,
        "model.input_shape": json.dumps(preprocessing["input_shape"]),
        "model.num_classes": str(NUM_CLASSES),
        "model.class_names": json.dumps(list(FASHION_MNIST_CLASSES)),
        "model.artifact_format_version": str(ARTIFACT_FORMAT_VERSION),
        "model.storage_status": "local-only",
        "model.local_artifact_scope": scope,
        "model.local_artifact_ref": reference,
        "config.source_sha256": source["sha256"],
        "config.effective_sha256": source["effective_sha256"],
        "metadata.provenance_artifact": PROVENANCE_ARTIFACT,
    }


def log_run_metadata(
    run_id: str,
    *,
    provenance: dict[str, Any],
    artifact_path: Path,
    project_root: Path,
) -> None:
    """Persist effective provenance and searchable tags in an existing run.

    Called only after atomic local artifact publication. A logging failure
    propagates to the run context (FAILED), but an already-published local
    artifact is deliberately not deleted or overwritten.
    """
    tags = model_run_tags(
        provenance,
        artifact_path=artifact_path,
        project_root=project_root,
    )
    settings = load_infrastructure_config().mlflow
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    client.log_dict(run_id, provenance, PROVENANCE_ARTIFACT)
    client.log_batch(
        run_id=run_id,
        tags=[RunTag(key, value) for key, value in tags.items()],
        synchronous=True,
    )


MINIO_STORAGE_STATUS = "minio-verified"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


def minio_artifact_tags(run_id: str, uploaded: UploadedModelArtifact) -> dict[str, str]:
    """Build searchable MLflow tags for an already verified S3 upload.

    This function never uploads anything or changes the artifact. The caller
    must have obtained `uploaded` from the post-upload integrity verifier.
    """
    expected_key = build_model_object_key(run_id)
    if uploaded.bucket != REQUIRED_MINIO_BUCKET:
        raise ValueError(f"MinIO reference must use bucket {REQUIRED_MINIO_BUCKET!r}.")
    if uploaded.object_key != expected_key:
        raise ValueError("MinIO object key does not belong to the MLflow run ID.")
    if type(uploaded.size_bytes) is not int or uploaded.size_bytes <= 0:
        raise ValueError("MinIO object size must be a positive integer.")
    if not isinstance(uploaded.sha256, str) or not _SHA256.fullmatch(uploaded.sha256):
        raise ValueError("MinIO object SHA-256 must be a lowercase 64-digit hexadecimal digest.")

    return {
        "model.storage_status": MINIO_STORAGE_STATUS,
        "model.minio_bucket": uploaded.bucket,
        "model.minio_object_key": uploaded.object_key,
        "model.minio_sha256": uploaded.sha256,
        "model.minio_size_bytes": str(uploaded.size_bytes),
    }


def log_minio_artifact_reference(run_id: str, uploaded: UploadedModelArtifact) -> None:
    """Attach the verified MinIO reference to the matching live MLflow run.

    Must run after the checked S3 upload but before run finalization.
    Logging errors propagate: the surrounding run context then marks FAILED.
    An existing S3 object and local artifact are never deleted on failure.
    """
    tags = minio_artifact_tags(run_id, uploaded)
    tracking_uri = load_infrastructure_config().mlflow.tracking_uri
    client = MlflowClient(tracking_uri=tracking_uri)
    client.log_batch(
        run_id=run_id,
        tags=[RunTag(key, value) for key, value in tags.items()],
        synchronous=True,
    )
