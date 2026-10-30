from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, Subset

from fashion_mnist_mlops.webhook import (
    WebhookDeliveryError,
    send_training_completed_webhook,
)
from fashion_mnist_mlops.webhook import (
    TrainingCompletedPayload,
    load_webhook_settings,
)
from fashion_mnist_mlops.data import create_data_loaders
from fashion_mnist_mlops.storage import publish_model_artifact
from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.model_artifacts import (
    save_model_artifact,
)
from fashion_mnist_mlops.config import (
    TrainingConfig,
    load_training_config,
)
from fashion_mnist_mlops.config import (
    DEFAULT_CONFIG_PATH,
)

from fashion_mnist_mlops.tracking import (
    log_final_validation_accuracy,
    log_training_loss,
    log_validation_loss,
    log_training_parameters,
    log_run_metadata,
    log_minio_artifact_reference,
    start_tracking_run,
)

from fashion_mnist_mlops.paths import (
    get_project_root,
    resolve_project_path,
)

from .training_operations import create_loss_function
from .metrics import EpochRecord
from .training_operations import create_optimizer
from .training_operations import seed_everything
from .training_operations import train_one_epoch
from .training_operations import validate_one_epoch
from .provenance import build_training_provenance


@dataclass(frozen=True)
class TrainingRunResult:
    config: TrainingConfig
    artifact_path: Path
    history: tuple[EpochRecord, ...]
    mlflow_run_id: str | None = None
    minio_bucket: str | None = None
    minio_object_key: str | None = None

    @property
    def final_epoch(self) -> EpochRecord:
        return self.history[-1]


def _limit_loader(
    loader: DataLoader,
    limit: int | None,
    *,
    name: str,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Optionally restrict a DataLoader to its first N samples."""

    if limit is None:
        return loader

    available = len(loader.dataset)

    if limit > available:
        raise ValueError(
            f"{name} ({limit}) exceeds available "
            f"samples ({available})."
        )

    # The original partition was already randomly generated
    # using the configured seed.
    limited_dataset = Subset(
        loader.dataset,
        range(limit),
    )

    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        limited_dataset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


def _run_training_core(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    artifact_path: str | Path | None = None,
    epochs: int | None = None,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    project_root: str | Path | None = None,
    download_data: bool = False,
    mlflow_run_id: str | None = None,
    upload_minio: bool = False,
) -> TrainingRunResult:
    """Execute the existing training lifecycle with optional run provenance."""

    if type(download_data) is not bool:
        raise TypeError(
        "download_data must be a boolean."
    )
    # Load and validate configuration.
    root = get_project_root(project_root)

    config = load_training_config(
        config_path,
        project_root=root,
    )

    if epochs is not None:
        if type(epochs) is not int or epochs < 1:
            raise ValueError(
                "epochs override must be a positive integer."
            )

        config = replace(
            config,
            training=replace(
                config.training,
                epochs=epochs,
            ),
        )

    for name, value in (
        ("max_train_samples", max_train_samples),
        ("max_validation_samples", max_validation_samples),
    ):
        if value is not None and (
            type(value) is not int or value < 1
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    # Select artifact destination.
    if artifact_path is None:
        destination = (
            root
            / "artifacts"
            / "models"
            / uuid4().hex
            / "model.pt"
        )
    else:
        destination = resolve_project_path(
            artifact_path,
            project_root=root,
        )

    if destination.suffix != ".pt":
        raise ValueError(
            "Artifact path must have a .pt extension."
        )

    if destination.exists():
        raise FileExistsError(
            f"Artifact already exists: {destination}"
        )

    # Configure reproducibility.
    seed_everything(config.training.seed)

    # Load the original training and validation partitions.
    loaders = create_data_loaders(
        config,
        root=root / "data",
        download=download_data,
    )

    # Optionally limit dataset sizes for a smoke test.
    train_loader = _limit_loader(
        loaders.train,
        max_train_samples,
        name="max_train_samples",
        shuffle=True,
        seed=config.training.seed,
    )

    validation_loader = _limit_loader(
        loaders.validation,
        max_validation_samples,
        name="max_validation_samples",
        shuffle=False,
        seed=config.training.seed,
    )

    # Capture effective configuration and dataset provenance.
    provenance = build_training_provenance(
        config,
        config_path=config_path,
        project_root=root,
        data_root=root / "data",
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=loaders.test,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        download_data=download_data,
    )

    # Record the effective settings before any optimizer updates or artifact.
    if mlflow_run_id is not None:
        log_training_parameters(mlflow_run_id, provenance)

    # Construct training components.
    model = LeNet5()

    optimizer = create_optimizer(
        model,
        config.training,
    )

    criterion = create_loss_function(
        config.training,
    )

    # Run the training lifecycle.
    history: list[EpochRecord] = []

    print("Starting Fashion-MNIST training")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Optimizer: {config.training.optimizer}")
    print(f"Loss: {config.training.loss}")
    print(f"Seed: {config.training.seed}")
    print(
        "Dataset download:",
        "allowed" if download_data else "disabled (offline)",
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(
        f"Validation samples: {len(validation_loader.dataset)}",
        flush=True,
    )

    for epoch in range(1, config.training.epochs + 1):
        print(
            f"\nEpoch {epoch}/{config.training.epochs}",
            flush=True,
        )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            return_metrics=True,
        )

        # Log every completed training epoch before validation or artifact.
        if mlflow_run_id is not None:
            log_training_loss(
                mlflow_run_id,
                epoch=epoch,
                loss=train_metrics.loss,
            )

        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            return_metrics=True,
        )

        # Preserve one validation-loss point per completed epoch.
        if mlflow_run_id is not None:
            log_validation_loss(
                mlflow_run_id,
                epoch=epoch,
                loss=validation_metrics.loss,
            )

        record = EpochRecord(
            epoch=epoch,
            train=train_metrics,
            validation=validation_metrics,
        )

        history.append(record)

        print(f"  Train loss: {train_metrics.loss:.4f}")
        print(f"  Train accuracy: {train_metrics.accuracy:.2%}")
        print(
            f"  Validation loss: "
            f"{validation_metrics.loss:.4f}"
        )
        print(
            f"  Validation accuracy: "
            f"{validation_metrics.accuracy:.2%}",
            flush=True,
        )

    # Only the last completed epoch supplies final validation accuracy.
    final_record = history[-1]
    if mlflow_run_id is not None:
        log_final_validation_accuracy(
            mlflow_run_id,
            epoch=final_record.epoch,
            accuracy=final_record.validation.accuracy,
        )

    saved_path = save_model_artifact(
        model,
        destination,
        epoch=final_record.epoch,
        validation_accuracy=(
            final_record.validation.accuracy
        ),
        provenance=provenance,
        mlflow_run_id=mlflow_run_id,
    )

    # Record a reference only after the atomic artifact has been published.
    if mlflow_run_id is not None:
        log_run_metadata(
            mlflow_run_id,
            provenance=provenance,
            artifact_path=saved_path,
            project_root=root,
        )

    # Required MLflow parameters, losses, final metric and provenance are
    # logged before attempting storage. Failures propagate to FAILED run status.
    uploaded = None
    if upload_minio:
        if mlflow_run_id is None:
            raise ValueError("MinIO upload requires a tracked MLflow run ID.")
        uploaded = publish_model_artifact(saved_path, run_id=mlflow_run_id)
        # The S3 upload has passed its byte-integrity check. Only now publish
        # the exact bucket/key to this same MLflow run, before FINISHED.
        log_minio_artifact_reference(mlflow_run_id, uploaded)

    return TrainingRunResult(
        config=config,
        artifact_path=saved_path,
        history=tuple(history),
        mlflow_run_id=mlflow_run_id,
        minio_bucket=uploaded.bucket if uploaded else None,
        minio_object_key=uploaded.object_key if uploaded else None,
    )



def run_training(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    artifact_path: str | Path | None = None,
    epochs: int | None = None,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    project_root: str | Path | None = None,
    download_data: bool = False,
    track_mlflow: bool = False,
    upload_minio: bool = False,
) -> TrainingRunResult:
    """Train locally or create one MLflow run around the training lifecycle.

    Tracking is explicitly opt-in so existing offline tests
    and CLI behavior do not require a running MLflow server. A tracked run
    propagates its run ID into the existing artifact metadata and result.
    """
    if type(track_mlflow) is not bool:
        raise TypeError("track_mlflow must be a boolean.")
    if type(upload_minio) is not bool:
        raise TypeError("upload_minio must be a boolean.")
    if upload_minio and not track_mlflow:
        raise ValueError("--upload-minio requires --track-mlflow.")
    # Fail fast on invalid webhook settings before creating an MLflow run or
    # publishing a artifact. Local-only training does not require a webhook.
    if upload_minio:
        load_webhook_settings()

    run_context = start_tracking_run() if track_mlflow else nullcontext(None)
    with run_context as run_id:
        result = _run_training_core(
            config_path=config_path,
            artifact_path=artifact_path,
            epochs=epochs,
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            project_root=project_root,
            download_data=download_data,
            mlflow_run_id=run_id,
            upload_minio=upload_minio,
        )

    # Deliver the success event only after the MLflow context has finalized.
    # Required tracking and verified MinIO publication have already succeeded.
    webhook_receipt = None
    if result.minio_object_key is not None:
        if result.mlflow_run_id is None or result.minio_bucket is None:
            raise RuntimeError(
                "Stored training result is incomplete; refusing webhook delivery."
            )
        payload = TrainingCompletedPayload(
            mlflow_run_id=result.mlflow_run_id,
            minio_bucket=result.minio_bucket,
            minio_object_key=result.minio_object_key,
            validation_accuracy=result.final_epoch.validation.accuracy,
        )
        try:
            webhook_receipt = send_training_completed_webhook(payload)
        except WebhookDeliveryError as exc:
            # The MLflow run is already FINISHED; this is notification-only
            # failure, not a failed model build. Preserve the exception type.
            print(
                "Training and MinIO publication succeeded, but webhook "
                f"delivery failed for MLflow run {result.mlflow_run_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise

    print("\nTraining completed successfully.")
    print(f"Artifact saved: {result.artifact_path}")
    if result.mlflow_run_id is not None:
        print(f"MLflow run ID: {result.mlflow_run_id}")
    if result.minio_object_key is not None:
        print(f"MinIO artifact: {result.minio_bucket}/{result.minio_object_key}")
    if webhook_receipt is not None:
        print(f"Webhook delivered: HTTP {webhook_receipt.status_code}")

    return result
