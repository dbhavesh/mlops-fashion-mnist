"""Serialization and atomic publication of LeNet5 model artifacts."""

import json
import math
import os
import tempfile
from pathlib import (
    Path,
)

import torch

from fashion_mnist_mlops.labels import (
    FASHION_MNIST_CLASSES,
    NUM_CLASSES,
)
from fashion_mnist_mlops.preprocessing import (
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)

from .model import (
    LeNet5,
)


ARTIFACT_FORMAT_VERSION = 2
LEGACY_ARTIFACT_FORMAT_VERSION = 1
ARCHITECTURE_NAME = "LeNet5"


def _validate_training_metadata(
    epoch: int,
    validation_accuracy: float,
    mlflow_run_id: str | None,
) -> None:
    """Validate essential artifact metadata."""
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer.")

    if (
        isinstance(validation_accuracy, bool)
        or not isinstance(validation_accuracy, (int, float))
        or not math.isfinite(validation_accuracy)
        or not 0 <= validation_accuracy <= 1
    ):
        raise ValueError(
            "validation_accuracy must be a finite "
            "number between 0 and 1."
        )

    if mlflow_run_id is not None and (
        not isinstance(mlflow_run_id, str)
        or not mlflow_run_id.strip()
    ):
        raise ValueError(
            "mlflow_run_id must be a non-empty string or None."
        )


def _validate_provenance(value: dict) -> dict:
    """Validate and normalize the JSON-safe version-2 provenance payload."""
    if not isinstance(value, dict):
        raise ValueError("Artifact provenance must be a dictionary.")

    required_sections = (
        "training",
        "validation",
        "dataset",
        "source_config",
        "preprocessing",
        "software",
    )
    for section in required_sections:
        if not isinstance(value.get(section), dict):
            raise ValueError(
                f"Missing or invalid artifact provenance section: {section}."
            )

    training = value["training"]
    for key in ("epochs", "batch_size", "learning_rate", "optimizer", "loss", "seed"):
        if key not in training:
            raise ValueError(f"Artifact provenance missing training.{key}.")

    if "validation_split" not in value["validation"]:
        raise ValueError(
            "Artifact provenance missing validation.validation_split."
        )

    dataset = value["dataset"]
    for key in (
        "name",
        "location",
        "train_samples_used",
        "validation_samples_used",
        "test_samples_available",
        "max_train_samples",
        "max_validation_samples",
        "download_allowed",
    ):
        if key not in dataset:
            raise ValueError(f"Artifact provenance missing dataset.{key}.")

    source_config = value["source_config"]
    if "sha256" not in source_config or "location" not in source_config:
        raise ValueError("Artifact provenance missing source configuration identity.")

    if "version" not in value["preprocessing"]:
        raise ValueError("Artifact provenance missing preprocessing version.")

    try:
        # Keep serialized metadata limited to safe, portable JSON primitives.
        encoded = json.dumps(value, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Artifact provenance must contain JSON-compatible finite values."
        ) from exc


def _write_artifact_atomically(
    artifact: dict,
    destination: Path,
) -> None:
    """Publish a complete artifact without overwriting an existing file.

    Serialize to a temporary sibling file, flush and synchronize its
    contents, then publish using an atomic no-overwrite hard link.
    This requires hard-link support on the destination filesystem.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Artifact already exists: {destination}")

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_stream:
            temporary_path = Path(temporary_stream.name)

            torch.save(artifact, temporary_stream)
            temporary_stream.flush()
            os.fsync(temporary_stream.fileno())

        # os.link() cannot overwrite an existing destination, unlike
        # os.replace(). The final name becomes visible atomically.
        os.link(temporary_path, destination)

    finally:
        # A hard link keeps the published artifact accessible after
        # removing its temporary filename. On errors, clean up the temp.
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_model_artifact(
    model: LeNet5,
    path: str | Path,
    *,
    epoch: int,
    validation_accuracy: float,
    mlflow_run_id: str | None = None,
    provenance: dict | None = None,
) -> Path:
    """Save model weights with atomic publication and optional v2 provenance."""
    if not isinstance(model, LeNet5):
        raise TypeError("model must be a LeNet5 instance.")

    _validate_training_metadata(
        epoch,
        validation_accuracy,
        mlflow_run_id,
    )

    validated_provenance = (
        _validate_provenance(provenance)
        if provenance is not None
        else None
    )

    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("Artifact path must have a .pt extension.")

    # Detach parameters and move all tensors to CPU for portability.
    cpu_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }

    if not all(
        torch.isfinite(tensor).all().item()
        for tensor in cpu_state.values()
    ):
        raise ValueError(
            "Model contains non-finite weights; artifact not saved."
        )

    artifact = {
        "checkpoint_format_version": (
            ARTIFACT_FORMAT_VERSION
            if validated_provenance is not None
            else LEGACY_ARTIFACT_FORMAT_VERSION
        ),
        "architecture": ARCHITECTURE_NAME,
        "model_state_dict": cpu_state,
        "input_shape": [IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
        "num_classes": NUM_CLASSES,
        "class_names": list(FASHION_MNIST_CLASSES),
        "normalization_mean": list(NORMALIZATION_MEAN),
        "normalization_std": list(NORMALIZATION_STD),
        "epoch": epoch,
        "validation_accuracy": float(validation_accuracy),
        "mlflow_run_id": mlflow_run_id,
    }

    if validated_provenance is not None:
        artifact["provenance"] = validated_provenance

    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_artifact_atomically(artifact, destination)
    return destination


def load_model_artifact(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[LeNet5, dict]:
    """Restore a LeNet5 artifact and validate its compatibility."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model artifact not found: {source}")

    artifact = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(artifact, dict):
        raise ValueError("Artifact must contain a dictionary.")

    version = artifact.get("checkpoint_format_version")
    if type(version) is not int or version not in (
        LEGACY_ARTIFACT_FORMAT_VERSION,
        ARTIFACT_FORMAT_VERSION,
    ):
        raise ValueError(
            "Incompatible artifact metadata: checkpoint_format_version."
        )

    expected = {
        "architecture": ARCHITECTURE_NAME,
        "input_shape": [IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
        "num_classes": NUM_CLASSES,
        "class_names": list(FASHION_MNIST_CLASSES),
        "normalization_mean": list(NORMALIZATION_MEAN),
        "normalization_std": list(NORMALIZATION_STD),
    }

    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"Incompatible artifact metadata: {key}.")

    _validate_training_metadata(
        artifact.get("epoch"),
        artifact.get("validation_accuracy"),
        artifact.get("mlflow_run_id"),
    )

    if version == ARTIFACT_FORMAT_VERSION:
        _validate_provenance(artifact.get("provenance"))

    if not isinstance(artifact.get("model_state_dict"), dict):
        raise ValueError("Artifact is missing model_state_dict.")

    model = LeNet5()
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    metadata = {
        key: value
        for key, value in artifact.items()
        if key != "model_state_dict"
    }
    return model, metadata
