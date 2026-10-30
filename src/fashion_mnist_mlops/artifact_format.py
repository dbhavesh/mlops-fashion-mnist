"""Storage-readiness contract for the existing portable PyTorch model artifact.

The artifact writer uses a single serialization format. The writer in
``model_artifacts`` already publishes a CPU ``state_dict`` with versioned
metadata. This module validates those bytes *before* any future MinIO upload.
"""

import hashlib
from dataclasses import (
    dataclass,
)
from pathlib import (
    Path,
)

import torch

from fashion_mnist_mlops.preprocessing import (
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)

from .model_artifacts import (
    ARTIFACT_FORMAT_VERSION,
    load_model_artifact,
)


# The current inference preprocessing is version 1; other versions need an
# explicitly updated inference loader, rather than silently guessing.
SUPPORTED_PREPROCESSING_VERSION = 1


@dataclass(frozen=True)
class ModelArtifactInfo:
    """Facts verified from serialized bytes, suitable for a subsequent upload.

    The path is local-only; it is not embedded in the uploaded artifact.
    The SHA-256 digest identifies the complete serialized file, not only weights.
    """

    path: Path
    size_bytes: int
    sha256: str
    artifact_format_version: int
    architecture: str
    input_shape: tuple[int, int, int]
    num_classes: int
    class_names: tuple[str, ...]
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    preprocessing_version: int
    epoch: int
    validation_accuracy: float
    mlflow_run_id: str | None
    effective_config_sha256: str


def inspect_model_artifact(
    path: str | Path,
    *,
    expected_run_id: str | None = None,
) -> ModelArtifactInfo:
    """Validate and describe one ready-to-upload version-2 LeNet5 artifact.

    A legacy v1 artifact remains loadable for compatibility, but is not
    eligible for new object-store publication: it has no training provenance.
    This function is read-only; it does not upload, rewrite, or reserve keys.
    """
    if expected_run_id is not None and (
        not isinstance(expected_run_id, str) or not expected_run_id.strip()
    ):
        raise ValueError("expected_run_id must be a non-empty string or None.")

    source = Path(path)
    model, metadata = load_model_artifact(source, device="cpu")

    if metadata["checkpoint_format_version"] != ARTIFACT_FORMAT_VERSION:
        raise ValueError("MinIO model artifacts require artifact format version 2.")

    if expected_run_id is not None and metadata["mlflow_run_id"] != expected_run_id:
        raise ValueError("Artifact MLflow run ID does not match expected_run_id.")

    provenance = metadata["provenance"]
    preprocessing = provenance["preprocessing"]
    expected_shape = [IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]
    expected_mean = list(NORMALIZATION_MEAN)
    expected_std = list(NORMALIZATION_STD)
    if (
        type(preprocessing.get("version")) is not int
        or preprocessing["version"] != SUPPORTED_PREPROCESSING_VERSION
        or preprocessing.get("input_shape") != expected_shape
        or preprocessing.get("normalization_mean") != expected_mean
        or preprocessing.get("normalization_std") != expected_std
    ):
        raise ValueError("Artifact provenance preprocessing is incompatible with inference.")

    if (
        type(provenance["training"]["epochs"]) is not int
        or provenance["training"]["epochs"] != metadata["epoch"]
    ):
        raise ValueError("Artifact epoch differs from effective training provenance.")

    config_digest = provenance["source_config"].get("effective_sha256")
    if (
        not isinstance(config_digest, str)
        or len(config_digest) != 64
        or any(character not in "0123456789abcdef" for character in config_digest)
    ):
        raise ValueError("Artifact provenance is missing a valid effective config SHA-256.")

    if not all(torch.isfinite(t).all().item() for t in model.state_dict().values()):
        raise ValueError("Artifact contains non-finite model parameters.")

    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()

    return ModelArtifactInfo(
        path=source.resolve(),
        size_bytes=source.stat().st_size,
        sha256=digest,
        artifact_format_version=metadata["checkpoint_format_version"],
        architecture=metadata["architecture"],
        input_shape=tuple(metadata["input_shape"]),
        num_classes=metadata["num_classes"],
        class_names=tuple(metadata["class_names"]),
        normalization_mean=tuple(metadata["normalization_mean"]),
        normalization_std=tuple(metadata["normalization_std"]),
        preprocessing_version=preprocessing["version"],
        epoch=metadata["epoch"],
        validation_accuracy=metadata["validation_accuracy"],
        mlflow_run_id=metadata["mlflow_run_id"],
        effective_config_sha256=config_digest,
    )
