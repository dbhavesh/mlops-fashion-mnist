"""Restore verified model artifacts and run CPU inference.

Storage verification, model restoration, and preprocessing are separate.
This module performs no training, API serving, or MLflow writes.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from fashion_mnist_mlops.preprocessing import (
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)
from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.model_artifacts import (
    load_model_artifact,
)
from fashion_mnist_mlops.artifact_format import inspect_model_artifact
from fashion_mnist_mlops.model_keys import validate_model_object_key
from fashion_mnist_mlops.config_validation import REQUIRED_MINIO_BUCKET
from fashion_mnist_mlops.storage import (
    DownloadedModelArtifact,
    retrieve_model_artifact,
)


@dataclass(frozen=True)
class RestoredInferenceModel:
    """A version-2 LeNet5 artifact validated against its MinIO fingerprint."""

    model: LeNet5
    run_id: str
    artifact_path: Path
    class_names: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class InferencePrediction:
    """One Fashion-MNIST class prediction from a preprocessed grayscale image."""

    class_index: int
    class_name: str


def _fingerprint(path: Path) -> tuple[int, str]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return path.stat().st_size, digest


def restore_downloaded_model(artifact: DownloadedModelArtifact) -> RestoredInferenceModel:
    """Reload only a byte-verified, run-matched, compatible version-2 artifact.

    A DownloadedModelArtifact is returned by the verified downloader. Recheck
    its bytes here because a local file can be modified after downloading.
    Keeping the model on CPU makes inference portable and reproducible.
    """
    if not isinstance(artifact, DownloadedModelArtifact):
        raise TypeError("Expected a verified DownloadedModelArtifact from MinIO retrieval.")
    if artifact.bucket != REQUIRED_MINIO_BUCKET:
        raise ValueError("Downloaded model has an unexpected MinIO bucket.")
    validate_model_object_key(artifact.object_key)
    run_id = artifact.object_key.split("/")[1]
    source = Path(artifact.path)
    if source.is_symlink():
        raise ValueError("Refusing to restore a downloaded artifact through a symlink.")

    # The checker verifies the complete serialized file and compatible
    # architecture/preprocessing/provenance before the inference model is used.
    info = inspect_model_artifact(source, expected_run_id=run_id)
    if info.sha256 != artifact.sha256 or info.size_bytes != artifact.size_bytes:
        raise ValueError("Downloaded model no longer matches its verified MinIO fingerprint.")

    model, metadata = load_model_artifact(source, device="cpu")
    if metadata["mlflow_run_id"] != run_id:
        raise ValueError("Restored artifact does not belong to the selected MLflow run.")
    # Detect a file changed during the two read operations, not merely after
    # the initial remote download verification.
    if _fingerprint(source) != (artifact.size_bytes, artifact.sha256):
        raise ValueError("Downloaded artifact was modified during model restoration.")

    model.eval()
    return RestoredInferenceModel(
        model=model,
        run_id=run_id,
        artifact_path=source.resolve(),
        class_names=info.class_names,
        sha256=info.sha256,
    )


def retrieve_inference_model(
    destination: str | Path,
    *,
    object_key: str | None = None,
) -> RestoredInferenceModel:
    """Retrieve one selected remote model and restore it, without retraining.

    The storage layer enforces FINISHED MLflow run, key/metadata match, signed download
    integrity and an atomic local no-overwrite publication.
    """
    artifact = retrieve_model_artifact(destination, object_key=object_key)
    return restore_downloaded_model(artifact)


def predict_preprocessed_image(
    restored: RestoredInferenceModel,
    image: torch.Tensor,
) -> InferencePrediction:
    """Classify one already preprocessed grayscale image in evaluation mode.

    Use ``preprocessing.preprocess_image_for_inference`` to transform PIL images. This
    function neither preprocesses raw uploads nor starts model training.
    """
    if not isinstance(restored, RestoredInferenceModel):
        raise TypeError("restored must be a RestoredInferenceModel.")
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a PyTorch tensor.")
    if image.shape != (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("Expected image tensor shape [1, 1, 28, 28].")
    if image.dtype != torch.float32 or image.device.type != "cpu":
        raise ValueError("Expected a float32 image tensor on CPU.")
    if not torch.isfinite(image).all().item():
        raise ValueError("Image tensor contains non-finite values.")
    if restored.model.training:
        raise ValueError("Restored model must be in evaluation mode for inference.")

    with torch.inference_mode():
        logits = restored.model(image)
    if logits.shape != (1, len(restored.class_names)) or not torch.isfinite(logits).all().item():
        raise ValueError("Restored model produced invalid class logits.")
    index = int(logits.argmax(dim=1).item())
    return InferencePrediction(class_index=index, class_name=restored.class_names[index])
