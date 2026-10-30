"""Explicit model selection and lazy, verified API model caching."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

from fashion_mnist_mlops.model_keys import validate_model_object_key
from fashion_mnist_mlops.config_validation import REQUIRED_MINIO_BUCKET


@dataclass(frozen=True)
class APISettings:
    """Presentation metadata; no infrastructure connection is required."""

    title: str = "Fashion-MNIST Prediction API"
    version: str = "0.1.0"


@dataclass(frozen=True)
class ModelSelection:
    """Exact bucket/key identity of one stored model, not a loaded artifact.

    Only the established bucket and object-key convention is accepted.
    Nothing here checks whether the object exists or whether MLflow confirms it.
    """

    bucket: str
    object_key: str

    def __post_init__(self) -> None:
        if self.bucket != REQUIRED_MINIO_BUCKET:
            raise ValueError(
                f"MINIO_BUCKET must be exactly {REQUIRED_MINIO_BUCKET!r}."
            )
        try:
            validate_model_object_key(self.object_key)
        except ValueError as exc:
            raise ValueError(
                "MODEL_OBJECT_KEY must follow 'model/<run-id>/model.pt' "
                "with a safe run identifier."
            ) from exc

    @property
    def run_id(self) -> str:
        """Version identifier embedded in the validated canonical object key."""
        return self.object_key.split("/")[1]


def load_model_selection(environ: Mapping[str, str] | None = None) -> ModelSelection:
    """Select one model explicitly via MINIO_BUCKET and MODEL_OBJECT_KEY.

    Reads the current process environment on every call unless a mapping is
    supplied. A missing/invalid selection fails rather than selecting `latest`.
    This function performs no I/O or service authentication and never requires
    access keys, MLflow, MinIO, or an active API server.
    """
    env = os.environ if environ is None else environ
    bucket = env.get("MINIO_BUCKET")
    key = env.get("MODEL_OBJECT_KEY")

    if not isinstance(bucket, str) or not bucket:
        raise ValueError("MINIO_BUCKET is required to select an API model.")
    if not isinstance(key, str) or not key:
        raise ValueError(
            "MODEL_OBJECT_KEY is required; select a specific "
            "'model/<run-id>/model.pt' artifact."
        )

    return ModelSelection(bucket=bucket, object_key=key)

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING

from fastapi import Request


if TYPE_CHECKING:
    from fashion_mnist_mlops.inference import RestoredInferenceModel


class APIModelLoadError(RuntimeError):
    """A specifically selected model could not be retrieved or initialized."""


def _retrieve_verified_model(path: Path, *, object_key: str) -> RestoredInferenceModel:
    """Invoke the existing integrity-verified restoration path.

    Imported lazily so FastAPI startup and /health remain independent of
    PyTorch, MLflow and MinIO availability.
    """
    from fashion_mnist_mlops.inference import retrieve_inference_model

    return retrieve_inference_model(path, object_key=object_key)


class APIModelLoader:
    """One cached, CPU inference model per FastAPI application instance.

    The first load downloads a versioned, validated model to a private temporary
    directory and retains that directory as long as the model is cached. Calls
    for the exact same ModelSelection return the same in-memory instance.
    A different selection reloads atomically under a thread lock; a failure
    leaves the last known-good selection/model in place. ``clear()`` removes
    the cached artifact and releases the model. Never call clear() while
    another thread needs the artifact path outside the loader API.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._selection: ModelSelection | None = None
        self._restored: RestoredInferenceModel | None = None
        self._directory: TemporaryDirectory[str] | None = None

    @property
    def cached_selection(self) -> ModelSelection | None:
        """Read the identity of the currently cached model, if any."""
        with self._lock:
            return self._selection

    def load(self, selection: ModelSelection | None = None) -> RestoredInferenceModel:
        """Resolve and lazily load an explicit model; never select `latest`.

        When selection is omitted, resolve the current environment on each call.
        ModelSelection validation precedes any network access. The model is
        held in eval mode on CPU by the existing restoration function.
        """
        selected = load_model_selection() if selection is None else selection
        if not isinstance(selected, ModelSelection):
            raise TypeError("selection must be a ModelSelection instance.")

        with self._lock:
            if self._selection == selected and self._restored is not None:
                return self._restored

            # downloader requires a new, nonexisting destination; use
            # a fresh private directory per attempted load (never overwrite).
            pending = TemporaryDirectory(prefix="fashion-mnist-api-model-")
            path = Path(pending.name) / "model.pt"
            try:
                restored = _retrieve_verified_model(path, object_key=selected.object_key)
                # Defend against accidental changes to the integration
                # that could return another version or a training-mode model.
                if restored.run_id != selected.run_id:
                    raise ValueError("Restored model run ID differs from the selection.")
                if restored.model.training:
                    raise ValueError("Restored model must be in eval mode.")
                if Path(restored.artifact_path).resolve() != path.resolve():
                    raise ValueError("Restored artifact is not the selected download.")
                if not path.is_file():
                    raise ValueError("Verified artifact was not published.")
            except Exception as exc:
                pending.cleanup()
                raise APIModelLoadError(
                    f"Could not initialize selected model run {selected.run_id}."
                ) from exc

            # Commit the cache only after every check succeeds. Keep the file
            # alive until replacement or application shutdown.
            previous_directory = self._directory
            self._directory = pending
            self._selection = selected
            self._restored = restored
            if previous_directory is not None:
                previous_directory.cleanup()
            return restored

    def clear(self) -> None:
        """Drop the model cache and delete its local artifact."""
        with self._lock:
            previous_directory = self._directory
            self._directory = None
            self._selection = None
            self._restored = None
            if previous_directory is not None:
                previous_directory.cleanup()


def get_app_model(request: Request) -> RestoredInferenceModel:
    """Deferred FastAPI dependency for the prediction route ()."""
    return request.app.state.model_loader.load()
