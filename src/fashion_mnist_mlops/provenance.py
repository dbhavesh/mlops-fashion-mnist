"""Portable provenance for a completed Fashion-MNIST training run."""

import hashlib
import json
import platform
from dataclasses import (
    asdict,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import torch
import torchvision
from torch.utils.data import (
    DataLoader,
)

from fashion_mnist_mlops.preprocessing import (
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)
from fashion_mnist_mlops.config import (
    TrainingConfig,
)
from fashion_mnist_mlops.paths import (
    resolve_project_path,
)


PREPROCESSING_VERSION = 1


def _portable_location(path: Path, project_root: Path) -> dict[str, str]:
    """Avoid embedding a developer's absolute home directory in artifacts."""
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return {"scope": "external", "name": resolved.name}
    return {"scope": "project", "path": relative.as_posix()}


def build_training_provenance(
    config: TrainingConfig,
    *,
    config_path: str | Path,
    project_root: Path,
    data_root: Path,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    max_train_samples: int | None,
    max_validation_samples: int | None,
    download_data: bool,
) -> dict[str, Any]:
    """Snapshot the effective settings and environment for model lineage.

    The snapshot is JSON-compatible and excludes credentials and user home
    directories. Source TOML identity is a SHA-256 hash of its file bytes.
    """
    resolved_config = resolve_project_path(
        config_path,
        project_root=project_root,
    )
    source_digest = hashlib.sha256(resolved_config.read_bytes()).hexdigest()

    effective_config = {
        "training": asdict(config.training),
        "validation": asdict(config.validation),
    }
    config_digest = hashlib.sha256(
        json.dumps(effective_config, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()

    return {
        **effective_config,
        "dataset": {
            "name": "FashionMNIST",
            "source": "torchvision.datasets.FashionMNIST",
            "location": _portable_location(data_root, project_root),
            "train_samples_used": len(train_loader.dataset),
            "validation_samples_used": len(validation_loader.dataset),
            "test_samples_available": len(test_loader.dataset),
            "max_train_samples": max_train_samples,
            "max_validation_samples": max_validation_samples,
            "download_allowed": download_data,
        },
        "source_config": {
            "location": _portable_location(resolved_config, project_root),
            "sha256": source_digest,
            "effective_sha256": config_digest,
        },
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "input_shape": [IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
            "normalization_mean": list(NORMALIZATION_MEAN),
            "normalization_std": list(NORMALIZATION_STD),
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
    }
