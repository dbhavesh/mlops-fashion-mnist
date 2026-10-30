"""Real-data, external-CWD training -> artifact -> inference contract.

Uses a small bounded dataset and never accesses the network. If Fashion-MNIST
is not already cached, pytest explicitly SKIPS this test. A skip does not
count as successful end-to-end verification.
"""

import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST

from fashion_mnist_mlops.data import create_data_loaders
from fashion_mnist_mlops.preprocessing import preprocess_image_for_inference
from fashion_mnist_mlops.model_artifacts import load_model_artifact
from fashion_mnist_mlops.config import load_training_config
from fashion_mnist_mlops.training_operations import (
    create_loss_function,
    validate_one_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require_cached_fashion_mnist():
    """Explicitly check both partitions without permitting any download."""
    data_root = PROJECT_ROOT / "data"
    try:
        train = FashionMNIST(root=str(data_root), train=True, download=False)
        test = FashionMNIST(root=str(data_root), train=False, download=False)
    except RuntimeError as exc:
        if "Dataset not found" in str(exc):
            pytest.skip(
                "Fashion-MNIST is not cached; download it explicitly "
                "before running the end-to-end acceptance test."
            )
        raise
    assert len(train) == 60000
    assert len(test) == 10000
    return train


def test_real_data_cli_artifact_and_inference_outside_project(tmp_path):
    raw_train = require_cached_fashion_mnist()

    external_cwd = tmp_path / "elsewhere"
    external_cwd.mkdir()
    destination = tmp_path / "artifacts" / "smoke_model.pt"

    env = os.environ.copy()
    env.pop("FASHION_MNIST_PROJECT_ROOT", None)

    command = [
        sys.executable,
        "-m",
        "fashion_mnist_mlops.train",
        "--offline",
        "--epochs", "1",
        "--max-train-samples", "32",
        "--max-validation-samples", "16",
        "--artifact-path", str(destination),
    ]
    process = subprocess.run(
        command,
        cwd=external_cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    assert process.returncode == 0, (
        f"CLI failed (exit {process.returncode})\n"
        f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
    )
    assert "Training completed successfully." in process.stdout
    assert "Training samples: 32" in process.stdout
    assert "Validation samples: 16" in process.stdout
    assert destination.is_file()
    assert not (external_cwd / "artifacts").exists()

    model, metadata = load_model_artifact(destination)
    assert not model.training
    assert metadata["checkpoint_format_version"] == 2
    assert metadata["epoch"] == 1
    assert math.isfinite(metadata["validation_accuracy"])

    provenance = metadata["provenance"]
    assert provenance["training"]["epochs"] == 1
    assert provenance["training"]["seed"] == 42
    assert provenance["dataset"]["train_samples_used"] == 32
    assert provenance["dataset"]["validation_samples_used"] == 16
    assert provenance["dataset"]["test_samples_available"] == 10000
    assert provenance["dataset"]["max_train_samples"] == 32
    assert provenance["dataset"]["max_validation_samples"] == 16
    assert provenance["dataset"]["download_allowed"] is False
    assert provenance["dataset"]["location"] == {
        "scope": "project", "path": "data"
    }
    config_path = PROJECT_ROOT / "configs" / "train.toml"
    assert provenance["source_config"]["sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()

    # Reconstruct precisely the same 16 validation images and labels.
    config = load_training_config(project_root=PROJECT_ROOT)
    loaders = create_data_loaders(
        config, root=PROJECT_ROOT / "data", download=False
    )
    train_indices = set(loaders.train.dataset.indices)
    validation_indices = set(loaders.validation.dataset.indices)
    assert train_indices.isdisjoint(validation_indices)
    assert len(train_indices) == 54000
    assert len(validation_indices) == 6000

    subset_loader = DataLoader(
        Subset(loaders.validation.dataset, range(16)),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    metrics = validate_one_epoch(
        model,
        subset_loader,
        create_loss_function(config.training),
        return_metrics=True,
    )
    assert metrics.samples == 16
    assert metrics.accuracy == metadata["validation_accuracy"]
    assert math.isfinite(metrics.loss)

    # Prove inference uses the shared single-image preprocessing contract.
    raw_image, _ = raw_train[0]
    tensor = preprocess_image_for_inference(raw_image)
    assert tensor.shape == (1, 1, 28, 28)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
    assert logits.shape == (1, 10)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(1), rtol=1e-6, atol=1e-6
    )
