"""Bad artifacts and failed training must not masquerade as success."""

import importlib
import pickle
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fashion_mnist_mlops.data import DataLoaders
from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.model_artifacts import (
    load_model_artifact,
    save_model_artifact,
)
from fashion_mnist_mlops.training import run_training


PROJECT_ROOT = Path(__file__).resolve().parents[2]
runner = importlib.import_module("fashion_mnist_mlops.training")


def test_truncated_artifact_is_rejected(tmp_path):
    original = tmp_path / "complete.pt"
    truncated = tmp_path / "truncated.pt"
    save_model_artifact(
        LeNet5(), original, epoch=1, validation_accuracy=0.5
    )

    serialized = original.read_bytes()
    truncated.write_bytes(serialized[: len(serialized) // 2])

    # The underlying PyTorch zip reader raises RuntimeError in typical
    # versions; other supported failures depend on the PyTorch release.
    with pytest.raises((RuntimeError, OSError, EOFError, pickle.UnpicklingError)):
        load_model_artifact(truncated)

    assert original.is_file()
    assert truncated.is_file()


def test_artifact_without_state_dict_is_rejected(tmp_path):
    source = tmp_path / "valid.pt"
    malformed = tmp_path / "missing_weights.pt"
    save_model_artifact(
        LeNet5(), source, epoch=1, validation_accuracy=0.5
    )

    payload = torch.load(source, map_location="cpu", weights_only=True)
    del payload["model_state_dict"]
    torch.save(payload, malformed)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_model_artifact(malformed)


def test_v2_artifact_without_provenance_is_rejected(tmp_path):
    source = tmp_path / "legacy.pt"
    malformed = tmp_path / "invalid_v2.pt"
    save_model_artifact(
        LeNet5(), source, epoch=1, validation_accuracy=0.5
    )

    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["checkpoint_format_version"] = 2
    torch.save(payload, malformed)

    with pytest.raises(ValueError, match="provenance"):
        load_model_artifact(malformed)


def fake_small_loaders(config, root=None, download=False):
    images = torch.zeros(8, 1, 28, 28)
    labels = torch.arange(8, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    return DataLoaders(train=loader, validation=loader, test=loader)


def test_validation_exception_does_not_publish_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", fake_small_loaders)

    def failing_validation(*args, **kwargs):
        raise RuntimeError("Simulated validation failure")

    monkeypatch.setattr(runner, "validate_one_epoch", failing_validation)
    destination = tmp_path / "validation_failure.pt"

    with pytest.raises(RuntimeError, match="Simulated validation failure"):
        run_training(
            config_path=PROJECT_ROOT / "configs" / "train.toml",
            project_root=PROJECT_ROOT,
            artifact_path=destination,
            epochs=1,
            download_data=False,
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".validation_failure.pt.*.tmp")) == []


def test_provenance_exception_does_not_publish_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", fake_small_loaders)

    def failing_provenance(*args, **kwargs):
        raise RuntimeError("Simulated provenance failure")

    monkeypatch.setattr(runner, "build_training_provenance", failing_provenance)
    destination = tmp_path / "provenance_failure.pt"

    with pytest.raises(RuntimeError, match="Simulated provenance failure"):
        run_training(
            config_path=PROJECT_ROOT / "configs" / "train.toml",
            project_root=PROJECT_ROOT,
            artifact_path=destination,
            epochs=1,
            download_data=False,
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".provenance_failure.pt.*.tmp")) == []
