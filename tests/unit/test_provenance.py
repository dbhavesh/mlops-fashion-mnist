"""Regression tests for versioned artifact provenance."""

import hashlib
from dataclasses import (
    replace,
)

import pytest
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.model_artifacts import (
    load_model_artifact,
    save_model_artifact,
)
from fashion_mnist_mlops.config import (
    load_training_config,
)
from fashion_mnist_mlops.provenance import (
    build_training_provenance,
)


def _provenance(tmp_path, *, epochs=4):
    source = tmp_path / "training.toml"
    source.write_text(
        "[training]\nepochs = 10\nbatch_size = 64\n",
        encoding="utf-8",
    )

    original = load_training_config()
    effective = replace(
        original,
        training=replace(original.training, epochs=epochs),
    )

    examples = TensorDataset(
        torch.zeros(8, 1, 28, 28),
        torch.zeros(8, dtype=torch.long),
    )
    loader = DataLoader(examples, batch_size=4)

    provenance = build_training_provenance(
        effective,
        config_path=source,
        project_root=tmp_path,
        data_root=tmp_path / "data",
        train_loader=loader,
        validation_loader=loader,
        test_loader=loader,
        max_train_samples=8,
        max_validation_samples=8,
        download_data=False,
    )
    return provenance, source


def test_provenance_contains_effective_configuration(tmp_path):
    provenance, source = _provenance(tmp_path, epochs=4)

    assert provenance["training"]["epochs"] == 4
    assert provenance["training"]["batch_size"] == 64
    assert provenance["training"]["seed"] == 42
    assert provenance["validation"]["validation_split"] == 0.1
    assert provenance["dataset"]["train_samples_used"] == 8
    assert provenance["dataset"]["validation_samples_used"] == 8
    assert provenance["dataset"]["max_train_samples"] == 8
    assert provenance["dataset"]["download_allowed"] is False
    assert provenance["dataset"]["location"] == {
        "scope": "project", "path": "data"
    }
    assert provenance["source_config"]["sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert provenance["preprocessing"]["version"] == 1
    assert provenance["software"]["torch"]
    assert provenance["software"]["torchvision"]


def test_v2_artifact_preserves_provenance_and_logits(tmp_path):
    provenance, _ = _provenance(tmp_path)
    model = LeNet5().eval()
    images = torch.randn(2, 1, 28, 28)

    with torch.inference_mode():
        initial_logits = model(images)

    path = tmp_path / "version2.pt"
    save_model_artifact(
        model,
        path,
        epoch=4,
        validation_accuracy=0.5,
        provenance=provenance,
    )
    restored_model, metadata = load_model_artifact(path)

    with torch.inference_mode():
        restored_logits = restored_model(images)

    assert metadata["checkpoint_format_version"] == 2
    assert metadata["provenance"] == provenance
    assert torch.equal(initial_logits, restored_logits)
    assert list(tmp_path.glob(".version2.pt.*.tmp")) == []


def test_legacy_v1_artifact_still_loads(tmp_path):
    path = tmp_path / "legacy.pt"
    save_model_artifact(
        LeNet5(),
        path,
        epoch=1,
        validation_accuracy=0.25,
    )
    _, metadata = load_model_artifact(path)

    assert metadata["checkpoint_format_version"] == 1
    assert "provenance" not in metadata


def test_invalid_provenance_is_rejected_before_saving(tmp_path):
    target = tmp_path / "invalid.pt"
    with pytest.raises(ValueError, match="provenance"):
        save_model_artifact(
            LeNet5(),
            target,
            epoch=1,
            validation_accuracy=0.5,
            provenance={"training": {"epochs": 1}},
        )

    assert not target.exists()


def test_v2_artifact_without_provenance_is_rejected(tmp_path):
    provenance, _ = _provenance(tmp_path)
    original = tmp_path / "model.pt"
    save_model_artifact(
        LeNet5(),
        original,
        epoch=1,
        validation_accuracy=0.5,
        provenance=provenance,
    )

    malformed = torch.load(original, weights_only=True)
    malformed.pop("provenance")
    bad_path = tmp_path / "malformed.pt"
    torch.save(malformed, bad_path)

    with pytest.raises(ValueError, match="provenance"):
        load_model_artifact(bad_path)
