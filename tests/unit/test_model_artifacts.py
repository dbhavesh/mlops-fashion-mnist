import math

import pytest
import torch

from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.model_artifacts import (
    load_model_artifact,
    save_model_artifact,
)


def test_artifact_round_trip(tmp_path):
    torch.manual_seed(42)

    model = LeNet5()
    model.eval()

    images = torch.randn(3, 1, 28, 28)

    with torch.inference_mode():
        original_logits = model(images)

    path = tmp_path / "model.pt"

    save_model_artifact(
        model,
        path,
        epoch=2,
        validation_accuracy=0.75,
        mlflow_run_id="test-run-123",
    )

    restored, metadata = load_model_artifact(path)

    with torch.inference_mode():
        restored_logits = restored(images)

    assert torch.equal(
        original_logits,
        restored_logits,
    )

    assert not restored.training

    assert metadata["epoch"] == 2
    assert metadata["validation_accuracy"] == 0.75
    assert metadata["mlflow_run_id"] == "test-run-123"
    assert metadata["architecture"] == "LeNet5"

    assert metadata["input_shape"] == [1, 28, 28]
    assert metadata["num_classes"] == 10
    assert len(metadata["class_names"]) == 10


def test_artifact_preserves_model_parameters(tmp_path):
    model = LeNet5()

    path = tmp_path / "model.pt"

    save_model_artifact(
        model,
        path,
        epoch=1,
        validation_accuracy=0.5,
    )

    restored, _ = load_model_artifact(path)

    for name, original in model.state_dict().items():
        assert torch.equal(
            original,
            restored.state_dict()[name],
        )


def test_artifact_cannot_overwrite_existing_file(tmp_path):
    model = LeNet5()
    path = tmp_path / "model.pt"

    save_model_artifact(
        model,
        path,
        epoch=1,
        validation_accuracy=0.5,
    )

    with pytest.raises(FileExistsError):
        save_model_artifact(
            model,
            path,
            epoch=2,
            validation_accuracy=0.6,
        )


def test_missing_artifact_is_rejected(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found",
    ):
        load_model_artifact(
            tmp_path / "missing.pt"
        )


@pytest.mark.parametrize(
    "epoch, accuracy",
    [
        (0, 0.5),
        (1, -0.1),
        (1, 1.1),
        (1, float("nan")),
        (1, float("inf")),
    ],
)
def test_invalid_artifact_metadata_is_rejected(
    tmp_path,
    epoch,
    accuracy,
):
    model = LeNet5()

    with pytest.raises(ValueError):
        save_model_artifact(
            model,
            tmp_path / "model.pt",
            epoch=epoch,
            validation_accuracy=accuracy,
        )


def test_incompatible_architecture_is_rejected(tmp_path):
    model = LeNet5()

    original = tmp_path / "model.pt"

    save_model_artifact(
        model,
        original,
        epoch=1,
        validation_accuracy=0.5,
    )

    payload = torch.load(
        original,
        weights_only=True,
    )

    payload["architecture"] = "IncompatibleModel"

    invalid = tmp_path / "invalid.pt"

    torch.save(payload, invalid)

    with pytest.raises(
        ValueError,
        match="Incompatible artifact metadata",
    ):
        load_model_artifact(invalid)


def test_incompatible_weights_are_rejected(tmp_path):
    model = LeNet5()

    original = tmp_path / "model.pt"

    save_model_artifact(
        model,
        original,
        epoch=1,
        validation_accuracy=0.5,
    )

    payload = torch.load(
        original,
        weights_only=True,
    )

    # Wrong number of output classes.
    payload["model_state_dict"]["fc3.weight"] = (
        torch.zeros(9, 84)
    )

    invalid = tmp_path / "invalid_weights.pt"

    torch.save(payload, invalid)

    with pytest.raises(
        RuntimeError,
        match="size mismatch",
    ):
        load_model_artifact(invalid)


def test_invalid_extension_is_rejected(tmp_path):
    model = LeNet5()

    with pytest.raises(
        ValueError,
        match=".pt extension",
    ):
        save_model_artifact(
            model,
            tmp_path / "model.txt",
            epoch=1,
            validation_accuracy=0.5,
        )


def test_non_finite_model_weights_are_rejected(tmp_path):
    model = LeNet5()

    with torch.no_grad():
        model.fc3.weight[0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="non-finite weights",
    ):
        save_model_artifact(
            model,
            tmp_path / "model.pt",
            epoch=1,
            validation_accuracy=0.5,
        )
