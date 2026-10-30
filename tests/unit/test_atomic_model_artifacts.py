from pathlib import (
    Path,
)

import pytest
import torch

import fashion_mnist_mlops.model_artifacts as model_artifacts

from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.model_artifacts import (
    load_model_artifact,
    save_model_artifact,
)


def assert_no_temporary_files(directory: Path):
    leftovers = list(
        directory.glob(".model.pt.*.tmp")
    )

    assert leftovers == []


def test_atomic_save_creates_valid_artifact(tmp_path):
    model = LeNet5()
    destination = tmp_path / "model.pt"

    save_model_artifact(
        model,
        destination,
        epoch=1,
        validation_accuracy=0.5,
    )

    assert destination.is_file()

    restored, metadata = load_model_artifact(
        destination
    )

    for name, tensor in model.state_dict().items():
        assert torch.equal(
            tensor,
            restored.state_dict()[name],
        )

    assert metadata["epoch"] == 1

    assert_no_temporary_files(tmp_path)


def test_serialization_failure_leaves_no_destination(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "model.pt"

    def failing_save(artifact, stream):
        # Simulate a partially written temporary file.
        stream.write(b"INCOMPLETE DATA")

        raise OSError(
            "Simulated serialization failure"
        )

    monkeypatch.setattr(
        model_artifacts.torch,
        "save",
        failing_save,
    )

    with pytest.raises(
        OSError,
        match="Simulated serialization failure",
    ):
        save_model_artifact(
            LeNet5(),
            destination,
            epoch=1,
            validation_accuracy=0.5,
        )

    assert not destination.exists()
    assert_no_temporary_files(tmp_path)


def test_publication_failure_cleans_temporary_file(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "model.pt"

    def failing_link(source, destination):
        raise OSError(
            "Simulated publication failure"
        )

    monkeypatch.setattr(
        model_artifacts.os,
        "link",
        failing_link,
    )

    with pytest.raises(
        OSError,
        match="Simulated publication failure",
    ):
        save_model_artifact(
            LeNet5(),
            destination,
            epoch=1,
            validation_accuracy=0.5,
        )

    assert not destination.exists()
    assert_no_temporary_files(tmp_path)


def test_existing_artifact_is_never_overwritten(
    tmp_path,
):
    destination = tmp_path / "model.pt"

    original_bytes = b"EXISTING ARTIFACT"

    destination.write_bytes(original_bytes)

    with pytest.raises(FileExistsError):
        save_model_artifact(
            LeNet5(),
            destination,
            epoch=1,
            validation_accuracy=0.5,
        )

    assert destination.read_bytes() == original_bytes
    assert_no_temporary_files(tmp_path)


def test_concurrent_destination_creation_is_protected(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "model.pt"

    original_link = model_artifacts.os.link

    competing_data = b"OTHER PROCESS ARTIFACT"

    def simulate_competing_writer(source, target):
        # Another process creates the destination
        # just before our atomic publication.
        Path(target).write_bytes(competing_data)

        # Must fail rather than overwrite it.
        return original_link(source, target)

    monkeypatch.setattr(
        model_artifacts.os,
        "link",
        simulate_competing_writer,
    )

    with pytest.raises(FileExistsError):
        save_model_artifact(
            LeNet5(),
            destination,
            epoch=1,
            validation_accuracy=0.5,
        )

    assert destination.read_bytes() == competing_data

    assert_no_temporary_files(tmp_path)
