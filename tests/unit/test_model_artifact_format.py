"""portable version-2 artifact inspection before object storage."""

import hashlib
from dataclasses import FrozenInstanceError

import pytest
import torch

from fashion_mnist_mlops.model import LeNet5
from fashion_mnist_mlops.model_artifacts import load_model_artifact, save_model_artifact
from fashion_mnist_mlops.artifact_format import inspect_model_artifact


@pytest.fixture
def artifact(tmp_path):
    # Mirror the production v2 provenance schema without importing the
    # training package; this test must not require a live MLflow server.
    provenance = {
        "training": {
            "epochs": 4,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "cross_entropy",
            "seed": 42,
        },
        "validation": {"validation_split": 0.1},
        "dataset": {
            "name": "FashionMNIST",
            "location": {"scope": "project", "path": "data"},
            "train_samples_used": 8,
            "validation_samples_used": 8,
            "test_samples_available": 8,
            "max_train_samples": 8,
            "max_validation_samples": 8,
            "download_allowed": False,
        },
        "source_config": {
            "location": {"scope": "project", "path": "configs/train.toml"},
            "sha256": "a" * 64,
            "effective_sha256": "b" * 64,
        },
        "preprocessing": {
            "version": 1,
            "input_shape": [1, 28, 28],
            "normalization_mean": [0.2860406],
            "normalization_std": [0.3530242],
        },
        "software": {"python": "3.12", "torch": "2.x", "torchvision": "0.x"},
    }
    model = LeNet5().eval()
    path = tmp_path / "model.pt"
    save_model_artifact(
        model,
        path,
        epoch=4,
        validation_accuracy=0.625,
        mlflow_run_id="1234567890abcdef1234567890abcdef",
        provenance=provenance,
    )
    return path, model


def _corrupt(artifact, tmp_path, name, change):
    payload = torch.load(artifact[0], map_location="cpu", weights_only=True)
    change(payload)
    destination = tmp_path / name
    torch.save(payload, destination)
    return destination


def test_inspection_exposes_portable_reload_contract(artifact):
    path, _ = artifact
    info = inspect_model_artifact(path, expected_run_id="1234567890abcdef1234567890abcdef")
    assert info.path == path.resolve()
    assert info.artifact_format_version == 2
    assert info.architecture == "LeNet5"
    assert info.input_shape == (1, 28, 28)
    assert info.num_classes == 10
    assert len(info.class_names) == 10
    assert info.normalization_mean == (0.2860406,)
    assert info.normalization_std == (0.3530242,)
    assert info.preprocessing_version == 1
    assert info.epoch == 4
    assert info.validation_accuracy == 0.625
    assert info.mlflow_run_id == "1234567890abcdef1234567890abcdef"
    assert len(info.effective_config_sha256) == 64


def test_artifact_digest_matches_bytes_and_is_read_only(artifact):
    path, _ = artifact
    original = path.read_bytes()
    info = inspect_model_artifact(path)
    assert info.sha256 == hashlib.sha256(original).hexdigest()
    assert info.size_bytes == len(original)
    assert path.read_bytes() == original


def test_descriptor_is_immutable(artifact):
    info = inspect_model_artifact(artifact[0])
    with pytest.raises(FrozenInstanceError):
        info.sha256 = "other"


def test_inspected_artifact_restores_identical_cpu_logits(artifact):
    path, original = artifact
    inspect_model_artifact(path)
    restored, _ = load_model_artifact(path)
    images = torch.randn(3, 1, 28, 28)
    with torch.inference_mode():
        assert torch.equal(original(images), restored(images))
    assert not restored.training
    assert all(t.device.type == "cpu" for t in restored.state_dict().values())


def test_tracked_run_identity_mismatch_is_rejected(artifact):
    with pytest.raises(ValueError, match="run ID"):
        inspect_model_artifact(artifact[0], expected_run_id="other-run")


@pytest.mark.parametrize("run_id", ["", " ", 123, False])
def test_invalid_expected_run_id_is_rejected(artifact, run_id):
    with pytest.raises(ValueError, match="expected_run_id"):
        inspect_model_artifact(artifact[0], expected_run_id=run_id)


def test_missing_artifact_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        inspect_model_artifact(tmp_path / "missing.pt")


def test_legacy_v1_is_loadable_but_not_storage_ready(tmp_path):
    path = tmp_path / "legacy.pt"
    save_model_artifact(LeNet5(), path, epoch=1, validation_accuracy=0.2)
    assert load_model_artifact(path)[1]["checkpoint_format_version"] == 1
    with pytest.raises(ValueError, match="format version 2"):
        inspect_model_artifact(path)


def test_untracked_v2_artifact_remains_eligible(artifact, tmp_path):
    path = _corrupt(artifact, tmp_path, "untracked.pt", lambda d: d.update(mlflow_run_id=None))
    assert inspect_model_artifact(path).mlflow_run_id is None
    with pytest.raises(ValueError, match="run ID"):
        inspect_model_artifact(path, expected_run_id="tracked-run")


@pytest.mark.parametrize("key", ["architecture", "input_shape", "class_names", "normalization_std"])
def test_incompatible_model_identity_is_rejected(artifact, tmp_path, key):
    path = _corrupt(artifact, tmp_path, f"bad-{key}.pt", lambda d: d.update({key: "invalid"}))
    with pytest.raises(ValueError, match="Incompatible artifact metadata"):
        inspect_model_artifact(path)


@pytest.mark.parametrize("field,value", [
    ("version", 2),
    ("input_shape", [3, 28, 28]),
    ("normalization_mean", [0.0]),
    ("normalization_std", [1.0]),
])
def test_incompatible_preprocessing_provenance_is_rejected(artifact, tmp_path, field, value):
    def mutate(d):
        d["provenance"]["preprocessing"][field] = value
    path = _corrupt(artifact, tmp_path, f"bad-pre-{field}.pt", mutate)
    with pytest.raises(ValueError, match="preprocessing"):
        inspect_model_artifact(path)


def test_effective_epoch_mismatch_is_rejected(artifact, tmp_path):
    path = _corrupt(
        artifact, tmp_path, "bad-epochs.pt",
        lambda d: d["provenance"]["training"].update(epochs=5),
    )
    with pytest.raises(ValueError, match="epoch"):
        inspect_model_artifact(path)


@pytest.mark.parametrize("digest", [None, "bad", "A" * 64, 123])
def test_effective_config_digest_is_required(artifact, tmp_path, digest):
    path = _corrupt(
        artifact, tmp_path, f"bad-digest-{str(digest)[:5]}.pt",
        lambda d: d["provenance"]["source_config"].update(effective_sha256=digest),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_model_artifact(path)


def test_non_finite_reloaded_weights_are_rejected(artifact, tmp_path):
    def mutate(d):
        d["model_state_dict"]["fc3.bias"][0] = float("nan")
    path = _corrupt(artifact, tmp_path, "bad-weights.pt", mutate)
    with pytest.raises(ValueError, match="non-finite"):
        inspect_model_artifact(path)


def test_mismatched_state_dict_is_rejected(artifact, tmp_path):
    def mutate(d):
        d["model_state_dict"]["fc3.weight"] = torch.zeros(9, 84)
    path = _corrupt(artifact, tmp_path, "bad-state.pt", mutate)
    with pytest.raises(RuntimeError, match="size mismatch"):
        inspect_model_artifact(path)


def test_artifact_contains_only_cpu_tensors_and_metadata(artifact):
    payload = torch.load(artifact[0], map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    assert payload["checkpoint_format_version"] == 2
    assert payload["architecture"] == "LeNet5"
    assert isinstance(payload["model_state_dict"], dict)
    assert all(
        isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
        for tensor in payload["model_state_dict"].values()
    )


def test_byte_for_byte_copy_is_relocatable_and_preserves_predictions(artifact, tmp_path):
    original_path, original_model = artifact
    copied = tmp_path / "downloaded" / "model.pt"
    copied.parent.mkdir()
    copied.write_bytes(original_path.read_bytes())
    original_info = inspect_model_artifact(original_path)
    copied_info = inspect_model_artifact(copied)
    assert copied_info.sha256 == original_info.sha256
    assert copied_info.path == copied.resolve()
    recovered, _ = load_model_artifact(copied)
    batch = torch.randn(2, 1, 28, 28)
    with torch.inference_mode():
        assert torch.equal(original_model(batch), recovered(batch))


def test_inspection_cli_prints_metadata_without_uploading(artifact, monkeypatch, capsys):
    import sys
    from scripts.inspect_model_artifact import main

    monkeypatch.setattr(sys, "argv", ["inspect_model_artifact.py", str(artifact[0])])
    assert main() == 0
    output = capsys.readouterr().out
    assert "PASS: Version-2 model artifact is ready for storage" in output
    assert "LeNet5" in output
    assert "SHA-256:" in output
    assert "No object was uploaded or modified." in output


def test_inspection_cli_returns_nonzero_for_run_mismatch(artifact, monkeypatch, capsys):
    import sys
    from scripts.inspect_model_artifact import main

    monkeypatch.setattr(sys, "argv", [
        "inspect_model_artifact.py", str(artifact[0]), "--run-id", "another-run",
    ])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 1
    assert "MLflow run ID" in capsys.readouterr().err


def test_inspection_cli_rejects_truncated_file_with_nonzero_exit(artifact, tmp_path, monkeypatch, capsys):
    import sys
    from scripts.inspect_model_artifact import main

    damaged = tmp_path / "truncated.pt"
    source = artifact[0].read_bytes()
    damaged.write_bytes(source[: len(source) // 2])
    monkeypatch.setattr(sys, "argv", ["inspect_model_artifact.py", str(damaged)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 1
    assert "FAIL:" in capsys.readouterr().err
