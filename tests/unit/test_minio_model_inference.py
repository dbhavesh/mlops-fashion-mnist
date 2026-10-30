"""restore exactly the model version retrieved from verified storage."""

import hashlib
import io
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from PIL import Image

from fashion_mnist_mlops.labels import (
    FASHION_MNIST_CLASSES,
)
from fashion_mnist_mlops.preprocessing import (
    preprocess_image_for_inference,
)
from fashion_mnist_mlops.inference import (
    predict_preprocessed_image,
    restore_downloaded_model,
    retrieve_inference_model,
)
from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.model_artifacts import (
    load_model_artifact,
    save_model_artifact,
)
from fashion_mnist_mlops.storage import (
    DownloadedModelArtifact,
    download_model_artifact,
)

RUN_ID = "1234567890abcdef1234567890abcdef"
KEY = f"model/{RUN_ID}/model.pt"
BUCKET = "fashion-minst-bucket"


@pytest.fixture
def trained_artifact(tmp_path):
    provenance = {
        "training": {
            "epochs": 1, "batch_size": 64, "learning_rate": 0.001,
            "optimizer": "adam", "loss": "cross_entropy", "seed": 42,
        },
        "validation": {"validation_split": 0.1},
        "dataset": {
            "name": "FashionMNIST",
            "location": {"scope": "project", "path": "data"},
            "train_samples_used": 8, "validation_samples_used": 8,
            "test_samples_available": 8, "max_train_samples": 8,
            "max_validation_samples": 8, "download_allowed": False,
        },
        "source_config": {
            "location": {"scope": "project", "path": "configs/train.toml"},
            "sha256": "a" * 64, "effective_sha256": "b" * 64,
        },
        "preprocessing": {
            "version": 1, "input_shape": [1, 28, 28],
            "normalization_mean": [0.2860406],
            "normalization_std": [0.3530242],
        },
        "software": {"python": "3.12", "torch": "2.x", "torchvision": "0.x"},
    }
    model = LeNet5().eval()
    source = tmp_path / "original" / "model.pt"
    save_model_artifact(
        model, source, epoch=1, validation_accuracy=0.1875,
        mlflow_run_id=RUN_ID, provenance=provenance,
    )
    return source, model


class FakeResponse:
    def __init__(self, data):
        self.stream = io.BytesIO(data)
        self.closed = False
        self.released = False

    def read(self, size):
        return self.stream.read(size)

    def close(self):
        self.closed = True

    def release_conn(self):
        self.released = True


class FakeMinio:
    def __init__(self, data):
        self.data = data
        self.response = None

    def bucket_exists(self, bucket):
        assert bucket == BUCKET
        return True

    def get_object(self, bucket, key):
        assert (bucket, key) == (BUCKET, KEY)
        self.response = FakeResponse(self.data)
        return self.response


@pytest.fixture
def downloaded(trained_artifact, tmp_path):
    original_path, model = trained_artifact
    data = original_path.read_bytes()
    client = FakeMinio(data)
    result = download_model_artifact(
        client, bucket_name=BUCKET, object_key=KEY,
        destination=tmp_path / "download" / "model.pt",
        expected_sha256=hashlib.sha256(data).hexdigest(),
        expected_size_bytes=len(data),
    )
    assert client.response.closed and client.response.released
    return result, model


def test_real_storage_copy_restores_the_original_logits(downloaded):
    artifact, original_model = downloaded
    restored = restore_downloaded_model(artifact)
    assert restored.run_id == RUN_ID
    assert restored.artifact_path == artifact.path
    assert restored.sha256 == artifact.sha256
    assert restored.class_names == tuple(FASHION_MNIST_CLASSES)
    assert isinstance(restored.model, LeNet5)
    assert not restored.model.training
    assert all(p.device.type == "cpu" for p in restored.model.parameters())
    image = preprocess_image_for_inference(Image.new("L", (28, 28), color=128))
    with torch.inference_mode():
        assert torch.equal(original_model(image), restored.model(image))


def test_inference_label_matches_argmax_and_does_not_change_parameters(downloaded):
    restored = restore_downloaded_model(downloaded[0])
    image = preprocess_image_for_inference(Image.new("L", (28, 28), color=128))
    before = {key: tensor.clone() for key, tensor in restored.model.state_dict().items()}
    with torch.inference_mode():
        idx = int(restored.model(image).argmax(dim=1).item())
    prediction = predict_preprocessed_image(restored, image)
    assert prediction.class_index == idx
    assert prediction.class_name == FASHION_MNIST_CLASSES[idx]
    assert not restored.model.training
    assert all(torch.equal(before[k], v) for k, v in restored.model.state_dict().items())


def test_repeated_predictions_are_deterministic(downloaded):
    restored = restore_downloaded_model(downloaded[0])
    image = preprocess_image_for_inference(Image.new("L", (28, 28), color=128))
    assert predict_preprocessed_image(restored, image) == predict_preprocessed_image(restored, image)


def test_downloaded_artifact_is_read_only(downloaded):
    artifact, _ = downloaded
    before = artifact.path.read_bytes()
    restored = restore_downloaded_model(artifact)
    predict_preprocessed_image(restored, torch.zeros((1, 1, 28, 28)))
    assert artifact.path.read_bytes() == before


def test_restore_descriptor_is_immutable(downloaded):
    restored = restore_downloaded_model(downloaded[0])
    with pytest.raises(FrozenInstanceError):
        restored.run_id = "other"


def test_bad_artifact_type_rejected_before_load(downloaded):
    with pytest.raises(TypeError, match="DownloadedModelArtifact"):
        restore_downloaded_model(downloaded[0].path)


@pytest.mark.parametrize("bucket", ["other-bucket", "fashion-mnist-bucket", ""])
def test_rejects_other_bucket(downloaded, bucket):
    artifact, _ = downloaded
    modified = DownloadedModelArtifact(
        bucket=bucket, object_key=artifact.object_key, path=artifact.path,
        size_bytes=artifact.size_bytes, sha256=artifact.sha256,
    )
    with pytest.raises(ValueError, match="bucket"):
        restore_downloaded_model(modified)


@pytest.mark.parametrize("key", ["model/../model.pt", "model/run/wrong.pt", "other/run/model.pt"])
def test_rejects_noncanonical_object_key(downloaded, key):
    artifact, _ = downloaded
    modified = DownloadedModelArtifact(
        bucket=artifact.bucket, object_key=key, path=artifact.path,
        size_bytes=artifact.size_bytes, sha256=artifact.sha256,
    )
    with pytest.raises(ValueError, match="Object key"):
        restore_downloaded_model(modified)


def test_artifact_of_another_run_is_rejected(downloaded):
    artifact, _ = downloaded
    modified = DownloadedModelArtifact(
        bucket=artifact.bucket, object_key="model/another-run/model.pt",
        path=artifact.path, size_bytes=artifact.size_bytes, sha256=artifact.sha256,
    )
    with pytest.raises(ValueError, match="run ID"):
        restore_downloaded_model(modified)


def test_local_file_tampering_after_download_is_detected(downloaded):
    artifact, _ = downloaded
    with artifact.path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="fingerprint"):
        restore_downloaded_model(artifact)


@pytest.mark.parametrize("field", ["size_bytes", "sha256"])
def test_tampered_reference_metadata_is_detected(downloaded, field):
    artifact, _ = downloaded
    values = dict(bucket=artifact.bucket, object_key=artifact.object_key,
                  path=artifact.path, size_bytes=artifact.size_bytes, sha256=artifact.sha256)
    values[field] = 123 if field == "size_bytes" else "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        restore_downloaded_model(DownloadedModelArtifact(**values))


def test_deleted_download_is_rejected(downloaded):
    artifact, _ = downloaded
    artifact.path.unlink()
    with pytest.raises(FileNotFoundError):
        restore_downloaded_model(artifact)


def test_legacy_artifact_refused_even_if_fingerprint_matches(tmp_path):
    path = tmp_path / "legacy.pt"
    save_model_artifact(LeNet5(), path, epoch=1, validation_accuracy=0.1,
                          mlflow_run_id=RUN_ID)
    artifact = DownloadedModelArtifact(
        bucket=BUCKET, object_key=KEY, path=path,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="format version 2"):
        restore_downloaded_model(artifact)


def test_incompatible_artifact_refused(downloaded):
    artifact, _ = downloaded
    payload = torch.load(artifact.path, weights_only=True, map_location="cpu")
    payload["architecture"] = "OtherCNN"
    torch.save(payload, artifact.path)
    altered = DownloadedModelArtifact(
        bucket=BUCKET, object_key=KEY, path=artifact.path,
        size_bytes=artifact.path.stat().st_size,
        sha256=hashlib.sha256(artifact.path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="Incompatible artifact metadata"):
        restore_downloaded_model(altered)


@pytest.mark.parametrize("tensor,error", [
    (torch.zeros((1, 28, 28)), "shape"),
    (torch.zeros((2, 1, 28, 28)), "shape"),
    (torch.zeros((1, 3, 28, 28)), "shape"),
    (torch.zeros((1, 1, 28, 28), dtype=torch.float64), "float32"),
    (torch.full((1, 1, 28, 28), float("nan")), "non-finite"),
    (torch.full((1, 1, 28, 28), float("inf")), "non-finite"),
])
def test_invalid_image_tensor_fails(downloaded, tensor, error):
    restored = restore_downloaded_model(downloaded[0])
    with pytest.raises(ValueError, match=error):
        predict_preprocessed_image(restored, tensor)


def test_non_tensor_image_rejected(downloaded):
    with pytest.raises(TypeError, match="PyTorch"):
        predict_preprocessed_image(restore_downloaded_model(downloaded[0]), "wrong")


def test_invalid_restored_model_rejected():
    with pytest.raises(TypeError, match="RestoredInferenceModel"):
        predict_preprocessed_image(None, torch.zeros((1, 1, 28, 28)))


def test_nonfinite_logits_rejected(downloaded, monkeypatch):
    restored = restore_downloaded_model(downloaded[0])
    monkeypatch.setattr(restored.model, "forward", lambda image: torch.full((1, 10), float("nan")))
    with pytest.raises(ValueError, match="logits"):
        predict_preprocessed_image(restored, torch.zeros((1, 1, 28, 28)))


def test_retrieve_inference_model_chains_verified_download(downloaded, monkeypatch, tmp_path):
    from fashion_mnist_mlops import inference as restoration
    artifact, _ = downloaded
    called = []

    def retrieve(destination, *, object_key=None):
        called.append((Path(destination), object_key))
        return artifact

    monkeypatch.setattr(restoration, "retrieve_model_artifact", retrieve)
    destination = tmp_path / "next" / "model.pt"
    restored = retrieve_inference_model(destination, object_key=KEY)
    assert called == [(destination, KEY)]
    assert restored.run_id == RUN_ID


def test_remote_retrieval_failure_prevents_inference(monkeypatch, tmp_path):
    from fashion_mnist_mlops import inference as restoration
    monkeypatch.setattr(restoration, "retrieve_model_artifact", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("missing MinIO object")))
    with pytest.raises(FileNotFoundError, match="missing"):
        retrieve_inference_model(tmp_path / "missing.pt", object_key=KEY)


def test_cli_smoke_restores_predicts_and_compares_original(downloaded, trained_artifact, monkeypatch, capsys, tmp_path):
    from scripts import verify_model_inference as cli
    artifact, _ = downloaded
    monkeypatch.setattr(cli, "retrieve_inference_model", lambda *a, **kw: restore_downloaded_model(artifact))
    monkeypatch.setattr(sys, "argv", [
        "verify_model_inference.py", "--destination", str(tmp_path / "another.pt"),
        "--object-key", KEY, "--compare-local-artifact", str(trained_artifact[0]),
    ])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "PASS: Downloaded and original artifact logits match exactly" in out
    assert "PASS: Verified MinIO model restored and used for inference" in out
    assert "No training" in out


def test_cli_rejects_mismatched_reference(downloaded, tmp_path, monkeypatch, capsys):
    from scripts import verify_model_inference as cli
    artifact, _ = downloaded
    different = tmp_path / "other.pt"
    save_model_artifact(LeNet5(), different, epoch=1, validation_accuracy=0.5,
                          mlflow_run_id="other-run")
    monkeypatch.setattr(cli, "retrieve_inference_model", lambda *a, **kw: restore_downloaded_model(artifact))
    monkeypatch.setattr(sys, "argv", [
        "verify_model_inference.py", "--destination", str(tmp_path / "another.pt"),
        "--compare-local-artifact", str(different),
    ])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "different MLflow run" in capsys.readouterr().err


def test_prediction_refuses_training_mode(downloaded):
    restored = restore_downloaded_model(downloaded[0])
    restored.model.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        predict_preprocessed_image(restored, torch.zeros((1, 1, 28, 28)))
