"""tracked-training/upload sequencing and failure boundaries."""

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fashion_mnist_mlops.data import DataLoaders
from fashion_mnist_mlops.storage import UploadedModelArtifact
import fashion_mnist_mlops.training as runner


RUN_ID = "abc0123456789defabc0123456789def"
BUCKET = "fashion-minst-bucket"
KEY = f"model/{RUN_ID}/model.pt"


def make_config(tmp_path):
    path = tmp_path / "training.toml"
    path.write_text(
        "[training]\nepochs=1\nbatch_size=4\nlearning_rate=0.001\n"
        "optimizer='adam'\nloss='cross_entropy'\nseed=42\n"
        "[validation]\nvalidation_split=0.1\n",
        encoding="utf-8",
    )
    return path


def make_loaders(config, root=None, download=False):
    images = torch.randn(12, 1, 28, 28)
    labels = torch.arange(12) % 10
    return DataLoaders(
        train=DataLoader(TensorDataset(images[:8], labels[:8]), batch_size=4),
        validation=DataLoader(TensorDataset(images[8:], labels[8:]), batch_size=4),
        test=DataLoader(TensorDataset(images[8:], labels[8:]), batch_size=4),
    )


def fake_mlflow(monkeypatch, events):
    @contextmanager
    def tracking():
        events.append("start")
        try:
            yield RUN_ID
        except BaseException:
            events.append("FAILED")
            raise
        else:
            events.append("FINISHED")

    monkeypatch.setattr(runner, "create_data_loaders", make_loaders)
    monkeypatch.setattr(runner, "start_tracking_run", tracking)
    monkeypatch.setattr(runner, "log_training_parameters", lambda *a: events.append("parameters"))
    monkeypatch.setattr(runner, "log_training_loss", lambda *a, **kw: events.append("train_loss"))
    monkeypatch.setattr(runner, "log_validation_loss", lambda *a, **kw: events.append("val_loss"))
    monkeypatch.setattr(runner, "log_final_validation_accuracy", lambda *a, **kw: events.append("val_accuracy"))
    monkeypatch.setattr(runner, "log_run_metadata", lambda *a, **kw: events.append("provenance"))
    monkeypatch.setattr(runner, "log_minio_artifact_reference", lambda *a, **kw: events.append("minio-reference"))


def test_upload_follows_all_required_tracking_and_artifact(tmp_path, monkeypatch):

    # Isolate this test from live webhook configuration.
    # Webhook delivery has separate tests.
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    monkeypatch.delenv("WEBHOOK_TIMEOUT_SECONDS", raising=False)

    events = []
    fake_mlflow(monkeypatch, events)

    def fake_publish(path, *, run_id):
        assert Path(path).is_file()
        assert run_id == RUN_ID
        events.append("upload")
        return UploadedModelArtifact(BUCKET, KEY, 123, "a" * 64)

    monkeypatch.setattr(runner, "publish_model_artifact", fake_publish)
    result = runner.run_training(
        config_path=make_config(tmp_path), artifact_path=tmp_path / "model.pt",
        epochs=1, track_mlflow=True, upload_minio=True,
    )
    assert events == [
        "start", "parameters", "train_loss", "val_loss",
        "val_accuracy", "provenance", "upload", "minio-reference", "FINISHED",
    ]
    assert result.minio_bucket == BUCKET
    assert result.minio_object_key == KEY
    assert result.artifact_path.exists()


def test_mlflow_metadata_failure_blocks_upload(tmp_path, monkeypatch):
    events = []
    fake_mlflow(monkeypatch, events)

    def fail_log(*args, **kwargs):
        events.append("provenance-failed")
        raise ConnectionError("MLflow artifact upload failed")

    monkeypatch.setattr(runner, "log_run_metadata", fail_log)
    monkeypatch.setattr(runner, "publish_model_artifact", lambda *a, **kw: pytest.fail("upload must not start"))
    path = tmp_path / "model.pt"
    with pytest.raises(ConnectionError, match="MLflow artifact upload failed"):
        runner.run_training(
            config_path=make_config(tmp_path), artifact_path=path,
            epochs=1, track_mlflow=True, upload_minio=True,
        )
    assert path.is_file()
    assert events[-2:] == ["provenance-failed", "FAILED"]


def test_minio_failure_marks_run_failed_without_removing_artifact(tmp_path, monkeypatch):
    events = []
    fake_mlflow(monkeypatch, events)

    def fail_upload(*args, **kwargs):
        events.append("upload-failed")
        raise ConnectionError("MinIO unavailable")

    monkeypatch.setattr(runner, "publish_model_artifact", fail_upload)
    path = tmp_path / "model.pt"
    with pytest.raises(ConnectionError, match="MinIO unavailable"):
        runner.run_training(
            config_path=make_config(tmp_path), artifact_path=path,
            epochs=1, track_mlflow=True, upload_minio=True,
        )
    assert path.is_file()
    assert events[-2:] == ["upload-failed", "FAILED"]


def test_upload_requires_tracking_and_fails_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", lambda *a, **kw: pytest.fail("training started"))
    with pytest.raises(ValueError, match="requires --track-mlflow"):
        runner.run_training(config_path=make_config(tmp_path), upload_minio=True)


def test_invalid_upload_flag_rejected_before_training(tmp_path):
    with pytest.raises(TypeError, match="upload_minio must be a boolean"):
        runner.run_training(config_path=make_config(tmp_path), upload_minio="yes")


def test_minio_reference_failure_marks_run_failed_and_preserves_uploaded_model(tmp_path, monkeypatch):
    events = []
    fake_mlflow(monkeypatch, events)
    path = tmp_path / "model.pt"

    def publish(artifact_path, *, run_id):
        assert Path(artifact_path).is_file()
        assert run_id == RUN_ID
        events.append("upload")
        return UploadedModelArtifact(BUCKET, KEY, 123, "b" * 64)

    def fail_reference(run_id, uploaded):
        assert run_id == RUN_ID
        assert uploaded.object_key == KEY
        events.append("reference-failed")
        raise ConnectionError("MLflow tag write unavailable")

    monkeypatch.setattr(runner, "publish_model_artifact", publish)
    monkeypatch.setattr(runner, "log_minio_artifact_reference", fail_reference)
    with pytest.raises(ConnectionError, match="MLflow tag write unavailable"):
        runner.run_training(
            config_path=make_config(tmp_path), artifact_path=path,
            epochs=1, track_mlflow=True, upload_minio=True,
        )
    assert path.is_file()
    assert events[-3:] == ["upload", "reference-failed", "FAILED"]
    assert "FINISHED" not in events


def test_upload_failure_does_not_log_minio_reference(tmp_path, monkeypatch):
    events = []
    fake_mlflow(monkeypatch, events)
    monkeypatch.setattr(runner, "log_minio_artifact_reference",
                        lambda *a, **kw: pytest.fail("Must not log before verified upload"))

    def fail_upload(*args, **kwargs):
        raise ConnectionError("S3 upload failed")

    monkeypatch.setattr(runner, "publish_model_artifact", fail_upload)
    with pytest.raises(ConnectionError, match="S3 upload failed"):
        runner.run_training(
            config_path=make_config(tmp_path), artifact_path=tmp_path / "model.pt",
            epochs=1, track_mlflow=True, upload_minio=True,
        )
    assert events[-1] == "FAILED"


def test_tracked_training_without_upload_does_not_log_minio_reference(tmp_path, monkeypatch):
    events = []
    fake_mlflow(monkeypatch, events)
    monkeypatch.setattr(runner, "log_minio_artifact_reference",
                        lambda *a, **kw: pytest.fail("No MinIO reference without upload"))
    runner.run_training(
        config_path=make_config(tmp_path), artifact_path=tmp_path / "model.pt",
        epochs=1, track_mlflow=True, upload_minio=False,
    )
    assert "FINISHED" == events[-1]
    assert "minio-reference" not in events
