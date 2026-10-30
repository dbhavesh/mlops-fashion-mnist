"""failure boundaries and post-training webhook diagnostics.

These tests never connect to MLflow, MinIO or a webhook receiver.
"""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fashion_mnist_mlops.data import DataLoaders
from fashion_mnist_mlops.webhook import (
    WebhookDeliveryError,
    WebhookHTTPError,
    WebhookTimeoutError,
)
from fashion_mnist_mlops.storage import UploadedModelArtifact
import fashion_mnist_mlops.training as runner


RUN_ID = "abc0123456789defabc0123456789def"
BUCKET = "fashion-minst-bucket"
KEY = f"model/{RUN_ID}/model.pt"


def _config(tmp_path):
    path = tmp_path / "training.toml"
    path.write_text(
        "[training]\nepochs=1\nbatch_size=4\nlearning_rate=0.001\n"
        "optimizer='adam'\nloss='cross_entropy'\nseed=42\n"
        "[validation]\nvalidation_split=0.1\n",
        encoding="utf-8",
    )
    return path


def _loaders(*args, **kwargs):
    images = torch.zeros((12, 1, 28, 28), dtype=torch.float32)
    labels = torch.arange(12) % 10
    return DataLoaders(
        train=DataLoader(TensorDataset(images[:8], labels[:8]), batch_size=4),
        validation=DataLoader(TensorDataset(images[8:], labels[8:]), batch_size=4),
        test=DataLoader(TensorDataset(images[8:], labels[8:]), batch_size=4),
    )


@pytest.fixture
def harness(monkeypatch):
    """Provide a real runner with controlled external side effects."""
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    monkeypatch.delenv("WEBHOOK_TIMEOUT_SECONDS", raising=False)
    events = []

    @contextmanager
    def tracked():
        events.append("start")
        try:
            yield RUN_ID
        except BaseException:
            events.append("FAILED")
            raise
        else:
            events.append("FINISHED")

    monkeypatch.setattr(runner, "start_tracking_run", tracked)
    monkeypatch.setattr(runner, "create_data_loaders", _loaders)
    for name in (
        "log_training_parameters",
        "log_training_loss",
        "log_validation_loss",
        "log_final_validation_accuracy",
        "log_run_metadata",
        "log_minio_artifact_reference",
    ):
        monkeypatch.setattr(
            runner, name,
            lambda *args, _name=name, **kwargs: events.append(_name),
        )

    def published(path, *, run_id):
        assert Path(path).is_file()
        assert run_id == RUN_ID
        events.append("publish_model_artifact")
        return UploadedModelArtifact(BUCKET, KEY, 123, "a" * 64)

    monkeypatch.setattr(runner, "publish_model_artifact", published)

    def unexpected_notification(*args, **kwargs):
        events.append("unexpected_webhook")
        pytest.fail("A failed upstream stage must not issue a success webhook")

    monkeypatch.setattr(
        runner, "send_training_completed_webhook", unexpected_notification
    )
    return events


@pytest.mark.parametrize(
    "failure_stage",
    [
        "train_one_epoch",
        "log_training_parameters",
        "log_training_loss",
        "log_validation_loss",
        "log_final_validation_accuracy",
        "log_run_metadata",
        "publish_model_artifact",
        "log_minio_artifact_reference",
    ],
)
def test_upstream_failure_marks_run_failed_and_suppresses_success(
    tmp_path, monkeypatch, harness, failure_stage,
):
    def broken(*args, **kwargs):
        harness.append("injected_failure")
        raise RuntimeError("controlled upstream failure")

    monkeypatch.setattr(runner, failure_stage, broken)
    artifact = tmp_path / "artifact.pt"

    with pytest.raises(RuntimeError, match="controlled upstream failure"):
        runner.run_training(
            config_path=_config(tmp_path),
            artifact_path=artifact,
            epochs=1,
            track_mlflow=True,
            upload_minio=True,
        )

    assert harness[-2:] == ["injected_failure", "FAILED"]
    assert "unexpected_webhook" not in harness
    if failure_stage in (
        "train_one_epoch",
        "log_training_parameters",
        "log_training_loss",
        "log_validation_loss",
        "log_final_validation_accuracy",
        "log_run_metadata",
    ):
        assert "publish_model_artifact" not in harness
    if failure_stage == "log_minio_artifact_reference":
        assert "publish_model_artifact" in harness
    if failure_stage in (
        "log_run_metadata", "publish_model_artifact",
        "log_minio_artifact_reference",
    ):
        assert artifact.is_file()


def _success_result():
    return runner.TrainingRunResult(
        config=SimpleNamespace(),
        artifact_path=Path("artifacts/models/mock/model.pt"),
        history=(SimpleNamespace(validation=SimpleNamespace(accuracy=0.25)),),
        mlflow_run_id=RUN_ID,
        minio_bucket=BUCKET,
        minio_object_key=KEY,
    )


@pytest.mark.parametrize(
    "exception",
    [
        WebhookDeliveryError("HTTP transport unavailable"),
        WebhookTimeoutError("Webhook delivery timed out"),
        WebhookHTTPError(503),
    ],
)
def test_delivery_failure_is_reported_after_finished_without_retry(
    monkeypatch, capsys, exception,
):
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    events = []

    @contextmanager
    def tracked():
        events.append("started")
        try:
            yield RUN_ID
        except BaseException:
            events.append("FAILED")
            raise
        else:
            events.append("FINISHED")

    monkeypatch.setattr(runner, "start_tracking_run", tracked)
    monkeypatch.setattr(runner, "_run_training_core", lambda **kwargs: _success_result())

    def failed_delivery(payload):
        events.append("delivery_attempt")
        assert payload.minio_object_key == KEY
        raise exception

    monkeypatch.setattr(
        runner, "send_training_completed_webhook", failed_delivery
    )
    with pytest.raises(type(exception)) as error:
        runner.run_training(track_mlflow=True, upload_minio=True)

    assert error.value is exception
    assert events == ["started", "FINISHED", "delivery_attempt"]
    captured = capsys.readouterr()
    assert "Training and MinIO publication succeeded" in captured.err
    assert "webhook delivery failed" in captured.err
    assert RUN_ID in captured.err
    assert "http://" not in captured.err
    assert "Training completed successfully." not in captured.out


@pytest.mark.parametrize(
    "variable,value",
    [
        ("WEBHOOK_URL", "ftp://invalid.example/webhook"),
        ("WEBHOOK_TIMEOUT_SECONDS", "0"),
    ],
)
def test_invalid_webhook_configuration_fails_before_tracking_or_training(
    monkeypatch, variable, value,
):
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    monkeypatch.delenv("WEBHOOK_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv(variable, value)

    def must_not_run(*args, **kwargs):
        pytest.fail("Invalid webhook configuration must fail before training")

    monkeypatch.setattr(runner, "start_tracking_run", must_not_run)
    monkeypatch.setattr(runner, "_run_training_core", must_not_run)

    with pytest.raises(ValueError, match=variable):
        runner.run_training(track_mlflow=True, upload_minio=True)


def test_invalid_webhook_configuration_does_not_block_local_training(
    monkeypatch,
):
    monkeypatch.setenv("WEBHOOK_URL", "ftp://invalid.example/webhook")
    monkeypatch.setattr(
        runner, "send_training_completed_webhook", lambda *args: pytest.fail(
            "Local-only run must never notify"
        ),
    )
    # Local-only runs return no remote artifact and never validate the webhook.
    existing = _success_result()
    monkeypatch.setattr(
        runner,
        "_run_training_core",
        lambda **kwargs: runner.TrainingRunResult(
            config=existing.config,
            artifact_path=existing.artifact_path,
            history=existing.history,
        ),
    )
    result = runner.run_training(track_mlflow=False, upload_minio=False)
    assert result.minio_object_key is None
