"""webhook placement in the completed training lifecycle."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import fashion_mnist_mlops.training as runner


RUN_ID = "97461abfc4804cb0a2a1473c71737739"
BUCKET = "fashion-minst-bucket"
KEY = f"model/{RUN_ID}/model.pt"


def _result(*, stored: bool) -> runner.TrainingRunResult:
    record = SimpleNamespace(
        validation=SimpleNamespace(accuracy=0.625),
    )
    return runner.TrainingRunResult(
        config=SimpleNamespace(),
        artifact_path=Path("artifacts/models/test/model.pt"),
        history=(record,),
        mlflow_run_id=RUN_ID,
        minio_bucket=BUCKET if stored else None,
        minio_object_key=KEY if stored else None,
    )


def _tracking_context(events):
    @contextmanager
    def fake_start_tracking_run():
        events.append("mlflow_enter")
        try:
            yield RUN_ID
        except BaseException:
            events.append("mlflow_failed")
            raise
        else:
            events.append("mlflow_finished")

    return fake_start_tracking_run


def test_webhook_runs_only_after_mlflow_context_finishes(monkeypatch):
    events = []

    monkeypatch.setattr(
        runner,
        "start_tracking_run",
        _tracking_context(events),
    )

    def fake_core(**kwargs):
        events.append("core")
        assert kwargs["mlflow_run_id"] == RUN_ID
        assert kwargs["upload_minio"] is True
        return _result(stored=True)

    def fake_send(payload):
        events.append("webhook")
        assert payload.mlflow_run_id == RUN_ID
        assert payload.minio_bucket == BUCKET
        assert payload.minio_object_key == KEY
        assert payload.validation_accuracy == 0.625
        return SimpleNamespace(status_code=204)

    monkeypatch.setattr(runner, "_run_training_core", fake_core)
    monkeypatch.setattr(
        runner,
        "send_training_completed_webhook",
        fake_send,
    )

    result = runner.run_training(
        track_mlflow=True,
        upload_minio=True,
    )

    assert result.minio_object_key == KEY
    assert events == [
        "mlflow_enter",
        "core",
        "mlflow_finished",
        "webhook",
    ]


def test_no_webhook_without_successful_minio_artifact(monkeypatch):
    events = []

    monkeypatch.setattr(
        runner,
        "start_tracking_run",
        _tracking_context(events),
    )
    monkeypatch.setattr(
        runner,
        "_run_training_core",
        lambda **kwargs: _result(stored=False),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Webhook must not run without a MinIO artifact.")

    monkeypatch.setattr(
        runner,
        "send_training_completed_webhook",
        fail_if_called,
    )

    result = runner.run_training(
        track_mlflow=True,
        upload_minio=False,
    )

    assert result.minio_object_key is None
    assert events == ["mlflow_enter", "mlflow_finished"]


def test_upstream_failure_suppresses_webhook(monkeypatch):
    events = []

    monkeypatch.setattr(
        runner,
        "start_tracking_run",
        _tracking_context(events),
    )

    def fail_core(**kwargs):
        events.append("core")
        raise RuntimeError("simulated MinIO or training failure")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Success webhook must be suppressed.")

    monkeypatch.setattr(runner, "_run_training_core", fail_core)
    monkeypatch.setattr(
        runner,
        "send_training_completed_webhook",
        fail_if_called,
    )

    with pytest.raises(RuntimeError, match="simulated"):
        runner.run_training(
            track_mlflow=True,
            upload_minio=True,
        )

    assert events == ["mlflow_enter", "core", "mlflow_failed"]


def test_webhook_failure_occurs_after_mlflow_is_finished(monkeypatch):
    events = []

    monkeypatch.setattr(
        runner,
        "start_tracking_run",
        _tracking_context(events),
    )
    monkeypatch.setattr(
        runner,
        "_run_training_core",
        lambda **kwargs: _result(stored=True),
    )

    def fail_webhook(payload):
        events.append("webhook")
        raise RuntimeError("simulated webhook outage")

    monkeypatch.setattr(
        runner,
        "send_training_completed_webhook",
        fail_webhook,
    )

    with pytest.raises(RuntimeError, match="webhook outage"):
        runner.run_training(
            track_mlflow=True,
            upload_minio=True,
        )

    assert events == [
        "mlflow_enter",
        "mlflow_finished",
        "webhook",
    ]


@pytest.mark.parametrize(
    "run_id,bucket",
    [
        (None, BUCKET),
        (RUN_ID, None),
    ],
)
def test_incomplete_stored_result_is_rejected_before_webhook(
    monkeypatch,
    run_id,
    bucket,
):
    events = []

    monkeypatch.setattr(
        runner,
        "start_tracking_run",
        _tracking_context(events),
    )

    broken = _result(stored=True)
    broken = runner.TrainingRunResult(
        config=broken.config,
        artifact_path=broken.artifact_path,
        history=broken.history,
        mlflow_run_id=run_id,
        minio_bucket=bucket,
        minio_object_key=KEY,
    )
    monkeypatch.setattr(
        runner,
        "_run_training_core",
        lambda **kwargs: broken,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Malformed lifecycle result must not be sent.")

    monkeypatch.setattr(
        runner,
        "send_training_completed_webhook",
        fail_if_called,
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        runner.run_training(
            track_mlflow=True,
            upload_minio=True,
        )

    assert events == ["mlflow_enter", "mlflow_finished"]
