"""MLflow run creation without a live tracking server."""

from types import SimpleNamespace

import pytest

import fashion_mnist_mlops.tracking as tracking_runs



def test_tracking_context_finishes_created_run(monkeypatch):
    statuses = []

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://127.0.0.1:5000"

        def get_experiment_by_name(self, name):
            assert name == "fashion-mnist-classifier"
            return SimpleNamespace(experiment_id="1", lifecycle_stage="active")

        def create_run(self, experiment_id):
            assert experiment_id == "1"
            return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

        def set_terminated(self, run_id, status):
            statuses.append((run_id, status))

    monkeypatch.setattr(tracking_runs, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        tracking_runs,
        "load_infrastructure_config",
        lambda: SimpleNamespace(mlflow=SimpleNamespace(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="fashion-mnist-classifier",
        )),
    )
    with tracking_runs.start_tracking_run() as run_id:
        assert run_id == "run-123"
    assert statuses == [("run-123", "FINISHED")]


def test_tracking_context_marks_failed_on_error(monkeypatch):
    statuses = []

    class FakeClient:
        def __init__(self, tracking_uri):
            pass

        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="1", lifecycle_stage="active")

        def create_run(self, experiment_id):
            return SimpleNamespace(info=SimpleNamespace(run_id="run-err"))

        def set_terminated(self, run_id, status):
            statuses.append((run_id, status))

    monkeypatch.setattr(tracking_runs, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        tracking_runs,
        "load_infrastructure_config",
        lambda: SimpleNamespace(mlflow=SimpleNamespace(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="fashion-mnist-classifier",
        )),
    )
    with pytest.raises(ValueError, match="deliberate failure"):
        with tracking_runs.start_tracking_run():
            raise ValueError("deliberate failure")
    assert statuses == [("run-err", "FAILED")]


def test_missing_named_experiment_is_error(monkeypatch):
    class FakeClient:
        def __init__(self, tracking_uri):
            pass

        def get_experiment_by_name(self, name):
            return None

    monkeypatch.setattr(tracking_runs, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        tracking_runs,
        "load_infrastructure_config",
        lambda: SimpleNamespace(mlflow=SimpleNamespace(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="fashion-mnist-classifier",
        )),
    )
    with pytest.raises(RuntimeError, match="does not exist"):
        with tracking_runs.start_tracking_run():
            pass


# lifecycle status update failures are never silently ignored.


def _configure_fake_tracking(monkeypatch, *, fail_statuses=(), run_id="run-59"):
    calls = []

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://127.0.0.1:5000"

        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="1", lifecycle_stage="active")

        def create_run(self, experiment_id):
            return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

        def set_terminated(self, supplied_run_id, status):
            assert supplied_run_id == run_id
            calls.append(status)
            if status in fail_statuses:
                raise ConnectionError(f"tracking server unavailable: {status}")

    monkeypatch.setattr(tracking_runs, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        tracking_runs,
        "load_infrastructure_config",
        lambda: SimpleNamespace(
            mlflow=SimpleNamespace(
                tracking_uri="http://127.0.0.1:5000",
                experiment_name="fashion-mnist-classifier",
            )
        ),
    )
    return calls


def test_finishing_failure_fails_closed_and_attempts_failed_status(monkeypatch):
    calls = _configure_fake_tracking(monkeypatch, fail_statuses={"FINISHED"})
    with pytest.raises(RuntimeError, match="Could not confirm MLflow run") as info:
        with tracking_runs.start_tracking_run() as run_id:
            assert run_id == "run-59"
    assert calls == ["FINISHED", "FAILED"]
    assert isinstance(info.value.__cause__, ConnectionError)
    assert "run-59" in str(info.value)


def test_finishing_failure_and_failed_cleanup_both_reported(monkeypatch, capsys):
    calls = _configure_fake_tracking(
        monkeypatch, fail_statuses={"FINISHED", "FAILED"}
    )
    with pytest.raises(RuntimeError, match="Could not confirm") as info:
        with tracking_runs.start_tracking_run():
            pass
    assert calls == ["FINISHED", "FAILED"]
    assert isinstance(info.value.__cause__, ConnectionError)
    assert any("could not set status FAILED" in n for n in info.value.__notes__)
    assert "remote status is unconfirmed" in capsys.readouterr().err


def test_primary_training_failure_not_masked_when_failed_status_unavailable(
    monkeypatch, capsys
):
    calls = _configure_fake_tracking(monkeypatch, fail_statuses={"FAILED"})
    with pytest.raises(ValueError, match="original training failure") as info:
        with tracking_runs.start_tracking_run():
            raise ValueError("original training failure")
    assert calls == ["FAILED"]
    assert any("could not set status FAILED" in n for n in info.value.__notes__)
    assert "WARNING:" in capsys.readouterr().err


def test_keyboard_interrupt_attempts_failed_status(monkeypatch):
    calls = _configure_fake_tracking(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        with tracking_runs.start_tracking_run():
            raise KeyboardInterrupt()
    assert calls == ["FAILED"]


def test_unavailable_server_before_run_creation_propagates(monkeypatch):
    class FakeClient:
        def __init__(self, tracking_uri):
            pass

        def get_experiment_by_name(self, name):
            raise ConnectionError("tracking server offline before creation")

    monkeypatch.setattr(tracking_runs, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        tracking_runs,
        "load_infrastructure_config",
        lambda: SimpleNamespace(
            mlflow=SimpleNamespace(
                tracking_uri="http://127.0.0.1:5000",
                experiment_name="fashion-mnist-classifier",
            )
        ),
    )
    with pytest.raises(ConnectionError, match="before creation"):
        with tracking_runs.start_tracking_run():
            pytest.fail("Training must not begin without tracking")


def test_empty_created_run_id_is_rejected(monkeypatch):
    calls = _configure_fake_tracking(monkeypatch, run_id="")
    with pytest.raises(RuntimeError, match="valid run ID"):
        with tracking_runs.start_tracking_run():
            pytest.fail("Training must not begin with an unknown run ID")
    assert calls == []
