from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fashion_mnist_mlops.model_artifacts import (
    load_model_artifact,
)
from fashion_mnist_mlops.training import (
    TrainingRunResult,
    run_training,
)
import fashion_mnist_mlops.training as runner


def make_config_file(tmp_path: Path) -> Path:
    path = tmp_path / "training.toml"

    path.write_text(
        """
[training]
epochs = 2
batch_size = 4
learning_rate = 0.001
optimizer = "adam"
loss = "cross_entropy"
seed = 42

[validation]
validation_split = 0.1
""".strip(),
        encoding="utf-8",
    )

    return path


def make_small_loaders(
    config,
    root=None,
    download=False,
):
    from fashion_mnist_mlops.data import DataLoaders

    images = torch.linspace(
        -1.0,
        1.0,
        steps=12 * 28 * 28,
    ).reshape(12, 1, 28, 28)

    labels = torch.arange(12) % 10

    train_dataset = TensorDataset(
        images[:8],
        labels[:8],
    )

    validation_dataset = TensorDataset(
        images[8:],
        labels[8:],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    return DataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=validation_loader,
    )


def test_training_runner_saves_final_artifact(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "create_data_loaders",
        make_small_loaders,
    )

    config_path = make_config_file(tmp_path)
    artifact_path = tmp_path / "model.pt"

    result = run_training(
        config_path=config_path,
        artifact_path=artifact_path,
    )

    assert isinstance(result, TrainingRunResult)

    assert len(result.history) == 2
    assert result.history[0].epoch == 1
    assert result.history[1].epoch == 2

    assert result.config.training.epochs == 2
    assert result.artifact_path == artifact_path
    assert artifact_path.is_file()

    model, metadata = load_model_artifact(
        artifact_path
    )

    assert metadata["epoch"] == 2

    assert (
        metadata["validation_accuracy"]
        == result.final_epoch.validation.accuracy
    )

    assert not model.training


def test_epochs_override_is_applied(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "create_data_loaders",
        make_small_loaders,
    )

    config_path = make_config_file(tmp_path)

    result = run_training(
        config_path=config_path,
        artifact_path=tmp_path / "model.pt",
        epochs=1,
    )

    assert result.config.training.epochs == 1
    assert len(result.history) == 1
    assert result.final_epoch.epoch == 1


def test_existing_artifact_is_not_overwritten(
    tmp_path,
):
    config_path = make_config_file(tmp_path)

    artifact_path = tmp_path / "model.pt"
    artifact_path.write_bytes(b"existing artifact")

    with pytest.raises(
        FileExistsError,
        match="Artifact already exists",
    ):
        run_training(
            config_path=config_path,
            artifact_path=artifact_path,
        )

    assert artifact_path.read_bytes() == (
        b"existing artifact"
    )


def test_invalid_epochs_override_is_rejected(
    tmp_path,
):
    config_path = make_config_file(tmp_path)

    with pytest.raises(
        ValueError,
        match="epochs override",
    ):
        run_training(
            config_path=config_path,
            epochs=0,
        )


def test_invalid_artifact_extension_is_rejected(
    tmp_path,
):
    config_path = make_config_file(tmp_path)

    with pytest.raises(
        ValueError,
        match=".pt extension",
    ):
        run_training(
            config_path=config_path,
            artifact_path=tmp_path / "model.txt",
        )


def test_training_failure_does_not_save_artifact(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "create_data_loaders",
        make_small_loaders,
    )

    def failing_training(*args, **kwargs):
        raise RuntimeError(
            "Simulated training failure"
        )

    monkeypatch.setattr(
        runner,
        "train_one_epoch",
        failing_training,
    )

    config_path = make_config_file(tmp_path)
    artifact_path = tmp_path / "model.pt"

    with pytest.raises(
        RuntimeError,
        match="Simulated training failure",
    ):
        run_training(
            config_path=config_path,
            artifact_path=artifact_path,
        )

    assert not artifact_path.exists()


def test_sample_limits_are_applied(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "create_data_loaders",
        make_small_loaders,
    )

    result = run_training(
        config_path=make_config_file(tmp_path),
        artifact_path=tmp_path / "limited.pt",
        epochs=1,
        max_train_samples=6,
        max_validation_samples=3,
    )

    assert result.final_epoch.train.samples == 6
    assert result.final_epoch.validation.samples == 3

    assert result.artifact_path.is_file()


def test_invalid_sample_limit_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="max_train_samples",
    ):
        run_training(
            config_path=make_config_file(tmp_path),
            max_train_samples=0,
        )


def test_sample_limit_cannot_exceed_dataset(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "create_data_loaders",
        make_small_loaders,
    )

    with pytest.raises(
        ValueError,
        match="exceeds available samples",
    ):
        run_training(
            config_path=make_config_file(tmp_path),
            max_validation_samples=5,
        )


def test_runner_paths_are_independent_of_cwd(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "create_data_loaders",
        make_small_loaders,
    )

    project_root = tmp_path / "project"
    project_root.mkdir()

    config_path = make_config_file(project_root)

    # Simulate launching from a different directory.
    external_cwd = tmp_path / "elsewhere"
    external_cwd.mkdir()
    monkeypatch.chdir(external_cwd)

    result = run_training(
        config_path=config_path,
        project_root=project_root,
        artifact_path="artifacts/models/model.pt",
        epochs=1,
    )

    expected_path = (
        project_root
        / "artifacts"
        / "models"
        / "model.pt"
    )

    assert result.artifact_path == expected_path
    assert expected_path.is_file()

    # Nothing should be written under the external CWD.
    assert not (
        external_cwd / "artifacts"
    ).exists()


@pytest.mark.parametrize(
    "download_flag",
    [False, True],
)
def test_runner_forwards_download_policy(
    tmp_path,
    monkeypatch,
    download_flag,
):
    observed = []

    def recording_data_loaders(
        config,
        root=None,
        download=False,
    ):
        observed.append(download)

        return make_small_loaders(
            config,
            root=root,
            download=download,
        )

    monkeypatch.setattr(
        runner,
        "create_data_loaders",
        recording_data_loaders,
    )

    result = run_training(
        config_path=make_config_file(tmp_path),
        artifact_path=tmp_path / "model.pt",
        epochs=1,
        download_data=download_flag,
    )

    assert observed == [download_flag]
    assert result.artifact_path.is_file()


def test_tracked_runner_propagates_id_to_artifact_and_result(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)

    @contextmanager
    def fake_tracking_run():
        yield "test-run-id-54"

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *_, **__: None)
    # isolates run-ID propagation from parameter/metric logging.
    monkeypatch.setattr(runner, "log_training_parameters", lambda *_: None)
    monkeypatch.setattr(runner, "log_training_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_validation_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_final_validation_accuracy", lambda *_, **__: None)
    result = runner.run_training(
        config_path=make_config_file(tmp_path),
        artifact_path=tmp_path / "tracked.pt",
        epochs=1,
        track_mlflow=True,
    )
    _, metadata = load_model_artifact(result.artifact_path)
    assert result.mlflow_run_id == "test-run-id-54"
    assert metadata["mlflow_run_id"] == result.mlflow_run_id
    assert metadata["checkpoint_format_version"] == 2
    assert metadata["provenance"]["training"]["epochs"] == 1


def test_untracked_runner_retains_none_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)

    def fail_if_called():
        raise AssertionError("MLflow should not be accessed")

    monkeypatch.setattr(runner, "start_tracking_run", fail_if_called)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *_, **__: fail_if_called())
    monkeypatch.setattr(runner, "log_training_loss", lambda *_, **__: fail_if_called())
    monkeypatch.setattr(runner, "log_validation_loss", lambda *_, **__: fail_if_called())
    monkeypatch.setattr(runner, "log_final_validation_accuracy", lambda *_, **__: fail_if_called())
    result = runner.run_training(
        config_path=make_config_file(tmp_path),
        artifact_path=tmp_path / "untracked.pt",
        epochs=1,
    )
    _, metadata = load_model_artifact(result.artifact_path)
    assert result.mlflow_run_id is None
    assert metadata["mlflow_run_id"] is None


def test_mlflow_startup_failure_does_not_save_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)

    def unavailable():
        raise ConnectionError("Tracking service unavailable")

    monkeypatch.setattr(runner, "start_tracking_run", unavailable)
    path = tmp_path / "must_not_exist.pt"
    with pytest.raises(ConnectionError, match="Tracking service unavailable"):
        runner.run_training(
            config_path=make_config_file(tmp_path),
            artifact_path=path,
            epochs=1,
            track_mlflow=True,
        )
    assert not path.exists()


def test_tracked_runner_logs_effective_parameters_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)
    observed = []

    @contextmanager
    def fake_tracking_run():
        yield "test-run-id-55"

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_training_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_validation_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_final_validation_accuracy", lambda *_, **__: None)
    monkeypatch.setattr(
        runner,
        "log_training_parameters",
        lambda run_id, provenance: observed.append((run_id, provenance)),
    )
    result = runner.run_training(
        config_path=make_config_file(tmp_path),
        artifact_path=tmp_path / "tracked55.pt",
        epochs=1,
        max_train_samples=4,
        max_validation_samples=2,
        track_mlflow=True,
    )
    assert len(observed) == 1
    run_id, provenance = observed[0]
    assert run_id == "test-run-id-55"
    assert provenance["training"]["epochs"] == 1
    assert provenance["dataset"]["train_samples_used"] == 4
    assert provenance["dataset"]["validation_samples_used"] == 2
    assert result.mlflow_run_id == run_id
    assert result.artifact_path.is_file()


def test_parameter_logging_failure_prevents_training_and_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)

    @contextmanager
    def fake_tracking_run():
        yield "test-run-id-55-failed"

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *_, **__: None)

    def fail_logging(run_id, provenance):
        raise ConnectionError("Failed to log required parameters")

    def fail_if_training_starts(*args, **kwargs):
        raise AssertionError("Training should not begin after MLflow failure")

    monkeypatch.setattr(runner, "log_training_parameters", fail_logging)
    monkeypatch.setattr(runner, "train_one_epoch", fail_if_training_starts)
    destination = tmp_path / "must_not_exist55.pt"
    with pytest.raises(ConnectionError, match="required parameters"):
        runner.run_training(
            config_path=make_config_file(tmp_path),
            artifact_path=destination,
            epochs=1,
            track_mlflow=True,
        )
    assert not destination.exists()


def test_tracked_runner_logs_train_loss_at_each_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)
    observed = []

    @contextmanager
    def fake_tracking_run():
        yield "train-loss-run"

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_training_parameters", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "log_training_loss",
        lambda run_id, *, epoch, loss: observed.append((run_id, epoch, loss)),
    )
    monkeypatch.setattr(runner, "log_validation_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_final_validation_accuracy", lambda *_, **__: None)

    result = runner.run_training(
        config_path=make_config_file(tmp_path),
        artifact_path=tmp_path / "three-epochs.pt",
        epochs=3,
        track_mlflow=True,
    )
    assert [run_id for run_id, _, _ in observed] == ["train-loss-run"] * 3
    assert [epoch_index for _, epoch_index, _ in observed] == [1, 2, 3]
    assert [loss for _, _, loss in observed] == [
        record.train.loss for record in result.history
    ]
    assert result.artifact_path.is_file()


def test_train_loss_logging_failure_prevents_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)
    statuses = []

    @contextmanager
    def fake_tracking_run():
        try:
            yield "failed-loss-run"
        except ConnectionError:
            statuses.append("FAILED")
            raise

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_training_parameters", lambda *args: None)

    def fail_log(run_id, *, epoch, loss):
        raise ConnectionError("MLflow training loss unavailable")

    def should_not_validate(*args, **kwargs):
        raise AssertionError("Validation must not run after failed loss logging")

    monkeypatch.setattr(runner, "log_training_loss", fail_log)
    monkeypatch.setattr(runner, "validate_one_epoch", should_not_validate)
    path = tmp_path / "missing-artifact.pt"

    with pytest.raises(ConnectionError, match="training loss unavailable"):
        runner.run_training(
            config_path=make_config_file(tmp_path),
            artifact_path=path,
            epochs=1,
            track_mlflow=True,
        )

    assert statuses == ["FAILED"]
    assert not path.exists()



def test_tracked_runner_logs_validation_loss_each_epoch_and_final_accuracy(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)
    validation_calls = []
    accuracy_calls = []

    @contextmanager
    def fake_tracking_run():
        yield "validation-run-57"

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_training_parameters", lambda *args: None)
    monkeypatch.setattr(runner, "log_training_loss", lambda *_, **__: None)
    monkeypatch.setattr(
        runner,
        "log_validation_loss",
        lambda run_id, *, epoch, loss: validation_calls.append(
            (run_id, epoch, loss)
        ),
    )
    monkeypatch.setattr(
        runner,
        "log_final_validation_accuracy",
        lambda run_id, *, epoch, accuracy: accuracy_calls.append(
            (run_id, epoch, accuracy)
        ),
    )

    result = runner.run_training(
        config_path=make_config_file(tmp_path),
        artifact_path=tmp_path / "validation57.pt",
        epochs=3,
        track_mlflow=True,
    )
    assert [run_id for run_id, _, _ in validation_calls] == [
        "validation-run-57"
    ] * 3
    assert [epoch_index for _, epoch_index, _ in validation_calls] == [1, 2, 3]
    assert [loss for _, _, loss in validation_calls] == [
        record.validation.loss for record in result.history
    ]
    assert accuracy_calls == [
        ("validation-run-57", 3, result.final_epoch.validation.accuracy)
    ]
    assert result.artifact_path.is_file()
    _, metadata = load_model_artifact(result.artifact_path)
    assert metadata["validation_accuracy"] == result.final_epoch.validation.accuracy


def test_validation_loss_logging_failure_prevents_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)
    statuses = []

    @contextmanager
    def fake_tracking_run():
        try:
            yield "failed-validation-loss"
        except ConnectionError:
            statuses.append("FAILED")
            raise

    def unavailable(*args, **kwargs):
        raise ConnectionError("MLflow validation loss unavailable")

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_training_parameters", lambda *args: None)
    monkeypatch.setattr(runner, "log_training_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_validation_loss", unavailable)
    monkeypatch.setattr(
        runner,
        "log_final_validation_accuracy",
        lambda *_, **__: pytest.fail("Must not log accuracy after loss failure"),
    )
    destination = tmp_path / "missing-val-loss.pt"
    with pytest.raises(ConnectionError, match="validation loss unavailable"):
        runner.run_training(
            config_path=make_config_file(tmp_path),
            artifact_path=destination,
            epochs=1,
            track_mlflow=True,
        )
    assert statuses == ["FAILED"]
    assert not destination.exists()


def test_final_validation_accuracy_failure_prevents_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)
    statuses = []
    calls = []

    @contextmanager
    def fake_tracking_run():
        try:
            yield "failed-final-accuracy"
        except ConnectionError:
            statuses.append("FAILED")
            raise

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_training_parameters", lambda *args: None)
    monkeypatch.setattr(runner, "log_training_loss", lambda *_, **__: None)
    monkeypatch.setattr(
        runner,
        "log_validation_loss",
        lambda *_, **__: calls.append("val_loss"),
    )

    def unavailable(*args, **kwargs):
        raise ConnectionError("MLflow final accuracy unavailable")

    monkeypatch.setattr(runner, "log_final_validation_accuracy", unavailable)
    destination = tmp_path / "missing-final-accuracy.pt"
    with pytest.raises(ConnectionError, match="final accuracy unavailable"):
        runner.run_training(
            config_path=make_config_file(tmp_path),
            artifact_path=destination,
            epochs=2,
            track_mlflow=True,
        )
    assert calls == ["val_loss", "val_loss"]
    assert statuses == ["FAILED"]
    assert not destination.exists()


def test_tracked_runner_logs_metadata_after_artifact_save(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)
    observed = []

    @contextmanager
    def fake_tracking_run():
        yield "metadata-run-58"

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_training_parameters", lambda *_: None)
    monkeypatch.setattr(runner, "log_training_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_validation_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_final_validation_accuracy", lambda *_, **__: None)

    def inspect_metadata(run_id, *, provenance, artifact_path, project_root):
        assert artifact_path.is_file()  # Reference cannot precede publication.
        observed.append((run_id, provenance, artifact_path, project_root))

    monkeypatch.setattr(runner, "log_run_metadata", inspect_metadata)
    root = tmp_path / "project"
    root.mkdir()
    result = runner.run_training(
        config_path=make_config_file(root),
        artifact_path="artifacts/models/model.pt",
        project_root=root,
        epochs=1,
        track_mlflow=True,
    )
    assert len(observed) == 1
    run_id, provenance, artifact_path, project_root = observed[0]
    assert run_id == result.mlflow_run_id == "metadata-run-58"
    assert artifact_path == result.artifact_path
    assert project_root == root
    assert provenance["training"]["epochs"] == 1
    assert provenance["source_config"]["effective_sha256"]
    assert load_model_artifact(result.artifact_path)[1]["mlflow_run_id"] == run_id


def test_metadata_failure_preserves_atomic_artifact_and_reports_error(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)
    statuses = []

    @contextmanager
    def fake_tracking_run():
        try:
            yield "metadata-failed-58"
        except ConnectionError:
            statuses.append("FAILED")
            raise

    monkeypatch.setattr(runner, "start_tracking_run", fake_tracking_run)
    monkeypatch.setattr(runner, "log_training_parameters", lambda *_: None)
    monkeypatch.setattr(runner, "log_training_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_validation_loss", lambda *_, **__: None)
    monkeypatch.setattr(runner, "log_final_validation_accuracy", lambda *_, **__: None)

    def fail_metadata(run_id, *, provenance, artifact_path, project_root):
        assert artifact_path.is_file()
        raise ConnectionError("MLflow metadata upload unavailable")

    monkeypatch.setattr(runner, "log_run_metadata", fail_metadata)
    destination = tmp_path / "retained-model.pt"
    with pytest.raises(ConnectionError, match="metadata upload unavailable"):
        runner.run_training(
            config_path=make_config_file(tmp_path),
            artifact_path=destination,
            epochs=1,
            track_mlflow=True,
        )
    assert statuses == ["FAILED"]
    assert destination.is_file()  # never erase a published artifact.
    _, metadata = load_model_artifact(destination)
    assert metadata["mlflow_run_id"] == "metadata-failed-58"
    assert metadata["checkpoint_format_version"] == 2


def test_finalization_failure_preserves_artifact_and_does_not_report_success(
    tmp_path, monkeypatch, capsys
):
    """a saved artifact is not proof of MLflow success."""
    from types import SimpleNamespace
    import fashion_mnist_mlops.tracking as tracking_runs

    statuses = []

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://127.0.0.1:5000"

        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="1", lifecycle_stage="active")

        def create_run(self, experiment_id):
            return SimpleNamespace(info=SimpleNamespace(run_id="finalize-failed-59"))

        def set_terminated(self, run_id, status):
            statuses.append(status)
            if status == "FINISHED":
                raise ConnectionError("server disconnected during finalization")

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
    monkeypatch.setattr(runner, "create_data_loaders", make_small_loaders)
    monkeypatch.setattr(runner, "log_training_parameters", lambda *args: None)
    monkeypatch.setattr(runner, "log_training_loss", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "log_validation_loss", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "log_final_validation_accuracy", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "log_run_metadata", lambda *args, **kwargs: None)

    destination = tmp_path / "finalization-failed.pt"
    with pytest.raises(RuntimeError, match="could not confirm|Could not confirm") as info:
        runner.run_training(
            config_path=make_config_file(tmp_path),
            artifact_path=destination,
            epochs=1,
            track_mlflow=True,
        )

    assert statuses == ["FINISHED", "FAILED"]
    assert isinstance(info.value.__cause__, ConnectionError)
    assert destination.is_file()  # Atomic publication must be preserved.
    _, saved_metadata = load_model_artifact(destination)
    assert saved_metadata["mlflow_run_id"] == "finalize-failed-59"
    assert "Training completed successfully" not in capsys.readouterr().out
