"""effective MLflow parameter conversion and client operations."""

from types import SimpleNamespace

import pytest

import fashion_mnist_mlops.tracking as parameters


def _provenance(*, max_train_samples=32, max_validation_samples=16):
    return {
        "training": {
            "epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "cross_entropy",
            "seed": 42,
        },
        "validation": {"validation_split": 0.1},
        "dataset": {
            "train_samples_used": 32,
            "validation_samples_used": 16,
            "max_train_samples": max_train_samples,
            "max_validation_samples": max_validation_samples,
            "download_allowed": False,
        },
    }


def test_effective_parameters_capture_all_required_values():
    assert parameters.effective_training_parameters(_provenance()) == {
        "epochs": "1",
        "batch_size": "64",
        "learning_rate": "0.001",
        "optimizer": "adam",
        "loss": "cross_entropy",
        "seed": "42",
        "validation_split": "0.1",
        "train_samples_used": "32",
        "validation_samples_used": "16",
        "max_train_samples": "32",
        "max_validation_samples": "16",
        "download_allowed": "false",
    }
    params = parameters.effective_training_parameters(
        _provenance(max_train_samples=None, max_validation_samples=None)
    )
    assert params["max_train_samples"] == "none"
    assert params["max_validation_samples"] == "none"


def test_parameter_logging_sends_one_batch_to_selected_endpoint(monkeypatch):
    observed = []

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://127.0.0.1:5000"

        def log_batch(self, **kwargs):
            observed.append(kwargs)

    monkeypatch.setattr(parameters, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        parameters,
        "load_infrastructure_config",
        lambda: SimpleNamespace(mlflow=SimpleNamespace(
            tracking_uri="http://127.0.0.1:5000",
        )),
    )
    parameters.log_training_parameters("run-id-55", _provenance())
    assert len(observed) == 1
    batch = observed[0]
    assert batch["run_id"] == "run-id-55"
    assert {p.key: p.value for p in batch["params"]} == (
        parameters.effective_training_parameters(_provenance())
    )


def test_parameter_logging_propagates_client_errors(monkeypatch):
    class FakeClient:
        def __init__(self, tracking_uri):
            pass

        def log_batch(self, **kwargs):
            raise ConnectionError("MLflow parameter logging unavailable")

    monkeypatch.setattr(parameters, "MlflowClient", FakeClient)
    with pytest.raises(ConnectionError, match="parameter logging unavailable"):
        parameters.log_training_parameters("run-id-55", _provenance())
