"""per-epoch training loss logging without a live MLflow server."""

from types import SimpleNamespace

import pytest

import fashion_mnist_mlops.tracking as tracking_metrics


def test_loss_logging_uses_configured_endpoint_and_epoch_index(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://127.0.0.1:5000"

        def log_metric(self, run_id, *, key, value, step):
            calls.append((run_id, key, value, step))

    monkeypatch.setattr(tracking_metrics, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        tracking_metrics,
        "load_infrastructure_config",
        lambda: SimpleNamespace(
            mlflow=SimpleNamespace(tracking_uri="http://127.0.0.1:5000")
        ),
    )

    tracking_metrics.log_training_loss("run-56", epoch=1, loss=2.25)
    tracking_metrics.log_training_loss("run-56", epoch=2, loss=2.0)

    assert calls == [
        ("run-56", "train_loss", 2.25, 1),
        ("run-56", "train_loss", 2.0, 2),
    ]


def test_loss_logging_propagates_client_failure(monkeypatch):
    class FakeClient:
        def __init__(self, tracking_uri):
            pass

        def log_metric(self, run_id, *, key, value, step):
            raise ConnectionError("MLflow metric logging unavailable")

    monkeypatch.setattr(tracking_metrics, "MlflowClient", FakeClient)

    with pytest.raises(ConnectionError, match="metric logging unavailable"):
        tracking_metrics.log_training_loss("run-56", epoch=1, loss=2.25)


@pytest.mark.parametrize("epoch", [0, -1, 1.5, True])
def test_loss_logging_rejects_invalid_epochs(epoch):
    with pytest.raises(ValueError, match="positive integer"):
        tracking_metrics.log_training_loss("run-56", epoch=epoch, loss=2.25)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf")])
def test_loss_logging_rejects_nonfinite_values(loss):
    with pytest.raises(ValueError, match="finite"):
        tracking_metrics.log_training_loss("run-56", epoch=1, loss=loss)
