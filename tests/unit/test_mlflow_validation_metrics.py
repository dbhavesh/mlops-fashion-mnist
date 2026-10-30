"""validation loss per epoch and final validation accuracy."""

from types import SimpleNamespace

import pytest

import fashion_mnist_mlops.tracking as tracking_metrics


def test_validation_metrics_use_configured_endpoint_and_correct_epochs(monkeypatch):
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

    tracking_metrics.log_validation_loss("run-57", epoch=1, loss=2.25)
    tracking_metrics.log_validation_loss("run-57", epoch=2, loss=2.0)
    tracking_metrics.log_validation_loss("run-57", epoch=3, loss=1.75)
    tracking_metrics.log_final_validation_accuracy(
        "run-57", epoch=3, accuracy=0.625
    )

    assert calls == [
        ("run-57", "val_loss", 2.25, 1),
        ("run-57", "val_loss", 2.0, 2),
        ("run-57", "val_loss", 1.75, 3),
        ("run-57", "val_accuracy", 0.625, 3),
    ]


@pytest.mark.parametrize("function_name,keyword", [
    ("log_validation_loss", "loss"),
    ("log_final_validation_accuracy", "accuracy"),
])
def test_validation_logging_propagates_client_failure(
    monkeypatch, function_name, keyword
):
    class FakeClient:
        def __init__(self, tracking_uri):
            pass

        def log_metric(self, run_id, *, key, value, step):
            raise ConnectionError("MLflow validation metric unavailable")

    monkeypatch.setattr(tracking_metrics, "MlflowClient", FakeClient)
    kwargs = {keyword: 0.5}
    with pytest.raises(ConnectionError, match="validation metric unavailable"):
        getattr(tracking_metrics, function_name)("run-57", epoch=1, **kwargs)


@pytest.mark.parametrize("epoch", [0, -1, 1.5, True])
@pytest.mark.parametrize("function_name,keyword", [
    ("log_validation_loss", "loss"),
    ("log_final_validation_accuracy", "accuracy"),
])
def test_validation_logging_rejects_invalid_epoch(
    epoch, function_name, keyword
):
    with pytest.raises(ValueError, match="positive integer"):
        getattr(tracking_metrics, function_name)(
            "run-57", epoch=epoch, **{keyword: 0.5}
        )


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf")])
def test_validation_loss_rejects_nonfinite_values(loss):
    with pytest.raises(ValueError, match="finite"):
        tracking_metrics.log_validation_loss("run-57", epoch=1, loss=loss)


@pytest.mark.parametrize("accuracy", [
    float("nan"), float("inf"), float("-inf"), -0.01, 1.01
])
def test_final_validation_accuracy_rejects_invalid_values(accuracy):
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        tracking_metrics.log_final_validation_accuracy(
            "run-57", epoch=1, accuracy=accuracy
        )
