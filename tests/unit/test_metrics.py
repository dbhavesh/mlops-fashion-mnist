import math

import pytest
import torch
from torch import (
    nn,
)
from torch.optim import (
    SGD,
)
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from fashion_mnist_mlops.metrics import (
    EpochMetrics,
    EpochRecord,
    count_correct_predictions,
)
from fashion_mnist_mlops.training_operations import (
    train_one_epoch,
    validate_one_epoch,
)


def make_known_predictions():
    """Create a classifier with four predictable outputs."""

    model = nn.Linear(
        2,
        3,
        bias=False,
    )

    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
            ])
        )

    images = torch.tensor([
        [2.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [-1.0, -1.0],
        [2.0, 0.0],
    ])

    labels = torch.tensor(
        [0, 1, 2, 2, 0],
        dtype=torch.int64,
    )

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
        shuffle=False,
    )

    return model, loader


def test_correct_prediction_count():
    logits = torch.tensor([
        [0.1, 2.0, 0.3],
        [3.0, 0.2, 0.1],
        [0.2, 0.4, 1.5],
    ])

    labels = torch.tensor(
        [1, 0, 1],
        dtype=torch.int64,
    )

    assert count_correct_predictions(
        logits,
        labels,
    ) == 2


def test_validation_reports_correct_accuracy():
    model, loader = make_known_predictions()

    metrics = validate_one_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        return_metrics=True,
    )

    assert isinstance(metrics, EpochMetrics)
    assert metrics.correct == 4
    assert metrics.samples == 5
    assert metrics.accuracy == pytest.approx(0.8)
    assert math.isfinite(metrics.loss)


def test_training_reports_correct_accuracy():
    model, loader = make_known_predictions()

    optimizer = SGD(
        model.parameters(),
        lr=0.0,
    )

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        return_metrics=True,
    )

    assert isinstance(metrics, EpochMetrics)
    assert metrics.correct == 4
    assert metrics.samples == 5
    assert metrics.accuracy == pytest.approx(0.8)
    assert math.isfinite(metrics.loss)


def test_legacy_float_return_is_preserved():
    model, loader = make_known_predictions()

    optimizer = SGD(
        model.parameters(),
        lr=0.0,
    )

    train_loss = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
    )

    validation_loss = validate_one_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
    )

    assert isinstance(train_loss, float)
    assert isinstance(validation_loss, float)


def test_epoch_record_has_tracking_metric_names():
    train = EpochMetrics(
        loss=0.5,
        accuracy=0.8,
        correct=8,
        samples=10,
    )

    validation = EpochMetrics(
        loss=0.6,
        accuracy=0.7,
        correct=7,
        samples=10,
    )

    record = EpochRecord(
        epoch=1,
        train=train,
        validation=validation,
    )

    assert record.as_tracking_metrics() == {
        "train_loss": 0.5,
        "train_accuracy": 0.8,
        "val_loss": 0.6,
        "val_accuracy": 0.7,
    }


def test_invalid_accuracy_inputs_are_rejected():
    logits = torch.randn(2, 10)

    invalid_labels = torch.zeros(
        2,
        1,
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="Accuracy requires",
    ):
        count_correct_predictions(
            logits,
            invalid_labels,
        )
