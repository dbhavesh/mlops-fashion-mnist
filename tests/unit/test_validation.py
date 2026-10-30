import math

import pytest
import torch
from torch import (
    nn,
)
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.training_operations import (
    validate_one_epoch,
)


def test_validation_does_not_update_parameters():
    torch.manual_seed(42)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    model = LeNet5()
    model.train()

    criterion = nn.CrossEntropyLoss()

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss = validate_one_epoch(
        model,
        loader,
        criterion,
    )

    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss > 0

    assert not model.training

    for old, new in zip(
        before,
        model.parameters(),
    ):
        assert torch.equal(old, new)

    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_validation_loss_is_sample_weighted():
    images = torch.tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [10.0],
    ])

    labels = torch.zeros_like(images)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
    )

    model = nn.Linear(
        1,
        1,
        bias=False,
    )

    with torch.no_grad():
        model.weight.fill_(1.0)

    criterion = nn.MSELoss()

    loss = validate_one_epoch(
        model,
        loader,
        criterion,
    )

    assert loss == pytest.approx(26.0)


def test_empty_validation_loader_is_rejected():
    images = torch.empty(0, 1)
    labels = torch.empty(0, 1)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
    )

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()

    with pytest.raises(
        ValueError,
        match="no samples",
    ):
        validate_one_epoch(
            model,
            loader,
            criterion,
        )


def test_non_finite_validation_loss_is_rejected():
    images = torch.ones(2, 1)
    labels = torch.zeros(2, 1)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
    )

    model = nn.Linear(1, 1)

    class NonFiniteLoss(nn.Module):
        def forward(self, predictions, targets):
            return predictions.mean() * float("nan")

    with pytest.raises(
        RuntimeError,
        match="Non-finite validation loss",
    ):
        validate_one_epoch(
            model,
            loader,
            NonFiniteLoss(),
        )
