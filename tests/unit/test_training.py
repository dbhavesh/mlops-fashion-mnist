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

from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.training_operations import (
    train_one_epoch,
)


def test_one_epoch_updates_parameters():
    torch.manual_seed(42)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    model = LeNet5()
    model.eval()

    optimizer = SGD(
        model.parameters(),
        lr=0.01,
    )

    criterion = nn.CrossEntropyLoss()

    before = model.fc3.weight.detach().clone()

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
    )

    after = model.fc3.weight.detach().clone()

    assert model.training
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss > 0
    assert not torch.equal(before, after)


def test_epoch_loss_is_sample_weighted():
    images = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [10.0]]
    )

    labels = torch.zeros(
        5,
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
        shuffle=False,
    )

    model = nn.Linear(
        1,
        1,
        bias=False,
    )

    with torch.no_grad():
        model.weight.fill_(1.0)

    optimizer = SGD(
        model.parameters(),
        lr=0.0,
    )

    class MeanPredictionLoss(nn.Module):
        def forward(self, predictions, targets):
            return predictions.mean()

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        MeanPredictionLoss(),
    )

    # Correct: (1 + 2 + 3 + 4 + 10) / 5 = 4
    # Incorrect unweighted batch mean would be 5.
    assert loss == pytest.approx(4.0)


def test_empty_training_loader_is_rejected():
    images = torch.empty(0, 1)
    labels = torch.empty(0, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
        shuffle=False,
    )

    model = nn.Linear(1, 2)

    optimizer = SGD(
        model.parameters(),
        lr=0.01,
    )

    criterion = nn.CrossEntropyLoss()

    with pytest.raises(
        ValueError,
        match="no samples",
    ):
        train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
        )
