from dataclasses import (
    replace,
)

import pytest
import torch
from torch import (
    nn,
)

from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.config import (
    load_training_config,
)
from fashion_mnist_mlops.training_operations import (
    create_loss_function,
)


def test_create_cross_entropy_loss():
    settings = load_training_config().training

    criterion = create_loss_function(
        settings
    )

    assert isinstance(
        criterion,
        nn.CrossEntropyLoss,
    )


def test_cross_entropy_accepts_model_output():
    settings = load_training_config().training

    criterion = create_loss_function(
        settings
    )

    model = LeNet5()

    inputs = torch.randn(
        4,
        1,
        28,
        28,
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.int64,
    )

    logits = model(inputs)

    loss = criterion(
        logits,
        labels,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_unsupported_loss_is_rejected():
    settings = replace(
        load_training_config().training,
        loss="mse",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported loss function",
    ):
        create_loss_function(
            settings
        )
