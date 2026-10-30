from dataclasses import (
    replace,
)

import pytest
from torch.optim import (
    Adam,
    SGD,
)

from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.config import (
    load_training_config,
)
from fashion_mnist_mlops.training_operations import (
    create_optimizer,
)


def test_create_adam_optimizer():
    settings = load_training_config().training

    model = LeNet5()

    optimizer = create_optimizer(
        model,
        settings,
    )

    assert isinstance(optimizer, Adam)

    assert (
        optimizer.param_groups[0]["lr"]
        == settings.learning_rate
    )


def test_create_sgd_optimizer():
    settings = replace(
        load_training_config().training,
        optimizer="sgd",
        learning_rate=0.01,
    )

    model = LeNet5()

    optimizer = create_optimizer(
        model,
        settings,
    )

    assert isinstance(optimizer, SGD)

    assert (
        optimizer.param_groups[0]["lr"]
        == 0.01
    )


def test_unsupported_optimizer_is_rejected():
    settings = replace(
        load_training_config().training,
        optimizer="rmsprop",
    )

    model = LeNet5()

    with pytest.raises(
        ValueError,
        match="Unsupported optimizer",
    ):
        create_optimizer(
            model,
            settings,
        )
