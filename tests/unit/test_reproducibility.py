import random

import numpy as np
import pytest
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.config import (
    load_training_config,
)
from fashion_mnist_mlops.training_operations import (
    create_loss_function,
    create_optimizer,
    seed_everything,
    train_one_epoch,
)


def test_random_generators_are_reproducible():
    seed_everything(42)

    python_a = random.random()
    numpy_a = np.random.rand()
    torch_a = torch.rand(3)

    seed_everything(42)

    assert random.random() == python_a
    assert np.random.rand() == numpy_a
    assert torch.equal(torch.rand(3), torch_a)


def test_model_initialization_is_reproducible():
    seed_everything(42)
    model_a = LeNet5()

    seed_everything(42)
    model_b = LeNet5()

    for parameter_a, parameter_b in zip(
        model_a.parameters(),
        model_b.parameters(),
    ):
        assert torch.equal(
            parameter_a,
            parameter_b,
        )


def test_deterministic_algorithms_enabled():
    seed_everything(42)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


@pytest.mark.parametrize(
    "invalid_seed",
    [-1, True, 3.14, "42", 2**64],
)
def test_invalid_seed_is_rejected(invalid_seed):
    with pytest.raises(
        ValueError,
        match="seed must be an integer",
    ):
        seed_everything(invalid_seed)


def run_small_training(seed):
    config = load_training_config()

    seed_everything(seed)

    images = torch.linspace(
        -1.0,
        1.0,
        steps=8 * 28 * 28,
    ).reshape(8, 1, 28, 28)

    labels = torch.arange(8)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = LeNet5()

    optimizer = create_optimizer(
        model,
        config.training,
    )

    criterion = create_loss_function(
        config.training,
    )

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        return_metrics=True,
    )

    parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    return metrics, parameters


def test_controlled_training_is_reproducible():
    metrics_a, parameters_a = run_small_training(42)
    metrics_b, parameters_b = run_small_training(42)

    assert metrics_a == metrics_b

    for parameter_a, parameter_b in zip(
        parameters_a,
        parameters_b,
    ):
        assert torch.equal(
            parameter_a,
            parameter_b,
        )
