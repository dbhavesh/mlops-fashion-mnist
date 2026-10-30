"""Independently lock down the LeNet output and parameter contracts."""

import pytest
import torch
from torch import (
    nn,
)

from fashion_mnist_mlops.model import (
    LeNet5,
)


@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_lenet_parameter_count_and_logits_shape(batch_size):
    model = LeNet5()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 44_426

    inputs = torch.zeros(batch_size, 1, 28, 28)
    targets = torch.arange(batch_size) % 10

    outputs = model(inputs)
    assert outputs.shape == (batch_size, 10)
    assert torch.isfinite(outputs).all()

    # CrossEntropyLoss must be able to consume raw model outputs.
    loss = nn.CrossEntropyLoss()(outputs, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
