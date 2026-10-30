import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, Optimizer, SGD
from torch.utils.data import DataLoader

from .config import TrainingSettings
from .metrics import EpochMetrics, count_correct_predictions


def create_loss_function(
    settings: TrainingSettings,
) -> nn.Module:
    loss_name = settings.loss.lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(
        f"Unsupported loss function: {settings.loss}"
    )


def create_optimizer(
    model: nn.Module,
    settings: TrainingSettings,
) -> Optimizer:
    optimizer_name = settings.optimizer.lower()

    if optimizer_name == "adam":
        return Adam(
            model.parameters(),
            lr=settings.learning_rate,
        )

    if optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=settings.learning_rate,
        )

    raise ValueError(
        f"Unsupported optimizer: {settings.optimizer}"
    )


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
) -> None:
    """Configure random seeds and deterministic PyTorch behavior."""

    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= (2**64 - 1)
    ):
        raise ValueError(
            "seed must be an integer between 0 "
            "and 2**64 - 1."
        )

    # Python standard-library randomness.
    random.seed(seed)

    # NumPy's legacy global RNG accepts 32-bit seeds.
    np.random.seed(seed % (2**32))

    # Seed PyTorch on CPU and available accelerators.
    torch.manual_seed(seed)

    # Configure deterministic PyTorch operations.
    torch.use_deterministic_algorithms(
        deterministic
    )

    # Configure cuDNN if a CUDA-enabled build is used.
    torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    *,
    return_metrics: bool = False,
) -> float | EpochMetrics:
    """Train for one epoch and return loss or detailed metrics."""

    model.train()

    device = next(model.parameters()).device

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(logits, labels)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Non-finite training loss encountered."
            )

        # Calculate accuracy before updating parameters.
        if return_metrics:
            total_correct += count_correct_predictions(
                logits.detach(),
                labels,
            )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Training DataLoader contains no samples."
        )

    average_loss = total_loss / total_samples

    if return_metrics:
        return EpochMetrics(
            loss=average_loss,
            accuracy=total_correct / total_samples,
            correct=total_correct,
            samples=total_samples,
        )

    return average_loss


def validate_one_epoch(
    model: nn.Module,
    validation_loader: DataLoader,
    criterion: nn.Module,
    *,
    return_metrics: bool = False,
) -> float | EpochMetrics:
    """Evaluate an epoch and return loss or detailed metrics."""

    model.eval()

    device = next(model.parameters()).device

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Non-finite validation loss encountered."
                )

            if return_metrics:
                total_correct += count_correct_predictions(
                    logits,
                    labels,
                )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Validation DataLoader contains no samples."
        )

    average_loss = total_loss / total_samples

    if return_metrics:
        return EpochMetrics(
            loss=average_loss,
            accuracy=total_correct / total_samples,
            correct=total_correct,
            samples=total_samples,
        )

    return average_loss
