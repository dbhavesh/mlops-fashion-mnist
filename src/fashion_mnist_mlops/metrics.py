from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics aggregated across one dataset pass."""

    loss: float
    accuracy: float
    correct: int
    samples: int


@dataclass(frozen=True)
class EpochRecord:
    """Training and validation results for one epoch."""

    epoch: int
    train: EpochMetrics
    validation: EpochMetrics

    def as_tracking_metrics(self) -> dict[str, float]:
        """Return flat metric names suitable for MLflow."""

        return {
            "train_loss": self.train.loss,
            "train_accuracy": self.train.accuracy,
            "val_loss": self.validation.loss,
            "val_accuracy": self.validation.accuracy,
        }


def count_correct_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> int:
    """Count correct predictions in a classification batch."""

    if (
        logits.ndim != 2
        or labels.ndim != 1
        or logits.size(0) != labels.size(0)
        or labels.dtype != torch.int64
    ):
        raise ValueError(
            "Accuracy requires logits with shape [N, C] "
            "and int64 labels with shape [N]."
        )

    predictions = logits.argmax(dim=1)

    correct = (
        predictions == labels
    ).sum().item()

    return int(correct)
