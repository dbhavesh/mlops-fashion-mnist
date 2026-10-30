import torch
from torch import (
    nn,
)

from fashion_mnist_mlops.preprocessing import (
    IMAGE_CHANNELS,
)
from fashion_mnist_mlops.labels import (
    NUM_CLASSES,
)


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=IMAGE_CHANNELS,
            out_channels=6,
            kernel_size=5,
        )

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )

        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(
            in_features=16 * 4 * 4,
            out_features=120,
        )

        self.fc2 = nn.Linear(
            in_features=120,
            out_features=84,
        )

        self.fc3 = nn.Linear(
            in_features=84,
            out_features=NUM_CLASSES,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x
