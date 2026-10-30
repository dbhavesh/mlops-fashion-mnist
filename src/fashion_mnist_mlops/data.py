from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import FashionMNIST

from .config import TrainingConfig
from .paths import resolve_project_path
from .labels import get_class_name
from .preprocessing import build_image_transform


DEFAULT_DATA_DIR = Path("data")


def load_fashion_mnist(
    root: str | Path = DEFAULT_DATA_DIR,
    *,
    transform=None,
    download: bool = True,
    project_root: str | Path | None = None,
) -> tuple[FashionMNIST, FashionMNIST]:
    """Load Fashion-MNIST with an explicit download policy."""

    if type(download) is not bool:
        raise TypeError(
            "download must be a boolean."
        )

    data_root = resolve_project_path(
        root,
        project_root=project_root,
    )

    try:
        train_dataset = FashionMNIST(
            root=data_root,
            train=True,
            transform=transform,
            download=download,
        )

        test_dataset = FashionMNIST(
            root=data_root,
            train=False,
            transform=transform,
            download=download,
        )

    except RuntimeError as exc:
        # TorchVision raises RuntimeError when the dataset
        # is unavailable and downloading is disabled.
        if (
            not download
            and "Dataset not found" in str(exc)
        ):
            raise FileNotFoundError(
                "Fashion-MNIST is unavailable in offline mode. "
                f"Dataset root: {data_root}. "
                "Run training with --download-data to "
                "retrieve missing dataset files."
            ) from exc

        # Preserve unrelated errors, including failures
        # while downloading or reading corrupted files.
        raise

    return train_dataset, test_dataset


def download_fashion_mnist(
    root: str | Path = DEFAULT_DATA_DIR,
    *,
    project_root: str | Path | None = None,
) -> tuple[FashionMNIST, FashionMNIST]:
    """Explicitly permit downloading Fashion-MNIST."""

    return load_fashion_mnist(
        root=root,
        transform=None,
        download=True,
        project_root=project_root,
    )


def split_train_validation(
    dataset: Dataset,
    validation_split: float,
    seed: int,
) -> tuple[Subset, Subset]:
    dataset_size = len(dataset)

    if dataset_size < 2:
        raise ValueError(
            "Dataset must contain at least two samples."
        )

    if not 0 < validation_split < 1:
        raise ValueError(
            "validation_split must be greater than 0 "
            "and less than 1."
        )

    validation_size = int(
        dataset_size * validation_split
    )

    if validation_size == 0:
        raise ValueError(
            "validation_split produces an empty "
            "validation dataset."
        )

    if validation_size >= dataset_size:
        raise ValueError(
            "validation_split produces an empty "
            "training dataset."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    shuffled_indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    validation_indices = shuffled_indices[
        :validation_size
    ]

    training_indices = shuffled_indices[
        validation_size:
    ]

    training_dataset = Subset(
        dataset,
        training_indices,
    )

    validation_dataset = Subset(
        dataset,
        validation_indices,
    )

    return training_dataset, validation_dataset


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def create_data_loaders(
    config: TrainingConfig,
    root: str | Path = DEFAULT_DATA_DIR,
    *,
    download: bool = False,
) -> DataLoaders:
    """Create reproducible Fashion-MNIST DataLoaders."""

    # Validate the dataset download policy.
    if type(download) is not bool:
        raise TypeError(
            "download must be a boolean."
        )

    # Construct the shared preprocessing pipeline.
    transform = build_image_transform()

    # Load the official training and test datasets.
    full_training_dataset, test_dataset = load_fashion_mnist(
        root=root,
        transform=transform,
        download=download,
    )

    # Split the official training dataset into
    # training and validation subsets.
    training_dataset, validation_dataset = (
        split_train_validation(
            full_training_dataset,
            validation_split=(
                config.validation.validation_split
            ),
            seed=config.training.seed,
        )
    )

    # Create a dedicated generator for reproducible
    # training batch shuffling.
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(
        config.training.seed
    )

    # Create the training DataLoader.
    train_loader = DataLoader(
        training_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        generator=shuffle_generator,
    )

    # Create the validation DataLoader.
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create the independent test DataLoader.
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Return all three DataLoaders.
    return DataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
    )


def summarize_raw_dataset(
    dataset: FashionMNIST,
) -> dict[str, Any]:
    image, label = dataset[0]

    return {
        "samples": len(dataset),
        "data_shape": tuple(dataset.data.shape),
        "target_shape": tuple(dataset.targets.shape),
        "data_dtype": str(dataset.data.dtype),
        "target_dtype": str(dataset.targets.dtype),
        "pixel_min": dataset.data.min().item(),
        "pixel_max": dataset.data.max().item(),
        "first_image_type": type(image).__name__,
        "first_image_size": image.size,
        "first_image_mode": image.mode,
        "first_label": label,
        "first_class_name": get_class_name(label),
    }


def summarize_processed_sample(
    dataset,
    index: int = 0,
) -> dict[str, Any]:
    image, label = dataset[index]

    return {
        "index": index,
        "image_shape": tuple(image.shape),
        "image_dtype": str(image.dtype),
        "pixel_min": image.min().item(),
        "pixel_max": image.max().item(),
        "label": label,
        "class_name": get_class_name(label),
    }
