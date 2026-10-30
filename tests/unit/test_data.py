from PIL import Image
import pytest
import torch
from torch.utils.data import Dataset

from fashion_mnist_mlops.labels import (
    FASHION_MNIST_CLASSES,
    NUM_CLASSES,
    get_class_name,
)
from fashion_mnist_mlops.preprocessing import (
    build_image_transform,
    preprocess_image_for_inference,
)
from fashion_mnist_mlops.data import (
    create_data_loaders,
    split_train_validation,
    summarize_processed_sample,
    summarize_raw_dataset,
)
from fashion_mnist_mlops.preprocessing import (
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)
from fashion_mnist_mlops.config import (
    TrainingConfig,
    TrainingSettings,
    ValidationSettings,
)


class SyntheticFashionDataset(Dataset):
    def __init__(
        self,
        size: int,
        transform=None,
    ):
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = Image.new(
            "L",
            (28, 28),
            color=index % 256,
        )

        label = index % 10

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class SyntheticRawFashionDataset:
    def __init__(self):
        self.data = torch.stack(
            [
                torch.zeros(
                    (28, 28),
                    dtype=torch.uint8,
                ),
                torch.full(
                    (28, 28),
                    255,
                    dtype=torch.uint8,
                ),
            ]
        )

        self.targets = torch.tensor(
            [9, 1],
            dtype=torch.int64,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(
            self.data[index].numpy(),
            mode="L",
        )

        return image, self.targets[index].item()


def make_test_config(
    *,
    batch_size: int = 8,
    validation_split: float = 0.2,
    seed: int = 42,
) -> TrainingConfig:
    return TrainingConfig(
        training=TrainingSettings(
            epochs=1,
            batch_size=batch_size,
            learning_rate=0.001,
            optimizer="adam",
            loss="cross_entropy",
            seed=seed,
        ),
        validation=ValidationSettings(
            validation_split=validation_split,
        ),
    )


def test_class_mapping_contains_ten_classes():
    assert NUM_CLASSES == 10
    assert len(FASHION_MNIST_CLASSES) == 10


def test_class_mapping_has_expected_values():
    assert get_class_name(0) == "T-shirt/top"
    assert get_class_name(1) == "Trouser"
    assert get_class_name(7) == "Sneaker"
    assert get_class_name(9) == "Ankle boot"


@pytest.mark.parametrize(
    "class_index",
    [-1, 10],
)
def test_invalid_class_index_is_rejected(
    class_index,
):
    with pytest.raises(
        ValueError,
        match="between 0 and 9",
    ):
        get_class_name(class_index)


@pytest.mark.parametrize(
    "class_index",
    ["7", 7.0, True],
)
def test_invalid_class_index_type_is_rejected(
    class_index,
):
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        get_class_name(class_index)


def test_transform_produces_expected_shape_and_dtype():
    image = Image.new(
        "RGB",
        (100, 60),
        color=(100, 150, 200),
    )

    tensor = build_image_transform()(image)

    assert tensor.shape == (1, 28, 28)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_black_image_uses_expected_normalization():
    image = Image.new(
        "L",
        (28, 28),
        color=0,
    )

    tensor = build_image_transform()(image)

    expected = (
        0.0 - NORMALIZATION_MEAN[0]
    ) / NORMALIZATION_STD[0]

    assert tensor.min().item() == pytest.approx(
        expected,
        abs=1e-6,
    )

    assert tensor.max().item() == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_white_image_uses_expected_normalization():
    image = Image.new(
        "L",
        (28, 28),
        color=255,
    )

    tensor = build_image_transform()(image)

    expected = (
        1.0 - NORMALIZATION_MEAN[0]
    ) / NORMALIZATION_STD[0]

    assert tensor.min().item() == pytest.approx(
        expected,
        abs=1e-6,
    )

    assert tensor.max().item() == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_preprocessing_is_deterministic():
    image = Image.new(
        "RGB",
        (50, 75),
        color=(80, 120, 160),
    )

    transform = build_image_transform()

    first = transform(image)
    second = transform(image)

    assert torch.equal(first, second)


def test_split_sizes_are_correct():
    dataset = list(range(100))

    train_dataset, validation_dataset = (
        split_train_validation(
            dataset,
            validation_split=0.2,
            seed=42,
        )
    )

    assert len(train_dataset) == 80
    assert len(validation_dataset) == 20


def test_split_has_no_overlap_and_full_coverage():
    dataset = list(range(100))

    train_dataset, validation_dataset = (
        split_train_validation(
            dataset,
            validation_split=0.2,
            seed=42,
        )
    )

    train_indices = set(
        train_dataset.indices
    )

    validation_indices = set(
        validation_dataset.indices
    )

    assert train_indices.isdisjoint(
        validation_indices
    )

    assert (
        train_indices | validation_indices
        == set(range(100))
    )


def test_split_is_reproducible_for_same_seed():
    dataset = list(range(100))

    train_a, validation_a = (
        split_train_validation(
            dataset,
            validation_split=0.2,
            seed=42,
        )
    )

    train_b, validation_b = (
        split_train_validation(
            dataset,
            validation_split=0.2,
            seed=42,
        )
    )

    assert train_a.indices == train_b.indices

    assert (
        validation_a.indices
        == validation_b.indices
    )


def test_different_seed_changes_split():
    dataset = list(range(100))

    train_a, validation_a = (
        split_train_validation(
            dataset,
            validation_split=0.2,
            seed=42,
        )
    )

    train_b, validation_b = (
        split_train_validation(
            dataset,
            validation_split=0.2,
            seed=123,
        )
    )

    assert train_a.indices != train_b.indices

    assert (
        validation_a.indices
        != validation_b.indices
    )


@pytest.mark.parametrize(
    "validation_split",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_validation_split_is_rejected(
    validation_split,
):
    with pytest.raises(
        ValueError,
        match="validation_split",
    ):
        split_train_validation(
            list(range(100)),
            validation_split=validation_split,
            seed=42,
        )


def test_inference_preprocessing_matches_training():
    image = Image.new(
        "RGB",
        (80, 120),
        color=(40, 100, 180),
    )

    training_tensor = build_image_transform()(
        image
    )

    inference_tensor = (
        preprocess_image_for_inference(image)
    )

    assert inference_tensor.shape == (
        1,
        1,
        28,
        28,
    )

    assert torch.equal(
        training_tensor,
        inference_tensor.squeeze(0),
    )


def test_invalid_inference_input_is_rejected():
    with pytest.raises(
        TypeError,
        match="PIL.Image.Image",
    ):
        preprocess_image_for_inference(
            "not-an-image"
        )


def test_data_loaders_use_configured_pipeline(
    monkeypatch,
):
    def fake_load_fashion_mnist(
        root,
        *,
        transform,
        download,
    ):
        return (
            SyntheticFashionDataset(
                size=100,
                transform=transform,
            ),
            SyntheticFashionDataset(
                size=20,
                transform=transform,
            ),
        )

    monkeypatch.setattr(
        (
            "fashion_mnist_mlops.data."
            "load_fashion_mnist"
        ),
        fake_load_fashion_mnist,
    )

    config = make_test_config(
        batch_size=8,
        validation_split=0.2,
        seed=42,
    )

    loaders = create_data_loaders(config)

    assert len(loaders.train.dataset) == 80
    assert len(loaders.validation.dataset) == 20
    assert len(loaders.test.dataset) == 20

    assert loaders.train.batch_size == 8
    assert loaders.validation.batch_size == 8
    assert loaders.test.batch_size == 8

    images, labels = next(
        iter(loaders.train)
    )

    assert images.shape == (
        8,
        1,
        28,
        28,
    )

    assert labels.shape == (8,)

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_raw_dataset_summary():
    dataset = SyntheticRawFashionDataset()

    summary = summarize_raw_dataset(
        dataset
    )

    assert summary["samples"] == 2
    assert summary["data_shape"] == (
        2,
        28,
        28,
    )
    assert summary["target_shape"] == (2,)
    assert summary["data_dtype"] == "torch.uint8"
    assert summary["target_dtype"] == "torch.int64"
    assert summary["pixel_min"] == 0
    assert summary["pixel_max"] == 255
    assert summary["first_image_size"] == (
        28,
        28,
    )
    assert summary["first_image_mode"] == "L"
    assert summary["first_label"] == 9
    assert (
        summary["first_class_name"]
        == "Ankle boot"
    )


def test_processed_sample_summary():
    dataset = SyntheticFashionDataset(
        size=10,
        transform=build_image_transform(),
    )

    summary = summarize_processed_sample(
        dataset,
        index=7,
    )

    assert summary["index"] == 7
    assert summary["image_shape"] == (
        1,
        28,
        28,
    )
    assert summary["image_dtype"] == (
        "torch.float32"
    )
    assert summary["label"] == 7
    assert summary["class_name"] == "Sneaker"
