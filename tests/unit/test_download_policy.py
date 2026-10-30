import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from fashion_mnist_mlops.config import (
    load_training_config,
)


datasets = importlib.import_module(
    "fashion_mnist_mlops.data"
)

loaders_module = importlib.import_module(
    "fashion_mnist_mlops.data"
)

cli = importlib.import_module(
    "fashion_mnist_mlops.train"
)


@pytest.mark.parametrize("download", [False, True])
def test_dataset_receives_download_flag(
    tmp_path,
    monkeypatch,
    download,
):
    observed = []

    class FakeFashionMNIST:
        def __init__(
            self,
            *,
            root,
            train,
            transform,
            download,
        ):
            observed.append(
                (Path(root), train, download)
            )

    monkeypatch.setattr(
        datasets,
        "FashionMNIST",
        FakeFashionMNIST,
    )

    datasets.load_fashion_mnist(
        root=tmp_path,
        download=download,
    )

    assert observed == [
        (tmp_path.resolve(), True, download),
        (tmp_path.resolve(), False, download),
    ]


def test_missing_dataset_gives_clear_offline_error(
    tmp_path,
    monkeypatch,
):
    class MissingFashionMNIST:
        def __init__(self, **kwargs):
            raise RuntimeError(
                "Dataset not found. "
                "You can use download=True to download it"
            )

    monkeypatch.setattr(
        datasets,
        "FashionMNIST",
        MissingFashionMNIST,
    )

    with pytest.raises(
        FileNotFoundError,
        match="--download-data",
    ) as exc_info:
        datasets.load_fashion_mnist(
            root=tmp_path,
            download=False,
        )

    assert str(tmp_path) in str(exc_info.value)


def test_unrelated_runtime_errors_are_preserved(
    tmp_path,
    monkeypatch,
):
    class BrokenFashionMNIST:
        def __init__(self, **kwargs):
            raise RuntimeError("Corrupted tensor data")

    monkeypatch.setattr(
        datasets,
        "FashionMNIST",
        BrokenFashionMNIST,
    )

    with pytest.raises(
        RuntimeError,
        match="Corrupted tensor data",
    ):
        datasets.load_fashion_mnist(
            root=tmp_path,
            download=False,
        )


def test_invalid_download_type_is_rejected(tmp_path):
    with pytest.raises(
        TypeError,
        match="download must be a boolean",
    ):
        datasets.load_fashion_mnist(
            root=tmp_path,
            download="false",
        )


def test_explicit_download_helper_enables_download(
    tmp_path,
    monkeypatch,
):
    observed = []

    class FakeFashionMNIST:
        def __init__(self, **kwargs):
            observed.append(kwargs["download"])

    monkeypatch.setattr(
        datasets,
        "FashionMNIST",
        FakeFashionMNIST,
    )

    datasets.download_fashion_mnist(
        root=tmp_path,
    )

    assert observed == [True, True]


@pytest.mark.parametrize("download", [False, True])
def test_dataloader_factory_forwards_policy(
    tmp_path,
    monkeypatch,
    download,
):
    observed = []

    def fake_load_fashion_mnist(
        *,
        root,
        transform,
        download,
    ):
        observed.append(download)

        train = TensorDataset(
            torch.zeros(20, 1, 28, 28),
            torch.zeros(20, dtype=torch.long),
        )

        test = TensorDataset(
            torch.zeros(10, 1, 28, 28),
            torch.zeros(10, dtype=torch.long),
        )

        return train, test

    monkeypatch.setattr(
        loaders_module,
        "load_fashion_mnist",
        fake_load_fashion_mnist,
    )

    config = load_training_config()

    result = loaders_module.create_data_loaders(
        config,
        root=tmp_path,
        download=download,
    )

    assert observed == [download]
    assert len(result.train.dataset) == 18
    assert len(result.validation.dataset) == 2
    assert len(result.test.dataset) == 10


@pytest.mark.parametrize(
    "flag, expected",
    [
        ("--offline", False),
        ("--download-data", True),
    ],
)
def test_cli_forwards_download_policy(
    monkeypatch,
    flag,
    expected,
):
    observed = {}

    class FakeEpochRecord:
        def as_tracking_metrics(self):
            return {
                "train_loss": 1.0,
                "val_accuracy": 0.2,
            }

    def fake_run_training(**kwargs):
        observed.update(kwargs)

        return SimpleNamespace(
            final_epoch=FakeEpochRecord(),
            artifact_path=Path("fake-model.pt"),
        )

    monkeypatch.setattr(
        cli,
        "run_training",
        fake_run_training,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["training", flag, "--epochs", "1"],
    )

    assert cli.main() == 0
    assert observed["download_data"] is expected


def test_cli_rejects_conflicting_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training",
            "--offline",
            "--download-data",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
