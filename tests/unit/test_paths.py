import os
import subprocess
import sys
from pathlib import Path

import pytest

from fashion_mnist_mlops import data as datasets
from fashion_mnist_mlops.config import (
    load_training_config,
)
from fashion_mnist_mlops.paths import (
    PROJECT_ROOT_ENV,
    get_project_root,
    resolve_project_path,
)
from fashion_mnist_mlops.config_validation import (
    ConfigurationError,
)


def test_default_config_works_outside_project_root(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    config = load_training_config()

    assert config.training.seed == 42
    assert config.training.batch_size == 64


def test_relative_paths_use_explicit_project_root(tmp_path):
    result = resolve_project_path(
        "artifacts/models/model.pt",
        project_root=tmp_path,
    )

    assert result == (
        tmp_path / "artifacts/models/model.pt"
    ).resolve()


def test_absolute_paths_are_preserved(tmp_path):
    absolute = tmp_path / "external-model.pt"

    assert resolve_project_path(absolute) == absolute.resolve()


def test_explicit_project_root_overrides_environment(
    tmp_path,
    monkeypatch,
):
    environment_root = tmp_path / "environment"
    explicit_root = tmp_path / "explicit"

    environment_root.mkdir()
    explicit_root.mkdir()

    monkeypatch.setenv(
        PROJECT_ROOT_ENV,
        str(environment_root),
    )

    assert get_project_root() == environment_root.resolve()

    assert get_project_root(
        explicit_root
    ) == explicit_root.resolve()


def test_missing_project_root_produces_clear_error(
    monkeypatch,
):
    import fashion_mnist_mlops.paths as paths

    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)

    monkeypatch.setattr(
        paths,
        "_discover_source_project_root",
        lambda: None,
    )

    with pytest.raises(
        ConfigurationError,
        match="Unable to discover the project root",
    ):
        get_project_root()


def test_dataset_uses_project_root(
    tmp_path,
    monkeypatch,
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
        root="data",
        download=False,
        project_root=tmp_path,
    )

    expected = (tmp_path / "data").resolve()

    assert observed == [
        (expected, True, False),
        (expected, False, False),
    ]


def test_subprocess_loads_config_from_external_cwd(
    tmp_path,
):
    project_root = get_project_root()

    script = (
        "from fashion_mnist_mlops.config import "
        "load_training_config; "
        "print(load_training_config().training.seed)"
    )

    environment = os.environ.copy()
    environment.pop(PROJECT_ROOT_ENV, None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "42"
    assert project_root.is_dir()
