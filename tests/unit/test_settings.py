import pytest

from fashion_mnist_mlops.config import (
    ConfigurationError,
    load_infrastructure_config,
    load_training_config,
    validate_infrastructure_config,
)


def test_default_training_config_loads():
    config = load_training_config()

    assert config.training.epochs == 10
    assert config.training.batch_size == 64
    assert config.training.learning_rate == 0.001
    assert config.training.optimizer == "adam"
    assert config.training.loss == "cross_entropy"
    assert config.training.seed == 42
    assert config.validation.validation_split == 0.1


def test_invalid_epochs_are_rejected(tmp_path):
    config_path = tmp_path / "training.toml"

    config_path.write_text(
        """
[training]
epochs = -1
batch_size = 64
learning_rate = 0.001
optimizer = "adam"
loss = "cross_entropy"
seed = 42

[validation]
validation_split = 0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="epochs",
    ):
        load_training_config(config_path)


def test_missing_training_setting_is_rejected(tmp_path):
    config_path = tmp_path / "training.toml"

    config_path.write_text(
        """
[training]
epochs = 10
batch_size = 64
optimizer = "adam"
loss = "cross_entropy"
seed = 42

[validation]
validation_split = 0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="learning_rate",
    ):
        load_training_config(config_path)


def test_unsupported_optimizer_is_rejected(tmp_path):
    config_path = tmp_path / "training.toml"

    config_path.write_text(
        """
[training]
epochs = 10
batch_size = 64
learning_rate = 0.001
optimizer = "banana"
loss = "cross_entropy"
seed = 42

[validation]
validation_split = 0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="optimizer",
    ):
        load_training_config(config_path)


def test_invalid_validation_split_is_rejected(tmp_path):
    config_path = tmp_path / "training.toml"

    config_path.write_text(
        """
[training]
epochs = 10
batch_size = 64
learning_rate = 0.001
optimizer = "adam"
loss = "cross_entropy"
seed = 42

[validation]
validation_split = 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="validation_split",
    ):
        load_training_config(config_path)


def test_wrong_minio_bucket_is_rejected(monkeypatch):
    monkeypatch.setenv(
        "MINIO_BUCKET",
        "fashion-mnist-bucket",
    )

    with pytest.raises(
        ConfigurationError,
        match="fashion-minst-bucket",
    ):
        load_infrastructure_config()


def test_incomplete_minio_credentials_are_rejected(
    monkeypatch,
):
    monkeypatch.setenv(
        "MINIO_ACCESS_KEY",
        "test-access-key",
    )
    monkeypatch.delenv(
        "MINIO_SECRET_KEY",
        raising=False,
    )

    with pytest.raises(
        ConfigurationError,
        match="supplied together",
    ):
        load_infrastructure_config()


def test_invalid_model_object_key_is_rejected(
    monkeypatch,
):
    monkeypatch.setenv(
        "MODEL_OBJECT_KEY",
        "latest-model.pt",
    )

    with pytest.raises(
        ConfigurationError,
        match="MODEL_OBJECT_KEY",
    ):
        load_infrastructure_config()


def test_minio_credentials_can_be_required(monkeypatch):
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    config = load_infrastructure_config()

    with pytest.raises(
        ConfigurationError,
        match="MinIO credentials are required",
    ):
        validate_infrastructure_config(
            config,
            require_minio_credentials=True,
        )
