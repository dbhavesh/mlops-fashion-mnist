from dataclasses import dataclass
from pathlib import Path
import tomllib
import os

from .paths import resolve_project_path
from .config_validation import (
    ConfigurationError,
    validate_training_config,
    validate_infrastructure_config,
)


DEFAULT_CONFIG_PATH = Path("configs/train.toml")


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    loss: str
    seed: int


@dataclass(frozen=True)
class ValidationSettings:
    validation_split: float


@dataclass(frozen=True)
class TrainingConfig:
    training: TrainingSettings
    validation: ValidationSettings


def load_training_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: str | Path | None = None,
) -> TrainingConfig:
    config_path = resolve_project_path(
        path,
        project_root=project_root,
    )

    try:
        with config_path.open("rb") as file:
            raw_config = tomllib.load(file)
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"Training configuration file not found: "
            f"{config_path}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(
            f"Invalid TOML in training configuration "
            f"'{config_path}': {exc}"
        ) from exc

    validate_training_config(raw_config)

    training = raw_config["training"]
    validation = raw_config["validation"]

    return TrainingConfig(
        training=TrainingSettings(
            epochs=training["epochs"],
            batch_size=training["batch_size"],
            learning_rate=training["learning_rate"],
            optimizer=training["optimizer"].lower(),
            loss=training["loss"].lower(),
            seed=training["seed"],
        ),
        validation=ValidationSettings(
            validation_split=validation["validation_split"],
        ),
    )


DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5000"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "fashion-mnist-classifier"

DEFAULT_MINIO_ENDPOINT = "localhost:9000"
DEFAULT_MINIO_BUCKET = "fashion-minst-bucket"

DEFAULT_FASTAPI_BASE_URL = "http://localhost:8000"


@dataclass(frozen=True)
class MLflowSettings:
    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class MinIOSettings:
    endpoint: str
    access_key: str | None
    secret_key: str | None
    secure: bool
    bucket: str


@dataclass(frozen=True)
class WebhookSettings:
    url: str | None


@dataclass(frozen=True)
class ModelSettings:
    object_key: str | None


@dataclass(frozen=True)
class ServiceSettings:
    fastapi_base_url: str


@dataclass(frozen=True)
class InfrastructureConfig:
    mlflow: MLflowSettings
    minio: MinIOSettings
    webhook: WebhookSettings
    model: ModelSettings
    services: ServiceSettings


def _get_optional_environment_variable(
    name: str,
) -> str | None:
    value = os.getenv(name)

    if value is None:
        return None

    stripped_value = value.strip()

    if not stripped_value:
        return None

    return stripped_value


def _get_boolean_environment_variable(
    name: str,
    default: bool = False,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    normalized_value = value.strip().lower()

    if normalized_value in {"1", "true", "yes", "on"}:
        return True

    if normalized_value in {"0", "false", "no", "off"}:
        return False

    raise ConfigurationError(
        f"{name} must be one of: "
        "true, false, 1, 0, yes, no, on, off."
    )


def load_infrastructure_config() -> InfrastructureConfig:
    config = InfrastructureConfig(
        mlflow=MLflowSettings(
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI",
                DEFAULT_MLFLOW_TRACKING_URI,
            ),
            experiment_name=os.getenv(
                "MLFLOW_EXPERIMENT_NAME",
                DEFAULT_MLFLOW_EXPERIMENT_NAME,
            ),
        ),
        minio=MinIOSettings(
            endpoint=os.getenv(
                "MINIO_ENDPOINT",
                DEFAULT_MINIO_ENDPOINT,
            ),
            access_key=_get_optional_environment_variable(
                "MINIO_ACCESS_KEY"
            ),
            secret_key=_get_optional_environment_variable(
                "MINIO_SECRET_KEY"
            ),
            secure=_get_boolean_environment_variable(
                "MINIO_SECURE",
                default=False,
            ),
            bucket=os.getenv(
                "MINIO_BUCKET",
                DEFAULT_MINIO_BUCKET,
            ),
        ),
        webhook=WebhookSettings(
            url=_get_optional_environment_variable(
                "TRAINING_WEBHOOK_URL"
            ),
        ),
        model=ModelSettings(
            object_key=_get_optional_environment_variable(
                "MODEL_OBJECT_KEY"
            ),
        ),
        services=ServiceSettings(
            fastapi_base_url=os.getenv(
                "FASTAPI_BASE_URL",
                DEFAULT_FASTAPI_BASE_URL,
            ),
        ),
    )

    validate_infrastructure_config(config)

    return config
