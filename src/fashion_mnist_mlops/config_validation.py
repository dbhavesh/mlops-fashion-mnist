from urllib.parse import urlparse

from .model_keys import validate_model_object_key


REQUIRED_MINIO_BUCKET = "fashion-minst-bucket"

SUPPORTED_OPTIMIZERS = {"adam", "sgd"}
SUPPORTED_LOSSES = {"cross_entropy"}


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _require_non_empty_string(
    value: object,
    setting_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            f"{setting_name} must be a non-empty string."
        )

    return value.strip()


def _validate_http_url(
    value: str,
    setting_name: str,
) -> None:
    parsed = urlparse(value)

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
    ):
        raise ConfigurationError(
            f"{setting_name} must be a valid HTTP or HTTPS URL."
        )


def validate_training_config(raw_config: dict) -> None:
    if not isinstance(raw_config, dict):
        raise ConfigurationError(
            "Training configuration must be a TOML table."
        )

    if "training" not in raw_config:
        raise ConfigurationError(
            "Missing required [training] section."
        )

    if "validation" not in raw_config:
        raise ConfigurationError(
            "Missing required [validation] section."
        )

    training = raw_config["training"]
    validation = raw_config["validation"]

    if not isinstance(training, dict):
        raise ConfigurationError(
            "[training] must be a TOML table."
        )

    if not isinstance(validation, dict):
        raise ConfigurationError(
            "[validation] must be a TOML table."
        )

    required_training_settings = {
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
        "loss",
        "seed",
    }

    missing_training_settings = (
        required_training_settings - training.keys()
    )

    if missing_training_settings:
        missing = ", ".join(
            sorted(missing_training_settings)
        )
        raise ConfigurationError(
            f"Missing required training setting(s): {missing}."
        )

    if "validation_split" not in validation:
        raise ConfigurationError(
            "Missing required validation setting: "
            "validation_split."
        )

    epochs = training["epochs"]
    if not _is_integer(epochs) or epochs <= 0:
        raise ConfigurationError(
            "training.epochs must be an integer greater than 0."
        )

    batch_size = training["batch_size"]
    if not _is_integer(batch_size) or batch_size <= 0:
        raise ConfigurationError(
            "training.batch_size must be an integer "
            "greater than 0."
        )

    learning_rate = training["learning_rate"]
    if (
        not _is_number(learning_rate)
        or learning_rate <= 0
    ):
        raise ConfigurationError(
            "training.learning_rate must be a number "
            "greater than 0."
        )

    optimizer = _require_non_empty_string(
        training["optimizer"],
        "training.optimizer",
    ).lower()

    if optimizer not in SUPPORTED_OPTIMIZERS:
        supported = ", ".join(
            sorted(SUPPORTED_OPTIMIZERS)
        )
        raise ConfigurationError(
            f"training.optimizer must be one of: "
            f"{supported}."
        )

    loss = _require_non_empty_string(
        training["loss"],
        "training.loss",
    ).lower()

    if loss not in SUPPORTED_LOSSES:
        supported = ", ".join(
            sorted(SUPPORTED_LOSSES)
        )
        raise ConfigurationError(
            f"training.loss must be one of: {supported}."
        )

    seed = training["seed"]
    if not _is_integer(seed) or seed < 0:
        raise ConfigurationError(
            "training.seed must be an integer "
            "greater than or equal to 0."
        )

    validation_split = validation["validation_split"]

    if (
        not _is_number(validation_split)
        or not 0 < validation_split < 1
    ):
        raise ConfigurationError(
            "validation.validation_split must be "
            "greater than 0 and less than 1."
        )


def validate_infrastructure_config(
    config,
    *,
    require_minio_credentials: bool = False,
    require_webhook_url: bool = False,
    require_model_object_key: bool = False,
) -> None:
    tracking_uri = _require_non_empty_string(
        config.mlflow.tracking_uri,
        "MLFLOW_TRACKING_URI",
    )
    _validate_http_url(
        tracking_uri,
        "MLFLOW_TRACKING_URI",
    )

    _require_non_empty_string(
        config.mlflow.experiment_name,
        "MLFLOW_EXPERIMENT_NAME",
    )

    minio_endpoint = _require_non_empty_string(
        config.minio.endpoint,
        "MINIO_ENDPOINT",
    )

    if "://" in minio_endpoint:
        raise ConfigurationError(
            "MINIO_ENDPOINT must not contain a URL scheme. "
            "Use a value such as 'localhost:9000' "
            "instead of 'http://localhost:9000'."
        )

    if config.minio.bucket != REQUIRED_MINIO_BUCKET:
        raise ConfigurationError(
            "MINIO_BUCKET must be exactly "
            f"'{REQUIRED_MINIO_BUCKET}'."
        )

    access_key = config.minio.access_key
    secret_key = config.minio.secret_key

    if bool(access_key) != bool(secret_key):
        raise ConfigurationError(
            "MINIO_ACCESS_KEY and MINIO_SECRET_KEY "
            "must be supplied together."
        )

    if require_minio_credentials and not (
        access_key and secret_key
    ):
        raise ConfigurationError(
            "MinIO credentials are required for this operation. "
            "Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY."
        )

    if not isinstance(config.minio.secure, bool):
        raise ConfigurationError(
            "MINIO_SECURE must resolve to a boolean value."
        )

    webhook_url = config.webhook.url

    if webhook_url is not None:
        _validate_http_url(
            webhook_url,
            "TRAINING_WEBHOOK_URL",
        )

    if require_webhook_url and webhook_url is None:
        raise ConfigurationError(
            "TRAINING_WEBHOOK_URL is required "
            "for this operation."
        )

    object_key = config.model.object_key

    if object_key is not None:
        try:
            validate_model_object_key(object_key)
        except ValueError as exc:
            raise ConfigurationError(
                f"MODEL_OBJECT_KEY is invalid: {exc}"
            ) from exc

    if require_model_object_key and object_key is None:
        raise ConfigurationError(
            "MODEL_OBJECT_KEY is required "
            "for this operation."
        )

    fastapi_url = _require_non_empty_string(
        config.services.fastapi_base_url,
        "FASTAPI_BASE_URL",
    )
    _validate_http_url(
        fastapi_url,
        "FASTAPI_BASE_URL",
    )
