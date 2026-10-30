"""Validated training-complete payloads and single-attempt HTTP delivery."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol
from urllib.parse import urlsplit

import requests

from .config_validation import REQUIRED_MINIO_BUCKET
from .model_keys import build_model_object_key


WEBHOOK_SCHEMA_VERSION = 1
WEBHOOK_EVENT = "training.completed"
WEBHOOK_STATUS = "success"
DEFAULT_WEBHOOK_TIMEOUT_SECONDS = 5.0
MAX_WEBHOOK_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class WebhookSettings:
    """Externally supplied webhook delivery settings."""

    url: str | None
    timeout_seconds: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS


@dataclass(frozen=True)
class TrainingCompletedPayload:
    """Versioned success payload emitted only after the training lifecycle succeeds."""

    mlflow_run_id: str
    minio_bucket: str
    minio_object_key: str
    validation_accuracy: float
    schema_version: int = WEBHOOK_SCHEMA_VERSION
    event: str = WEBHOOK_EVENT
    status: str = WEBHOOK_STATUS

    def __post_init__(self) -> None:
        if self.schema_version != WEBHOOK_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {WEBHOOK_SCHEMA_VERSION}."
            )
        if self.event != WEBHOOK_EVENT:
            raise ValueError(f"event must be {WEBHOOK_EVENT!r}.")
        if self.status != WEBHOOK_STATUS:
            raise ValueError(f"status must be {WEBHOOK_STATUS!r}.")

        expected_key = build_model_object_key(self.mlflow_run_id)

        if self.minio_bucket != REQUIRED_MINIO_BUCKET:
            raise ValueError(
                f"minio_bucket must be exactly {REQUIRED_MINIO_BUCKET!r}."
            )
        if self.minio_object_key != expected_key:
            raise ValueError(
                "minio_object_key must be the canonical key for mlflow_run_id: "
                f"{expected_key!r}."
            )

        if isinstance(self.validation_accuracy, bool) or not isinstance(
            self.validation_accuracy, (int, float)
        ):
            raise ValueError("validation_accuracy must be a finite number from 0.0 to 1.0.")

        accuracy = float(self.validation_accuracy)
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError("validation_accuracy must be a finite number from 0.0 to 1.0.")

    def as_json_dict(self) -> dict[str, object]:
        """Return the stable JSON-ready representation of this payload."""
        return asdict(self)


def _validate_webhook_url(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()
    if not value:
        return None

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("WEBHOOK_URL is not a valid absolute HTTP(S) URL.") from exc

    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ValueError("WEBHOOK_URL must be an absolute http:// or https:// URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("WEBHOOK_URL must not embed credentials.")
    if parsed.fragment:
        raise ValueError("WEBHOOK_URL must not contain a URL fragment.")

    return value


def _parse_timeout(value: str | None) -> float:
    if value is None or not value.strip():
        return DEFAULT_WEBHOOK_TIMEOUT_SECONDS

    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError("WEBHOOK_TIMEOUT_SECONDS must be numeric.") from exc

    if not math.isfinite(timeout) or not 0.0 < timeout <= MAX_WEBHOOK_TIMEOUT_SECONDS:
        raise ValueError(
            "WEBHOOK_TIMEOUT_SECONDS must be greater than 0 and at most "
            f"{MAX_WEBHOOK_TIMEOUT_SECONDS:g}."
        )
    return timeout


def load_webhook_settings(
    environ: Mapping[str, str] | None = None,
) -> WebhookSettings:
    """Load webhook settings without sending any network request."""
    env = os.environ if environ is None else environ
    return WebhookSettings(
        url=_validate_webhook_url(env.get("WEBHOOK_URL")),
        timeout_seconds=_parse_timeout(env.get("WEBHOOK_TIMEOUT_SECONDS")),
    )


class WebhookDeliveryError(RuntimeError):
    """Base class for a configured webhook delivery failure."""


class WebhookTimeoutError(WebhookDeliveryError):
    """Raised when the webhook request exceeds its configured timeout."""


class WebhookHTTPError(WebhookDeliveryError):
    """Raised when the receiver responds outside the 2xx success range."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"Webhook receiver returned HTTP {status_code}.")


@dataclass(frozen=True)
class WebhookDeliveryReceipt:
    """Minimal evidence that a receiver accepted the webhook request."""

    status_code: int


class _Response(Protocol):
    status_code: int


class _Poster(Protocol):
    def post(self, url: str, **kwargs: object) -> _Response: ...


def send_training_completed_webhook(
    payload: TrainingCompletedPayload,
    *,
    settings: WebhookSettings | None = None,
    session: _Poster | None = None,
) -> WebhookDeliveryReceipt | None:
    """POST one validated success payload to the configured webhook receiver.

    A disabled webhook is an explicit no-op and returns ``None``. When enabled,
    the function makes exactly one HTTP request and never retries automatically;
    retry policy belongs to a higher-level lifecycle decision because duplicate
    webhook deliveries may have side effects at the receiver.
    """
    if not isinstance(payload, TrainingCompletedPayload):
        raise TypeError("payload must be a TrainingCompletedPayload instance.")

    effective = load_webhook_settings() if settings is None else settings
    if not isinstance(effective, WebhookSettings):
        raise TypeError("settings must be a WebhookSettings instance.")
    if effective.url is None:
        return None

    poster: _Poster = requests if session is None else session
    try:
        response = poster.post(
            effective.url,
            json=payload.as_json_dict(),
            timeout=effective.timeout_seconds,
            allow_redirects=False,
            headers={"Accept": "application/json"},
        )
    except requests.Timeout as exc:
        raise WebhookTimeoutError(
            "Webhook delivery timed out before the receiver confirmed success."
        ) from exc
    except requests.RequestException as exc:
        raise WebhookDeliveryError(
            "Webhook delivery failed due to an HTTP transport error."
        ) from exc

    status_code = response.status_code
    if type(status_code) is not int:
        raise WebhookDeliveryError("Webhook receiver returned an invalid HTTP status.")
    if not 200 <= status_code < 300:
        raise WebhookHTTPError(status_code)

    return WebhookDeliveryReceipt(status_code=status_code)
