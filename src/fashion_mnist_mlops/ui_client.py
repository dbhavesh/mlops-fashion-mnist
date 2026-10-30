"""HTTP client boundary used by the Streamlit presentation layer.

The Streamlit UI sends uploaded image bytes to FastAPI and consumes the
backend's prediction contract. It never imports model, preprocessing, storage,
or PyTorch inference code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

import requests


PREDICT_PATH = "/predict"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15.0
_ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}


class PredictionFailure(str, Enum):
    """Sanitized failure categories that the presentation layer may interpret."""

    INVALID_UPLOAD = "invalid_upload"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    TRANSPORT = "transport"
    HTTP = "http"
    INVALID_RESPONSE = "invalid_response"


class PredictionClientError(RuntimeError):
    """A prediction request or response could not be safely consumed."""

    def __init__(
        self,
        message: str,
        *,
        failure: PredictionFailure,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Validated prediction values returned by the FastAPI backend."""

    class_index: int
    class_name: str
    model_run_id: str


class _ResponseLike(Protocol):
    status_code: int

    def json(self) -> Any: ...


_PostFunction = Callable[..., _ResponseLike]


def _invalid_response() -> PredictionClientError:
    return PredictionClientError(
        "FastAPI returned an invalid prediction response.",
        failure=PredictionFailure.INVALID_RESPONSE,
    )


def _validate_prediction_payload(payload: Any) -> PredictionResult:
    """Validate the small JSON contract without duplicating class mappings."""
    if not isinstance(payload, dict):
        raise _invalid_response()

    class_index = payload.get("class_index")
    class_name = payload.get("class_name")
    model_run_id = payload.get("model_run_id")

    if isinstance(class_index, bool) or not isinstance(class_index, int):
        raise _invalid_response()
    if not 0 <= class_index <= 9:
        raise _invalid_response()
    if not isinstance(class_name, str) or not class_name.strip():
        raise _invalid_response()
    if not isinstance(model_run_id, str) or not model_run_id.strip():
        raise _invalid_response()

    return PredictionResult(
        class_index=class_index,
        class_name=class_name,
        model_run_id=model_run_id,
    )


def predict_image(
    *,
    base_url: str,
    filename: str,
    content: bytes,
    content_type: str,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    post: _PostFunction | None = None,
) -> PredictionResult:
    """POST one image to FastAPI's `/predict` endpoint and validate the result.

    The client classifies failures without exposing raw exception text or HTTP
    response bodies. Streamlit converts these sanitized categories/status codes
    into user-facing guidance.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise PredictionClientError(
            "An uploaded filename is required.",
            failure=PredictionFailure.INVALID_UPLOAD,
        )
    if not isinstance(content, bytes) or not content:
        raise PredictionClientError(
            "The uploaded image is empty.",
            failure=PredictionFailure.INVALID_UPLOAD,
        )
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise PredictionClientError(
            "The uploaded image type is unsupported.",
            failure=PredictionFailure.INVALID_UPLOAD,
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero.")

    url = f"{base_url}{PREDICT_PATH}"
    send = requests.post if post is None else post

    try:
        response = send(
            url,
            files={"file": (filename, content, content_type)},
            timeout=timeout_seconds,
        )
    except requests.Timeout as exc:
        raise PredictionClientError(
            "FastAPI prediction request timed out.",
            failure=PredictionFailure.TIMEOUT,
        ) from exc
    except requests.ConnectionError as exc:
        raise PredictionClientError(
            "FastAPI prediction connection failed.",
            failure=PredictionFailure.CONNECTION,
        ) from exc
    except requests.RequestException as exc:
        raise PredictionClientError(
            "FastAPI prediction request failed.",
            failure=PredictionFailure.TRANSPORT,
        ) from exc

    if response.status_code != 200:
        raise PredictionClientError(
            "FastAPI prediction request failed.",
            failure=PredictionFailure.HTTP,
            status_code=response.status_code,
        )

    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        raise _invalid_response() from exc

    return _validate_prediction_payload(payload)
