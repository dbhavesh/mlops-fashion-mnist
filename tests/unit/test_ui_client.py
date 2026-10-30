"""Offline tests for FastAPI HTTP client behavior."""

from __future__ import annotations

import pytest
import requests

from fashion_mnist_mlops.ui_client import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    PredictionClientError,
    PredictionFailure,
    PredictionResult,
    predict_image,
)


RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"


class FakeResponse:
    def __init__(self, *, status_code=200, payload=None, json_error=None):
        self.status_code = status_code
        self.payload = payload
        self.json_error = json_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


def successful_payload(**overrides):
    payload = {
        "class_index": 8,
        "class_name": "Bag",
        "model_run_id": RUN_ID,
    }
    payload.update(overrides)
    return payload


def test_predict_posts_exact_multipart_file_to_configured_backend():
    observed = []

    def fake_post(url, **kwargs):
        observed.append((url, kwargs))
        return FakeResponse(payload=successful_payload())

    result = predict_image(
        base_url="http://api:8000",
        filename="bag.png",
        content=b"png-bytes",
        content_type="image/png",
        post=fake_post,
    )

    assert result == PredictionResult(
        class_index=8,
        class_name="Bag",
        model_run_id=RUN_ID,
    )
    assert observed == [
        (
            "http://api:8000/predict",
            {
                "files": {"file": ("bag.png", b"png-bytes", "image/png")},
                "timeout": DEFAULT_REQUEST_TIMEOUT_SECONDS,
            },
        )
    ]


def test_jpeg_multipart_preserves_filename_and_media_type():
    observed = []

    def fake_post(url, **kwargs):
        observed.append(kwargs["files"]["file"])
        return FakeResponse(payload=successful_payload())

    predict_image(
        base_url="http://127.0.0.1:8000",
        filename="coat.jpeg",
        content=b"jpeg-bytes",
        content_type="image/jpeg",
        post=fake_post,
    )

    assert observed == [("coat.jpeg", b"jpeg-bytes", "image/jpeg")]


@pytest.mark.parametrize("status_code", [400, 413, 415, 422, 500, 503, 418])
def test_non_200_response_preserves_status_without_exposing_body(status_code):
    with pytest.raises(PredictionClientError) as caught:
        predict_image(
            base_url="http://api:8000",
            filename="shirt.png",
            content=b"content",
            content_type="image/png",
            post=lambda *args, **kwargs: FakeResponse(
                status_code=status_code,
                payload={"detail": "secret infrastructure message"},
            ),
        )

    assert caught.value.failure == PredictionFailure.HTTP
    assert caught.value.status_code == status_code
    assert str(caught.value) == "FastAPI prediction request failed."
    assert "secret" not in str(caught.value)


def test_timeout_failure_is_distinguished_and_sanitized():
    def fail(*args, **kwargs):
        raise requests.Timeout("socket timeout details that should not leak")

    with pytest.raises(PredictionClientError) as caught:
        predict_image(
            base_url="http://api:8000",
            filename="shirt.png",
            content=b"content",
            content_type="image/png",
            post=fail,
        )

    assert caught.value.failure == PredictionFailure.TIMEOUT
    assert caught.value.status_code is None
    assert str(caught.value) == "FastAPI prediction request timed out."
    assert "socket" not in str(caught.value)


def test_connection_failure_is_distinguished_and_sanitized():
    def fail(*args, **kwargs):
        raise requests.ConnectionError("connection details that should not leak")

    with pytest.raises(PredictionClientError) as caught:
        predict_image(
            base_url="http://api:8000",
            filename="shirt.png",
            content=b"content",
            content_type="image/png",
            post=fail,
        )

    assert caught.value.failure == PredictionFailure.CONNECTION
    assert caught.value.status_code is None
    assert str(caught.value) == "FastAPI prediction connection failed."
    assert "connection details" not in str(caught.value)


def test_other_transport_failure_is_sanitized():
    def fail(*args, **kwargs):
        raise requests.RequestException("transport internals that should not leak")

    with pytest.raises(PredictionClientError) as caught:
        predict_image(
            base_url="http://api:8000",
            filename="shirt.png",
            content=b"content",
            content_type="image/png",
            post=fail,
        )

    assert caught.value.failure == PredictionFailure.TRANSPORT
    assert caught.value.status_code is None
    assert str(caught.value) == "FastAPI prediction request failed."
    assert "transport internals" not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"class_index": 1, "class_name": "Trouser"},
        successful_payload(class_index=True),
        successful_payload(class_index="8"),
        successful_payload(class_index=-1),
        successful_payload(class_index=10),
        successful_payload(class_name=""),
        successful_payload(class_name=None),
        successful_payload(model_run_id=""),
        successful_payload(model_run_id=None),
    ],
)
def test_malformed_or_incomplete_success_payload_is_rejected(payload):
    with pytest.raises(PredictionClientError) as caught:
        predict_image(
            base_url="http://api:8000",
            filename="bag.png",
            content=b"content",
            content_type="image/png",
            post=lambda *args, **kwargs: FakeResponse(payload=payload),
        )

    assert caught.value.failure == PredictionFailure.INVALID_RESPONSE
    assert caught.value.status_code is None
    assert str(caught.value) == "FastAPI returned an invalid prediction response."


def test_non_json_success_response_is_rejected():
    with pytest.raises(PredictionClientError) as caught:
        predict_image(
            base_url="http://api:8000",
            filename="bag.png",
            content=b"content",
            content_type="image/png",
            post=lambda *args, **kwargs: FakeResponse(
                json_error=ValueError("not json")
            ),
        )

    assert caught.value.failure == PredictionFailure.INVALID_RESPONSE
    assert str(caught.value) == "FastAPI returned an invalid prediction response."


@pytest.mark.parametrize(
    ("filename", "content", "content_type"),
    [
        ("", b"content", "image/png"),
        ("shirt.png", b"", "image/png"),
        ("shirt.gif", b"content", "image/gif"),
    ],
)
def test_invalid_local_upload_contract_fails_before_network(
    filename, content, content_type
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("network must not be called")

    with pytest.raises(PredictionClientError) as caught:
        predict_image(
            base_url="http://api:8000",
            filename=filename,
            content=content,
            content_type=content_type,
            post=fail_if_called,
        )

    assert caught.value.failure == PredictionFailure.INVALID_UPLOAD
    assert caught.value.status_code is None


def test_timeout_must_be_positive_before_network():
    with pytest.raises(ValueError, match="greater than zero"):
        predict_image(
            base_url="http://api:8000",
            filename="shirt.png",
            content=b"content",
            content_type="image/png",
            timeout_seconds=0,
            post=lambda *args, **kwargs: None,
        )
