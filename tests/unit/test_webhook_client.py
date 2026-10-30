"""webhook HTTP client and local receiver tests."""

import json
import threading

import pytest
import requests

from fashion_mnist_mlops.webhook import (
    WebhookDeliveryError,
    WebhookHTTPError,
    WebhookTimeoutError,
    send_training_completed_webhook,
)
from fashion_mnist_mlops.webhook import (
    TrainingCompletedPayload,
    WebhookSettings,
)
from fashion_mnist_mlops.webhook_receiver import (
    create_webhook_receiver,
    validate_received_payload,
)

RUN_ID = "97461abfc4804cb0a2a1473c71737739"
BUCKET = "fashion-minst-bucket"
KEY = f"model/{RUN_ID}/model.pt"


def _payload():
    return TrainingCompletedPayload(
        mlflow_run_id=RUN_ID,
        minio_bucket=BUCKET,
        minio_object_key=KEY,
        validation_accuracy=0.8125,
    )


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeSession:
    def __init__(self, result=None, error=None):
        self.result = FakeResponse(204) if result is None else result
        self.error = error
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize("status", [200, 201, 202, 204, 299])
def test_all_2xx_statuses_are_accepted(status):
    session = FakeSession(result=FakeResponse(status))
    receipt = send_training_completed_webhook(
        _payload(),
        settings=WebhookSettings(url="http://127.0.0.1:8765/webhook", timeout_seconds=3),
        session=session,
    )
    assert receipt.status_code == status
    assert len(session.calls) == 1


def test_client_posts_exact_contract_with_timeout_and_redirects_disabled():
    session = FakeSession()
    payload = _payload()
    send_training_completed_webhook(
        payload,
        settings=WebhookSettings(url="http://127.0.0.1:8765/webhook", timeout_seconds=2.5),
        session=session,
    )
    assert session.calls == [
        (
            "http://127.0.0.1:8765/webhook",
            {
                "json": payload.as_json_dict(),
                "timeout": 2.5,
                "allow_redirects": False,
                "headers": {"Accept": "application/json"},
            },
        )
    ]


def test_disabled_webhook_is_noop_without_http_call():
    session = FakeSession()
    assert send_training_completed_webhook(
        _payload(), settings=WebhookSettings(url=None), session=session
    ) is None
    assert session.calls == []


@pytest.mark.parametrize("status", [300, 301, 302, 400, 404, 409, 429, 500, 503])
def test_non_2xx_status_is_explicit_failure_without_retry(status):
    session = FakeSession(result=FakeResponse(status))
    with pytest.raises(WebhookHTTPError) as exc_info:
        send_training_completed_webhook(
            _payload(),
            settings=WebhookSettings(url="http://127.0.0.1:8765/webhook"),
            session=session,
        )
    assert exc_info.value.status_code == status
    assert str(status) in str(exc_info.value)
    assert len(session.calls) == 1


def test_timeout_is_distinct_failure_and_not_retried():
    session = FakeSession(error=requests.Timeout("secret receiver detail"))
    with pytest.raises(WebhookTimeoutError, match="timed out") as exc_info:
        send_training_completed_webhook(
            _payload(),
            settings=WebhookSettings(url="https://example.com/webhook?token=secret"),
            session=session,
        )
    assert len(session.calls) == 1
    assert "secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("secret connection detail"),
        requests.RequestException("secret transport detail"),
    ],
)
def test_transport_failure_is_sanitized_and_not_retried(error):
    session = FakeSession(error=error)
    with pytest.raises(WebhookDeliveryError, match="transport error") as exc_info:
        send_training_completed_webhook(
            _payload(),
            settings=WebhookSettings(url="https://example.com/webhook?token=secret"),
            session=session,
        )
    assert len(session.calls) == 1
    assert "secret" not in str(exc_info.value)
    assert "example.com" not in str(exc_info.value)


def test_invalid_response_status_is_rejected():
    session = FakeSession(result=FakeResponse("204"))
    with pytest.raises(WebhookDeliveryError, match="invalid HTTP status"):
        send_training_completed_webhook(
            _payload(),
            settings=WebhookSettings(url="http://127.0.0.1:8765/webhook"),
            session=session,
        )


def test_wrong_payload_type_is_rejected_before_http():
    session = FakeSession()
    with pytest.raises(TypeError, match="TrainingCompletedPayload"):
        send_training_completed_webhook(
            {"status": "success"},
            settings=WebhookSettings(url="http://127.0.0.1:8765/webhook"),
            session=session,
        )
    assert session.calls == []


def test_wrong_settings_type_is_rejected_before_http():
    session = FakeSession()
    with pytest.raises(TypeError, match="WebhookSettings"):
        send_training_completed_webhook(_payload(), settings={}, session=session)
    assert session.calls == []


def test_receiver_accepts_exact_contract():
    assert validate_received_payload(_payload().as_json_dict()) == _payload().as_json_dict()


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "text",
        {"status": "success"},
        {**_payload().as_json_dict(), "extra": True},
        {**_payload().as_json_dict(), "schema_version": 2},
        {**_payload().as_json_dict(), "event": "other"},
        {**_payload().as_json_dict(), "status": "failed"},
        {**_payload().as_json_dict(), "minio_object_key": "model/another-run/model.pt"},
    ],
)
def test_receiver_rejects_noncanonical_payloads(value):
    with pytest.raises((KeyError, TypeError, ValueError)):
        validate_received_payload(value)


def test_receiver_factory_validates_address_contract():
    with pytest.raises(ValueError, match="host"):
        create_webhook_receiver(host="")
    with pytest.raises(ValueError, match="port"):
        create_webhook_receiver(port=-1)
    with pytest.raises(ValueError, match="path"):
        create_webhook_receiver(path="webhook")


def test_real_loopback_delivery_reaches_strict_receiver_once():
    server = create_webhook_receiver(port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        host, port = server.server_address[:2]
        receipt = send_training_completed_webhook(
            _payload(),
            settings=WebhookSettings(
                url=f"http://{host}:{port}/webhook", timeout_seconds=2
            ),
        )
        assert receipt.status_code == 204
        assert server.captured_payloads == [_payload().as_json_dict()]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_real_receiver_wrong_path_is_reported_as_http_failure():
    server = create_webhook_receiver(port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        host, port = server.server_address[:2]
        with pytest.raises(WebhookHTTPError) as exc_info:
            send_training_completed_webhook(
                _payload(),
                settings=WebhookSettings(
                    url=f"http://{host}:{port}/wrong", timeout_seconds=2
                ),
            )
        assert exc_info.value.status_code == 404
        assert server.captured_payloads == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
