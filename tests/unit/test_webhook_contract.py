"""tests for webhook configuration and payload contracts."""

import math

import pytest

from fashion_mnist_mlops.webhook import (
    DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    WEBHOOK_EVENT,
    WEBHOOK_SCHEMA_VERSION,
    WEBHOOK_STATUS,
    TrainingCompletedPayload,
    load_webhook_settings,
)

RUN_ID = "97461abfc4804cb0a2a1473c71737739"
BUCKET = "fashion-minst-bucket"
OBJECT_KEY = f"model/{RUN_ID}/model.pt"


def _payload(**overrides):
    values = {
        "mlflow_run_id": RUN_ID,
        "minio_bucket": BUCKET,
        "minio_object_key": OBJECT_KEY,
        "validation_accuracy": 0.8125,
    }
    values.update(overrides)
    return TrainingCompletedPayload(**values)


def test_payload_exposes_required_success_contract():
    payload = _payload()
    assert payload.schema_version == WEBHOOK_SCHEMA_VERSION == 1
    assert payload.event == WEBHOOK_EVENT == "training.completed"
    assert payload.status == WEBHOOK_STATUS == "success"
    assert payload.mlflow_run_id == RUN_ID
    assert payload.minio_bucket == BUCKET
    assert payload.minio_object_key == OBJECT_KEY
    assert payload.validation_accuracy == 0.8125


def test_payload_json_representation_is_stable():
    assert _payload().as_json_dict() == {
        "mlflow_run_id": RUN_ID,
        "minio_bucket": BUCKET,
        "minio_object_key": OBJECT_KEY,
        "validation_accuracy": 0.8125,
        "schema_version": 1,
        "event": "training.completed",
        "status": "success",
    }


def test_payload_is_frozen():
    payload = _payload()
    with pytest.raises(Exception):
        payload.status = "changed"


@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_valid_accuracy_values_are_accepted(value):
    assert _payload(validation_accuracy=value).validation_accuracy == value


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf"), True, False, "0.5", None],
)
def test_invalid_accuracy_values_are_rejected(value):
    with pytest.raises(ValueError, match="validation_accuracy"):
        _payload(validation_accuracy=value)


@pytest.mark.parametrize(
    "bucket",
    ["fashion-mnist-bucket", "", "other", " fashion-minst-bucket"],
)
def test_wrong_bucket_is_rejected(bucket):
    with pytest.raises(ValueError, match="minio_bucket"):
        _payload(minio_bucket=bucket)


def test_object_key_must_match_same_run_id():
    other = "f38e2a9c552747b3ba69621b7f71c518"
    with pytest.raises(ValueError, match="canonical key"):
        _payload(minio_object_key=f"model/{other}/model.pt")


@pytest.mark.parametrize("run_id", ["", "..", "with space", "run/id"])
def test_unsafe_run_ids_are_rejected(run_id):
    with pytest.raises(ValueError):
        _payload(mlflow_run_id=run_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("event", "training.failed"),
        ("event", ""),
        ("status", "failed"),
        ("status", ""),
    ],
)
def test_fixed_contract_fields_cannot_be_changed(field, value):
    with pytest.raises(ValueError):
        _payload(**{field: value})


def test_missing_webhook_url_disables_notifications():
    settings = load_webhook_settings({})
    assert settings.url is None
    assert settings.timeout_seconds == DEFAULT_WEBHOOK_TIMEOUT_SECONDS


@pytest.mark.parametrize("url", [None, "", "   "])
def test_blank_webhook_url_is_treated_as_disabled(url):
    env = {} if url is None else {"WEBHOOK_URL": url}
    assert load_webhook_settings(env).url is None


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/hook",
        "https://example.com/webhooks/training",
        "http://localhost:9999/",
        "https://example.com/hook?source=fashion",
    ],
)
def test_valid_absolute_http_urls_are_accepted(url):
    assert load_webhook_settings({"WEBHOOK_URL": url}).url == url


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/hook",
        "example.com/hook",
        "/hook",
        "http:///hook",
        "https://",
        "https://user:password@example.com/hook",
        "https://example.com/hook#fragment",
    ],
)
def test_invalid_webhook_urls_are_rejected(url):
    with pytest.raises(ValueError, match="WEBHOOK_URL"):
        load_webhook_settings({"WEBHOOK_URL": url})


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.1", 0.1),
        ("1", 1.0),
        ("5", 5.0),
        ("60", 60.0),
        (" 2.5 ", 2.5),
    ],
)
def test_valid_timeouts_are_parsed(raw, expected):
    assert (
        load_webhook_settings({"WEBHOOK_TIMEOUT_SECONDS": raw}).timeout_seconds
        == expected
    )


@pytest.mark.parametrize(
    "raw",
    ["0", "-1", "60.1", "nan", "inf", "-inf", "abc"],
)
def test_invalid_timeouts_are_rejected(raw):
    with pytest.raises(ValueError, match="WEBHOOK_TIMEOUT_SECONDS"):
        load_webhook_settings({"WEBHOOK_TIMEOUT_SECONDS": raw})


def test_blank_timeout_uses_default():
    assert (
        load_webhook_settings({"WEBHOOK_TIMEOUT_SECONDS": "   "}).timeout_seconds
        == DEFAULT_WEBHOOK_TIMEOUT_SECONDS
    )


def test_settings_loader_has_no_network_side_effects(monkeypatch):
    called = False

    def fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network access should not occur while loading settings")

    monkeypatch.setattr("socket.create_connection", fail)
    settings = load_webhook_settings(
        {
            "WEBHOOK_URL": "http://127.0.0.1:9999/hook",
            "WEBHOOK_TIMEOUT_SECONDS": "3",
        }
    )
    assert settings.url == "http://127.0.0.1:9999/hook"
    assert settings.timeout_seconds == 3.0
    assert called is False
