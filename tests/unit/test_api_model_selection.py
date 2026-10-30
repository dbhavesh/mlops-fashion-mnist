"""externally selected, canonical and offline MinIO model identity."""

from dataclasses import FrozenInstanceError
import importlib

import pytest
from fastapi.testclient import TestClient

from fashion_mnist_mlops.model_loading import (
    ModelSelection,
    load_model_selection,
)
from fashion_mnist_mlops.api import create_app
from fashion_mnist_mlops.model_keys import build_model_object_key
from fashion_mnist_mlops.config_validation import REQUIRED_MINIO_BUCKET


RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"
SECOND_RUN_ID = "1d2e046e9f2f475dab88e6146db43015"
KEY = build_model_object_key(RUN_ID)


def _env(key=KEY, bucket=REQUIRED_MINIO_BUCKET):
    return {"MINIO_BUCKET": bucket, "MODEL_OBJECT_KEY": key}


def test_selection_reads_canonical_bucket_key_and_version_without_network():
    selection = load_model_selection(_env())
    assert selection == ModelSelection(bucket=REQUIRED_MINIO_BUCKET, object_key=KEY)
    assert selection.bucket == "fashion-minst-bucket"
    assert selection.object_key == KEY
    assert selection.run_id == RUN_ID


def test_selection_is_immutable():
    selection = load_model_selection(_env())
    with pytest.raises(FrozenInstanceError):
        selection.object_key = "other"


def test_explicit_environment_mapping_does_not_depend_on_os_environ(monkeypatch):
    monkeypatch.setenv("MINIO_BUCKET", "wrong-bucket")
    monkeypatch.setenv("MODEL_OBJECT_KEY", "invalid")
    assert load_model_selection(_env()).run_id == RUN_ID


def test_selection_uses_current_environment_not_import_time_cache(monkeypatch):
    monkeypatch.setenv("MINIO_BUCKET", REQUIRED_MINIO_BUCKET)
    monkeypatch.setenv("MODEL_OBJECT_KEY", KEY)
    assert load_model_selection().run_id == RUN_ID
    monkeypatch.setenv("MODEL_OBJECT_KEY", build_model_object_key(SECOND_RUN_ID))
    assert load_model_selection().run_id == SECOND_RUN_ID


@pytest.mark.parametrize(
    "environment,part",
    [
        ({}, "MINIO_BUCKET"),
        ({"MINIO_BUCKET": REQUIRED_MINIO_BUCKET}, "MODEL_OBJECT_KEY"),
        ({"MINIO_BUCKET": "", "MODEL_OBJECT_KEY": KEY}, "MINIO_BUCKET"),
        ({"MINIO_BUCKET": REQUIRED_MINIO_BUCKET, "MODEL_OBJECT_KEY": ""}, "MODEL_OBJECT_KEY"),
        (_env(bucket="fashion-mnist-bucket"), "MINIO_BUCKET"),
        (_env(bucket="other"), "MINIO_BUCKET"),
    ],
)
def test_missing_or_incorrect_selection_fails_closed(environment, part):
    with pytest.raises(ValueError, match=part):
        load_model_selection(environment)


@pytest.mark.parametrize(
    "key",
    [
        "model.pt",
        RUN_ID,
        "model/../model.pt",
        "model//model.pt",
        "model/run name/model.pt",
        "model/run/../model.pt",
        "model/run/model.pth",
        "models/run/model.pt",
        "model/run/MODEL.pt",
        "model/run/model.pt/extra",
        "model/run\\model.pt",
        "model/run%2Fname/model.pt",
        " model/run/model.pt",
        "model/run/model.pt ",
        "model/" + "a" * 129 + "/model.pt",
    ],
)
def test_invalid_or_ambiguous_keys_are_rejected_by_inference_validator(key):
    with pytest.raises(ValueError, match="MODEL_OBJECT_KEY"):
        load_model_selection(_env(key=key))


@pytest.mark.parametrize("value", [None, False, 23, [], {}])
def test_non_string_keys_are_rejected(value):
    with pytest.raises(ValueError, match="MODEL_OBJECT_KEY"):
        load_model_selection(_env(key=value))


@pytest.mark.parametrize("value", [None, False, 23, [], {}])
def test_non_string_bucket_is_rejected(value):
    with pytest.raises(ValueError, match="MINIO_BUCKET"):
        load_model_selection(_env(bucket=value))


def test_direct_dataclass_instantiation_enforces_same_contract():
    with pytest.raises(ValueError, match="MINIO_BUCKET"):
        ModelSelection(bucket="wrong", object_key=KEY)
    with pytest.raises(ValueError, match="MODEL_OBJECT_KEY"):
        ModelSelection(bucket=REQUIRED_MINIO_BUCKET, object_key="model/latest")


def test_api_app_and_health_are_healthy_without_selection_or_services(monkeypatch):
    monkeypatch.delenv("MODEL_OBJECT_KEY", raising=False)
    monkeypatch.setenv("MINIO_BUCKET", "incorrect")
    monkeypatch.setenv("MINIO_ENDPOINT", "127.0.0.1:1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")

    importlib.import_module("fashion_mnist_mlops.model_loading")
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/openapi.json").status_code == 200


def test_credentials_are_not_part_of_selection_or_its_repr():
    selection = load_model_selection({**_env(), "MINIO_ACCESS_KEY": "do-not-print", "MINIO_SECRET_KEY": "secret-do-not-print"})
    assert "do-not-print" not in repr(selection)
    assert "MINIO_SECRET_KEY" not in repr(selection)


def test_key_is_an_exact_version_not_latest_fallback():
    first = load_model_selection(_env())
    second = load_model_selection(_env(key=build_model_object_key(SECOND_RUN_ID)))
    assert first.object_key != second.object_key
    assert first.run_id != second.run_id
