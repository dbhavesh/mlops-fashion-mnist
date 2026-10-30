"""Deterministic and safe model artifact key contract."""

import pytest

from fashion_mnist_mlops.config import load_infrastructure_config
from fashion_mnist_mlops.config_validation import ConfigurationError
from fashion_mnist_mlops.model_keys import (
    build_model_object_key,
    validate_model_object_key,
)


def test_mlflow_run_id_maps_to_exact_required_prefix_and_filename():
    run_id = "1d2e046e9f2f475dab88e6146db43015"
    assert build_model_object_key(run_id) == (
        "model/1d2e046e9f2f475dab88e6146db43015/model.pt"
    )


def test_repeated_generation_for_same_id_is_stable():
    assert build_model_object_key("run-id-001") == build_model_object_key("run-id-001")


def test_different_runs_have_distinct_keys():
    keys = {build_model_object_key(run_id) for run_id in ("run-id-001", "run-id-002")}
    assert keys == {"model/run-id-001/model.pt", "model/run-id-002/model.pt"}


@pytest.mark.parametrize(
    "run_id",
    ["a", "1", "A0", "run-id-001", "run_id_001", "a" * 128],
)
def test_safe_identifiers_round_trip(run_id):
    key = build_model_object_key(run_id)
    assert validate_model_object_key(key) == key


@pytest.mark.parametrize(
    "run_id",
    [
        None, 123, "", " ", "..", ".", "_run", "-run", "../run",
        "run/../../x", "run\\x", "run id", "run\n", "rún", "run.pt",
        "run%2Fid", "run?version=2", "run#id", "a" * 129,
    ],
)
def test_invalid_run_ids_are_rejected_before_building_keys(run_id):
    with pytest.raises(ValueError, match="run_id"):
        build_model_object_key(run_id)


@pytest.mark.parametrize(
    "object_key",
    [
        None, "", "model.pt", "Model/run-id/model.pt", "model//model.pt",
        "model/../model.pt", "model/run/../model.pt", "model/run/other.pt",
        "model/run/model.pt/", "/model/run/model.pt", "model/run/model.pt?x=y",
        "model\\run\\model.pt", "model/run//model.pt", "model/with space/model.pt",
    ],
)
def test_invalid_object_keys_are_rejected(object_key):
    with pytest.raises(ValueError, match="Object key"):
        validate_model_object_key(object_key)


def test_config_accepts_canonical_object_key(monkeypatch):
    key = build_model_object_key("run-id-001")
    monkeypatch.setenv("MODEL_OBJECT_KEY", key)
    assert load_infrastructure_config().model.object_key == key


@pytest.mark.parametrize(
    "key", ["model/../model.pt", "model/with space/model.pt", "model/run/../model.pt"]
)
def test_config_rejects_noncanonical_model_object_key(monkeypatch, key):
    monkeypatch.setenv("MODEL_OBJECT_KEY", key)
    with pytest.raises(ConfigurationError, match="MODEL_OBJECT_KEY"):
        load_infrastructure_config()


def test_config_does_not_require_key_before_inference(monkeypatch):
    monkeypatch.delenv("MODEL_OBJECT_KEY", raising=False)
    assert load_infrastructure_config().model.object_key is None
