"""lazy, isolated, version-pinned model retrieval and caching.

These tests never contact MLflow, MinIO, or the network and work on Windows.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fashion_mnist_mlops import model_loading
from fashion_mnist_mlops.model_loading import ModelSelection
from fashion_mnist_mlops.api import create_app
from fashion_mnist_mlops.model_keys import build_model_object_key
from fashion_mnist_mlops.config_validation import REQUIRED_MINIO_BUCKET


RUN_A = "9629c8e5feb84a0c940926b6ecbc726f"
RUN_B = "1d2e046e9f2f475dab88e6146db43015"
SELECT_A = ModelSelection(REQUIRED_MINIO_BUCKET, build_model_object_key(RUN_A))
SELECT_B = ModelSelection(REQUIRED_MINIO_BUCKET, build_model_object_key(RUN_B))


def fake_retriever(monkeypatch, *, fail=False, wrong_run=False, training=False,
                   wrong_path=False, skip_file=False, cause=None):
    calls = []
    paths = []

    def retrieve(path, *, object_key):
        calls.append(object_key)
        paths.append(path)
        if cause is not None:
            raise cause
        if fail:
            raise RuntimeError("broken download")
        if not skip_file:
            path.write_bytes(b"verified test artifact")
        return SimpleNamespace(
            run_id="other" if wrong_run else object_key.split("/")[1],
            artifact_path=path.parent / "wrong.pt" if wrong_path else path.resolve(),
            model=SimpleNamespace(training=training),
            sha256="a" * 64,
        )

    monkeypatch.setattr(model_loading, "_retrieve_verified_model", retrieve)
    return calls, paths


def test_app_creates_independent_unloaded_model_cache_without_services(monkeypatch):
    monkeypatch.delenv("MODEL_OBJECT_KEY", raising=False)
    monkeypatch.setenv("MINIO_ENDPOINT", "127.0.0.1:1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    first = create_app()
    second = create_app()
    assert first.state.model_loader is not second.state.model_loader
    assert first.state.model_loader.cached_selection is None
    with TestClient(first) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/openapi.json").status_code == 200
    assert first.state.model_loader.cached_selection is None


def test_load_exact_key_is_single_download_and_identity_cached(monkeypatch):
    calls, paths = fake_retriever(monkeypatch)
    loader = model_loading.APIModelLoader()
    try:
        first = loader.load(SELECT_A)
        second = loader.load(SELECT_A)
        assert first is second
        assert calls == [SELECT_A.object_key]
        assert first.run_id == RUN_A
        assert loader.cached_selection == SELECT_A
        assert paths[0].is_file()
    finally:
        loader.clear()
    assert not paths[0].exists()
    assert loader.cached_selection is None


def test_reload_new_run_replaces_cache_only_after_success(monkeypatch):
    calls, paths = fake_retriever(monkeypatch)
    loader = model_loading.APIModelLoader()
    try:
        first = loader.load(SELECT_A)
        second = loader.load(SELECT_B)
        assert second is not first
        assert calls == [SELECT_A.object_key, SELECT_B.object_key]
        assert paths[0] != paths[1]
        assert not paths[0].exists()
        assert paths[1].is_file()
        assert loader.cached_selection == SELECT_B
    finally:
        loader.clear()
    assert not paths[1].exists()


def test_missing_selection_fails_before_network_and_keeps_old_cache(monkeypatch):
    calls, paths = fake_retriever(monkeypatch)
    loader = model_loading.APIModelLoader()
    try:
        initial = loader.load(SELECT_A)
        monkeypatch.setenv("MINIO_BUCKET", REQUIRED_MINIO_BUCKET)
        monkeypatch.delenv("MODEL_OBJECT_KEY", raising=False)
        with pytest.raises(ValueError, match="MODEL_OBJECT_KEY"):
            loader.load()
        assert loader.cached_selection == SELECT_A
        assert loader.load(SELECT_A) is initial
        assert calls == [SELECT_A.object_key]
        assert paths[0].exists()
    finally:
        loader.clear()


def test_environment_selects_live_version_each_call(monkeypatch):
    calls, _ = fake_retriever(monkeypatch)
    monkeypatch.setenv("MINIO_BUCKET", REQUIRED_MINIO_BUCKET)
    monkeypatch.setenv("MODEL_OBJECT_KEY", SELECT_A.object_key)
    loader = model_loading.APIModelLoader()
    try:
        assert loader.load().run_id == RUN_A
        monkeypatch.setenv("MODEL_OBJECT_KEY", SELECT_B.object_key)
        assert loader.load().run_id == RUN_B
        assert calls == [SELECT_A.object_key, SELECT_B.object_key]
    finally:
        loader.clear()


def test_wrong_selection_type_rejected_before_download(monkeypatch):
    calls, _ = fake_retriever(monkeypatch)
    loader = model_loading.APIModelLoader()
    with pytest.raises(TypeError, match="ModelSelection"):
        loader.load("model/latest/model.pt")
    assert calls == []


@pytest.mark.parametrize("cause", [
    FileNotFoundError("NoSuchKey"),
    ConnectionError("MinIO unavailable"),
    ValueError("corrupted checksum"),
    RuntimeError("invalid artifact schema"),
])
def test_first_load_failure_does_not_cache_or_leave_partial_file(monkeypatch, cause):
    calls, paths = fake_retriever(monkeypatch, cause=cause)
    loader = model_loading.APIModelLoader()
    with pytest.raises(model_loading.APIModelLoadError, match=RUN_A) as error:
        loader.load(SELECT_A)
    assert isinstance(error.value.__cause__, type(cause))
    assert calls == [SELECT_A.object_key]
    assert loader.cached_selection is None
    assert not paths[0].exists()


def test_failed_reload_keeps_previous_cache_and_artifact(monkeypatch):
    calls, paths = fake_retriever(monkeypatch)
    loader = model_loading.APIModelLoader()
    try:
        first = loader.load(SELECT_A)
        fake_retriever(monkeypatch, fail=True)
        with pytest.raises(model_loading.APIModelLoadError, match=RUN_B):
            loader.load(SELECT_B)
        assert loader.cached_selection == SELECT_A
        assert loader.load(SELECT_A) is first
        assert calls == [SELECT_A.object_key]
        assert paths[0].exists()
    finally:
        loader.clear()


@pytest.mark.parametrize("options", [
    {"wrong_run": True},
    {"training": True},
    {"wrong_path": True},
    {"skip_file": True},
])
def test_unverified_retriever_result_is_not_cached(monkeypatch, options):
    _, paths = fake_retriever(monkeypatch, **options)
    loader = model_loading.APIModelLoader()
    with pytest.raises(model_loading.APIModelLoadError) as error:
        loader.load(SELECT_A)
    assert isinstance(error.value.__cause__, ValueError)
    assert loader.cached_selection is None
    assert not paths[0].exists()


def test_clear_can_be_called_more_than_once(monkeypatch):
    _, paths = fake_retriever(monkeypatch)
    loader = model_loading.APIModelLoader()
    loader.load(SELECT_A)
    loader.clear()
    loader.clear()
    assert loader.cached_selection is None
    assert not paths[0].exists()


def test_app_lifespan_cleans_downloaded_artifact(monkeypatch):
    _, paths = fake_retriever(monkeypatch)
    application = create_app()
    with TestClient(application) as client:
        restored = application.state.model_loader.load(SELECT_A)
        assert restored.run_id == RUN_A
        assert paths[0].is_file()
        assert client.get("/health").status_code == 200
    assert application.state.model_loader.cached_selection is None
    assert not paths[0].exists()


def test_parallel_requests_share_one_download(monkeypatch):
    calls, _ = fake_retriever(monkeypatch)
    loader = model_loading.APIModelLoader()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            models = list(pool.map(loader.load, [SELECT_A] * 24))
        assert all(model is models[0] for model in models)
        assert calls == [SELECT_A.object_key]
    finally:
        loader.clear()


def test_app_dependency_lazily_delegates_to_own_loader(monkeypatch):
    calls, _ = fake_retriever(monkeypatch)
    application = create_app()
    monkeypatch.setenv("MINIO_BUCKET", REQUIRED_MINIO_BUCKET)
    monkeypatch.setenv("MODEL_OBJECT_KEY", SELECT_A.object_key)
    with TestClient(application):
        request = SimpleNamespace(app=application)
        first = model_loading.get_app_model(request)
        second = model_loading.get_app_model(request)
        assert first is second
        assert calls == [SELECT_A.object_key]
    assert application.state.model_loader.cached_selection is None


def test_prediction_route_does_not_load_a_model_without_a_file():
    application = create_app()
    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict").status_code == 422
        assert application.state.model_loader.cached_selection is None


def test_fastapi_dependency_injection_loads_model_on_demand(monkeypatch):
    from fastapi import Depends

    calls, _ = fake_retriever(monkeypatch)
    monkeypatch.setenv("MINIO_BUCKET", REQUIRED_MINIO_BUCKET)
    monkeypatch.setenv("MODEL_OBJECT_KEY", SELECT_A.object_key)
    application = create_app()

    def internal_test_route(restored=Depends(model_loading.get_app_model)):
        return {"run_id": restored.run_id}

    application.add_api_route("/_test_lazy_model", internal_test_route, methods=["GET"])
    with TestClient(application) as client:
        assert calls == []
        assert client.get("/health").json() == {"status": "ok"}
        assert calls == []
        assert client.get("/_test_lazy_model").json() == {"run_id": RUN_A}
        assert client.get("/_test_lazy_model").json() == {"run_id": RUN_A}
    assert calls == [SELECT_A.object_key]
