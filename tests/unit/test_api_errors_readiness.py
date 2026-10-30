"""safe prediction errors and model readiness contracts.

No actual MLflow, MinIO, socket interception or network dependency is used.
These tests are compatible with Windows asyncio/TestClient.
"""

from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from fashion_mnist_mlops.api import app, create_app
from fashion_mnist_mlops.model_loading import APIModelLoadError
from fashion_mnist_mlops import api as predict_route

RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"
SECRET = "very-sensitive-example-secret"


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("L", (28, 28), color=128).save(output, format="PNG")
    return output.getvalue()


def submit(client, payload=None, *, name="sample.png", mime="image/png"):
    if payload is None:
        payload = png_bytes()
    return client.post("/predict", files={"file": (name, payload, mime)})


def prepare(monkeypatch, application, *, index=0, run_id=RUN_ID, failure=None):
    calls = []

    def load():
        calls.append("load")
        if failure is not None:
            raise failure
        return SimpleNamespace(run_id=run_id, fake_index=index)

    monkeypatch.setattr(application.state.model_loader, "load", load)
    monkeypatch.setattr(
        predict_route,
        "predict_uploaded_image",
        lambda restored, image: SimpleNamespace(class_index=restored.fake_index),
    )
    return calls


def test_process_liveness_is_independent_of_missing_model(monkeypatch):
    application = create_app()
    calls = prepare(monkeypatch, application, failure=ValueError("MODEL_OBJECT_KEY missing"))
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
    assert calls == []


def test_readiness_explicitly_initializes_selected_model(monkeypatch):
    application = create_app()
    calls = prepare(monkeypatch, application)
    with TestClient(application) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_run_id": RUN_ID}
    assert calls == ["load"]


def test_readiness_rechecks_current_selection_per_request(monkeypatch):
    application = create_app()
    ids = iter([RUN_ID, "different_version"])
    calls = []

    def load():
        calls.append(1)
        return SimpleNamespace(run_id=next(ids))

    monkeypatch.setattr(application.state.model_loader, "load", load)
    with TestClient(application) as client:
        assert client.get("/ready").json()["model_run_id"] == RUN_ID
        assert client.get("/ready").json()["model_run_id"] == "different_version"
    assert len(calls) == 2


@pytest.mark.parametrize("failure", [
    ValueError("MODEL_OBJECT_KEY not configured"),
    APIModelLoadError("Remote MinIO host, access_key=" + SECRET),
])
def test_readiness_returns_safe_503_when_model_is_unavailable(monkeypatch, failure):
    application = create_app()
    prepare(monkeypatch, application, failure=failure)
    with TestClient(application) as client:
        response = client.get("/ready")
        health = client.get("/health")
    assert health.status_code == 200
    assert response.status_code == 503
    assert response.json() == {
        "detail": "The selected prediction model is unavailable.",
        "code": "MODEL_UNAVAILABLE",
    }
    assert SECRET not in response.text
    assert "MODEL_OBJECT_KEY" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_readiness_unexpected_failure_is_safe_500(monkeypatch):
    application = create_app()
    prepare(monkeypatch, application, failure=RuntimeError(SECRET))
    with TestClient(application) as client:
        response = client.get("/ready")
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Model readiness could not be determined.",
        "code": "READINESS_FAILED",
    }
    assert SECRET not in response.text


def test_readiness_rejects_invalid_restoration_result(monkeypatch):
    application = create_app()
    monkeypatch.setattr(application.state.model_loader, "load", lambda: object())
    with TestClient(application) as client:
        response = client.get("/ready")
    assert response.status_code == 500
    assert response.json()["code"] == "READINESS_FAILED"


def test_readiness_is_get_only(monkeypatch):
    application = create_app()
    calls = prepare(monkeypatch, application)
    with TestClient(application) as client:
        assert client.post("/ready").status_code == 405
    assert calls == []


def test_two_app_instances_are_isolated(monkeypatch):
    a, b = create_app(), create_app()
    monkeypatch.setattr(a.state.model_loader, "load", lambda: SimpleNamespace(run_id="run-A"))
    monkeypatch.setattr(b.state.model_loader, "load", lambda: SimpleNamespace(run_id="run-B"))
    with TestClient(a) as first, TestClient(b) as second:
        assert first.get("/ready").json()["model_run_id"] == "run-A"
        assert second.get("/ready").json()["model_run_id"] == "run-B"


def test_default_app_exposes_both_liveness_and_readiness():
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert "get" in schema["paths"]["/ready"]
    assert "503" in schema["paths"]["/ready"]["get"]["responses"]


@pytest.mark.parametrize("status,code,payload,filename,mime", [
    (400, "INVALID_IMAGE", b"", "sample.png", "image/png"),
    (400, "INVALID_IMAGE", b"not image", "sample.png", "image/png"),
    (415, "UNSUPPORTED_MEDIA_TYPE", png_bytes(), "sample.png", "image/jpeg"),
    (415, "UNSUPPORTED_MEDIA_TYPE", png_bytes(), "sample.jpg", "image/jpeg"),
    (415, "UNSUPPORTED_MEDIA_TYPE", png_bytes(), "sample.gif", "image/gif"),
])
def test_invalid_uploads_get_stable_codes_before_model_loading(
    monkeypatch, status, code, payload, filename, mime
):
    application = create_app()
    calls = prepare(monkeypatch, application)
    with TestClient(application) as client:
        response = submit(client, payload, name=filename, mime=mime)
    assert response.status_code == status, response.text
    assert response.json()["code"] == code
    assert isinstance(response.json()["detail"], str)
    assert response.headers["cache-control"] == "no-store"
    assert calls == []


def test_duplicate_upload_gets_400_before_model_loading(monkeypatch):
    application = create_app()
    calls = prepare(monkeypatch, application)
    with TestClient(application) as client:
        response = client.post("/predict", files=[
            ("file", ("a.png", png_bytes(), "image/png")),
            ("file", ("b.png", png_bytes(), "image/png")),
        ])
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_UPLOAD"
    assert calls == []


def test_oversized_upload_gets_413_before_model_loading(monkeypatch):
    from fashion_mnist_mlops.prediction import MAX_UPLOAD_BYTES
    application = create_app()
    calls = prepare(monkeypatch, application)
    with TestClient(application) as client:
        response = submit(client, b"a" * (MAX_UPLOAD_BYTES + 1))
    assert response.status_code == 413
    assert response.json()["code"] == "PAYLOAD_TOO_LARGE"
    assert calls == []


def test_framework_missing_file_still_returns_422(monkeypatch):
    application = create_app()
    calls = prepare(monkeypatch, application)
    with TestClient(application) as client:
        response = client.post("/predict")
    assert response.status_code == 422
    assert "detail" in response.json()
    assert calls == []


@pytest.mark.parametrize("failure", [
    APIModelLoadError("S3 secret=" + SECRET),
    ValueError("Selection invalid " + SECRET),
])
def test_predict_model_failure_503_does_not_leak_details(monkeypatch, failure):
    application = create_app()
    calls = prepare(monkeypatch, application, failure=failure)
    with TestClient(application) as client:
        response = submit(client)
    assert calls == ["load"]
    assert response.status_code == 503
    assert response.json() == {
        "detail": "The selected prediction model is unavailable.",
        "code": "MODEL_UNAVAILABLE",
    }
    assert SECRET not in response.text


def test_predict_inference_failure_is_safe_500(monkeypatch):
    application = create_app()
    calls = prepare(monkeypatch, application)

    def fail(*args):
        raise RuntimeError("confidential artifact path=" + SECRET)

    monkeypatch.setattr(predict_route, "predict_uploaded_image", fail)
    with TestClient(application) as client:
        response = submit(client)
    assert calls == ["load"]
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Prediction could not be completed.",
        "code": "PREDICTION_FAILED",
    }
    assert SECRET not in response.text


def test_predict_unexpected_loader_failure_is_safe_500(monkeypatch):
    application = create_app()
    calls = prepare(monkeypatch, application, failure=RuntimeError(SECRET))
    with TestClient(application) as client:
        response = submit(client)
    assert calls == ["load"]
    assert response.status_code == 500
    assert response.json()["code"] == "PREDICTION_FAILED"
    assert SECRET not in response.text


def test_predict_response_construction_failure_is_safe_500(monkeypatch):
    application = create_app()
    prepare(monkeypatch, application)
    monkeypatch.setattr(predict_route, "predict_uploaded_image", lambda *args: object())
    with TestClient(application) as client:
        response = submit(client)
    assert response.status_code == 500
    assert response.json()["code"] == "PREDICTION_FAILED"


def test_successful_response_is_unchanged(monkeypatch):
    application = create_app()
    calls = prepare(monkeypatch, application, index=0)
    with TestClient(application) as client:
        response = submit(client)
    assert response.status_code == 200
    assert response.json() == {
        "class_index": 0,
        "class_name": "T-shirt/top",
        "model_run_id": RUN_ID,
    }
    assert calls == ["load"]


def test_openapi_documents_all_known_error_codes():
    with TestClient(app) as client:
        schemas = client.get("/openapi.json").json()
    post = schemas["paths"]["/predict"]["post"]
    assert {"200", "400", "413", "415", "422", "500", "503"} <= set(post["responses"])
    for status in ("400", "413", "415", "500", "503"):
        assert "application/json" in post["responses"][status]["content"]
    assert {"200", "500", "503"} <= set(schemas["paths"]["/ready"]["get"]["responses"])


def test_readiness_fails_closed_when_new_run_cannot_replace_cached_model(monkeypatch):
    from fashion_mnist_mlops import model_loading
    from fashion_mnist_mlops.config_validation import REQUIRED_MINIO_BUCKET

    first_run = RUN_ID
    second_run = "0123456789abcdef0123456789abcdef"
    first_key = f"model/{first_run}/model.pt"
    second_key = f"model/{second_run}/model.pt"
    calls = []

    def fake_restore(path, *, object_key):
        calls.append(object_key)
        if object_key == second_key:
            raise ConnectionError("MinIO secret=" + SECRET)
        path.write_bytes(b"verified placeholder")
        return SimpleNamespace(
            run_id=first_run,
            artifact_path=path.resolve(),
            model=SimpleNamespace(training=False),
        )

    monkeypatch.setattr(model_loading, "_retrieve_verified_model", fake_restore)
    monkeypatch.setenv("MINIO_BUCKET", REQUIRED_MINIO_BUCKET)
    monkeypatch.setenv("MODEL_OBJECT_KEY", first_key)
    application = create_app()
    with TestClient(application) as client:
        first = client.get("/ready")
        assert first.status_code == 200
        assert first.json()["model_run_id"] == first_run
        assert client.get("/ready").status_code == 200  # cached
        monkeypatch.setenv("MODEL_OBJECT_KEY", second_key)
        second = client.get("/ready")
        assert second.status_code == 503
        assert second.json()["code"] == "MODEL_UNAVAILABLE"
        assert SECRET not in second.text
        assert client.get("/health").status_code == 200
        assert application.state.model_loader.cached_selection.run_id == first_run
    assert calls == [first_key, second_key]
