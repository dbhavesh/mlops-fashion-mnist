"""offline, Windows-safe multipart prediction route acceptance tests."""

from io import BytesIO
import sys
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from fashion_mnist_mlops import prediction
from fashion_mnist_mlops.api import app, create_app
from fashion_mnist_mlops import api as predict_route
from fashion_mnist_mlops.model_loading import APIModelLoadError

RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"


def image_bytes(*, fmt="PNG", size=(28, 28), mode="L"):
    stream = BytesIO()
    Image.new(mode, size, color=100).save(stream, format=fmt)
    return stream.getvalue()


def fake_model_calls(monkeypatch, application, *, index=6, error=None):
    loads = []
    predictions = []

    def load():
        loads.append(True)
        if error is not None:
            raise error
        return SimpleNamespace(run_id=RUN_ID, fake_index=index)

    def predict(restored, image):
        predictions.append((restored.run_id, image.size, image.mode))
        return SimpleNamespace(class_index=restored.fake_index)

    monkeypatch.setattr(application.state.model_loader, "load", load)
    monkeypatch.setattr(predict_route, "predict_uploaded_image", predict)
    return loads, predictions


def test_post_predict_returns_numeric_class_and_exact_version(monkeypatch):
    application = create_app()
    loads, predictions = fake_model_calls(monkeypatch, application, index=7)
    with TestClient(application) as client:
        response = client.post("/predict", files={"file": ("sample.png", image_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    assert response.json() == {"class_index": 7, "class_name": "Sneaker", "model_run_id": RUN_ID}
    assert loads == [True]
    assert predictions == [(RUN_ID, (28, 28), "L")]


def test_rgb_png_passes_pil_image_to_pipeline(monkeypatch):
    application = create_app()
    _, predictions = fake_model_calls(monkeypatch, application, index=4)
    with TestClient(application) as client:
        response = client.post("/predict", files={"file": ("sample.png", image_bytes(mode="RGB"), "image/png")})
    assert response.status_code == 200
    assert predictions == [(RUN_ID, (28, 28), "RGB")]


def test_jpeg_accepted_by_initial_decoder_before_format_policy(monkeypatch):
    application = create_app()
    loads, _ = fake_model_calls(monkeypatch, application)
    with TestClient(application) as client:
        response = client.post("/predict", files={"file": ("sample.jpg", image_bytes(fmt="JPEG"), "image/jpeg")})
    assert response.status_code == 200
    assert loads == [True]


def test_predict_method_is_post_only_and_health_stays_independent(monkeypatch):
    application = create_app()
    loads, _ = fake_model_calls(monkeypatch, application)
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/predict").status_code == 405
        assert client.post("/predict").status_code == 422
    assert loads == []


def test_openapi_lists_multipart_file_field_and_response(monkeypatch):
    application = create_app()
    with TestClient(application) as client:
        schema = client.get("/openapi.json").json()
        operation = schema["paths"]["/predict"]["post"]
        assert "multipart/form-data" in operation["requestBody"]["content"]
        assert "200" in operation["responses"]
        assert "503" in operation["responses"]  # assert "/health" in schema["paths"]


def test_module_import_and_app_startup_do_not_load_selected_model(monkeypatch):
    application = create_app()
    loads, _ = fake_model_calls(monkeypatch, application)
    with TestClient(application) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/health").status_code == 200
    assert loads == []


@pytest.mark.parametrize("bad", [b"", b"not an image", b"\x89PNG\r\n\x1a\n", b"\x00" * 17])
def test_empty_and_unreadable_images_do_not_trigger_model_loading(monkeypatch, bad):
    application = create_app()
    loads, _ = fake_model_calls(monkeypatch, application)
    with TestClient(application) as client:
        response = client.post("/predict", files={"file": ("broken.png", bad, "image/png")})
    assert response.status_code == 400
    assert loads == []


def test_large_upload_returns_413_before_model_load(monkeypatch):
    application = create_app()
    loads, _ = fake_model_calls(monkeypatch, application)
    with TestClient(application) as client:
        response = client.post("/predict", files={"file": ("oversize.png", b"x" * (prediction.MAX_UPLOAD_BYTES + 1), "image/png")})
    assert response.status_code == 413
    assert loads == []


def test_multiple_uploads_rejected_without_model(monkeypatch):
    application = create_app()
    loads, _ = fake_model_calls(monkeypatch, application)
    with TestClient(application) as client:
        response = client.post("/predict", files=[
            ("file", ("a.png", image_bytes(), "image/png")),
            ("file", ("b.png", image_bytes(), "image/png")),
        ])
        other = client.post("/predict", files=[
            ("file", ("a.png", image_bytes(), "image/png")),
            ("other", ("b.png", image_bytes(), "image/png")),
        ])
    assert response.status_code == 400
    assert other.status_code == 400
    assert loads == []


def test_upload_wrong_field_and_json_body_return_422(monkeypatch):
    application = create_app()
    loads, _ = fake_model_calls(monkeypatch, application)
    with TestClient(application) as client:
        wrong_field = client.post("/predict", files={"image": ("a.png", image_bytes(), "image/png")})
        json_body = client.post("/predict", json={"file": "a.png"})
    assert wrong_field.status_code == 422
    assert json_body.status_code == 422
    assert loads == []


@pytest.mark.parametrize("error", [
    ValueError("MODEL_OBJECT_KEY not set"),
    APIModelLoadError("MinIO credentials should not leak"),
])
def test_missing_selection_or_service_down_returns_generic_503(monkeypatch, error):
    application = create_app()
    loads, _ = fake_model_calls(monkeypatch, application, error=error)
    with TestClient(application) as client:
        response = client.post("/predict", files={"file": ("a.png", image_bytes(), "image/png")})
    assert response.status_code == 503
    assert response.json() == {"detail": "The selected prediction model is unavailable.", "code": "MODEL_UNAVAILABLE"}
    assert "MODEL_OBJECT_KEY" not in response.text
    assert "credentials" not in response.text
    assert loads == [True]


def test_response_is_per_application(monkeypatch):
    first, second = create_app(), create_app()
    calls_a, _ = fake_model_calls(monkeypatch, first, index=2)
    calls_b, _ = fake_model_calls(monkeypatch, second, index=9)
    payload = {"file": ("a.png", image_bytes(), "image/png")}
    with TestClient(first) as client_a, TestClient(second) as client_b:
        assert client_a.post("/predict", files=payload).json()["class_index"] == 2
        assert client_b.post("/predict", files=payload).json()["class_index"] == 9
    assert calls_a == [True]
    assert calls_b == [True]


def test_image_decoder_closes_upload_and_returns_detached_image():
    decoded = prediction.decode_uploaded_image(image_bytes())
    try:
        assert decoded.size == (28, 28)
        assert decoded.getpixel((0, 0)) == 100
    finally:
        decoded.close()


def test_image_decoder_rejects_pixels_above_limit():
    content = image_bytes(size=(4097, 4097), mode="1")
    with pytest.raises(prediction.InvalidUploadedImage, match="dimensions"):
        prediction.decode_uploaded_image(content)


def test_prediction_orchestrator_reuses_existing_preprocess_and_inference(monkeypatch):
    calls = []
    data_mod = ModuleType("fashion_mnist_mlops.preprocessing")
    inference_mod = ModuleType("fashion_mnist_mlops.inference")

    import torch
    tensor = torch.zeros((1, 1, 28, 28), dtype=torch.float32)

    def preprocess(image):
        calls.append(("preprocess", image.size))
        return tensor

    def predict(restored, tensor):
        calls.append(("inference", restored.run_id, tensor))
        return SimpleNamespace(class_index=8, class_name="Bag")

    data_mod.preprocess_image_for_inference = preprocess
    data_mod.IMAGE_CHANNELS = 1
    data_mod.IMAGE_HEIGHT = 28
    data_mod.IMAGE_WIDTH = 28
    # Preserve the real class order in this stub; serving checks index AND label.
    from fashion_mnist_mlops.labels import FASHION_MNIST_CLASSES
    data_mod.FASHION_MNIST_CLASSES = tuple(FASHION_MNIST_CLASSES)
    inference_mod.predict_preprocessed_image = predict
    monkeypatch.setitem(sys.modules, "fashion_mnist_mlops.preprocessing", data_mod)
    monkeypatch.setitem(sys.modules, "fashion_mnist_mlops.inference", inference_mod)
    image = Image.new("L", (28, 28))
    restored = SimpleNamespace(run_id=RUN_ID)
    result = prediction.predict_uploaded_image(restored, image)
    assert result.class_index == 8
    assert calls == [("preprocess", (28, 28)), ("inference", RUN_ID, tensor)]


def test_default_app_route_registered():
    with TestClient(app) as client:
        assert "/predict" in client.get("/openapi.json").json()["paths"]
