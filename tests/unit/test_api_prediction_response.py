"""canonical, labeled success response contract.

All tests are offline. Use a per-application fake model loader instead of
intercepting global sockets (which breaks asyncio on Windows).
"""

from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from fashion_mnist_mlops import prediction
from fashion_mnist_mlops.api import app, create_app
from fashion_mnist_mlops import api as predict_route
from fashion_mnist_mlops.labels import FASHION_MNIST_CLASSES

RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"
LABELS = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def _png():
    stream = BytesIO()
    with Image.new("L", (28, 28), color=128) as image:
        image.save(stream, format="PNG")
    return stream.getvalue()


def _fake_client(monkeypatch, *, index=0, returned_name=None):
    application = create_app()
    loads = []
    monkeypatch.setattr(application.state.model_loader, "load", lambda: (loads.append(1) or SimpleNamespace(run_id=RUN_ID)))
    def fake_predict(restored, image):
        return SimpleNamespace(class_index=index, class_name=returned_name)
    monkeypatch.setattr(predict_route, "predict_uploaded_image", fake_predict)
    return application, loads


def test_canonical_mapping_comes_from_shared_training_data():
    assert tuple(FASHION_MNIST_CLASSES) == LABELS
    for index, name in enumerate(LABELS):
        assert prediction.canonical_class_name(index) == name


@pytest.mark.parametrize("index", range(10))
def test_http_success_reports_canonical_name_for_every_class(monkeypatch, index):
    application, loads = _fake_client(monkeypatch, index=index)
    with TestClient(application) as client:
        response = client.post("/predict", files={"file": ("sample.png", _png(), "image/png")})
    assert response.status_code == 200, response.text
    assert response.json() == {
        "class_index": index,
        "class_name": LABELS[index],
        "model_run_id": RUN_ID,
    }
    assert loads == [1]


def test_openapi_includes_labeled_response_schema():
    with TestClient(create_app()) as client:
        operation = client.get("/openapi.json").json()["paths"]["/predict"]["post"]
        response_ref = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        schema = schemas[response_ref.split("/")[-1]]
        assert set(schema["required"]) == {"class_index", "class_name", "model_run_id"}
        assert schema["properties"]["class_name"]["type"] == "string"
        assert schema["properties"]["class_index"]["minimum"] == 0
        assert schema["properties"]["class_index"]["maximum"] == 9


@pytest.mark.parametrize("bad_index", [True, False, -1, 10, "7", 3.0, None])
def test_label_lookup_rejects_invalid_index_without_coercion(bad_index):
    with pytest.raises(prediction.InvalidInferencePrediction, match="invalid Fashion-MNIST"):
        prediction.canonical_class_name(bad_index)


def test_inference_disagreeing_label_is_not_accepted(monkeypatch):
    import fashion_mnist_mlops.inference as inference
    monkeypatch.setattr(
        inference,
        "predict_preprocessed_image",
        lambda restored, tensor: SimpleNamespace(class_index=8, class_name="Sneaker"),
    )
    with Image.new("L", (28, 28)) as image:
        with pytest.raises(prediction.InvalidInferencePrediction, match="inconsistent"):
            prediction.predict_uploaded_image(SimpleNamespace(run_id=RUN_ID), image)


def test_matching_inference_label_is_accepted(monkeypatch):
    import fashion_mnist_mlops.inference as inference
    monkeypatch.setattr(
        inference,
        "predict_preprocessed_image",
        lambda restored, tensor: SimpleNamespace(class_index=8, class_name="Bag"),
    )
    with Image.new("L", (28, 28)) as image:
        result = prediction.predict_uploaded_image(SimpleNamespace(run_id=RUN_ID), image)
    assert prediction.canonical_class_name(result.class_index) == result.class_name == "Bag"


def test_app_startup_and_health_do_not_load_or_retrieve_labels(monkeypatch):
    application, loads = _fake_client(monkeypatch)
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
    assert loads == []


def test_default_app_openapi_exposes_prediction_route():
    with TestClient(app) as client:
        assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_invalid_image_still_fails_before_loading(monkeypatch):
    application, loads = _fake_client(monkeypatch, index=1)
    with TestClient(application) as client:
        response = client.post("/predict", files={"file": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert loads == []
