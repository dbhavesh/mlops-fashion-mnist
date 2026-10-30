"""API and training preprocessing parity (offline on Windows/Linux).

No MinIO/MLflow services, no weights, and no global socket patching. Test the
real data-package transform, not a second API implementation of the transform.
"""

from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch
from torchvision import transforms

from fashion_mnist_mlops import prediction
from fashion_mnist_mlops.api import create_app
from fashion_mnist_mlops.preprocessing import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NORMALIZATION_MEAN, NORMALIZATION_STD, preprocess_image_for_inference

RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"


def _upload_bytes(*, fmt="PNG", mode="RGB", size=(28, 28)):
    with BytesIO() as data:
        image = Image.new(mode, size)
        # A nonuniform gradient exercises grayscale + resizing; a flat image
        # would not catch a missing conversion or incorrect interpolation.
        for y in range(size[1]):
            for x in range(size[0]):
                v = (11 * x + 7 * y) % 256
                if mode == "RGB":
                    pixel = (v, 255 - v, (3 * v) % 256)
                elif mode == "RGBA":
                    pixel = (v, 255 - v, (3 * v) % 256, 255)
                else:
                    pixel = v
                image.putpixel((x, y), pixel)
        image.save(data, format=fmt)
        image.close()
        return data.getvalue()


def _expected_training_tensor(image):
    # Reference specification from the training pipeline: test-only oracle.
    # Production calls data.preprocess_image_for_inference(), not this recipe.
    training_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=IMAGE_CHANNELS),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
    ])
    return training_transform(image).unsqueeze(0)


@pytest.mark.parametrize("fmt,mode,size", [
    ("PNG", "L", (28, 28)),
    ("PNG", "RGB", (28, 28)),
    ("PNG", "RGB", (37, 19)),
    ("PNG", "RGBA", (21, 43)),
    ("PNG", "L", (63, 31)),
    ("JPEG", "RGB", (28, 28)),
    ("JPEG", "RGB", (31, 47)),
    ("JPEG", "L", (41, 17)),
])
def test_upload_preprocessing_matches_training_contract(fmt, mode, size):
    payload = _upload_bytes(fmt=fmt, mode=mode, size=size)
    with prediction.decode_uploaded_image(payload, expected_format=fmt) as image:
        actual = prediction.preprocess_uploaded_image(image)
        canonical = preprocess_image_for_inference(image)
        reference = _expected_training_tensor(image)
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, canonical, rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=0, atol=1e-6)


@pytest.mark.parametrize("pixel_value", [0, 128, 255])
def test_pixel_normalization_uses_training_statistics_only_once(pixel_value):
    with Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), pixel_value) as image:
        tensor = prediction.preprocess_uploaded_image(image)
    expected_value = (pixel_value / 255.0 - NORMALIZATION_MEAN[0]) / NORMALIZATION_STD[0]
    torch.testing.assert_close(
        tensor,
        torch.full((1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), expected_value),
        atol=1e-6, rtol=0,
    )


def test_shared_transform_is_called_exactly_once(monkeypatch):
    import fashion_mnist_mlops.preprocessing as data
    import fashion_mnist_mlops.inference as inference

    calls = []
    original = data.preprocess_image_for_inference
    def spy(image):
        calls.append("shared-transform")
        return original(image)

    def fake_predict(restored, tensor):
        calls.append("inference")
        assert tensor.shape == (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        return SimpleNamespace(class_index=4)

    monkeypatch.setattr(data, "preprocess_image_for_inference", spy)
    monkeypatch.setattr(inference, "predict_preprocessed_image", fake_predict)
    with Image.new("RGB", (45, 19), color=(20, 90, 170)) as image:
        result = prediction.predict_uploaded_image(SimpleNamespace(run_id=RUN_ID), image)
    assert result.class_index == 4
    assert calls == ["shared-transform", "inference"]


@pytest.mark.parametrize("bad_tensor", [
    None,
    "not a tensor",
    torch.empty((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)),  # no batch
    torch.empty((2, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)),
    torch.empty((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)),
    torch.empty((1, IMAGE_CHANNELS, 32, 32)),
    torch.zeros((1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float64),
    torch.full((1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), float("nan")),
    torch.full((1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), float("inf")),
])
def test_malformed_shared_preprocessing_fails_closed(monkeypatch, bad_tensor):
    import fashion_mnist_mlops.preprocessing as data
    monkeypatch.setattr(data, "preprocess_image_for_inference", lambda image: bad_tensor)
    with Image.new("L", (28, 28)) as image:
        with pytest.raises(prediction.ImagePreprocessingError, match="invalid inference tensor"):
            prediction.preprocess_uploaded_image(image)


@pytest.mark.parametrize("fmt,mode,size,filename,mime", [
    ("PNG", "RGB", (49, 17), "garment.png", "image/png"),
    ("JPEG", "RGB", (15, 53), "garment.jpeg", "image/jpeg"),
])
def test_http_predict_passes_training_equivalent_tensor_into_inference(
    monkeypatch, fmt, mode, size, filename, mime
):
    import fashion_mnist_mlops.inference as inference

    seen = []
    application = create_app()
    def fake_load():
        return SimpleNamespace(run_id=RUN_ID)
    def fake_predict(restored, tensor):
        seen.append(tensor.detach().clone())
        return SimpleNamespace(class_index=8)
    monkeypatch.setattr(application.state.model_loader, "load", fake_load)
    monkeypatch.setattr(inference, "predict_preprocessed_image", fake_predict)
    payload = _upload_bytes(fmt=fmt, mode=mode, size=size)
    with TestClient(application) as client:
        response = client.post("/predict", files={"file": (filename, payload, mime)})
        assert response.status_code == 200, response.text
        assert response.json() == {"class_index": 8, "class_name": "Bag", "model_run_id": RUN_ID}
        assert client.get("/health").json() == {"status": "ok"}
    assert len(seen) == 1
    with prediction.decode_uploaded_image(payload, expected_format=fmt) as image:
        expected = _expected_training_tensor(image)
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=1e-6)
