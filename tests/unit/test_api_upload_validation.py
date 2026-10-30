"""offline PNG/JPEG upload-validation contract on Windows/Linux.

No MLflow/MinIO connection and no globally patched sockets: the ASGI test
client needs internal localhost sockets on Windows to initialize asyncio.
"""

from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from fashion_mnist_mlops import prediction
from fashion_mnist_mlops.api import create_app
from fashion_mnist_mlops import api as predict_route


RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"


def encoded_image(fmt="PNG", mode="L", *, size=(28, 28)):
    output = BytesIO()
    Image.new(mode, size, color=103).save(output, format=fmt)
    return output.getvalue()


def animated_png():
    stream = BytesIO()
    first = Image.new("RGBA", (28, 28), "black")
    second = Image.new("RGBA", (28, 28), "white")
    first.save(stream, format="PNG", save_all=True, append_images=[second], duration=100)
    return stream.getvalue()


@pytest.fixture
def client_and_loads(monkeypatch):
    application = create_app()
    loads = []

    def fake_load():
        loads.append(True)
        return SimpleNamespace(run_id=RUN_ID)

    def fake_predict(model, image):
        return SimpleNamespace(class_index=8)

    monkeypatch.setattr(application.state.model_loader, "load", fake_load)
    monkeypatch.setattr(predict_route, "predict_uploaded_image", fake_predict)
    with TestClient(application) as client:
        yield client, loads


def upload(client, data, *, filename="item.png", mime="image/png"):
    return client.post("/predict", files={"file": (filename, data, mime)})


@pytest.mark.parametrize("fmt,mime,filename,mode", [
    ("PNG", "image/png", "coat.png", "L"),
    ("PNG", "image/png", "coat.PNG", "RGB"),
    ("PNG", "IMAGE/PNG", "coat.png", "RGB"),
    ("JPEG", "image/jpeg", "coat.jpg", "L"),
    ("JPEG", "image/jpeg", "coat.jpeg", "RGB"),
    ("JPEG", "image/jpeg", "coat.JPEG", "RGB"),
])
def test_supported_images_accepted(client_and_loads, fmt, mime, filename, mode):
    client, loads = client_and_loads
    response = upload(client, encoded_image(fmt, mode), filename=filename, mime=mime)
    assert response.status_code == 200, response.text
    assert response.json() == {"class_index": 8, "class_name": "Bag", "model_run_id": RUN_ID}
    assert loads == [True]


@pytest.mark.parametrize("filename,mime", [
    ("out.gif", "image/gif"),
    ("out.webp", "image/webp"),
    ("out.bmp", "image/bmp"),
    ("out.svg", "image/svg+xml"),
    ("out.png", "application/octet-stream"),
    ("out.png", "text/plain"),
    ("out.png", ""),
    ("out.png", "image/jpg"),
    ("out", "image/png"),
    ("out.jpg", "image/png"),
    ("out.png", "image/jpeg"),
    ("out.gif", "image/png"),
])
def test_disallowed_filename_or_mime_is_415_before_model(client_and_loads, filename, mime):
    client, loads = client_and_loads
    response = upload(client, encoded_image(), filename=filename, mime=mime)
    assert response.status_code == 415, response.text
    assert "detail" in response.json()
    assert loads == []


@pytest.mark.parametrize("fmt,filename,mime", [
    ("JPEG", "fake.png", "image/png"),
    ("PNG", "fake.jpeg", "image/jpeg"),
    ("BMP", "fake.png", "image/png"),
    ("GIF", "fake.png", "image/png"),
])
def test_disguised_content_is_415_not_inferred_from_extension(
    client_and_loads, fmt, filename, mime
):
    client, loads = client_and_loads
    response = upload(client, encoded_image(fmt), filename=filename, mime=mime)
    assert response.status_code == 415, response.text
    assert loads == []


def test_svg_disguised_as_png_is_rejected_before_model(client_and_loads):
    client, loads = client_and_loads
    response = upload(client, b'<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    assert response.status_code == 400
    assert loads == []


@pytest.mark.parametrize("content", [
    b"\x89PNG\r\n\x1a\n",  # signature only
    encoded_image("PNG")[:-12],  # missing IEND
    encoded_image("JPEG")[:-32],  # truncated scan
    b"pretend image bytes",
])
def test_truncated_or_unreadable_images_are_400_before_model(client_and_loads, content):
    client, loads = client_and_loads
    response = upload(client, content)
    assert response.status_code == 400, response.text
    assert loads == []


def test_truncated_jpeg_with_matching_metadata_is_400(client_and_loads):
    client, loads = client_and_loads
    partial = encoded_image("JPEG", "RGB")[:-24]
    response = upload(client, partial, filename="item.jpg", mime="image/jpeg")
    assert response.status_code == 400, response.text
    assert loads == []


def test_png_bad_crc_rejected_before_model(client_and_loads):
    client, loads = client_and_loads
    original = encoded_image("PNG")
    changed = bytearray(original)
    position = original.index(b"IDAT") + 5
    changed[position] ^= 0x01
    response = upload(client, bytes(changed))
    assert response.status_code == 400, response.text
    assert loads == []


def test_animated_png_is_rejected_before_model(client_and_loads):
    client, loads = client_and_loads
    response = upload(client, animated_png())
    assert response.status_code == 400, response.text
    assert "multi-frame" in response.json()["detail"]
    assert loads == []


def test_multipart_additional_text_field_is_rejected(client_and_loads):
    client, loads = client_and_loads
    response = client.post("/predict", data={"note": "ignore"}, files={
        "file": ("item.png", encoded_image(), "image/png"),
    })
    assert response.status_code == 400, response.text
    assert loads == []


def test_size_limit_413_precedes_model_even_if_bytes_are_invalid(client_and_loads):
    client, loads = client_and_loads
    content = b"P" * (prediction.MAX_UPLOAD_BYTES + 1)
    response = upload(client, content)
    assert response.status_code == 413
    assert loads == []


def test_invalid_upload_preserves_other_api_endpoints(client_and_loads):
    client, loads = client_and_loads
    assert upload(client, b"garbage").status_code == 400
    assert client.get("/health").json() == {"status": "ok"}
    assert "/predict" in client.get("/openapi.json").json()["paths"]
    assert loads == []


def test_decoder_returns_detached_single_frame_image():
    content = encoded_image("JPEG", "RGB")
    decoded = prediction.decode_uploaded_image(content, expected_format="JPEG")
    try:
        assert decoded.mode == "RGB"
        assert decoded.size == (28, 28)
        decoded.getpixel((0, 0))  # source BytesIO is closed
    finally:
        decoded.close()


def test_low_level_decoder_rejects_non_byte_objects():
    with pytest.raises(prediction.InvalidUploadedImage):
        prediction.decode_uploaded_image(bytearray(encoded_image()))


def test_metadata_validation_is_explicit_and_platform_independent():
    assert prediction.validate_image_upload_metadata(
        filename=r"C:\fakepath\item.jpeg", content_type="image/jpeg"
    ) == "JPEG"
    with pytest.raises(prediction.UnsupportedUploadedImageType):
        prediction.validate_image_upload_metadata(filename=None, content_type="image/png")


def test_empty_filename_is_not_parsed_as_uploadfile(client_and_loads):
    client, loads = client_and_loads
    response = upload(client, encoded_image(), filename="")
    # Starlette parses an empty filename as a normal form field. FastAPI
    # rejects it at the required UploadFile contract (422).
    assert response.status_code == 422
    assert loads == []
