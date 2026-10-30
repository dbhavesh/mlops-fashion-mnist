"""Offline acceptance tests for the Streamlit UI."""

from __future__ import annotations

import ast
from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"


class StopCalled(RuntimeError):
    pass


class FakeStreamlit(ModuleType):
    def __init__(self, *, uploaded_file=None, button_clicked=False):
        super().__init__("streamlit")
        self.calls: list[tuple[str, object]] = []
        self.uploaded_file = uploaded_file
        self.button_clicked = button_clicked

    def set_page_config(self, **kwargs):
        self.calls.append(("set_page_config", kwargs))

    def title(self, value):
        self.calls.append(("title", value))

    def write(self, value):
        self.calls.append(("write", value))

    def subheader(self, value):
        self.calls.append(("subheader", value))

    def markdown(self, value):
        self.calls.append(("markdown", value))

    def file_uploader(self, label, **kwargs):
        self.calls.append(("file_uploader", {"label": label, **kwargs}))
        return self.uploaded_file

    def image(self, image, **kwargs):
        self.calls.append(("image", {"image": image, **kwargs}))

    def button(self, label, **kwargs):
        self.calls.append(("button", {"label": label, **kwargs}))
        return self.button_clicked

    def spinner(self, value):
        self.calls.append(("spinner", value))
        return nullcontext()

    def success(self, value):
        self.calls.append(("success", value))

    def caption(self, value):
        self.calls.append(("caption", value))

    def error(self, value):
        self.calls.append(("error", value))

    def stop(self):
        self.calls.append(("stop", None))
        raise StopCalled


def uploaded_file(*, name="shirt.png", media_type="image/png", content=b"image-bytes"):
    return SimpleNamespace(
        name=name,
        type=media_type,
        getvalue=lambda: content,
    )


def import_app_with_fake_streamlit(
    monkeypatch,
    *,
    upload=None,
    button_clicked=False,
):
    fake = FakeStreamlit(
        uploaded_file=upload,
        button_clicked=button_clicked,
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    sys.modules.pop("fashion_mnist_mlops.ui", None)
    module = importlib.import_module("fashion_mnist_mlops.ui")
    return module, fake


def test_import_is_side_effect_free_and_does_not_render(monkeypatch):
    _, fake = import_app_with_fake_streamlit(monkeypatch)
    assert fake.calls == []


def test_shell_renders_title_instructions_and_external_backend(monkeypatch):
    monkeypatch.setenv("FASTAPI_BASE_URL", "http://api:8000")
    app, fake = import_app_with_fake_streamlit(monkeypatch)

    app.render_app()

    assert ("title", "Fashion-MNIST Classifier") in fake.calls
    assert any(name == "markdown" and "Predict" in str(value) for name, value in fake.calls)
    assert ("caption", "FastAPI backend: `http://api:8000`") in fake.calls
    assert not any(name == "error" for name, _ in fake.calls)


def test_invalid_backend_configuration_is_sanitized(monkeypatch):
    monkeypatch.setenv("FASTAPI_BASE_URL", "http://user:secret@api:8000")
    app, fake = import_app_with_fake_streamlit(monkeypatch)

    try:
        app.render_app()
    except StopCalled:
        pass

    errors = [str(value) for name, value in fake.calls if name == "error"]
    assert errors == [
        "The FastAPI backend address is invalid. Set FASTAPI_BASE_URL to "
        "a valid HTTP(S) service origin."
    ]
    assert "secret" not in errors[0]


def test_uploader_accepts_one_png_or_jpeg_and_disables_predict_without_file(monkeypatch):
    app, fake = import_app_with_fake_streamlit(monkeypatch)

    app.render_app()

    upload_call = next(value for name, value in fake.calls if name == "file_uploader")
    assert upload_call["label"] == "Upload a PNG or JPEG image"
    assert upload_call["type"] == ["png", "jpg", "jpeg"]
    assert upload_call["accept_multiple_files"] is False

    button_call = next(value for name, value in fake.calls if name == "button")
    assert button_call == {
        "label": "Predict",
        "type": "primary",
        "disabled": True,
    }
    assert not any(name == "image" for name, _ in fake.calls)
    assert not any(name == "spinner" for name, _ in fake.calls)


def test_selected_file_is_previewed_and_enables_predict(monkeypatch):
    upload = uploaded_file()
    app, fake = import_app_with_fake_streamlit(monkeypatch, upload=upload)

    app.render_app()

    image_call = next(value for name, value in fake.calls if name == "image")
    assert image_call == {
        "image": upload,
        "caption": "Preview: shirt.png",
    }
    button_call = next(value for name, value in fake.calls if name == "button")
    assert button_call["disabled"] is False


def test_normal_rerun_with_selected_file_does_not_call_fastapi_or_show_spinner(monkeypatch):
    upload = uploaded_file(name="dress.jpg", media_type="image/jpeg")
    app, fake = import_app_with_fake_streamlit(
        monkeypatch,
        upload=upload,
        button_clicked=False,
    )

    def fail_if_called(**kwargs):
        raise AssertionError("FastAPI must only be called after an explicit click")

    monkeypatch.setattr(app, "predict_image", fail_if_called)
    app.render_app()

    assert not any(name == "spinner" for name, _ in fake.calls)
    assert not any(name == "success" for name, _ in fake.calls)


def test_predict_click_shows_progress_calls_fastapi_once_and_renders_result(monkeypatch):
    upload = uploaded_file(
        name="coat.jpeg",
        media_type="image/jpeg",
        content=b"jpeg-content",
    )
    monkeypatch.setenv("FASTAPI_BASE_URL", "http://api:8000")
    app, fake = import_app_with_fake_streamlit(
        monkeypatch,
        upload=upload,
        button_clicked=True,
    )
    observed = []

    def fake_predict_image(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(
            class_index=4,
            class_name="Coat",
            model_run_id=RUN_ID,
        )

    monkeypatch.setattr(app, "predict_image", fake_predict_image)
    app.render_app()

    assert observed == [
        {
            "base_url": "http://api:8000",
            "filename": "coat.jpeg",
            "content": b"jpeg-content",
            "content_type": "image/jpeg",
        }
    ]
    assert ("spinner", "Requesting prediction from FastAPI...") in fake.calls
    assert ("subheader", "Prediction result") in fake.calls
    assert ("success", "Predicted class: **Coat**") in fake.calls
    assert ("write", "Class index: `4`") in fake.calls
    assert ("caption", f"Model run ID: `{RUN_ID}`") in fake.calls
    assert not any("confidence" in str(value).lower() for _, value in fake.calls)


@pytest.mark.parametrize(
    ("failure", "status_code", "expected"),
    [
        (
            "timeout",
            None,
            "The prediction request timed out. Confirm FastAPI is responsive and try again.",
        ),
        (
            "connection",
            None,
            "Could not connect to the FastAPI backend. Confirm it is running and FASTAPI_BASE_URL is correct.",
        ),
        (
            "transport",
            None,
            "The prediction request could not reach FastAPI. Check the backend and network, then try again.",
        ),
        (
            "invalid_upload",
            None,
            "The selected file could not be sent. Choose a non-empty PNG or JPEG image.",
        ),
        (
            "invalid_response",
            None,
            "FastAPI returned an unexpected prediction response. Confirm the API and UI versions are compatible, then try again.",
        ),
        (
            "http",
            400,
            "FastAPI rejected the image as invalid or unreadable. Choose a valid PNG or JPEG image.",
        ),
        (
            "http",
            413,
            "The image is too large for FastAPI. Choose an image smaller than 5 MiB.",
        ),
        (
            "http",
            415,
            "FastAPI does not support this image type. Choose a PNG or JPEG image.",
        ),
        (
            "http",
            422,
            "FastAPI could not validate the upload request. Re-select a PNG or JPEG image and try again.",
        ),
        (
            "http",
            500,
            "FastAPI encountered an internal prediction error. Try again; if it persists, check the API logs.",
        ),
        (
            "http",
            503,
            "The selected model is currently unavailable. Confirm FastAPI readiness and its MLflow/MinIO dependencies.",
        ),
        (
            "http",
            418,
            "FastAPI returned HTTP 418. Prediction could not be completed.",
        ),
    ],
)
def test_prediction_failures_render_actionable_sanitized_guidance(
    monkeypatch, failure, status_code, expected
):
    from fashion_mnist_mlops.ui_client import PredictionClientError, PredictionFailure

    app, fake = import_app_with_fake_streamlit(
        monkeypatch,
        upload=uploaded_file(),
        button_clicked=True,
    )

    def fail(**kwargs):
        raise PredictionClientError(
            "hidden credentials/path/traceback detail",
            failure=PredictionFailure(failure),
            status_code=status_code,
        )

    monkeypatch.setattr(app, "predict_image", fail)
    app.render_app()

    errors = [str(value) for name, value in fake.calls if name == "error"]
    assert errors == [expected]
    assert "hidden" not in errors[0]
    assert "credentials" not in errors[0]
    assert "traceback" not in errors[0]
    assert ("spinner", "Requesting prediction from FastAPI...") in fake.calls


def test_streamlit_app_does_not_import_http_or_model_runtime_directly(monkeypatch):
    app, _ = import_app_with_fake_streamlit(monkeypatch)
    source = Path(app.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    forbidden_prefixes = (
        "torch",
        "torchvision",
        "mlflow",
        "minio",
        "requests",
        "httpx",
        "fashion_mnist_mlops.api",
        "fashion_mnist_mlops.data",
        "fashion_mnist_mlops.inference",
        "fashion_mnist_mlops.model",
        "fashion_mnist_mlops.model_artifacts",
        "fashion_mnist_mlops.model_loading",
        "fashion_mnist_mlops.prediction",
        "fashion_mnist_mlops.preprocessing",
        "fashion_mnist_mlops.labels",
        "fashion_mnist_mlops.storage",
        "fashion_mnist_mlops.tracking",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported
        for prefix in forbidden_prefixes
    )


def test_http_client_does_not_import_model_or_inference_runtime():
    import fashion_mnist_mlops.ui_client as client

    source = Path(client.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    forbidden_prefixes = (
        "torch",
        "torchvision",
        "mlflow",
        "minio",
        "fashion_mnist_mlops.api",
        "fashion_mnist_mlops.data",
        "fashion_mnist_mlops.inference",
        "fashion_mnist_mlops.model",
        "fashion_mnist_mlops.model_artifacts",
        "fashion_mnist_mlops.model_loading",
        "fashion_mnist_mlops.prediction",
        "fashion_mnist_mlops.preprocessing",
        "fashion_mnist_mlops.labels",
        "fashion_mnist_mlops.storage",
        "fashion_mnist_mlops.tracking",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported
        for prefix in forbidden_prefixes
    )
