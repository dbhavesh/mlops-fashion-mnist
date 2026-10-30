"""real HTTP integration between the Streamlit client and FastAPI.

These tests start the project's FastAPI application on an ephemeral loopback
port. Only the expensive model-loading/inference boundary is replaced with an
in-process fake; multipart transport, FastAPI request validation, response
serialization, and the Streamlit HTTP client remain real.
"""

from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
import socket
from threading import Thread
import time
from types import SimpleNamespace

from PIL import Image
import pytest
import uvicorn

from fashion_mnist_mlops.api import create_app
from fashion_mnist_mlops.model_loading import APIModelLoadError
from fashion_mnist_mlops import api as predict_route
from fashion_mnist_mlops.ui_client import (
    PredictionClientError,
    PredictionFailure,
    PredictionResult,
    predict_image,
)


RUN_ID = "9629c8e5feb84a0c940926b6ecbc726f"
INTEGRATION_REQUEST_TIMEOUT_SECONDS = 15.0


def _png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("L", (28, 28), color=128).save(stream, format="PNG")
    return stream.getvalue()


@contextmanager
def _live_fastapi(application):
    """Serve one FastAPI app on a Windows-safe ephemeral loopback port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]

    config = uvicorn.Config(
        application,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = Thread(
        target=server.run,
        kwargs={"sockets": [sock]},
        name="fastapi-test-server",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 5.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2.0)
        sock.close()
        raise RuntimeError("FastAPI integration server failed to start.")

    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        sock.close()
        if thread.is_alive():
            raise RuntimeError("FastAPI integration server failed to stop cleanly.")


def _configure_success(monkeypatch, application, *, class_index=8, class_name="Bag"):
    monkeypatch.setattr(
        application.state.model_loader,
        "load",
        lambda: SimpleNamespace(run_id=RUN_ID),
    )
    monkeypatch.setattr(
        predict_route,
        "predict_uploaded_image",
        lambda restored, image: SimpleNamespace(
            class_index=class_index,
            class_name=class_name,
        ),
    )


def test_streamlit_client_reaches_real_fastapi_predict_over_http(monkeypatch):
    application = create_app()
    _configure_success(monkeypatch, application)

    with _live_fastapi(application) as base_url:
        result = predict_image(
            base_url=base_url,
            filename="bag.png",
            content=_png_bytes(),
            content_type="image/png",
            timeout_seconds=INTEGRATION_REQUEST_TIMEOUT_SECONDS,
        )

    assert result == PredictionResult(
        class_index=8,
        class_name="Bag",
        model_run_id=RUN_ID,
    )


def test_invalid_image_crosses_http_boundary_and_is_rejected_before_model(monkeypatch):
    application = create_app()
    loads = []
    monkeypatch.setattr(
        application.state.model_loader,
        "load",
        lambda: loads.append(True),
    )

    with _live_fastapi(application) as base_url:
        with pytest.raises(PredictionClientError) as caught:
            predict_image(
                base_url=base_url,
                filename="broken.png",
                content=b"not actually a png",
                content_type="image/png",
                timeout_seconds=INTEGRATION_REQUEST_TIMEOUT_SECONDS,
            )

    assert caught.value.failure == PredictionFailure.HTTP
    assert caught.value.status_code == 400
    assert loads == []


def test_unavailable_selected_model_crosses_http_boundary_as_503(monkeypatch):
    application = create_app()

    def unavailable():
        raise APIModelLoadError("integration-only hidden model detail")

    monkeypatch.setattr(application.state.model_loader, "load", unavailable)

    with _live_fastapi(application) as base_url:
        with pytest.raises(PredictionClientError) as caught:
            predict_image(
                base_url=base_url,
                filename="shirt.png",
                content=_png_bytes(),
                content_type="image/png",
                timeout_seconds=INTEGRATION_REQUEST_TIMEOUT_SECONDS,
            )

    assert caught.value.failure == PredictionFailure.HTTP
    assert caught.value.status_code == 503
    assert "hidden model detail" not in str(caught.value)


def test_stopped_api_becomes_sanitized_unavailable_failure(monkeypatch):
    application = create_app()
    _configure_success(monkeypatch, application)

    with _live_fastapi(application) as base_url:
        assert predict_image(
            base_url=base_url,
            filename="shirt.png",
            content=_png_bytes(),
            content_type="image/png",
            timeout_seconds=INTEGRATION_REQUEST_TIMEOUT_SECONDS,
        ).model_run_id == RUN_ID

    with pytest.raises(PredictionClientError) as caught:
        predict_image(
            base_url=base_url,
            filename="shirt.png",
            content=_png_bytes(),
            content_type="image/png",
            timeout_seconds=2.0,
        )

    # On Windows, a port that has just been released can surface either an
    # immediate connection refusal or a short timeout depending on socket
    # teardown timing. Both are sanitized unavailable-backend outcomes.
    assert caught.value.failure in {
        PredictionFailure.CONNECTION,
        PredictionFailure.TIMEOUT,
    }
    assert caught.value.status_code is None
