"""offline FastAPI import/startup and modular structure tests."""

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fashion_mnist_mlops.model_loading import APISettings
from fashion_mnist_mlops.api import app, create_app
from fashion_mnist_mlops.api import api_router


def test_asgi_entrypoint_is_a_fastapi_app():
    assert isinstance(app, FastAPI)
    assert app.title == "Fashion-MNIST Prediction API"
    assert app.version == "0.1.0"


def test_openapi_metadata_endpoint_is_available_offline():
    with TestClient(app) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == app.title
    assert schema["info"]["version"] == app.version
    assert isinstance(schema["paths"], dict)



def test_factory_creates_distinct_instances():
    first = create_app()
    second = create_app()
    assert first is not second
    assert first is not app
    assert first.router is not second.router


def test_factory_accepts_explicit_presentation_settings():
    custom = create_app(APISettings(title="Test classifier", version="8.1"))
    assert custom.title == "Test classifier"
    assert custom.version == "8.1"
    with TestClient(custom) as client:
        assert client.get("/openapi.json").json()["info"]["title"] == "Test classifier"


@pytest.mark.parametrize("invalid", [False, "unexpected", {}, object()])
def test_factory_rejects_invalid_settings(invalid):
    with pytest.raises(TypeError, match="APISettings"):
        create_app(invalid)


def test_modular_boundaries_are_importable_without_loading_model():
    assert api_router is not None
    for module in (
        "fashion_mnist_mlops.api",
        "fashion_mnist_mlops.model_loading",
        "fashion_mnist_mlops.model_loading",
        "fashion_mnist_mlops.prediction",
    ):
        assert importlib.import_module(module) is not None


def test_package_exports_asgi_entrypoint():
    package = importlib.import_module("fashion_mnist_mlops.api")
    assert package.app is app
    assert package.create_app is create_app
