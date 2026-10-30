"""test the API process health route independently of services."""

import pytest
from fastapi.testclient import TestClient

from fashion_mnist_mlops.api import app, create_app
from fashion_mnist_mlops.api import HealthResponse


@pytest.mark.parametrize("application", [app, create_app()])
def test_health_get_returns_200_with_exact_json(application):
    with TestClient(application) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["content-type"].startswith("application/json")


def test_health_is_listed_in_openapi_with_response_schema():
    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()
    path = schema["paths"]["/health"]
    assert "get" in path
    assert path["get"]["tags"] == ["health"]
    assert "200" in path["get"]["responses"]
    assert (
        path["get"]["responses"]["200"]["content"]["application/json"]
        ["schema"]["$ref"].endswith("/HealthResponse")
    )
    assert "HealthResponse" in schema["components"]["schemas"]


def test_health_is_process_liveness_not_dependency_readiness(monkeypatch):
    # Intentionally wrong service addresses and no model selection configured.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    monkeypatch.setenv("MINIO_ENDPOINT", "127.0.0.1:1")
    monkeypatch.delenv("MODEL_OBJECT_KEY", raising=False)

    # Do not monkeypatch socket.socket.connect: on Windows, asyncio's
    # TestClient event loop uses internal loopback sockets to start up.
    # Invalid downstream URLs prove health does not require live services.
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_get_does_not_require_parameters():
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_health_post_is_not_allowed():
    with TestClient(create_app()) as client:
        response = client.post("/health")
    assert response.status_code == 405


def test_health_response_schema_rejects_non_ok_values():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HealthResponse(status="unhealthy")


def test_app_factory_exposes_one_health_operation_in_openapi():
    # Assert the public API contract rather than framework-internal route lists.
    for application in (app, create_app(), create_app()):
        with TestClient(application) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

            openapi_response = client.get("/openapi.json")
            assert openapi_response.status_code == 200
            operations = openapi_response.json()["paths"]["/health"]
            assert list(operations) == ["get"]
