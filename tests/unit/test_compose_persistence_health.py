"""Static contracts for persistence and health wiring."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "compose.yaml"
MINIO_DOCKERFILE = ROOT / "docker" / "minio.Dockerfile"


def _compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_minio_and_mlflow_use_named_persistent_volumes():
    text = _compose_text()
    assert "      - minio-data:/data" in text
    assert "      - mlflow-data:/mlflow" in text
    assert "  minio-data:\n    name: mlops-fashion-mnist-minio-data" in text
    assert "  mlflow-data:\n    name: mlops-fashion-mnist-mlflow-data" in text


def test_minio_healthcheck_uses_readiness_endpoint():
    text = _compose_text()
    assert 'http://127.0.0.1:9000/minio/health/ready' in text
    minio_dockerfile = MINIO_DOCKERFILE.read_text(encoding="utf-8")
    assert "apt-get install -y --no-install-recommends curl" in minio_dockerfile


def test_mlflow_cold_start_has_sufficient_start_period():
    text = _compose_text()
    mlflow_block = text.split("  mlflow:", 1)[1].split("\n  api:", 1)[0]

    assert "start_period: 240s" in mlflow_block
    assert "retries: 30" in mlflow_block


def test_mlflow_healthcheck_uses_native_health_endpoint():
    text = _compose_text()
    assert "http://127.0.0.1:5000/health" in text


def test_api_healthcheck_is_liveness_not_model_readiness():
    text = _compose_text()
    assert "http://127.0.0.1:8000/health" in text
    assert "http://127.0.0.1:8000/ready" not in text


def test_ui_healthcheck_uses_streamlit_health_endpoint():
    text = _compose_text()
    assert "http://127.0.0.1:8501/_stcore/health" in text


def test_api_waits_for_minio_and_mlflow_health():
    text = _compose_text()
    api_block = text.split("\n  api:\n", 1)[1].split("\n  ui:\n", 1)[0]
    assert "depends_on:" in api_block
    assert "minio:\n        condition: service_healthy" in api_block
    assert "mlflow:\n        condition: service_healthy" in api_block


def test_ui_waits_for_fastapi_process_health():
    text = _compose_text()
    ui_block = text.split("\n  ui:\n", 1)[1].split("\n  training:\n", 1)[0]
    assert "api:\n        condition: service_healthy" in ui_block


def test_training_waits_for_storage_and_tracking_health():
    text = _compose_text()
    training_block = text.split("\n  training:\n", 1)[1].split("\nnetworks:\n", 1)[0]
    assert "minio:\n        condition: service_healthy" in training_block
    assert "mlflow:\n        condition: service_healthy" in training_block


def test_compose_does_not_use_fixed_sleep_for_readiness():
    text = _compose_text()
    assert "sleep " not in text
