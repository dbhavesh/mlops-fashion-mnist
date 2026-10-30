"""Static contracts for Compose orchestration."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "compose.yaml"
TRAINING_DOCKERFILE = ROOT / "docker" / "training.Dockerfile"
TRAINING_REQUIREMENTS = ROOT / "docker" / "training.requirements.txt"
MLFLOW_DOCKERFILE = ROOT / "docker" / "mlflow.Dockerfile"
MINIO_DOCKERFILE = ROOT / "docker" / "minio.Dockerfile"


def _compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_compose_defines_expected_services_and_shared_network():
    text = _compose_text()
    for service in ("minio", "mlflow", "api", "ui", "training"):
        assert f"  {service}:\n" in text
    assert "name: mlops-fashion-mnist-network" in text
    assert text.count("      - mlops") == 5


def test_compose_uses_service_dns_for_internal_application_traffic():
    text = _compose_text()
    assert 'MLFLOW_TRACKING_URI: "http://mlflow:5000"' in text
    assert text.count('MINIO_ENDPOINT: "minio:9000"') == 2
    assert 'FASTAPI_BASE_URL: "http://api:8000"' in text


def test_compose_preserves_exact_bucket_and_external_model_selection():
    text = _compose_text()
    assert text.count('MINIO_BUCKET: "fashion-minst-bucket"') == 2
    assert 'MODEL_OBJECT_KEY: "${MODEL_OBJECT_KEY:-}"' in text
    assert "fashion-mnist-bucket" not in text


def test_compose_requires_credentials_without_embedding_values():
    text = _compose_text()
    assert text.count("${MINIO_ACCESS_KEY:?") == 3
    assert text.count("${MINIO_SECRET_KEY:?") == 3
    assert 'MINIO_ROOT_USER: "${MINIO_ACCESS_KEY:' in text
    assert 'MINIO_ROOT_PASSWORD: "${MINIO_SECRET_KEY:' in text
    assert "minioadmin" not in text
    assert "password123" not in text


def test_host_ports_are_configurable_while_container_ports_are_stable():
    text = _compose_text()
    for mapping in (
        '"127.0.0.1:${MINIO_API_PORT:-19000}:9000"',
        '"127.0.0.1:${MINIO_CONSOLE_PORT:-19001}:9001"',
        '"127.0.0.1:${MLFLOW_HOST_PORT:-15000}:5000"',
        '"127.0.0.1:${API_HOST_PORT:-18000}:8000"',
        '"127.0.0.1:${UI_HOST_PORT:-18501}:8501"',
    ):
        assert mapping in text


def test_training_is_explicit_profiled_one_shot_job():
    text = _compose_text()
    assert 'profiles: ["training"]' in text
    assert 'restart: "no"' in text
    assert "python scripts/provision_minio_bucket.py" in text
    assert "python -m fashion_mnist_mlops.train" in text
    assert "--project-root /app" in text
    assert "--download-data" in text
    assert "--track-mlflow" in text
    assert "--upload-minio" in text


def test_training_runtime_is_cpu_only_non_root_and_excludes_serving_packages():
    dockerfile = TRAINING_DOCKERFILE.read_text(encoding="utf-8")
    requirements = TRAINING_REQUIREMENTS.read_text(encoding="utf-8").lower()
    assert "FROM python:3.12-slim-bookworm" in dockerfile
    assert "--index-url https://download.pytorch.org/whl/cpu" in dockerfile
    assert "USER app" in dockerfile
    assert "streamlit" not in requirements
    assert "fastapi" not in requirements
    assert "uvicorn" not in requirements


def test_training_image_contains_config_and_bucket_provisioning_script():
    text = TRAINING_DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY --chown=app:app configs/ /app/configs/" in text
    assert (
        "COPY --chown=app:app scripts/provision_minio_bucket.py "
        "/app/scripts/provision_minio_bucket.py"
    ) in text


def test_mlflow_image_uses_single_worker_non_root_server_and_host_allowlist():
    text = MLFLOW_DOCKERFILE.read_text(encoding="utf-8")
    assert "USER app" in text
    assert '"--workers", "1"' in text
    assert '"--host", "0.0.0.0"' in text
    assert '"--allowed-hosts", "mlflow:5000,localhost:*,127.0.0.1:*"' in text
    assert "sqlite:////mlflow/tracking.db" in text
    assert "/mlflow/mlartifacts" in text


def test_minio_is_built_from_pinned_official_source_not_registry_mirror():
    compose = _compose_text()
    dockerfile = MINIO_DOCKERFILE.read_text(encoding="utf-8")
    assert "dockerfile: docker/minio.Dockerfile" in compose
    assert 'MINIO_VERSION: "${MINIO_VERSION:-RELEASE.2025-09-07T16-13-09Z}"' in compose
    assert "quay.io/minio/minio" not in compose
    assert "minio/minio:latest" not in compose
    assert "git clone --depth 1 --branch" in dockerfile
    assert "github.com/minio/minio.git" in dockerfile
    assert "MINIO_RELEASE=RELEASE go run buildscripts/gen-ldflags.go" in dockerfile
    assert "ARG MINIO_VERSION=RELEASE.2025-09-07T16-13-09Z" in dockerfile
    assert '/out/minio --version | grep -F "minio version ${MINIO_VERSION}"' in dockerfile


def test_minio_runtime_image_is_non_root_and_exposes_expected_ports():
    text = MINIO_DOCKERFILE.read_text(encoding="utf-8")
    assert "CGO_ENABLED=0" in text
    assert "USER app" in text
    assert "EXPOSE 9000 9001" in text
    assert 'ENTRYPOINT ["/usr/local/bin/minio"]' in text
