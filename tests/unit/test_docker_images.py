"""Static acceptance tests for API/UI container images."""

from __future__ import annotations

import re
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]
API_DOCKERFILE = ROOT / "docker" / "api.Dockerfile"
UI_DOCKERFILE = ROOT / "docker" / "ui.Dockerfile"
API_REQUIREMENTS = ROOT / "docker" / "api.requirements.txt"
UI_REQUIREMENTS = ROOT / "docker" / "ui.requirements.txt"
DOCKERIGNORE = ROOT / ".dockerignore"

SECRET_NAMES = {
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "WEBHOOK_URL",
}


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", line)
        assert match, f"Could not parse requirement line: {raw!r}"
        names.add(match.group(0).lower().replace("_", "-"))
    return names


def _project_dependency_names() -> set[str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]
    names: set[str] = set()
    for requirement in dependencies:
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        assert match
        names.add(match.group(0).lower().replace("_", "-"))
    return names


def test_runtime_requirement_lists_are_declared_project_dependencies():
    project = _project_dependency_names()
    assert _requirement_names(API_REQUIREMENTS) <= project
    assert _requirement_names(UI_REQUIREMENTS) <= project


def test_api_runtime_contains_serving_dependencies_but_not_streamlit():
    requirements = _requirement_names(API_REQUIREMENTS)
    assert {
        "numpy",
        "torch",
        "torchvision",
        "mlflow",
        "minio",
        "fastapi",
        "uvicorn",
        "pillow",
        "python-multipart",
    } <= requirements
    assert "streamlit" not in requirements


def test_ui_runtime_is_presentation_only():
    requirements = _requirement_names(UI_REQUIREMENTS)
    assert requirements == {"streamlit", "requests"}
    assert requirements.isdisjoint(
        {"torch", "torchvision", "mlflow", "minio", "fastapi", "uvicorn"}
    )


def test_api_dockerfile_uses_python_312_non_root_cpu_torch_and_expected_command():
    text = API_DOCKERFILE.read_text(encoding="utf-8")
    assert text.startswith("FROM python:3.12-slim-bookworm\n")
    assert "USER app" in text
    assert "EXPOSE 8000" in text
    assert "COPY --chown=app:app src/ /app/src/" in text
    assert "--index-url https://download.pytorch.org/whl/cpu" in text
    assert "torch torchvision" in text
    assert (
        'CMD ["python", "-m", "uvicorn", '
        '"fashion_mnist_mlops.api:app", "--host", "0.0.0.0", '
        '"--port", "8000"]'
    ) in text


def test_ui_dockerfile_contains_only_ui_project_source_and_expected_command():
    text = UI_DOCKERFILE.read_text(encoding="utf-8")
    assert text.startswith("FROM python:3.12-slim-bookworm\n")
    assert "USER app" in text
    assert "EXPOSE 8501" in text
    assert "COPY --chown=app:app src/ /app/src/" not in text
    for name in ("ui.py", "ui_client.py", "ui_config.py"):
        assert f"src/fashion_mnist_mlops/{name}" in text
    assert "src/fashion_mnist_mlops/model.py" not in text
    assert "src/fashion_mnist_mlops/training" not in text
    assert (
        'CMD ["python", "-m", "streamlit", "run", '
        '"src/fashion_mnist_mlops/ui.py", "--server.address=0.0.0.0", '
        '"--server.port=8501", "--server.headless=true", '
        '"--browser.gatherUsageStats=false"]'
    ) in text


def test_dockerfiles_do_not_bake_runtime_secrets_or_service_configuration():
    combined = (
        API_DOCKERFILE.read_text(encoding="utf-8")
        + "\n"
        + UI_DOCKERFILE.read_text(encoding="utf-8")
    )
    for name in SECRET_NAMES:
        assert name not in combined
    assert "MODEL_OBJECT_KEY" not in combined
    assert "MINIO_ENDPOINT" not in combined
    assert "MLFLOW_TRACKING_URI" not in combined
    assert "FASTAPI_BASE_URL" not in combined


def test_dockerfiles_avoid_broad_context_copy_and_install_without_pip_cache():
    api = API_DOCKERFILE.read_text(encoding="utf-8")
    ui = UI_DOCKERFILE.read_text(encoding="utf-8")
    for text in (api, ui):
        assert "COPY . " not in text
        assert "COPY ./" not in text
        assert "pip install --no-cache-dir" in text


def test_dockerignore_excludes_secrets_environments_and_generated_state():
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    for required in (
        ".git/",
        ".venv/",
        ".env",
        "data/",
        "artifacts/",
        "tests/",
        "docs/",
        "**/__pycache__/",
    ):
        assert required in text
    assert "!.env.example" in text
