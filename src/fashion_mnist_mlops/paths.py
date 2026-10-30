import os
from pathlib import Path

from .config_validation import ConfigurationError


PROJECT_ROOT_ENV = "FASHION_MNIST_PROJECT_ROOT"


def _discover_source_project_root() -> Path | None:
    """Find the project root from an editable source installation."""

    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "configs" / "train.toml").is_file()
        ):
            return candidate

    return None


def get_project_root(
    project_root: str | Path | None = None,
) -> Path:
    """Resolve an explicit, configured, or discoverable project root."""

    selected = project_root

    if selected is None:
        environment_value = os.environ.get(PROJECT_ROOT_ENV)

        if environment_value and environment_value.strip():
            selected = environment_value.strip()

    if selected is None:
        discovered = _discover_source_project_root()

        if discovered is None:
            raise ConfigurationError(
                "Unable to discover the project root. "
                "Supply --project-root or set "
                f"{PROJECT_ROOT_ENV} to an absolute directory."
            )

        return discovered

    root = Path(selected).expanduser()

    if not root.is_absolute():
        raise ConfigurationError(
            "Project root must be an absolute path."
        )

    root = root.resolve()

    if not root.is_dir():
        raise ConfigurationError(
            f"Project root directory does not exist: {root}"
        )

    return root


def resolve_project_path(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve relative paths against the project root, not the CWD."""

    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        return candidate.resolve()

    return (
        get_project_root(project_root) / candidate
    ).resolve()
