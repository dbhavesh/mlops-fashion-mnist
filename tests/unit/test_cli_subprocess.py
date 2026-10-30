"""Test the actual training CLI as a separate Python process.

All cases are offline and should not launch a training epoch.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_cli(*arguments: str, cwd: Path, timeout: int = 120):
    """Run exactly the interpreter executing pytest, independent of the CWD."""
    env = os.environ.copy()
    # Exercise editable source-tree discovery unless a test supplies --project-root.
    env.pop("FASHION_MNIST_PROJECT_ROOT", None)
    return subprocess.run(
        [sys.executable, "-m", "fashion_mnist_mlops.train", *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_cli_help_works_outside_repository(tmp_path):
    result = run_cli("--help", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "--epochs" in result.stdout
    assert "--project-root" in result.stdout
    assert "--max-train-samples" in result.stdout
    assert "--offline" in result.stdout
    assert "--download-data" in result.stdout


def test_cli_rejects_nonpositive_epochs(tmp_path):
    result = run_cli("--epochs", "0", cwd=tmp_path)

    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert "Training completed successfully" not in result.stdout


def test_cli_rejects_conflicting_download_flags(tmp_path):
    result = run_cli("--offline", "--download-data", cwd=tmp_path)

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_cli_rejects_missing_config(tmp_path):
    missing = tmp_path / "missing.toml"
    result = run_cli("--config", str(missing), cwd=tmp_path)

    assert result.returncode != 0
    assert "configuration file not found" in result.stderr.lower()
    assert not missing.exists()


def test_cli_rejects_invalid_artifact_extension(tmp_path):
    invalid = tmp_path / "artifact.txt"
    result = run_cli("--artifact-path", str(invalid), cwd=tmp_path)

    assert result.returncode != 0
    assert ".pt extension" in result.stderr
    assert not invalid.exists()


def test_cli_offline_missing_dataset_never_creates_artifact(tmp_path):
    """An explicit empty project root must not silently download data."""
    isolated_root = tmp_path / "isolated_project"
    (isolated_root / "configs").mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / "configs" / "train.toml",
        isolated_root / "configs" / "train.toml",
    )
    destination = isolated_root / "artifacts" / "models" / "model.pt"
    external_cwd = tmp_path / "elsewhere"
    external_cwd.mkdir()

    result = run_cli(
        "--project-root", str(isolated_root),
        "--offline", "--epochs", "1",
        "--max-train-samples", "8",
        "--max-validation-samples", "8",
        "--artifact-path", str(destination),
        cwd=external_cwd,
    )

    assert result.returncode != 0
    assert "offline mode" in result.stderr.lower()
    assert "--download-data" in result.stderr
    assert not destination.exists()
    assert not (external_cwd / "artifacts").exists()
