"""Read-only storage-readiness check for an existing local model artifact.

Example: python scripts/inspect_model_artifact.py artifacts/models/<id>/model.pt
"""

import argparse
import pickle
from pathlib import Path

from fashion_mnist_mlops.artifact_format import inspect_model_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Local .pt artifact path")
    parser.add_argument("--run-id", help="Expected MLflow run ID, if tracked")
    args = parser.parse_args()

    try:
        info = inspect_model_artifact(args.artifact, expected_run_id=args.run_id)
    except (
        ValueError, FileNotFoundError, RuntimeError, OSError,
        EOFError, pickle.UnpicklingError,
    ) as exc:
        parser.exit(1, f"FAIL: {exc}\n")

    print("PASS: Version-2 model artifact is ready for storage")
    print(f"Architecture: {info.architecture}")
    print(f"Input shape: {list(info.input_shape)}")
    print(f"Classes: {info.num_classes}")
    print(f"Preprocessing version: {info.preprocessing_version}")
    print(f"Artifact epoch: {info.epoch}")
    print(f"MLflow run ID: {info.mlflow_run_id or '(untracked)'}")
    print(f"Size (bytes): {info.size_bytes}")
    print(f"SHA-256: {info.sha256}")
    print("No object was uploaded or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
