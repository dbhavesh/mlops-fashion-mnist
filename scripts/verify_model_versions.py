"""Read-only verification of two completed MinIO-backed MLflow runs.

Usage:
    python scripts/verify_model_versions.py --older-run ID1 --newer-run ID2
"""

import argparse
import pickle

from fashion_mnist_mlops.config import load_infrastructure_config
from fashion_mnist_mlops.config_validation import validate_infrastructure_config
from fashion_mnist_mlops.model_verification import verify_model_versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older-run", required=True, help="Earlier FINISHED MLflow run ID")
    parser.add_argument("--newer-run", required=True, help="Later FINISHED MLflow run ID")
    args = parser.parse_args()

    try:
        config = load_infrastructure_config()
        validate_infrastructure_config(config, require_minio_credentials=True)

        from mlflow.tracking import MlflowClient
        from minio import Minio

        mlflow_client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
        minio_client = Minio(
            endpoint=config.minio.endpoint,
            access_key=config.minio.access_key,
            secret_key=config.minio.secret_key,
            secure=config.minio.secure,
        )
        older, newer = verify_model_versions(
            mlflow_client,
            minio_client,
            older_run_id=args.older_run,
            newer_run_id=args.newer_run,
            experiment_name=config.mlflow.experiment_name,
            bucket_name=config.minio.bucket,
        )
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError,
            RuntimeError, ConnectionError, OSError, EOFError,
            pickle.UnpicklingError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")

    print("PASS: Both MinIO model versions remain independent and byte-identical to their MLflow references")
    for label, info in (("Older", older), ("Newer", newer)):
        print(f"{label} run: {info.run_id}")
        print(f"  Bucket/key: {config.minio.bucket}/{info.object_key}")
        print(f"  Size (bytes): {info.size_bytes}")
        print(f"  SHA-256: {info.sha256}")
        print(f"  Synthetic prediction: {info.predicted_class}")
    print("No MinIO objects or MLflow runs were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
