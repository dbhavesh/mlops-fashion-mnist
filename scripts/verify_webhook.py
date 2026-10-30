"""Read-only acceptance check against MLflow and MinIO.

Usage:
    python scripts/verify_webhook.py --payload artifacts/evidence/webhook.json

The JSON must be the actual webhook receiver capture. No remote writes occur.
"""

import argparse
import json
from pathlib import Path

from fashion_mnist_mlops.webhook_verification import verify_webhook_evidence
from fashion_mnist_mlops.config import (
    load_infrastructure_config,
    validate_infrastructure_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True,
                        help="Actual JSON capture from the local receiver")
    args = parser.parse_args()
    try:
        with args.payload.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        settings = load_infrastructure_config()
        validate_infrastructure_config(settings, require_minio_credentials=True)

        from mlflow.tracking import MlflowClient
        from minio import Minio

        tracking = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
        storage = Minio(
            endpoint=settings.minio.endpoint,
            access_key=settings.minio.access_key,
            secret_key=settings.minio.secret_key,
            secure=settings.minio.secure,
        )
        evidence = verify_webhook_evidence(
            payload, tracking_client=tracking, minio_client=storage,
            experiment_name=settings.mlflow.experiment_name,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")

    print("PASS: Captured webhook matches a FINISHED MLflow run and verified MinIO bytes")
    print(f"MLflow run: {evidence.mlflow_run_id}")
    print(f"MinIO model: {evidence.minio_bucket}/{evidence.minio_object_key}")
    print(f"Validation accuracy: {evidence.validation_accuracy:.6f}")
    print(f"Model bytes: {evidence.model_size_bytes}")
    print(f"Model SHA-256: {evidence.model_sha256}")
    print("Read-only verification: no MLflow or MinIO records were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
