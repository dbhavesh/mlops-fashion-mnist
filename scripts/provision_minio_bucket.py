"""Create the exact Fashion-MNIST MinIO bucket if absent.

Run from the project root with configured MINIO_* environment variables:
    python scripts/provision_minio_bucket.py
"""

from fashion_mnist_mlops.config import (
    load_infrastructure_config,
    validate_infrastructure_config,
)
from fashion_mnist_mlops.storage import ensure_bucket_exists


def main() -> int:
    # Keep credentials out of output and source control.
    config = load_infrastructure_config()
    validate_infrastructure_config(config, require_minio_credentials=True)
    settings = config.minio

    from minio import Minio

    client = Minio(
        endpoint=settings.endpoint,
        access_key=settings.access_key,
        secret_key=settings.secret_key,
        secure=settings.secure,
    )
    created = ensure_bucket_exists(client, bucket_name=settings.bucket)
    print(f"PASS: Required MinIO bucket exists: {settings.bucket}")
    print("Action:", "created" if created else "already existed (no changes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, PermissionError, ConnectionError) as exc:
        raise SystemExit(f"FAIL: {exc}") from None
