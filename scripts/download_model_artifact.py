"""Download the selected version-verified MinIO model without overwriting files.

Example:
  MODEL_OBJECT_KEY=model/<run-id>/model.pt python scripts/download_model_artifact.py \\
    --destination artifacts/downloads/<run-id>/model.pt
"""

import argparse
from pathlib import Path

from fashion_mnist_mlops.storage import retrieve_model_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True,
                        help="New local artifact file (must not already exist)")
    parser.add_argument("--object-key", help="Override configured MODEL_OBJECT_KEY")
    args = parser.parse_args()
    try:
        result = retrieve_model_artifact(args.destination, object_key=args.object_key)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")

    print("PASS: MinIO model downloaded and verified")
    print(f"Bucket: {result.bucket}")
    print(f"Object key: {result.object_key}")
    print(f"Local file: {result.path}")
    print(f"Size (bytes): {result.size_bytes}")
    print(f"SHA-256: {result.sha256}")
    print("No model was loaded or used for inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
