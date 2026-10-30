"""download a selected MinIO artifact and run CPU inference.

The destination must NOT already exist. The example prediction uses a fixed
synthetic grayscale image, not a Fashion-MNIST test sample or an accuracy claim.
"""

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

from fashion_mnist_mlops.preprocessing import (
    preprocess_image_for_inference,
)
from fashion_mnist_mlops.inference import predict_preprocessed_image, retrieve_inference_model
from fashion_mnist_mlops.model_artifacts import (
    load_model_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True,
                        help="New local artifact filename (must not exist)")
    parser.add_argument("--object-key", help="Explicit model/<run-id>/model.pt key, or use MODEL_OBJECT_KEY")
    parser.add_argument("--compare-local-artifact", type=Path,
                        help="Optional original artifact for exact CPU-logits comparison")
    args = parser.parse_args()
    try:
        restored = retrieve_inference_model(args.destination, object_key=args.object_key)
        image = preprocess_image_for_inference(Image.new("L", (28, 28), color=128))
        prediction = predict_preprocessed_image(restored, image)

        if args.compare_local_artifact is not None:
            original, metadata = load_model_artifact(args.compare_local_artifact, device="cpu")
            if metadata["mlflow_run_id"] != restored.run_id:
                raise ValueError("Reference artifact belongs to a different MLflow run.")
            with torch.inference_mode():
                if not torch.equal(restored.model(image), original(image)):
                    raise ValueError("Downloaded and original model logits differ.")
            print("PASS: Downloaded and original artifact logits match exactly")
    except (ValueError, TypeError, FileNotFoundError, FileExistsError, RuntimeError,
            OSError, EOFError, pickle.UnpicklingError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")

    print("PASS: Verified MinIO model restored and used for inference")
    print(f"MLflow run ID: {restored.run_id}")
    print(f"Artifact: {restored.artifact_path}")
    print(f"SHA-256: {restored.sha256}")
    print(f"Synthetic-image prediction: {prediction.class_index} ({prediction.class_name})")
    print("No training, MLflow logging, or model upload was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
