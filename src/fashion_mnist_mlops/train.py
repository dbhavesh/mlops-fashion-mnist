import argparse
from pathlib import Path

from fashion_mnist_mlops.config import (
    DEFAULT_CONFIG_PATH,
)

from .training import run_training


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Value must be a positive integer."
        ) from exc

    if number < 1:
        raise argparse.ArgumentTypeError(
            "Value must be a positive integer."
        )

    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and validate a LeNet-5-style "
            "Fashion-MNIST classifier."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training TOML configuration.",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Absolute project root. Overrides the environment "
            "variable and automatic discovery."
        ),
    )

    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=None,
        help="Optional .pt path. Existing files are not overwritten.",
    )

    parser.add_argument(
        "--epochs",
        type=positive_integer,
        default=None,
        help="Override the number of training epochs.",
    )

    parser.add_argument(
        "--max-train-samples",
        type=positive_integer,
        default=None,
        help="Optional training sample limit for experiments.",
    )

    parser.add_argument(
        "--max-validation-samples",
        type=positive_integer,
        default=None,
        help="Optional validation sample limit for experiments.",
    )

    download_group = parser.add_mutually_exclusive_group()

    download_group.add_argument(
        "--download-data",
        dest="download_data",
        action="store_true",
        help=(
            "Permit downloading Fashion-MNIST files "
            "when they are missing."
        ),
    )

    download_group.add_argument(
        "--offline",
        dest="download_data",
        action="store_false",
        help=(
            "Use locally cached Fashion-MNIST files only. "
            "This is the default."
        ),
    )

    parser.add_argument(
        "--track-mlflow",
        action="store_true",
        help="Create a run in the configured MLflow experiment.",
    )

    parser.add_argument(
        "--upload-minio",
        action="store_true",
        help=(
            "After required MLflow logging and artifact publication, upload "
            "the model to MinIO. Requires --track-mlflow."
        ),
    )

    parser.set_defaults(download_data=False)

    args = parser.parse_args()

    if args.upload_minio and not args.track_mlflow:
        parser.error("--upload-minio requires --track-mlflow")

    result = run_training(
        config_path=args.config,
        artifact_path=args.artifact_path,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        project_root=args.project_root,
        download_data=args.download_data,
        track_mlflow=args.track_mlflow,
        upload_minio=args.upload_minio,
    )

    print("\nFinal training summary")

    for name, value in (
        result.final_epoch.as_tracking_metrics().items()
    ):
        print(f"  {name}: {value:.6f}")

    print(f"\nModel artifact: {result.artifact_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
