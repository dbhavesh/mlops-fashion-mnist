"""read-only verification of two independent Fashion-MNIST runs.

Run two *separate* tracked training commands first, with different epoch counts.
Then use their printed MLflow IDs with --run-a and --run-b.
The script reads MLflow records and local model artifacts; it creates no new runs.
"""

import argparse
import json
from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path

from mlflow.tracking import MlflowClient

from fashion_mnist_mlops.model_artifacts import load_model_artifact
from fashion_mnist_mlops.config import load_infrastructure_config
from fashion_mnist_mlops.paths import get_project_root
from fashion_mnist_mlops.tracking import (
    PROVENANCE_ARTIFACT, MINIO_STORAGE_STATUS, minio_artifact_tags,
)
from fashion_mnist_mlops.storage import UploadedModelArtifact
from fashion_mnist_mlops.tracking import effective_training_parameters


@dataclass(frozen=True)
class RunEvidence:
    run_id: str
    epochs: int
    parameters: dict[str, str]
    final_validation_accuracy: float
    artifact_reference: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metric_history(client, run_id: str, key: str, expected_epochs: list[int]):
    history = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
    _require(
        [metric.step for metric in history] == expected_epochs,
        f"Run {run_id}: {key} history must have epochs {expected_epochs}; "
        f"found {[metric.step for metric in history]}",
    )
    _require(
        all(isfinite(metric.value) for metric in history),
        f"Run {run_id}: {key} contains a nonfinite value",
    )
    return history


def _inspect_run(client, *, run_id: str, experiment_id: str, root: Path) -> RunEvidence:
    run = client.get_run(run_id)
    _require(run.info.run_id == run_id, f"Run identity mismatch: {run_id}")
    _require(
        str(run.info.experiment_id) == experiment_id,
        f"Run {run_id} is not in the expected experiment",
    )
    _require(
        run.info.status == "FINISHED" and run.info.end_time is not None,
        f"Run {run_id} is not confirmed FINISHED",
    )

    params = run.data.params
    try:
        epochs = int(params["epochs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Run {run_id} has no valid epochs parameter") from exc
    _require(epochs > 0 and params["epochs"] == str(epochs),
             f"Run {run_id} has an invalid epochs parameter")

    expected_epochs = list(range(1, epochs + 1))
    _metric_history(client, run_id, "train_loss", expected_epochs)
    _metric_history(client, run_id, "val_loss", expected_epochs)
    accuracy_history = _metric_history(client, run_id, "val_accuracy", [epochs])
    accuracy = accuracy_history[0].value
    _require(0.0 <= accuracy <= 1.0, f"Run {run_id} has invalid validation accuracy")

    tags = run.data.tags
    _require(tags.get("model.architecture") == "LeNet5", f"Run {run_id}: architecture mismatch")
    storage_status = tags.get("model.storage_status")
    _require(storage_status in {"local-only", MINIO_STORAGE_STATUS},
             f"Run {run_id}: unexpected storage status")
    if storage_status == MINIO_STORAGE_STATUS:
        try:
            uploaded = UploadedModelArtifact(
                bucket=tags["model.minio_bucket"],
                object_key=tags["model.minio_object_key"],
                size_bytes=int(tags["model.minio_size_bytes"]),
                sha256=tags["model.minio_sha256"],
            )
            expected_minio_tags = minio_artifact_tags(run_id, uploaded)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Run {run_id}: invalid MinIO artifact reference") from exc
        _require(all(tags.get(key) == value for key, value in expected_minio_tags.items()),
                 f"Run {run_id}: inconsistent MinIO artifact reference")
    _require(tags.get("model.local_artifact_scope") == "project",
             f"Run {run_id}: expected a project-local artifact")
    _require(tags.get("metadata.provenance_artifact") == PROVENANCE_ARTIFACT,
             f"Run {run_id}: provenance artifact tag mismatch")

    reference = tags.get("model.local_artifact_ref", "")
    _require(bool(reference), f"Run {run_id}: missing artifact reference")
    relative_path = Path(reference)
    _require(not relative_path.is_absolute(), f"Run {run_id}: artifact path is absolute")
    artifact = (root / relative_path).resolve()
    _require(artifact.is_relative_to(root), f"Run {run_id}: artifact escapes project root")
    _require(artifact.is_file(), f"Run {run_id}: artifact is missing: {reference}")

    _, metadata = load_model_artifact(artifact)
    _require(metadata.get("mlflow_run_id") == run_id,
             f"Run {run_id}: artifact belongs to a different run")
    _require(metadata.get("checkpoint_format_version") == 2,
             f"Run {run_id}: artifact format is not version 2")
    _require(metadata.get("epoch") == epochs,
             f"Run {run_id}: artifact epoch mismatch")
    _require(isclose(metadata["validation_accuracy"], accuracy, rel_tol=0, abs_tol=1e-12),
             f"Run {run_id}: final accuracy differs from artifact")
    provenance = metadata["provenance"]
    _require(tags.get("config.effective_sha256") == provenance["source_config"]["effective_sha256"],
             f"Run {run_id}: effective configuration hash mismatch")
    _require(effective_training_parameters(provenance) == {
        key: params.get(key) for key in effective_training_parameters(provenance)
    }, f"Run {run_id}: logged parameters differ from effective provenance")

    artifact_path = client.download_artifacts(run_id, PROVENANCE_ARTIFACT)
    artifact_provenance = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    _require(artifact_provenance == provenance,
             f"Run {run_id}: MLflow provenance differs from artifact provenance")

    return RunEvidence(
        run_id=run_id,
        epochs=epochs,
        parameters=dict(params),
        final_validation_accuracy=accuracy,
        artifact_reference=reference,
    )


def verify_run_comparison(
    client,
    *,
    experiment_name: str,
    run_a: str,
    run_b: str,
    project_root: Path,
    differing_parameter: str = "epochs",
) -> tuple[RunEvidence, RunEvidence]:
    """Verify two completed experiment runs remain independently identifiable."""
    _require(bool(run_a.strip()) and bool(run_b.strip()) and run_a != run_b,
             "Provide two distinct, nonempty MLflow run IDs")
    _require(bool(differing_parameter.strip()), "The comparison parameter is required")
    experiment = client.get_experiment_by_name(experiment_name)
    _require(experiment is not None and experiment.lifecycle_stage == "active",
             f"Active MLflow experiment not found: {experiment_name}")
    root = Path(project_root).resolve()
    _require(root.is_dir(), f"Project root not found: {root}")

    first = _inspect_run(client, run_id=run_a, experiment_id=str(experiment.experiment_id), root=root)
    second = _inspect_run(client, run_id=run_b, experiment_id=str(experiment.experiment_id), root=root)

    _require(
        differing_parameter in first.parameters and differing_parameter in second.parameters,
        f"Missing comparison parameter: {differing_parameter}",
    )
    _require(first.parameters[differing_parameter] != second.parameters[differing_parameter],
             f"Runs have identical {differing_parameter}; use two different settings")
    _require(first.artifact_reference != second.artifact_reference,
             "Different runs must not share the same artifact reference")
    return first, second


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, help="First completed MLflow run ID")
    parser.add_argument("--run-b", required=True, help="Second completed MLflow run ID")
    parser.add_argument("--difference-key", default="epochs", help="Parameter expected to differ (default: epochs)")
    parser.add_argument("--project-root", type=Path, help="Project root; autodetected by default")
    args = parser.parse_args()

    mlflow = load_infrastructure_config().mlflow
    client = MlflowClient(tracking_uri=mlflow.tracking_uri)
    try:
        first, second = verify_run_comparison(
            client,
            experiment_name=mlflow.experiment_name,
            run_a=args.run_a,
            run_b=args.run_b,
            project_root=get_project_root(args.project_root),
            differing_parameter=args.difference_key,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")

    print("PASS: Two independently tracked runs are comparable")
    print(f"Experiment: {mlflow.experiment_name}")
    print(f"Different parameter: {args.difference_key}")
    for label, result in (("A", first), ("B", second)):
        print(f"Run {label}: {result.run_id}")
        print(f"  {args.difference_key}: {result.parameters[args.difference_key]}")
        if args.difference_key != "epochs":
            print(f"  epochs: {result.epochs}")
        print(f"  final val_accuracy: {result.final_validation_accuracy:.6f}")
        print(f"  artifact: {result.artifact_reference}")
    print("Next: select these two runs in the MLflow experiment UI and click Compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
