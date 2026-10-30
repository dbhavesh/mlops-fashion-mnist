"""offline tests for the read-only multi-run verifier."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from fashion_mnist_mlops.tracking import effective_training_parameters
from scripts import compare_runs as comparison


def _provenance(epochs, digest):
    return {
        "training": {
            "epochs": epochs,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "cross_entropy",
            "seed": 42,
        },
        "validation": {"validation_split": 0.1},
        "dataset": {
            "train_samples_used": 32,
            "validation_samples_used": 16,
            "max_train_samples": 32,
            "max_validation_samples": 16,
            "download_allowed": False,
        },
        "source_config": {"sha256": "c" * 64, "effective_sha256": digest},
        "preprocessing": {"input_shape": [1, 28, 28]},
    }


@pytest.fixture
def runs(tmp_path, monkeypatch):
    root = tmp_path / "fashion-project"
    root.mkdir()
    records = {}
    saved_models = {}
    artifacts = {}
    histories = {}

    for run_id, epochs, digest in (("run-a", 1, "a" * 64), ("run-b", 2, "b" * 64)):
        provenance = _provenance(epochs, digest)
        ref = f"artifacts/models/{run_id}/model.pt"
        path = root / ref
        path.parent.mkdir(parents=True)
        path.write_bytes(b"mock-artifact")
        accuracy = 0.1875 if epochs == 1 else 0.3125
        saved_models[path.resolve()] = {
            "checkpoint_format_version": 2,
            "epoch": epochs,
            "validation_accuracy": accuracy,
            "mlflow_run_id": run_id,
            "provenance": provenance,
        }
        tags = {
            "model.architecture": "LeNet5",
            "model.storage_status": "local-only",
            "model.local_artifact_scope": "project",
            "model.local_artifact_ref": ref,
            "config.effective_sha256": digest,
            "metadata.provenance_artifact": comparison.PROVENANCE_ARTIFACT,
        }
        records[run_id] = SimpleNamespace(
            info=SimpleNamespace(
                run_id=run_id, experiment_id="1", status="FINISHED", end_time=1000
            ),
            data=SimpleNamespace(params=effective_training_parameters(provenance), tags=tags),
        )
        histories[(run_id, "train_loss")] = [
            SimpleNamespace(step=epoch, value=2.3 - epoch * 0.1)
            for epoch in range(1, epochs + 1)
        ]
        histories[(run_id, "val_loss")] = [
            SimpleNamespace(step=epoch, value=2.35 - epoch * 0.04)
            for epoch in range(1, epochs + 1)
        ]
        histories[(run_id, "val_accuracy")] = [
            SimpleNamespace(step=epochs, value=accuracy)
        ]
        artifact = tmp_path / f"{run_id}-provenance.json"
        artifact.write_text(json.dumps(provenance), encoding="utf-8")
        artifacts[run_id] = artifact

    class FakeClient:
        def __init__(self):
            self.history_calls = []
            self.artifact_calls = []

        def get_experiment_by_name(self, name):
            assert name == "fashion-mnist-classifier"
            return SimpleNamespace(experiment_id="1", lifecycle_stage="active")

        def get_run(self, run_id):
            return records[run_id]

        def get_metric_history(self, run_id, key):
            self.history_calls.append((run_id, key))
            return histories[(run_id, key)]

        def download_artifacts(self, run_id, path):
            assert path == comparison.PROVENANCE_ARTIFACT
            self.artifact_calls.append(run_id)
            return str(artifacts[run_id])

    client = FakeClient()
    monkeypatch.setattr(
        comparison, "load_model_artifact", lambda p: (None, saved_models[p.resolve()])
    )
    return client, root, records, histories, artifacts, saved_models


def _verify(runs, **changes):
    client, root, *_ = runs
    options = dict(
        experiment_name="fashion-mnist-classifier",
        run_a="run-a",
        run_b="run-b",
        project_root=root,
    )
    options.update(changes)
    return comparison.verify_run_comparison(client, **options)


def test_distinct_runs_have_independent_params_metrics_provenance_and_saved_models(runs):
    client, _, *_ = runs
    a, b = _verify(runs)
    assert (a.run_id, b.run_id) == ("run-a", "run-b")
    assert (a.epochs, b.epochs) == (1, 2)
    assert a.final_validation_accuracy == 0.1875
    assert b.final_validation_accuracy == 0.3125
    assert a.artifact_reference != b.artifact_reference
    assert set(client.history_calls) == {
        (rid, key)
        for rid in ("run-a", "run-b")
        for key in ("train_loss", "val_loss", "val_accuracy")
    }
    assert client.artifact_calls == ["run-a", "run-b"]


@pytest.mark.parametrize("run_a,run_b", [("run-a", "run-a"), ("", "run-b")])
def test_duplicate_or_empty_run_ids_are_rejected(runs, run_a, run_b):
    with pytest.raises(ValueError, match="distinct, nonempty"):
        _verify(runs, run_a=run_a, run_b=run_b)


def test_same_effective_hyperparameter_is_not_a_comparison(runs):
    with pytest.raises(ValueError, match="identical batch_size"):
        _verify(runs, differing_parameter="batch_size")


def test_artifact_cannot_be_shared_between_runs(runs):
    _, _, records, *_ = runs
    records["run-b"].data.tags["model.local_artifact_ref"] = (
        records["run-a"].data.tags["model.local_artifact_ref"]
    )
    with pytest.raises(ValueError, match="artifact belongs to a different run"):
        _verify(runs)


def test_artifact_cannot_escape_project_root(runs):
    _, _, records, *_ = runs
    records["run-a"].data.tags["model.local_artifact_ref"] = "../../secrets/model.pt"
    with pytest.raises(ValueError, match="escapes project root"):
        _verify(runs)


def test_wrong_experiment_or_run_status_fails(runs):
    _, _, records, *_ = runs
    records["run-b"].info.status = "FAILED"
    with pytest.raises(ValueError, match="not confirmed FINISHED"):
        _verify(runs)
    records["run-b"].info.status = "FINISHED"
    records["run-b"].info.experiment_id = "2"
    with pytest.raises(ValueError, match="not in the expected experiment"):
        _verify(runs)


def test_missing_loss_epoch_is_detected(runs):
    _, _, _, histories, *_ = runs
    histories[("run-b", "val_loss")] = histories[("run-b", "val_loss")][:1]
    with pytest.raises(ValueError, match="val_loss history"):
        _verify(runs)


def test_final_accuracy_must_only_be_at_last_epoch(runs):
    _, _, _, histories, *_ = runs
    histories[("run-b", "val_accuracy")][0].step = 1
    with pytest.raises(ValueError, match="val_accuracy history"):
        _verify(runs)


def test_swapped_provenance_artifact_is_detected(runs):
    _, _, _, _, artifacts, _ = runs
    artifacts["run-b"].write_text(
        artifacts["run-a"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="provenance differs"):
        _verify(runs)


def test_wrong_effective_params_are_detected(runs):
    _, _, records, *_ = runs
    records["run-b"].data.params["batch_size"] = "128"
    with pytest.raises(ValueError, match="logged parameters differ"):
        _verify(runs)


def test_original_run_data_is_not_modified(runs):
    _, _, records, histories, *_ = runs
    before = deepcopy((records, histories))
    _verify(runs)
    assert (records, histories) == before


def test_uploaded_runs_are_still_comparable(runs):
    _, _, records, *_ = runs
    for run_id in ("run-a", "run-b"):
        tags = records[run_id].data.tags
        tags.update({
            "model.storage_status": "minio-verified",
            "model.minio_bucket": "fashion-minst-bucket",
            "model.minio_object_key": f"model/{run_id}/model.pt",
            "model.minio_size_bytes": "182789",
            "model.minio_sha256": "a" * 64,
        })
    first, second = _verify(runs)
    assert (first.run_id, second.run_id) == ("run-a", "run-b")


def test_uploaded_run_wrong_bucket_is_detected(runs):
    _, _, records, *_ = runs
    records["run-a"].data.tags.update({
        "model.storage_status": "minio-verified",
        "model.minio_bucket": "fashion-mnist-bucket",
        "model.minio_object_key": "model/run-a/model.pt",
        "model.minio_size_bytes": "182789",
        "model.minio_sha256": "a" * 64,
    })
    with pytest.raises(ValueError, match="invalid MinIO artifact reference"):
        _verify(runs)
