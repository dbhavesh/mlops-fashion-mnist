"""safe model identity, config provenance, and local references."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import fashion_mnist_mlops.tracking as metadata


def make_provenance():
    return {
        "training": {"epochs": 1, "optimizer": "adam", "seed": 42},
        "validation": {"validation_split": 0.1},
        "dataset": {"name": "FashionMNIST"},
        "source_config": {
            "sha256": "a" * 64,
            "effective_sha256": "b" * 64,
        },
        "preprocessing": {"input_shape": [1, 28, 28], "version": 1},
        "software": {"python": "3.12"},
    }


def test_model_run_tags_identify_project_artifact_without_absolute_path(tmp_path):
    root = tmp_path / "project"
    artifact = root / "artifacts" / "models" / "uuid" / "model.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"saved")

    tags = metadata.model_run_tags(
        make_provenance(), artifact_path=artifact, project_root=root
    )
    assert tags["model.architecture"] == "LeNet5"
    assert tags["model.input_shape"] == "[1, 28, 28]"
    assert tags["model.num_classes"] == "10"
    assert tags["model.artifact_format_version"] == "2"
    assert tags["model.storage_status"] == "local-only"
    assert tags["model.local_artifact_scope"] == "project"
    assert tags["model.local_artifact_ref"] == "artifacts/models/uuid/model.pt"
    assert tags["config.source_sha256"] == "a" * 64
    assert tags["config.effective_sha256"] == "b" * 64
    assert tags["metadata.provenance_artifact"] == metadata.PROVENANCE_ARTIFACT
    assert str(tmp_path) not in str(tags)
    assert "minio" not in str(tags).lower()


def test_model_run_tags_external_path_redacts_parent_directories(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    artifact = tmp_path / "private" / "model.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"saved")
    tags = metadata.model_run_tags(
        make_provenance(), artifact_path=artifact, project_root=root
    )
    assert tags["model.local_artifact_scope"] == "external"
    assert tags["model.local_artifact_ref"] == "model.pt"
    assert str(artifact.parent) not in str(tags)


def test_model_run_tags_refuses_unpublished_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="not published"):
        metadata.model_run_tags(
            make_provenance(),
            artifact_path=tmp_path / "missing.pt",
            project_root=tmp_path,
        )


def test_log_run_metadata_uploads_provenance_and_searchable_tags(tmp_path, monkeypatch):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"saved")
    provenance = make_provenance()
    calls = []

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://127.0.0.1:5000"

        def log_dict(self, run_id, dictionary, artifact_file):
            calls.append(("artifact", run_id, dictionary, artifact_file))

        def log_batch(self, **kwargs):
            calls.append(("tags", kwargs))

    monkeypatch.setattr(metadata, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        metadata,
        "load_infrastructure_config",
        lambda: SimpleNamespace(
            mlflow=SimpleNamespace(tracking_uri="http://127.0.0.1:5000")
        ),
    )
    metadata.log_run_metadata(
        "run-58", provenance=provenance,
        artifact_path=artifact, project_root=tmp_path,
    )
    assert calls[0] == (
        "artifact", "run-58", provenance, "metadata/training_provenance.json"
    )
    assert calls[1][0] == "tags"
    kwargs = calls[1][1]
    assert kwargs["run_id"] == "run-58"
    assert kwargs["synchronous"] is True
    tags = {tag.key: tag.value for tag in kwargs["tags"]}
    assert tags["model.local_artifact_ref"] == "model.pt"
    assert tags["model.storage_status"] == "local-only"


@pytest.mark.parametrize("failure_stage", ["artifact", "tags"])
def test_log_run_metadata_propagates_upload_failure(
    tmp_path, monkeypatch, failure_stage
):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"saved")

    class FakeClient:
        def __init__(self, tracking_uri):
            pass

        def log_dict(self, run_id, dictionary, artifact_file):
            if failure_stage == "artifact":
                raise ConnectionError("metadata artifact unavailable")

        def log_batch(self, **kwargs):
            if failure_stage == "tags":
                raise ConnectionError("metadata tags unavailable")

    monkeypatch.setattr(metadata, "MlflowClient", FakeClient)
    with pytest.raises(ConnectionError, match="metadata"):
        metadata.log_run_metadata(
            "run-58", provenance=make_provenance(),
            artifact_path=artifact, project_root=tmp_path,
        )
