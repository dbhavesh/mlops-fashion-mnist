"""MLflow references to byte-verified MinIO model artifacts."""

from types import SimpleNamespace

import pytest

from fashion_mnist_mlops.storage import UploadedModelArtifact
from fashion_mnist_mlops import tracking as metadata


RUN = "c9df4f623d8c44de9f421ec762b48621"
BUCKET = "fashion-minst-bucket"
KEY = f"model/{RUN}/model.pt"
SHA = "a" * 64


def artifact(**overrides):
    data = dict(bucket=BUCKET, object_key=KEY, size_bytes=182789, sha256=SHA)
    data.update(overrides)
    return UploadedModelArtifact(**data)


def test_tags_capture_exact_verified_minio_reference():
    assert metadata.minio_artifact_tags(RUN, artifact()) == {
        "model.storage_status": "minio-verified",
        "model.minio_bucket": BUCKET,
        "model.minio_object_key": KEY,
        "model.minio_sha256": SHA,
        "model.minio_size_bytes": "182789",
    }


@pytest.mark.parametrize("bad_run", ["", "..", "has/slash", " bad", 123])
def test_rejects_invalid_run_before_logging(bad_run):
    with pytest.raises(ValueError):
        metadata.minio_artifact_tags(bad_run, artifact())


@pytest.mark.parametrize("changes", [
    {"bucket": "fashion-mnist-bucket"},
    {"object_key": "model/wrong-run/model.pt"},
    {"object_key": "model/../model.pt"},
    {"object_key": "model/c9df4f623d8c44de9f421ec762b48621/other.pt"},
    {"size_bytes": 0},
    {"size_bytes": -1},
    {"size_bytes": True},
    {"sha256": "bad"},
    {"sha256": "A" * 64},
    {"sha256": "g" * 64},
])
def test_rejects_incorrect_upload_facts(changes):
    with pytest.raises(ValueError):
        metadata.minio_artifact_tags(RUN, artifact(**changes))


def test_persists_all_tags_atomically_in_same_run(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://127.0.0.1:5000"

        def log_batch(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(metadata, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        metadata, "load_infrastructure_config",
        lambda: SimpleNamespace(mlflow=SimpleNamespace(tracking_uri="http://127.0.0.1:5000")),
    )
    metadata.log_minio_artifact_reference(RUN, artifact())
    assert len(calls) == 1
    call = calls[0]
    assert call["run_id"] == RUN
    assert call["synchronous"] is True
    assert {t.key: t.value for t in call["tags"]} == metadata.minio_artifact_tags(RUN, artifact())


def test_rejects_invalid_reference_before_creating_client(monkeypatch):
    monkeypatch.setattr(metadata, "MlflowClient", lambda *a, **kw: pytest.fail("No MLflow write"))
    with pytest.raises(ValueError, match="object key"):
        metadata.log_minio_artifact_reference(RUN, artifact(object_key="model/other/model.pt"))


def test_mlflow_logging_failure_propagates_without_deletion(monkeypatch):
    class FakeClient:
        def __init__(self, tracking_uri):
            pass

        def log_batch(self, **kwargs):
            raise ConnectionError("MLflow unavailable")

    monkeypatch.setattr(metadata, "MlflowClient", FakeClient)
    with pytest.raises(ConnectionError, match="MLflow unavailable"):
        metadata.log_minio_artifact_reference(RUN, artifact())
