"""Offline upload contracts using a conditional-write fake S3 client."""

import hashlib
from io import BytesIO
from types import SimpleNamespace

import pytest

from fashion_mnist_mlops.model import (
    LeNet5,
)
from fashion_mnist_mlops.model_artifacts import (
    save_model_artifact,
)
from fashion_mnist_mlops.storage import (
    MAX_SINGLE_PUT_BYTES,
    publish_model_artifact,
    upload_model_artifact,
)


RUN_ID = "abc0123456789defabc0123456789def"
BUCKET = "fashion-minst-bucket"
KEY = f"model/{RUN_ID}/model.pt"


@pytest.fixture
def artifact(tmp_path):
    provenance = {
        "training": {
            "epochs": 1, "batch_size": 4, "learning_rate": 0.001,
            "optimizer": "adam", "loss": "cross_entropy", "seed": 42,
        },
        "validation": {"validation_split": 0.1},
        "dataset": {
            "name": "FashionMNIST", "location": {"scope": "project", "path": "data"},
            "train_samples_used": 8, "validation_samples_used": 4,
            "test_samples_available": 4, "max_train_samples": 8,
            "max_validation_samples": 4, "download_allowed": False,
        },
        "source_config": {
            "location": {"scope": "project", "path": "config/training.toml"},
            "sha256": "a" * 64, "effective_sha256": "b" * 64,
        },
        "preprocessing": {
            "version": 1, "input_shape": [1, 28, 28],
            "normalization_mean": [0.2860406],
            "normalization_std": [0.3530242],
        },
        "software": {"python": "3.12", "torch": "2.x", "torchvision": "0.x"},
    }
    path = tmp_path / "model.pt"
    save_model_artifact(
        LeNet5(), path, epoch=1, validation_accuracy=0.25,
        mlflow_run_id=RUN_ID, provenance=provenance,
    )
    return path


class WriteConflict(Exception):
    code = "PreconditionFailed"


class FakeResponse(BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.released = False

    def release_conn(self):
        self.released = True


class FakeMinio:
    def __init__(self):
        self.objects = {}
        self.present = True
        self.put_calls = []
        self.get_calls = []
        self.response = None
        self.corrupt_download = False
        self.fail_put = None
        self.fail_get = None

    def bucket_exists(self, name):
        assert name == BUCKET
        return self.present

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        assert kwargs["bucket_name"] == BUCKET
        key = kwargs["object_name"]
        assert kwargs["extra_headers"] == {"If-None-Match": "*"}
        if self.fail_put:
            raise self.fail_put
        if key in self.objects:
            raise WriteConflict("existing object")
        data = kwargs["data"].read()
        assert len(data) == kwargs["length"]
        self.objects[key] = data
        return SimpleNamespace(object_name=key)

    def get_object(self, bucket, object_key):
        self.get_calls.append((bucket, object_key))
        if self.fail_get:
            raise self.fail_get
        content = self.objects[object_key]
        self.response = FakeResponse(content + b"bad" if self.corrupt_download else content)
        return self.response


def test_valid_artifact_uploaded_at_exact_key_and_bytes_verified(artifact):
    client = FakeMinio()
    result = upload_model_artifact(
        client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET
    )
    assert result.bucket == BUCKET
    assert result.object_key == KEY
    assert client.objects[KEY] == artifact.read_bytes()
    assert result.size_bytes == len(artifact.read_bytes())
    assert result.sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert client.get_calls == [(BUCKET, KEY)]
    assert client.response.closed and client.response.released
    assert client.put_calls[0]["extra_headers"] == {"If-None-Match": "*"}


def test_second_upload_must_not_overwrite_existing_object(artifact):
    client = FakeMinio()
    upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    original = client.objects[KEY]
    with pytest.raises(FileExistsError, match="already exists"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert client.objects[KEY] == original
    assert len(client.put_calls) == 2
    assert len(client.get_calls) == 1


def test_mismatched_run_id_prevents_network_write(artifact):
    client = FakeMinio()
    with pytest.raises(ValueError, match="run ID"):
        upload_model_artifact(client, artifact_path=artifact, run_id="another-run", bucket_name=BUCKET)
    assert not client.put_calls


@pytest.mark.parametrize("name", ["fashion-mnist-bucket", "other", ""])
def test_wrong_bucket_rejected(artifact, name):
    client = FakeMinio()
    with pytest.raises(ValueError, match="require bucket"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=name)
    assert not client.put_calls


def test_missing_bucket_never_creates_or_uploads(artifact):
    client = FakeMinio()
    client.present = False
    with pytest.raises(FileNotFoundError, match="bucket is missing"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert not client.put_calls


def test_corrupt_uploaded_content_fails_without_deletion(artifact):
    client = FakeMinio()
    client.corrupt_download = True
    with pytest.raises(RuntimeError, match="byte-integrity"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert KEY in client.objects
    assert client.response.closed and client.response.released


def test_put_permission_error_is_not_hidden(artifact):
    client = FakeMinio()
    client.fail_put = PermissionError("AccessDenied")
    with pytest.raises(PermissionError, match="AccessDenied"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert KEY not in client.objects


def test_get_failure_leaves_written_object_for_inspection(artifact):
    client = FakeMinio()
    client.fail_get = ConnectionError("download unavailable")
    with pytest.raises(ConnectionError, match="download unavailable"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert client.objects[KEY] == artifact.read_bytes()


def test_corrupt_artifact_fails_before_upload(artifact):
    artifact.write_bytes(b"invalid artifact")
    client = FakeMinio()
    with pytest.raises(Exception):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert not client.put_calls


def test_file_too_large_rejected_before_write(artifact, monkeypatch):
    import fashion_mnist_mlops.storage as uploads
    monkeypatch.setattr(uploads, "MAX_SINGLE_PUT_BYTES", 16)
    client = FakeMinio()
    with pytest.raises(ValueError, match="single-request"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert not client.put_calls


def test_changed_artifact_snapshot_is_rejected(artifact, monkeypatch):
    import fashion_mnist_mlops.storage as uploads
    genuine = uploads.inspect_model_artifact

    def altered(path, *, expected_run_id=None):
        info = genuine(path, expected_run_id=expected_run_id)
        artifact.write_bytes(artifact.read_bytes() + b"changed")
        return info

    monkeypatch.setattr(uploads, "inspect_model_artifact", altered)
    client = FakeMinio()
    with pytest.raises(RuntimeError, match="changed after"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert not client.put_calls


def test_live_client_factory_reads_settings_without_logging_secrets(artifact, monkeypatch, capsys):
    import sys
    from types import ModuleType
    fake = FakeMinio()
    constructed = []

    def factory(**kwargs):
        constructed.append(kwargs)
        return fake

    module = ModuleType("minio")
    module.Minio = factory
    monkeypatch.setitem(sys.modules, "minio", module)
    monkeypatch.setenv("MINIO_ENDPOINT", "127.0.0.1:9000")
    monkeypatch.setenv("MINIO_BUCKET", BUCKET)
    monkeypatch.setenv("MINIO_SECURE", "false")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "unit-user")
    monkeypatch.setenv("MINIO_SECRET_KEY", "unit-secret")
    result = publish_model_artifact(artifact, run_id=RUN_ID)
    assert result.object_key == KEY
    assert constructed == [{
        "endpoint": "127.0.0.1:9000", "access_key": "unit-user",
        "secret_key": "unit-secret", "secure": False,
    }]
    assert "unit-secret" not in capsys.readouterr().out


def test_older_sdk_signed_conditional_put_preserves_no_overwrite(artifact):
    class LegacyMinio(FakeMinio):
        # Matches an older SDK's public method: NO extra_headers parameter.
        def put_object(self, bucket_name, object_name, data, length, content_type="application/octet-stream"):
            pytest.fail("Unsafe fallback through put_object must never happen")

        def _execute(self, *, method, bucket_name, object_name, body, headers):
            assert method == "PUT"
            assert bucket_name == BUCKET
            assert headers["If-None-Match"] == "*"
            assert headers["Content-Type"] == "application/octet-stream"
            self.put_calls.append((method, bucket_name, object_name, headers.copy()))
            if object_name in self.objects:
                raise WriteConflict("existing object")
            self.objects[object_name] = body

    client = LegacyMinio()
    result = upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert result.object_key == KEY
    original = client.objects[KEY]
    with pytest.raises(FileExistsError, match="already exists"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert client.objects[KEY] == original
    assert len(client.put_calls) == 2
    assert len(client.get_calls) == 1


def test_sdk_without_conditional_transport_fails_closed(artifact):
    class UnsupportedMinio(FakeMinio):
        def put_object(self, bucket_name, object_name, data, length, content_type="application/octet-stream", headers=None):
            pytest.fail("Do not use headers= because SDK may reinterpret If-None-Match as metadata")

    client = UnsupportedMinio()
    with pytest.raises(RuntimeError, match="refusing to upload"):
        upload_model_artifact(client, artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET)
    assert client.objects == {}


def test_no_overwrite_limit_is_intentionally_single_put():
    assert MAX_SINGLE_PUT_BYTES == 5 * 1024 * 1024
