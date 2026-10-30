"""failure classification across provisioning, upload and download.

All cases run offline. Tests never modify a live MinIO bucket or reveal secrets.
"""

import hashlib
from io import BytesIO
from types import SimpleNamespace

import pytest
from urllib3.exceptions import MaxRetryError, NewConnectionError, ReadTimeoutError

from fashion_mnist_mlops.storage import ensure_bucket_exists
from fashion_mnist_mlops import storage as downloads
from fashion_mnist_mlops import storage as uploads
from fashion_mnist_mlops.storage_errors import (
    StorageBucketNotFoundError,
    StorageCredentialError,
    StorageObjectNotFoundError,
    StoragePermissionError,
    StorageUnavailableError,
    normalize_storage_error,
    storage_call,
)

BUCKET = "fashion-minst-bucket"
RUN_ID = "abcdef0123456789abcdef0123456789"
KEY = f"model/{RUN_ID}/model.pt"
BYTES = b"portable model bytes"
SHA = hashlib.sha256(BYTES).hexdigest()


class FakeS3Error(Exception):
    def __init__(self, code):
        super().__init__("secret-password-never-show")
        self.code = code


@pytest.mark.parametrize("code,exception,message", [
    ("InvalidAccessKeyId", StorageCredentialError, "MINIO_ACCESS_KEY"),
    ("SignatureDoesNotMatch", StorageCredentialError, "MINIO_ACCESS_KEY"),
    ("InvalidToken", StorageCredentialError, "MINIO_ACCESS_KEY"),
    ("ExpiredToken", StorageCredentialError, "MINIO_ACCESS_KEY"),
    ("AccessDenied", StoragePermissionError, "AccessDenied"),
    ("NoSuchBucket", StorageBucketNotFoundError, "bucket is missing"),
    ("NoSuchKey", StorageObjectNotFoundError, "model object not found"),
    ("NoSuchObject", StorageObjectNotFoundError, "model object not found"),
    ("PreconditionFailed", FileExistsError, "already exists"),
    ("ConditionalRequestConflict", FileExistsError, "already exists"),
])
def test_recognized_s3_failure_has_stable_type_and_no_secret(code, exception, message):
    error = normalize_storage_error(
        FakeS3Error(code), operation="download", bucket=BUCKET, object_key=KEY,
    )
    assert isinstance(error, exception)
    assert message in str(error)
    assert "secret-password-never-show" not in str(error)


def test_unknown_sdk_failure_is_not_hidden():
    original = FakeS3Error("UnexpectedServerCondition")
    assert normalize_storage_error(original, operation="download", bucket=BUCKET) is original
    with pytest.raises(FakeS3Error) as record:
        storage_call("download", BUCKET, lambda: (_ for _ in ()).throw(original))
    assert record.value is original


def test_native_connection_failure_remains_backward_compatible():
    original = ConnectionError("service disconnected")
    assert normalize_storage_error(original, operation="download", bucket=BUCKET) is original


@pytest.mark.parametrize("operation", ["probe", "creation", "verification"])
def test_provisioning_categorizes_unreachable_storage(operation):
    calls = []

    class Client:
        def bucket_exists(self, name):
            calls.append("probe")
            if operation in {"probe", "verification"} and (
                operation != "verification" or len(calls) > 2
            ):
                raise MaxRetryError(None, "/bucket", reason=Exception("offline"))
            return False

        def make_bucket(self, name):
            calls.append("create")
            if operation == "creation":
                raise MaxRetryError(None, "/bucket", reason=Exception("offline"))

    with pytest.raises(StorageUnavailableError, match="MINIO_ENDPOINT"):
        ensure_bucket_exists(Client(), bucket_name=BUCKET)


def test_missing_bucket_is_reported_on_initial_probe():
    class Client:
        def bucket_exists(self, name):
            raise FakeS3Error("NoSuchBucket")

    with pytest.raises(StorageBucketNotFoundError, match=BUCKET):
        ensure_bucket_exists(Client(), bucket_name=BUCKET)


def test_credentials_rejected_before_bucket_creation():
    class Client:
        def bucket_exists(self, name):
            raise FakeS3Error("InvalidAccessKeyId")

        def make_bucket(self, name):
            pytest.fail("Must not create a bucket with rejected credentials")

    with pytest.raises(StorageCredentialError, match="MINIO_ACCESS_KEY"):
        ensure_bucket_exists(Client(), bucket_name=BUCKET)


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(BYTES)
    monkeypatch.setattr(uploads, "inspect_model_artifact", lambda *args, **kwargs: SimpleNamespace(
        path=path, size_bytes=len(BYTES), sha256=SHA,
    ))
    return path


def test_upload_rejects_sdk_credential_error_before_put(artifact, monkeypatch):
    def unexpected(*a, **kw):
        pytest.fail("PUT must not execute")
    monkeypatch.setattr(uploads, "_put_if_absent", unexpected)

    class Client:
        def bucket_exists(self, name):
            raise FakeS3Error("SignatureDoesNotMatch")

    with pytest.raises(StorageCredentialError):
        uploads.upload_model_artifact(
            Client(), artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET,
        )
    assert artifact.read_bytes() == BYTES


@pytest.mark.parametrize("code,expected", [
    ("AccessDenied", StoragePermissionError),
    ("PreconditionFailed", FileExistsError),
    ("NoSuchBucket", StorageBucketNotFoundError),
])
def test_upload_reports_write_failure_without_retry(artifact, monkeypatch, code, expected):
    calls = []

    def failed(*a, **kw):
        calls.append("put")
        raise FakeS3Error(code)

    monkeypatch.setattr(uploads, "_put_if_absent", failed)

    class Client:
        def bucket_exists(self, name):
            return True

        def get_object(self, name, key):
            pytest.fail("Verification read not permitted after failed PUT")

    with pytest.raises(expected):
        uploads.upload_model_artifact(
            Client(), artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET,
        )
    assert calls == ["put"]
    assert artifact.read_bytes() == BYTES


def test_ambiguous_upload_network_failure_is_not_retried(artifact, monkeypatch):
    calls = []

    def interrupted(*a, **kw):
        calls.append("put")
        raise ReadTimeoutError(None, "/upload", "timed out")

    monkeypatch.setattr(uploads, "_put_if_absent", interrupted)

    class Client:
        def bucket_exists(self, name):
            return True

    with pytest.raises(StorageUnavailableError, match="MINIO_ENDPOINT"):
        uploads.upload_model_artifact(
            Client(), artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET,
        )
    assert calls == ["put"]


class ClosedResponse(BytesIO):
    def __init__(self, contents):
        super().__init__(contents)
        self.released = False

    def release_conn(self):
        self.released = True


@pytest.mark.parametrize("code,expected", [
    ("NoSuchBucket", StorageBucketNotFoundError),
    ("NoSuchKey", StorageObjectNotFoundError),
    ("InvalidAccessKeyId", StorageCredentialError),
    ("AccessDenied", StoragePermissionError),
])
def test_download_translates_sdk_failure_without_creating_files(tmp_path, code, expected):
    target = tmp_path / "downloads" / "model.pt"

    class Client:
        def bucket_exists(self, name):
            return True

        def get_object(self, name, key):
            raise FakeS3Error(code)

    with pytest.raises(expected):
        downloads.download_model_artifact(
            Client(), bucket_name=BUCKET, object_key=KEY, destination=target,
            expected_sha256=SHA, expected_size_bytes=len(BYTES),
        )
    assert not target.exists()
    assert not target.parent.exists()


def test_download_interrupted_stream_is_cleaned_and_connection_released(tmp_path):
    response = ClosedResponse(BYTES)
    def disconnected(_):
        raise NewConnectionError(None, "network dropped")
    response.read = disconnected

    class Client:
        def bucket_exists(self, name):
            return True

        def get_object(self, name, key):
            return response

    target = tmp_path / "download" / "model.pt"
    with pytest.raises(StorageUnavailableError, match="MINIO_ENDPOINT"):
        downloads.download_model_artifact(
            Client(), bucket_name=BUCKET, object_key=KEY, destination=target,
            expected_sha256=SHA, expected_size_bytes=len(BYTES),
        )
    assert not target.exists()
    assert not list(target.parent.glob("*.part"))
    assert not list(target.parent.glob(".*.part"))
    assert response.closed and response.released


def test_upload_verification_failure_leaves_published_bytes_untouched(artifact, monkeypatch):
    objects = {}

    def put(*args, **kwargs):
        objects[KEY] = BYTES

    monkeypatch.setattr(uploads, "_put_if_absent", put)

    class Client:
        def bucket_exists(self, name):
            return True

        def get_object(self, name, key):
            raise FakeS3Error("AccessDenied")

    with pytest.raises(StoragePermissionError):
        uploads.upload_model_artifact(
            Client(), artifact_path=artifact, run_id=RUN_ID, bucket_name=BUCKET,
        )
    assert objects[KEY] == BYTES


def test_bucket_creation_access_denied_is_explicit():
    class Client:
        def bucket_exists(self, name):
            return False

        def make_bucket(self, name):
            raise FakeS3Error("AccessDenied")

    with pytest.raises(StoragePermissionError, match="AccessDenied"):
        ensure_bucket_exists(Client(), bucket_name=BUCKET)
