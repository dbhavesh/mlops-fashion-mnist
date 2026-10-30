"""offline MinIO downloads and verified MLflow model selection."""

import hashlib
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from fashion_mnist_mlops import storage as downloads

RUN_ID = "97461abfc4804cb0a2a1473c71737739"
KEY = f"model/{RUN_ID}/model.pt"
BUCKET = "fashion-minst-bucket"
PAYLOAD = b"a portable model artifact" * 100
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse:
    def __init__(self, payload):
        self.stream = io.BytesIO(payload)
        self.closed = False
        self.released = False

    def read(self, size):
        return self.stream.read(size)

    def close(self):
        self.closed = True
        self.stream.close()

    def release_conn(self):
        self.released = True


class FakeClient:
    def __init__(self, *, payload=PAYLOAD, exists=True):
        self.payload = payload
        self.exists = exists
        self.calls = []
        self.response = None

    def bucket_exists(self, name):
        self.calls.append(("bucket_exists", name))
        return self.exists

    def get_object(self, bucket, key):
        self.calls.append(("get_object", bucket, key))
        self.response = FakeResponse(self.payload)
        return self.response


def download(tmp_path, client=None, **options):
    client = client or FakeClient()
    args = {
        "bucket_name": BUCKET,
        "object_key": KEY,
        "destination": tmp_path / "fetched" / "model.pt",
        "expected_sha256": SHA,
        "expected_size_bytes": len(PAYLOAD),
    }
    args.update(options)
    return downloads.download_model_artifact(client, **args), client


def test_streamed_bytes_are_published_under_exact_requested_destination(tmp_path):
    info, client = download(tmp_path)
    assert info.path == (tmp_path / "fetched" / "model.pt").resolve()
    assert info.bucket == BUCKET and info.object_key == KEY
    assert info.size_bytes == len(PAYLOAD) and info.sha256 == SHA
    assert info.path.read_bytes() == PAYLOAD
    assert client.calls == [("bucket_exists", BUCKET), ("get_object", BUCKET, KEY)]
    assert client.response.closed and client.response.released
    assert list(info.path.parent.glob("*.part")) == []
    assert list(info.path.parent.glob(".*.part")) == []


def test_existing_destination_is_never_modified_or_read_from_remote(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"keep-me")
    client = FakeClient()
    with pytest.raises(FileExistsError, match="overwrite"):
        download(tmp_path, client, destination=path)
    assert path.read_bytes() == b"keep-me"
    assert client.calls == []


def test_second_download_to_same_path_fails_without_overwriting(tmp_path):
    info, client = download(tmp_path)
    with pytest.raises(FileExistsError):
        download(tmp_path, client)
    assert info.path.read_bytes() == PAYLOAD


def test_destination_symlink_is_rejected(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"original")
    link = tmp_path / "link"
    try:
        link.symlink_to(original)
    except (NotImplementedError, OSError):
        pytest.skip("Symlink creation not permitted")
    client = FakeClient()
    with pytest.raises(FileExistsError):
        download(tmp_path, client, destination=link)
    assert original.read_bytes() == b"original"
    assert client.calls == []


@pytest.mark.parametrize("bad_key", ["model/../model.pt", "other/run/model.pt", "model//model.pt"])
def test_rejects_unsafe_keys_before_network(tmp_path, bad_key):
    client = FakeClient()
    with pytest.raises(ValueError):
        download(tmp_path, client, object_key=bad_key)
    assert client.calls == []


@pytest.mark.parametrize("bad_bucket", ["fashion-mnist-bucket", "", "another"])
def test_wrong_bucket_rejected_without_network(tmp_path, bad_bucket):
    client = FakeClient()
    with pytest.raises(ValueError, match="bucket"):
        download(tmp_path, client, bucket_name=bad_bucket)
    assert client.calls == []


@pytest.mark.parametrize("bad_digest", ["", "A" * 64, "z" * 64, None])
def test_rejects_invalid_expected_digest(tmp_path, bad_digest):
    client = FakeClient()
    with pytest.raises(ValueError, match="SHA-256"):
        download(tmp_path, client, expected_sha256=bad_digest)
    assert client.calls == []


@pytest.mark.parametrize("bad_size", [0, -1, True, 5*1024*1024+1, "100"])
def test_rejects_invalid_expected_size(tmp_path, bad_size):
    client = FakeClient()
    with pytest.raises(ValueError, match="size"):
        download(tmp_path, client, expected_size_bytes=bad_size)
    assert client.calls == []


def test_missing_bucket_does_not_leave_a_file(tmp_path):
    client = FakeClient(exists=False)
    with pytest.raises(FileNotFoundError, match="bucket"):
        download(tmp_path, client)
    assert not (tmp_path / "fetched").exists()


@pytest.mark.parametrize("remote", [b"corrupted", PAYLOAD[:-1], PAYLOAD + b"long"])
def test_corruption_truncation_or_excess_bytes_leaves_no_file(tmp_path, remote):
    client = FakeClient(payload=remote)
    with pytest.raises(ValueError):
        download(tmp_path, client)
    assert not (tmp_path / "fetched" / "model.pt").exists()
    assert not list((tmp_path / "fetched").glob("*.part"))
    assert not list((tmp_path / "fetched").glob(".*.part"))
    assert client.response.closed and client.response.released


def test_network_read_failure_cleans_partial_file(tmp_path):
    class BrokenResponse(FakeResponse):
        def read(self, size):
            raise ConnectionError("network disconnected")
    class BrokenClient(FakeClient):
        def get_object(self, bucket, key):
            self.response = BrokenResponse(PAYLOAD)
            return self.response
    client = BrokenClient()
    with pytest.raises(ConnectionError, match="disconnected"):
        download(tmp_path, client)
    assert not (tmp_path / "fetched" / "model.pt").exists()
    assert not list((tmp_path / "fetched").glob(".*.part"))
    assert client.response.closed and client.response.released


def test_missing_object_error_propagates_without_local_file(tmp_path):
    class MissingClient(FakeClient):
        def get_object(self, bucket, key):
            raise FileNotFoundError("NoSuchKey")
    with pytest.raises(FileNotFoundError, match="NoSuchKey"):
        download(tmp_path, MissingClient())
    assert not (tmp_path / "fetched").exists()


def test_sdk_missing_object_maps_to_readable_error(tmp_path):
    class MissingObjectError(Exception):
        code = "NoSuchKey"

    class MissingClient(FakeClient):
        def get_object(self, bucket, key):
            raise MissingObjectError("S3 NoSuchKey")

    with pytest.raises(FileNotFoundError, match="model object not found"):
        download(tmp_path, MissingClient())
    assert not (tmp_path / "fetched").exists()


def test_publication_race_must_not_replace_other_process_file(tmp_path, monkeypatch):
    target = tmp_path / "fetched" / "model.pt"
    original_link = downloads.os.link
    def racing_link(source, destination):
        Path(destination).write_bytes(b"other-writer")
        # Model the OS refusing the exclusive publication.
        raise FileExistsError("another writer created target")
    monkeypatch.setattr(downloads.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        download(tmp_path)
    assert target.read_bytes() == b"other-writer"
    assert not list(target.parent.glob(".*.part"))


def test_filesystem_without_hardlink_fails_closed(tmp_path, monkeypatch):
    def unsupported(*args):
        raise OSError("hard links unsupported")
    monkeypatch.setattr(downloads.os, "link", unsupported)
    with pytest.raises(OSError, match="unsupported"):
        download(tmp_path)
    assert not (tmp_path / "fetched" / "model.pt").exists()
    assert not list((tmp_path / "fetched").glob(".*.part"))


def _fake_services(monkeypatch, *, status="FINISHED", tag_changes=None):
    tags = {
        "model.storage_status": "minio-verified",
        "model.minio_bucket": BUCKET,
        "model.minio_object_key": KEY,
        "model.minio_sha256": SHA,
        "model.minio_size_bytes": str(len(PAYLOAD)),
    }
    tags.update(tag_changes or {})
    network = {"mlflow_calls": 0, "minio_clients": []}
    class MLflowClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://127.0.0.1:5000"
        def get_run(self, run_id):
            network["mlflow_calls"] += 1
            assert run_id == RUN_ID
            return SimpleNamespace(
                info=SimpleNamespace(run_id=run_id, status=status),
                data=SimpleNamespace(tags=tags),
            )
    tracking = ModuleType("mlflow.tracking")
    tracking.MlflowClient = MLflowClient
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)
    class Minio(FakeClient):
        def __init__(self, endpoint, access_key, secret_key, secure):
            assert (endpoint, access_key, secret_key, secure) == (
                "127.0.0.1:9000", "unit-user", "unit-secret", False
            )
            super().__init__()
            network["minio_clients"].append(self)
    sdk = ModuleType("minio")
    sdk.Minio = Minio
    monkeypatch.setitem(sys.modules, "minio", sdk)
    for key, value in {
        "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
        "MINIO_ENDPOINT": "127.0.0.1:9000",
        "MINIO_BUCKET": BUCKET,
        "MINIO_SECURE": "false",
        "MINIO_ACCESS_KEY": "unit-user",
        "MINIO_SECRET_KEY": "unit-secret",
        "MODEL_OBJECT_KEY": KEY,
    }.items():
        monkeypatch.setenv(key, value)
    return network


def test_configured_version_is_resolved_via_its_own_mlflow_tags(tmp_path, monkeypatch):
    network = _fake_services(monkeypatch)
    info = downloads.retrieve_model_artifact(tmp_path / "remote.pt")
    assert info.path.read_bytes() == PAYLOAD
    assert network["mlflow_calls"] == 1
    assert len(network["minio_clients"]) == 1


def test_explicit_object_key_overrides_optional_model_key(tmp_path, monkeypatch):
    network = _fake_services(monkeypatch)
    monkeypatch.delenv("MODEL_OBJECT_KEY")
    info = downloads.retrieve_model_artifact(tmp_path / "remote.pt", object_key=KEY)
    assert info.object_key == KEY


def test_missing_configured_key_fails_without_minio(tmp_path, monkeypatch):
    network = _fake_services(monkeypatch)
    monkeypatch.delenv("MODEL_OBJECT_KEY")
    with pytest.raises(ValueError, match="MODEL_OBJECT_KEY"):
        downloads.retrieve_model_artifact(tmp_path / "remote.pt")
    assert network["mlflow_calls"] == 0 and not network["minio_clients"]


@pytest.mark.parametrize("status", ["FAILED", "RUNNING", "KILLED"])
def test_only_finished_mlflow_runs_are_eligible(tmp_path, monkeypatch, status):
    network = _fake_services(monkeypatch, status=status)
    with pytest.raises(ValueError, match="FINISHED"):
        downloads.retrieve_model_artifact(tmp_path / "remote.pt")
    assert not network["minio_clients"]


@pytest.mark.parametrize("changes", [
    {"model.storage_status": "local-only"},
    {"model.minio_bucket": "wrong-bucket"},
    {"model.minio_object_key": "model/other/model.pt"},
])
def test_mismatched_remote_reference_fails_before_minio(tmp_path, monkeypatch, changes):
    network = _fake_services(monkeypatch, tag_changes=changes)
    with pytest.raises(ValueError, match="MLflow does not confirm"):
        downloads.retrieve_model_artifact(tmp_path / "remote.pt")
    assert not network["minio_clients"]


@pytest.mark.parametrize("changes", [
    {"model.minio_size_bytes": "not-integer"},
    {"model.minio_size_bytes": "0"},
    {"model.minio_size_bytes": str(6*1024*1024)},
    {"model.minio_sha256": "wrong"},
    {"model.minio_sha256": None},
])
def test_invalid_mlflow_fingerprint_is_rejected(tmp_path, monkeypatch, changes):
    network = _fake_services(monkeypatch, tag_changes=changes)
    with pytest.raises(ValueError):
        downloads.retrieve_model_artifact(tmp_path / "remote.pt")
    assert not network["minio_clients"]


def test_cli_outputs_reference_without_loading_model(tmp_path, monkeypatch, capsys):
    _fake_services(monkeypatch)
    from scripts import download_model_artifact as cli
    path = tmp_path / "model.pt"
    monkeypatch.setattr(sys, "argv", ["download_model_artifact.py", "--destination", str(path)])
    assert cli.main() == 0
    text = capsys.readouterr().out
    assert "PASS: MinIO model downloaded and verified" in text
    assert SHA in text and "No model was loaded" in text


def test_cli_refuses_existing_destination_with_exit_code_one(tmp_path, monkeypatch, capsys):
    _fake_services(monkeypatch)
    from scripts import download_model_artifact as cli
    path = tmp_path / "model.pt"
    path.write_bytes(b"existing")
    monkeypatch.setattr(sys, "argv", ["download_model_artifact.py", "--destination", str(path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "overwrite" in capsys.readouterr().err
    assert path.read_bytes() == b"existing"
