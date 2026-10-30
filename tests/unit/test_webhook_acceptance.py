"""real-payload evidence checks with no remote network or mutations."""

import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fashion_mnist_mlops.webhook_verification import verify_webhook_evidence

RUN_ID = "abc0123456789defabc0123456789def"
BUCKET = "fashion-minst-bucket"
KEY = f"model/{RUN_ID}/model.pt"
MODEL = b"verified model artifact bytes"
DIGEST = hashlib.sha256(MODEL).hexdigest()


def observed():
    return dict(
        mlflow_run_id=RUN_ID, minio_bucket=BUCKET,
        minio_object_key=KEY, validation_accuracy=0.25,
        schema_version=1, event="training.completed", status="success",
    )


class FakeTracking:
    def __init__(self):
        self.experiment = SimpleNamespace(experiment_id="1", lifecycle_stage="active")
        self.run = SimpleNamespace(
            info=SimpleNamespace(run_id=RUN_ID, experiment_id="1", status="FINISHED", end_time=1234),
            data=SimpleNamespace(
                tags={
                    "model.storage_status": "minio-verified",
                    "model.minio_bucket": BUCKET,
                    "model.minio_object_key": KEY,
                    "model.minio_size_bytes": str(len(MODEL)),
                    "model.minio_sha256": DIGEST,
                },
                metrics={"val_accuracy": 0.25},
            ),
        )
        self.queries = []

    def get_experiment_by_name(self, name):
        self.queries.append(("experiment", name))
        return self.experiment

    def get_run(self, run_id):
        self.queries.append(("run", run_id))
        return self.run


class FakeObject(io.BytesIO):
    def __init__(self, contents):
        super().__init__(contents)
        self.released = False

    def release_conn(self):
        self.released = True


class FakeMinio:
    def __init__(self, contents=MODEL):
        self.contents = contents
        self.calls = []
        self.object = None

    def get_object(self, bucket, key):
        self.calls.append((bucket, key))
        self.object = FakeObject(self.contents)
        return self.object


def verify(payload=None, tracking=None, storage=None):
    tracking = FakeTracking() if tracking is None else tracking
    storage = FakeMinio() if storage is None else storage
    result = verify_webhook_evidence(
        observed() if payload is None else payload,
        tracking_client=tracking,
        minio_client=storage,
    )
    return result, tracking, storage


def test_read_only_verification_matches_receiver_mlflow_and_full_minio_digest():
    result, tracking, storage = verify()
    assert result.mlflow_run_id == RUN_ID
    assert result.validation_accuracy == 0.25
    assert result.model_sha256 == DIGEST
    assert result.model_size_bytes == len(MODEL)
    assert tracking.queries == [
        ("experiment", "fashion-mnist-classifier"), ("run", RUN_ID)
    ]
    assert storage.calls == [(BUCKET, KEY)]
    assert storage.object.closed and storage.object.released


@pytest.mark.parametrize("field,value", [
    ("minio_bucket", "wrong"),
    ("minio_object_key", "model/another/model.pt"),
    ("status", "failed"),
    ("event", "wrong"),
    ("schema_version", 2),
    ("validation_accuracy", -1.0),
])
def test_invalid_capture_fails_before_remote_queries(field, value):
    payload = observed()
    payload[field] = value
    tracking, storage = FakeTracking(), FakeMinio()
    with pytest.raises((ValueError, TypeError)):
        verify(payload, tracking, storage)
    assert not tracking.queries
    assert not storage.calls


def test_rejects_noncanonical_extra_field():
    payload = observed()
    payload["extra"] = True
    with pytest.raises(ValueError, match="schema"):
        verify(payload)


@pytest.mark.parametrize("status,end_time", [
    ("RUNNING", None), ("FAILED", 1234), ("FINISHED", None),
])
def test_refuses_run_not_finished(status, end_time):
    tracking, storage = FakeTracking(), FakeMinio()
    tracking.run.info.status = status
    tracking.run.info.end_time = end_time
    with pytest.raises(ValueError, match="FINISHED"):
        verify(tracking=tracking, storage=storage)
    assert storage.calls == []


@pytest.mark.parametrize("tag,wrong", [
    ("model.storage_status", "local-only"),
    ("model.minio_bucket", "wrong"),
    ("model.minio_object_key", "model/incorrect/model.pt"),
    ("model.minio_sha256", "not-a-digest"),
    ("model.minio_size_bytes", "0"),
])
def test_refuses_invalid_mlflow_tags_without_reading_minio(tag, wrong):
    tracking, storage = FakeTracking(), FakeMinio()
    tracking.run.data.tags[tag] = wrong
    with pytest.raises(ValueError):
        verify(tracking=tracking, storage=storage)
    assert not storage.calls


def test_refuses_accuracy_disagreement_without_reading_minio():
    tracking, storage = FakeTracking(), FakeMinio()
    tracking.run.data.metrics["val_accuracy"] = 0.251
    with pytest.raises(ValueError, match="accuracy"):
        verify(tracking=tracking, storage=storage)
    assert storage.calls == []


def test_refuses_unrelated_mlflow_experiment():
    tracking, storage = FakeTracking(), FakeMinio()
    tracking.run.info.experiment_id = "other"
    with pytest.raises(ValueError, match="another experiment"):
        verify(tracking=tracking, storage=storage)
    assert storage.calls == []


@pytest.mark.parametrize("actual", [MODEL + b"extra", MODEL[:-1], MODEL[:-1] + b"x"])
def test_refuses_tampered_object_and_releases_connection(actual):
    storage = FakeMinio(actual)
    with pytest.raises(ValueError, match="(size|larger|SHA-256)"):
        verify(storage=storage)
    assert storage.object.closed and storage.object.released


def test_receiver_cli_saves_original_capture_as_new_json_file(tmp_path, monkeypatch):
    from scripts import webhook_receiver as cli

    class FakeReceiver:
        server_address = ("127.0.0.1", 8765)
        captured_payloads = []
        def handle_request(self):
            self.captured_payloads.append(observed())
        def server_close(self):
            pass

    monkeypatch.setattr(cli, "create_webhook_receiver", lambda **kwargs: FakeReceiver())
    target = tmp_path / "capture.json"
    monkeypatch.setattr(sys, "argv", ["receiver", "--once", "--capture-json", str(target)])
    assert cli.main() == 0
    assert json.loads(target.read_text(encoding="utf-8")) == observed()

    monkeypatch.setattr(sys, "argv", ["receiver", "--once", "--capture-json", str(target)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert json.loads(target.read_text(encoding="utf-8")) == observed()


def test_receiver_cli_refuses_capture_without_once(tmp_path, monkeypatch):
    from scripts import webhook_receiver as cli
    monkeypatch.setattr(sys, "argv", ["receiver", "--capture-json", str(tmp_path / "capture.json")])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_receiver_does_not_save_failed_capture(tmp_path, monkeypatch):
    from scripts import webhook_receiver as cli

    class FakeReceiver:
        server_address = ("127.0.0.1", 8765)
        captured_payloads = []
        def handle_request(self):
            pass
        def server_close(self):
            pass

    monkeypatch.setattr(cli, "create_webhook_receiver", lambda **kwargs: FakeReceiver())
    target = tmp_path / "capture.json"
    monkeypatch.setattr(sys, "argv", ["receiver", "--once", "--capture-json", str(target)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert not target.exists()
