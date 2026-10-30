"""multiple real artifact versions preserved by conditional MinIO writes."""

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fashion_mnist_mlops.model import LeNet5
from fashion_mnist_mlops.model_artifacts import save_model_artifact
from fashion_mnist_mlops.storage import upload_model_artifact
from fashion_mnist_mlops.model_verification import verify_model_versions
from fashion_mnist_mlops.tracking import minio_artifact_tags

BUCKET = "fashion-minst-bucket"
OLDER = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
NEWER = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


class Conflict(Exception):
    code = "PreconditionFailed"


class FakeResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.released = False

    def release_conn(self):
        self.released = True


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.put_calls = []
        self.get_calls = []
        self.responses = []

    def bucket_exists(self, bucket):
        assert bucket == BUCKET
        return True

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        assert kwargs["bucket_name"] == BUCKET
        assert kwargs["extra_headers"] == {"If-None-Match": "*"}
        key = kwargs["object_name"]
        if key in self.objects:
            raise Conflict()
        value = kwargs["data"].read()
        assert len(value) == kwargs["length"]
        self.objects[key] = value

    def get_object(self, bucket, key):
        assert bucket == BUCKET
        self.get_calls.append(key)
        if key not in self.objects:
            raise FileNotFoundError(key)
        response = FakeResponse(self.objects[key])
        self.responses.append(response)
        return response


class FakeTracking:
    def __init__(self, runs):
        self.runs = runs
        self.lookups = []

    def get_experiment_by_name(self, name):
        assert name == "fashion-mnist-classifier"
        return SimpleNamespace(experiment_id="1", lifecycle_stage="active")

    def get_run(self, run_id):
        self.lookups.append(run_id)
        return self.runs[run_id]


def _provenance(epochs):
    return {
        "training": {"epochs": epochs, "batch_size": 64, "learning_rate": 0.001,
                     "optimizer": "adam", "loss": "cross_entropy", "seed": 42},
        "validation": {"validation_split": 0.1},
        "dataset": {"name": "FashionMNIST", "location": {"scope": "project", "path": "data"},
                    "train_samples_used": 32, "validation_samples_used": 16,
                    "test_samples_available": 10000, "max_train_samples": 32,
                    "max_validation_samples": 16, "download_allowed": False},
        "source_config": {"location": {"scope": "project", "path": "configs/train.toml"},
                          "sha256": "a" * 64, "effective_sha256": "b" * 64},
        "preprocessing": {"version": 1, "input_shape": [1, 28, 28],
                          "normalization_mean": [0.2860406],
                          "normalization_std": [0.3530242]},
        "software": {"python": "3.12", "torch": "2.x", "torchvision": "0.x"},
    }


@pytest.fixture
def versions(tmp_path):
    s3 = FakeS3()
    records = {}
    paths = {}
    for run_id, epoch, end_time in ((OLDER, 1, 1000), (NEWER, 2, 2000)):
        torch.manual_seed(epoch + 100)
        path = tmp_path / run_id / "model.pt"
        save_model_artifact(LeNet5(), path, epoch=epoch, validation_accuracy=0.25,
                              mlflow_run_id=run_id, provenance=_provenance(epoch))
        paths[run_id] = path
        uploaded = upload_model_artifact(
            s3, artifact_path=path, run_id=run_id, bucket_name=BUCKET,
        )
        records[run_id] = SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, experiment_id="1", status="FINISHED",
                                 end_time=end_time),
            data=SimpleNamespace(tags=minio_artifact_tags(run_id, uploaded),
                                 params={"epochs": str(epoch)}),
        )
    return FakeTracking(records), s3, paths


def _verify(versions, **kwargs):
    client, s3, _ = versions
    arguments = dict(older_run_id=OLDER, newer_run_id=NEWER)
    arguments.update(kwargs)
    return verify_model_versions(client, s3, **arguments)


def test_two_versions_have_distinct_retrievable_objects_and_correct_provenance(versions):
    _, s3, paths = versions
    first, second = _verify(versions)
    assert (first.run_id, second.run_id) == (OLDER, NEWER)
    assert first.object_key == f"model/{OLDER}/model.pt"
    assert second.object_key == f"model/{NEWER}/model.pt"
    assert first.object_key != second.object_key
    assert first.sha256 == hashlib.sha256(paths[OLDER].read_bytes()).hexdigest()
    assert second.sha256 == hashlib.sha256(paths[NEWER].read_bytes()).hexdigest()
    assert first.sha256 != second.sha256
    assert first.size_bytes == len(paths[OLDER].read_bytes())
    assert all(r.closed and r.released for r in s3.responses)
    assert len(s3.put_calls) == 2  # Verifier added no PUTs.


def test_older_object_preserved_after_newer_run_uploaded(versions):
    _, s3, paths = versions
    before = bytes(s3.objects[f"model/{OLDER}/model.pt"])
    _verify(versions)
    assert s3.objects[f"model/{OLDER}/model.pt"] == before
    assert paths[OLDER].read_bytes() == before


def test_verification_is_independent_of_original_local_artifacts(versions):
    _, s3, paths = versions
    for path in paths.values():
        path.unlink()
    assert len(s3.objects) == 2
    assert _verify(versions)[0].run_id == OLDER


def test_conflicting_reupload_of_different_bytes_is_rejected_and_both_models_preserved(versions, tmp_path):
    _, s3, paths = versions
    key = f"model/{OLDER}/model.pt"
    before = dict(s3.objects)
    new_content = torch.load(paths[OLDER], map_location="cpu", weights_only=True)
    new_content["model_state_dict"]["fc3.bias"] += 1
    changed = tmp_path / "candidate" / "model.pt"
    changed.parent.mkdir()
    torch.save(new_content, changed)
    assert changed.read_bytes() != before[key]

    with pytest.raises(FileExistsError, match="already exists"):
        upload_model_artifact(s3, artifact_path=changed, run_id=OLDER, bucket_name=BUCKET)
    assert s3.objects == before
    assert s3.put_calls[-1]["extra_headers"] == {"If-None-Match": "*"}
    assert _verify(versions)[1].run_id == NEWER


@pytest.mark.parametrize("run_a,run_b", [(OLDER, OLDER), ("", NEWER), ("..", NEWER)])
def test_duplicate_or_unsafe_run_ids_are_rejected(versions, run_a, run_b):
    with pytest.raises(ValueError):
        _verify(versions, older_run_id=run_a, newer_run_id=run_b)


def test_swapped_run_order_is_rejected(versions):
    with pytest.raises(ValueError, match="earlier run"):
        _verify(versions, older_run_id=NEWER, newer_run_id=OLDER)


def test_unfinished_run_is_rejected_before_storage_download(versions):
    tracking, s3, _ = versions
    tracking.runs[OLDER].info.status = "FAILED"
    before = len(s3.get_calls)
    with pytest.raises(ValueError, match="not FINISHED"):
        _verify(versions)
    assert len(s3.get_calls) == before


def test_run_from_another_experiment_is_rejected(versions):
    tracking, _, _ = versions
    tracking.runs[NEWER].info.experiment_id = "2"
    with pytest.raises(ValueError, match="experiment"):
        _verify(versions)


@pytest.mark.parametrize("tag,value", [
    ("model.minio_object_key", f"model/{OLDER}/model.pt"),
    ("model.minio_bucket", "wrong-bucket"),
    ("model.storage_status", "local-only"),
    ("model.minio_sha256", "z" * 64),
    ("model.minio_size_bytes", "-1"),
])
def test_swapped_or_invalid_mlflow_reference_is_rejected(versions, tag, value):
    tracking, s3, _ = versions
    tracking.runs[NEWER].data.tags[tag] = value
    prior = len(s3.get_calls)
    with pytest.raises(ValueError, match="MinIO|tags"):
        _verify(versions)
    assert len(s3.get_calls) == prior


@pytest.mark.parametrize("target", [OLDER, NEWER])
def test_remote_corruption_is_detected_not_silently_accepted(versions, target):
    _, s3, _ = versions
    key = f"model/{target}/model.pt"
    s3.objects[key] = b"corrupted" + s3.objects[key]
    with pytest.raises(ValueError, match="integrity|recorded size"):
        _verify(versions)


def test_missing_older_object_is_not_hidden_by_newer_object(versions):
    _, s3, _ = versions
    del s3.objects[f"model/{OLDER}/model.pt"]
    with pytest.raises(FileNotFoundError):
        _verify(versions)


def test_empty_or_wrong_bucket_rejected_before_remote_contact(versions):
    _, s3, _ = versions
    for bad in ("", "fashion-mnist-bucket"):
        with pytest.raises(ValueError, match="bucket"):
            _verify(versions, bucket_name=bad)
    assert len(s3.put_calls) == 2
