"""required bucket provisioning with an isolated fake MinIO client."""

import pytest

from fashion_mnist_mlops.storage import ensure_bucket_exists


BUCKET = "fashion-minst-bucket"


class FakeMinio:
    def __init__(self, *, exists=False):
        self.exists = exists
        self.calls = []

    def bucket_exists(self, name):
        self.calls.append(("bucket_exists", name))
        return self.exists

    def make_bucket(self, name):
        self.calls.append(("make_bucket", name))
        self.exists = True


def test_creates_exact_bucket_and_verifies_it():
    client = FakeMinio()
    assert ensure_bucket_exists(client, bucket_name=BUCKET) is True
    assert client.calls == [
        ("bucket_exists", BUCKET),
        ("make_bucket", BUCKET),
        ("bucket_exists", BUCKET),
    ]


def test_existing_bucket_is_left_untouched():
    client = FakeMinio(exists=True)
    assert ensure_bucket_exists(client, bucket_name=BUCKET) is False
    assert client.calls == [("bucket_exists", BUCKET)]


def test_second_invocation_is_idempotent():
    client = FakeMinio()
    assert ensure_bucket_exists(client, bucket_name=BUCKET) is True
    assert ensure_bucket_exists(client, bucket_name=BUCKET) is False
    assert client.calls.count(("make_bucket", BUCKET)) == 1


@pytest.mark.parametrize("invalid_name", ["fashion-mnist-bucket", "", "other-bucket"])
def test_rejects_wrong_bucket_name_before_contacting_minio(invalid_name):
    client = FakeMinio()
    with pytest.raises(ValueError, match="must be exactly"):
        ensure_bucket_exists(client, bucket_name=invalid_name)
    assert client.calls == []


def test_bucket_existence_probe_error_is_not_masked():
    class BrokenMinio(FakeMinio):
        def bucket_exists(self, name):
            raise ConnectionError("MinIO unavailable")

    with pytest.raises(ConnectionError, match="unavailable"):
        ensure_bucket_exists(BrokenMinio(), bucket_name=BUCKET)


def test_creation_failure_propagates():
    class BrokenMinio(FakeMinio):
        def make_bucket(self, name):
            raise PermissionError("AccessDenied")

    with pytest.raises(PermissionError, match="AccessDenied"):
        ensure_bucket_exists(BrokenMinio(), bucket_name=BUCKET)


def test_postcreation_check_required():
    class EventuallyInvisibleMinio(FakeMinio):
        def make_bucket(self, name):
            self.calls.append(("make_bucket", name))
            # Deliberately leave exists=False.

    with pytest.raises(RuntimeError, match="not visible"):
        ensure_bucket_exists(EventuallyInvisibleMinio(), bucket_name=BUCKET)


def test_provisioning_cli_uses_validated_settings_without_printing_secrets(
    monkeypatch, capsys
):
    import sys
    from types import ModuleType

    from scripts import provision_minio_bucket

    monkeypatch.setenv("MINIO_ENDPOINT", "127.0.0.1:9000")
    monkeypatch.setenv("MINIO_BUCKET", BUCKET)
    monkeypatch.setenv("MINIO_SECURE", "false")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-user")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret-not-for-output")
    captured = []

    class FakeSDKClient(FakeMinio):
        def __init__(self, endpoint, access_key, secret_key, secure):
            captured.append((endpoint, access_key, secret_key, secure))
            super().__init__()

    fake_module = ModuleType("minio")
    fake_module.Minio = FakeSDKClient
    monkeypatch.setitem(sys.modules, "minio", fake_module)
    assert provision_minio_bucket.main() == 0
    assert captured == [
        ("127.0.0.1:9000", "test-user", "test-secret-not-for-output", False)
    ]
    output = capsys.readouterr().out
    assert f"PASS: Required MinIO bucket exists: {BUCKET}" in output
    assert "Action: created" in output
    assert "test-secret-not-for-output" not in output


def test_provisioning_cli_requires_credentials_before_client_creation(monkeypatch):
    from fashion_mnist_mlops.config_validation import ConfigurationError
    from scripts import provision_minio_bucket

    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="credentials are required"):
        provision_minio_bucket.main()
