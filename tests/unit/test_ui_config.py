"""Offline tests for the Streamlit backend-address setting."""

import pytest

from fashion_mnist_mlops.ui_config import (
    DEFAULT_FASTAPI_BASE_URL,
    FASTAPI_BASE_URL_ENV,
    UIConfigurationError,
    load_ui_settings,
    validate_fastapi_base_url,
)


def test_backend_address_defaults_to_local_fastapi_when_unset():
    settings = load_ui_settings({})
    assert settings.fastapi_base_url == DEFAULT_FASTAPI_BASE_URL
    assert settings.fastapi_base_url == "http://127.0.0.1:8000"


def test_backend_address_can_be_overridden_for_compose_dns():
    settings = load_ui_settings({FASTAPI_BASE_URL_ENV: "http://api:8000/"})
    assert settings.fastapi_base_url == "http://api:8000"


def test_explicit_environment_mapping_does_not_depend_on_process_environment(monkeypatch):
    monkeypatch.setenv(FASTAPI_BASE_URL_ENV, "http://wrong-process-value:9999")
    settings = load_ui_settings({FASTAPI_BASE_URL_ENV: "https://backend.example.test:8443"})
    assert settings.fastapi_base_url == "https://backend.example.test:8443"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "api:8000",
        "ftp://api:8000",
        "http://",
        "http://user:secret@api:8000",
        "http://api:8000/v1",
        "http://api:8000?debug=true",
        "http://api:8000#fragment",
        "http://api:not-a-port",
        "http://api:70000",
    ],
)
def test_invalid_backend_addresses_fail_closed(value):
    with pytest.raises(UIConfigurationError):
        validate_fastapi_base_url(value)


def test_non_string_backend_address_is_rejected():
    with pytest.raises(UIConfigurationError, match="string"):
        validate_fastapi_base_url(8000)  # type: ignore[arg-type]
