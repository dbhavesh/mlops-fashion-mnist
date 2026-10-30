"""Configuration for the Streamlit presentation layer.

The UI only needs to know where the FastAPI service is reachable. It must not
require MLflow, MinIO, model-selection, or inference settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from urllib.parse import urlsplit, urlunsplit


FASTAPI_BASE_URL_ENV = "FASTAPI_BASE_URL"
DEFAULT_FASTAPI_BASE_URL = "http://127.0.0.1:8000"


class UIConfigurationError(ValueError):
    """Raised when Streamlit presentation configuration is invalid."""


@dataclass(frozen=True, slots=True)
class UISettings:
    """Validated runtime settings used by the Streamlit application."""

    fastapi_base_url: str


def validate_fastapi_base_url(value: str) -> str:
    """Validate and normalize an HTTP(S) FastAPI origin.

    A bare origin keeps later endpoint construction unambiguous. The same
    setting therefore works for local development (for example,
    ``http://127.0.0.1:8000``) and a future Compose service name (for example,
    ``http://api:8000``).
    """
    if not isinstance(value, str):
        raise UIConfigurationError("FASTAPI_BASE_URL must be a string.")

    candidate = value.strip()
    if not candidate:
        raise UIConfigurationError("FASTAPI_BASE_URL must not be empty.")

    try:
        parsed = urlsplit(candidate)
        # Accessing .port performs additional validation such as rejecting
        # non-numeric or out-of-range ports.
        _ = parsed.port
    except ValueError as exc:
        raise UIConfigurationError("FASTAPI_BASE_URL is not a valid URL.") from exc

    if parsed.scheme not in {"http", "https"}:
        raise UIConfigurationError("FASTAPI_BASE_URL must use http or https.")
    if parsed.hostname is None:
        raise UIConfigurationError("FASTAPI_BASE_URL must include a host.")
    if parsed.username is not None or parsed.password is not None:
        raise UIConfigurationError("FASTAPI_BASE_URL must not contain credentials.")
    if parsed.path not in {"", "/"}:
        raise UIConfigurationError("FASTAPI_BASE_URL must be a service origin without a path.")
    if parsed.query or parsed.fragment:
        raise UIConfigurationError(
            "FASTAPI_BASE_URL must not contain a query string or fragment."
        )

    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def load_ui_settings(environ: Mapping[str, str] | None = None) -> UISettings:
    """Load Streamlit settings from an explicit mapping or the process env."""
    source = os.environ if environ is None else environ
    raw_base_url = source.get(FASTAPI_BASE_URL_ENV, DEFAULT_FASTAPI_BASE_URL)
    return UISettings(fastapi_base_url=validate_fastapi_base_url(raw_base_url))
