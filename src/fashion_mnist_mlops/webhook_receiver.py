"""Strict localhost receiver used to observe webhook payloads."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Final
from urllib.parse import urlsplit

from .webhook import TrainingCompletedPayload

DEFAULT_RECEIVER_HOST: Final[str] = "127.0.0.1"
DEFAULT_RECEIVER_PORT: Final[int] = 8765
DEFAULT_RECEIVER_PATH: Final[str] = "/webhook"
MAX_RECEIVER_BODY_BYTES: Final[int] = 64 * 1024

_REQUIRED_KEYS = {
    "schema_version",
    "event",
    "status",
    "mlflow_run_id",
    "minio_bucket",
    "minio_object_key",
    "validation_accuracy",
}


def validate_received_payload(value: object) -> dict[str, object]:
    """Validate that an observed JSON value exactly matches the webhook payload contract."""
    if not isinstance(value, dict):
        raise ValueError("Webhook JSON body must be an object.")
    if set(value) != _REQUIRED_KEYS:
        raise ValueError("Webhook JSON body does not match the required schema fields.")
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported webhook schema_version.")
    if value.get("event") != "training.completed":
        raise ValueError("Unexpected webhook event.")
    if value.get("status") != "success":
        raise ValueError("Unexpected webhook status.")

    candidate = TrainingCompletedPayload(
        mlflow_run_id=value["mlflow_run_id"],
        minio_bucket=value["minio_bucket"],
        minio_object_key=value["minio_object_key"],
        validation_accuracy=value["validation_accuracy"],
    )
    canonical = candidate.as_json_dict()
    if value != canonical:
        raise ValueError("Webhook JSON body is not in canonical contract form.")
    return canonical


class WebhookCaptureServer(HTTPServer):
    """Local receiver that retains validated payloads in memory."""

    def __init__(self, server_address: tuple[str, int], path: str) -> None:
        self.expected_path = path
        self.captured_payloads: list[dict[str, object]] = []
        super().__init__(server_address, _CaptureHandler)


class _CaptureHandler(BaseHTTPRequestHandler):
    server: WebhookCaptureServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _empty_response(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.headers.get_content_type() != "application/json":
            self._empty_response(415)
            return

        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            self._empty_response(400)
            return
        if length < 0:
            self._empty_response(411)
            return
        if length > MAX_RECEIVER_BODY_BYTES:
            self._empty_response(413)
            return

        body = self.rfile.read(length)

        # Drain a bounded POST body before replying 404. On Windows, closing
        # a connection with unread request bytes can reset the TCP connection,
        # causing clients to observe a transport error instead of HTTP 404.
        if urlsplit(self.path).path != self.server.expected_path:
            self._empty_response(404)
            return

        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._empty_response(400)
            return

        try:
            canonical = validate_received_payload(decoded)
        except (KeyError, TypeError, ValueError):
            self._empty_response(422)
            return

        self.server.captured_payloads.append(canonical)
        print("Received valid training-complete webhook:", flush=True)
        print(json.dumps(canonical, indent=2, sort_keys=True), flush=True)
        self._empty_response(204)


def create_webhook_receiver(
    *,
    host: str = DEFAULT_RECEIVER_HOST,
    port: int = DEFAULT_RECEIVER_PORT,
    path: str = DEFAULT_RECEIVER_PATH,
) -> WebhookCaptureServer:
    """Create a local capture server without starting its request loop."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string.")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 to 65535.")
    if not isinstance(path, str) or not path.startswith("/") or "?" in path or "#" in path:
        raise ValueError("path must be an absolute URL path without query or fragment.")
    return WebhookCaptureServer((host.strip(), port), path)
