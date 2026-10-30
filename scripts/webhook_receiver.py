"""Run the strict local webhook receiver.

Example:
    python scripts/webhook_receiver.py --once
"""

import argparse
import json
from pathlib import Path

from fashion_mnist_mlops.webhook_receiver import (
    DEFAULT_RECEIVER_HOST,
    DEFAULT_RECEIVER_PATH,
    DEFAULT_RECEIVER_PORT,
    create_webhook_receiver,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_RECEIVER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_RECEIVER_PORT)
    parser.add_argument("--path", default=DEFAULT_RECEIVER_PATH)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Handle one request and exit; useful for delivery verification.",
    )
    parser.add_argument(
        "--capture-json", type=Path,
        help="Save the accepted JSON payload to a new file; requires --once.",
    )
    args = parser.parse_args()
    if args.capture_json is not None and not args.once:
        parser.error("--capture-json requires --once")
    if args.capture_json is not None and (
        args.capture_json.exists() or args.capture_json.is_symlink()
    ):
        parser.error("--capture-json destination already exists (no overwrite)")

    server = create_webhook_receiver(host=args.host, port=args.port, path=args.path)
    actual_host, actual_port = server.server_address[:2]
    print(f"Webhook receiver listening on http://{actual_host}:{actual_port}{args.path}")
    print("Press Ctrl+C to stop." if not args.once else "Waiting for one request...")
    try:
        if args.once:
            server.handle_request()
        else:
            server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nWebhook receiver stopped.")
    finally:
        server.server_close()
    if args.capture_json is not None:
        if not server.captured_payloads:
            parser.exit(1, "FAIL: No valid webhook was received; no capture saved.\n")
        args.capture_json.parent.mkdir(parents=True, exist_ok=True)
        with args.capture_json.open("x", encoding="utf-8") as capture:
            json.dump(server.captured_payloads[0], capture, indent=2, sort_keys=True)
            capture.write("\n")
        print(f"Captured webhook evidence: {args.capture_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
