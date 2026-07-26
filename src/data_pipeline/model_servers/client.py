"""Generic, tiny client for the resident model-server harness.

Deliberately minimal: connect, send one request, block for the reply, exit 0 or
nonzero. Must never import torch or any model library — it knows nothing about
what the adapter does, only that it sends a JSON payload of paths and waits.

This is what a Snakemake `shell:` line would call in place of today's
`{RUN} -m data_pipeline.pipeline_orchestrator.tasks ...` invocation, IF this
prototype were ever wired in (it is not, in this pass). Snakemake's file-exists
contract is preserved because the client blocks until the server finishes writing
the well's output file(s) and only then exits.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import uuid
from pathlib import Path

from data_pipeline.model_servers.protocol import ProtocolError, Request, Response, recv_frame, send_frame


def call_server(socket_path: Path, payload: dict, *, timeout: float | None = None) -> Response:
    """Connect to socket_path, send payload, block for the response. Raises on transport error."""
    request = Request(request_id=str(uuid.uuid4()), payload=payload)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        if timeout is not None:
            sock.settimeout(timeout)
        sock.connect(str(socket_path))
        send_frame(sock, request.to_json())
        frame = recv_frame(sock)
    return Response.from_json(frame)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send one request to a resident model server.")
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--payload-json", required=True, help="JSON object, e.g. '{\"well_id\": \"...\"}'")
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.payload_json)
    except json.JSONDecodeError as e:
        print(f"client: invalid --payload-json: {e}", file=sys.stderr)
        return 2

    try:
        response = call_server(args.socket_path, payload, timeout=args.timeout)
    except (OSError, ProtocolError) as e:
        print(f"client: transport error: {e}", file=sys.stderr)
        return 2

    if not response.ok:
        print(f"client: server reported failure: {response.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
