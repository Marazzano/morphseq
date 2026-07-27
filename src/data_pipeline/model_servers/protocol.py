"""Wire protocol for the resident model-server harness.

Shared by both `harness.py` (server side) and `client.py` (client side). This module
must stay dependency-light: stdlib only (json, socket, struct). No torch, no numpy,
no pandas — the client process must never import a model library, and the harness
itself must never import torch (see PATHS-NOT-PAYLOADS + harness-is-generic in the
package README).

Framing: length-prefixed JSON. Each message is:
    4 bytes big-endian uint32 length
    N bytes UTF-8 JSON payload

Request payload (client -> server):
    {"request_id": str, "payload": {...}}   # payload is adapter-defined, paths only

Response payload (server -> client):
    {"request_id": str, "ok": bool, "error": str | None}

`payload` for a request is intentionally an opaque dict as far as this module and
the harness are concerned — only the adapter interprets its contents. It should
carry file PATHS (input shards, output targets), never arrays/tensors: client and
server share a filesystem, so shipping data over the socket would be pure overhead
and would violate "the client never imports the model library."
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass, field
from typing import Any

_LENGTH_STRUCT = struct.Struct(">I")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # 64MB guard against a runaway/garbled length prefix


class ProtocolError(Exception):
    """Malformed frame, oversized message, or unexpected disconnect."""


@dataclass(frozen=True)
class Request:
    request_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "payload": self.payload}

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "Request":
        return Request(request_id=str(obj["request_id"]), payload=dict(obj.get("payload", {})))


@dataclass(frozen=True)
class Response:
    request_id: str
    ok: bool
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "ok": self.ok, "error": self.error}

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "Response":
        return Response(
            request_id=str(obj["request_id"]),
            ok=bool(obj["ok"]),
            error=obj.get("error"),
        )


def send_frame(sock: socket.socket, obj: dict[str, Any]) -> None:
    """Encode obj as length-prefixed JSON and write it fully to sock."""
    body = json.dumps(obj).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message too large: {len(body)} bytes > {MAX_MESSAGE_BYTES}")
    sock.sendall(_LENGTH_STRUCT.pack(len(body)) + body)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProtocolError("connection closed while reading frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    """Read one length-prefixed JSON frame from sock. Raises ProtocolError on EOF/garbage."""
    header = _recv_exact(sock, _LENGTH_STRUCT.size)
    (length,) = _LENGTH_STRUCT.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"declared frame length too large: {length} bytes")
    body = _recv_exact(sock, length)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ProtocolError(f"invalid JSON frame: {e}") from e
