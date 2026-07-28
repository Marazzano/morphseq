"""Generic, tiny client for the resident model-server harness.

Deliberately minimal: connect, send one request, block for the reply, exit 0 or
nonzero. Must never import torch or any model library — it knows nothing about
what the adapter does, only that it sends a JSON payload of paths and waits.

Called by the `*_served` rules in place of the in-process
`{RUN} -m data_pipeline.pipeline_orchestrator.tasks ...` invocation. Snakemake's
file-exists contract is preserved because the client blocks until the server has
finished writing the well's output file(s) and only then exits.

READINESS. Every client WAITS for its service to start accepting before sending
(see DEFAULT_READY_TIMEOUT_S). This is required, not defensive: Snakemake starts
the service and all of its consumers simultaneously, so without the wait every
client races the model load and dies. The wait lives in `call_server`, the one
function all clients go through.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import uuid
from pathlib import Path

from data_pipeline.model_servers.protocol import ProtocolError, Request, Response, recv_frame, send_frame


# How long a client waits for its service to finish loading before giving up.
#
# THIS IS LOAD-BEARING, NOT A NICETY. Snakemake starts a service() job and ALL of its consumers in
# the SAME instant -- from snakemake/executors.py:
#
#     futures = [self.run_single_job(j) for j in job]
#     n_non_service = sum(1 for j in job if not j.is_service)
#
# Every member of the group launches at once, and only then does it wait for the non-service jobs.
# That is correct for a PIPE group, whose members really do stream concurrently. For a model server
# it means every client starts while the model is still loading: the socket does not exist yet, so
# without a wait each client dies instantly with ENOENT/ECONNREFUSED and the run stalls. That is
# exactly how jobs 22825862 / 22827486 / 22829477 failed -- logs freeze on the harness's
# "listening on ..." line, zero clients finished, cpu=00:00:13 against 7m of wallclock.
#
# The harness binds the socket only AFTER the adapter finishes loading, so
# socket-accepts-connection IS the readiness signal; there is no separate handshake. This wait is
# what turns that signal into something a client can act on.
#
# Default 900s covers the slowest observed cold load (4x UNet ~70s, GroundingDINO ~47s) with wide
# margin: waiting too long costs a stalled job, waiting too short costs a failed run.
DEFAULT_READY_TIMEOUT_S = 900.0

# Poll interval while waiting. Short enough not to pad a fast load, long enough that 96 waiting
# clients do not busy-spin a node.
_READY_POLL_S = 0.5


def _connect_when_ready(
    socket_path: Path,
    *,
    ready_timeout: float,
    timeout: float | None,
) -> socket.socket:
    """Connect to ``socket_path``, waiting up to ``ready_timeout`` for the service to come up.

    Retries ONLY the errors that mean "not listening yet" -- FileNotFoundError (socket file not
    created) and ConnectionRefusedError (file exists, nothing accepting). Any other OSError is a
    real fault and propagates immediately instead of being masked by a long retry loop.
    """
    deadline = time.monotonic() + ready_timeout
    attempts = 0
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if timeout is not None:
            sock.settimeout(timeout)
        try:
            sock.connect(str(socket_path))
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            # ECONNREFUSED means "nothing is accepting here", which covers BOTH a service still
            # loading (retry) and a path that is not a socket at all (fatal). The errno cannot
            # tell them apart -- connecting to a directory or a regular file also raises
            # ConnectionRefusedError. So inspect the path: without this, a typo'd socket path
            # hangs for the full ready_timeout instead of failing immediately.
            if socket_path.exists() and not socket_path.is_socket():
                kind = "directory" if socket_path.is_dir() else "regular file"
                raise NotADirectoryError(
                    f"{socket_path} exists but is a {kind}, not a socket. The service rule and "
                    f"the client rule must name the SAME socket path."
                ) from None
            attempts += 1
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"service socket {socket_path} did not accept connections within "
                    f"{ready_timeout:.0f}s ({attempts} attempts). The service job either failed "
                    f"to start or is still loading its model -- check the service job's stderr "
                    f"for a 'listening on' line."
                ) from None
            time.sleep(_READY_POLL_S)
        except OSError:
            sock.close()
            raise
        else:
            if attempts:
                waited = attempts * _READY_POLL_S
                print(
                    f"client: service ready after ~{waited:.1f}s ({attempts} retries)",
                    file=sys.stderr,
                )
            return sock


def call_server(
    socket_path: Path,
    payload: dict,
    *,
    timeout: float | None = None,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
) -> Response:
    """Connect to socket_path, send payload, block for the response. Raises on transport error.

    Waits for the service to become ready before sending (see ``_connect_when_ready``). That wait
    lives HERE, in the one function every served client calls, so no rule or caller can forget it.
    """
    request = Request(request_id=str(uuid.uuid4()), payload=payload)
    with _connect_when_ready(
        socket_path, ready_timeout=ready_timeout, timeout=timeout
    ) as sock:
        send_frame(sock, request.to_json())
        frame = recv_frame(sock)
    return Response.from_json(frame)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send one request to a resident model server.")
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--payload-json", required=True, help="JSON object, e.g. '{\"well_id\": \"...\"}'")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_S,
        help="seconds to wait for the service to finish loading and start accepting "
             "(Snakemake launches clients at the same instant as the service)",
    )
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.payload_json)
    except json.JSONDecodeError as e:
        print(f"client: invalid --payload-json: {e}", file=sys.stderr)
        return 2

    try:
        response = call_server(
            args.socket_path, payload,
            timeout=args.timeout, ready_timeout=args.ready_timeout,
        )
    except TimeoutError as e:
        # Distinct from a transport fault: the service never came up. Named separately so the
        # failure reads as "server not ready" rather than a generic socket error.
        print(f"client: service never became ready: {e}", file=sys.stderr)
        return 3
    except (OSError, ProtocolError) as e:
        print(f"client: transport error: {e}", file=sys.stderr)
        return 2

    if not response.ok:
        print(f"client: server reported failure: {response.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
