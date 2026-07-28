"""The client must WAIT for its service to come up — not race it.

Snakemake starts a `service()` job and ALL of its consumers in the same instant
(`snakemake/executors.py`: `futures = [self.run_single_job(j) for j in job]`, then it waits for
the non-service jobs). That is correct for a PIPE group, whose members genuinely stream
concurrently. For a resident model server it means every client starts while the model is still
loading, so the socket does not exist yet.

Without the wait, each client dies instantly on ENOENT/ECONNREFUSED and the run stalls with the
service idling: jobs 22825862 / 22827486 / 22829477 all froze on the harness's "listening on ..."
line with zero clients finished and cpu=00:00:13 against 7 minutes of wallclock.

These tests pin the wait so it cannot be quietly removed. They use a bare AF_UNIX echo server
rather than a real adapter -- what is under test is the CONNECT retry, not any model.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from data_pipeline.model_servers.client import (
    DEFAULT_READY_TIMEOUT_S,
    _connect_when_ready,
    call_server,
)
from data_pipeline.model_servers.protocol import Response, recv_frame, send_frame


def _serve_one(socket_path: Path, *, delay: float, ready: threading.Event) -> None:
    """Bind after ``delay`` seconds, answer exactly one request, exit.

    The delay stands in for model load. Binding only AFTER it mirrors the harness, which creates
    the socket once the adapter is loaded -- that ordering is the readiness signal.
    """
    time.sleep(delay)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(8)
    ready.set()
    conn, _ = server.accept()
    with conn:
        request = recv_frame(conn)
        send_frame(conn, Response(request_id=request["request_id"], ok=True).to_json())
    server.close()


def test_client_waits_for_a_slow_service_instead_of_failing(tmp_path):
    """A client started BEFORE the service binds must wait, then succeed."""
    socket_path = tmp_path / "s.sock"
    ready = threading.Event()
    thread = threading.Thread(
        target=_serve_one, args=(socket_path,), kwargs={"delay": 1.5, "ready": ready}, daemon=True
    )
    thread.start()

    # Deliberately do NOT wait for `ready` -- calling while the socket is absent is the exact
    # race Snakemake creates, and is what this test exists to cover.
    assert not socket_path.exists()
    response = call_server(socket_path, {"hello": "world"}, ready_timeout=30.0)

    assert response.ok
    thread.join(timeout=5)


def test_missing_service_times_out_with_an_actionable_error(tmp_path):
    """When no service ever appears, fail with a diagnosis — not a bare socket error."""
    with pytest.raises(TimeoutError) as excinfo:
        _connect_when_ready(tmp_path / "never.sock", ready_timeout=1.0, timeout=None)

    message = str(excinfo.value)
    assert "did not accept connections" in message
    assert "listening on" in message, "error must point at the service log line to check"


def test_real_faults_are_not_swallowed_by_the_retry_loop(tmp_path):
    """Only 'not listening yet' retries. Other OSErrors must propagate immediately.

    A retry loop that catches everything would turn a genuine misconfiguration (a socket path that
    is a directory, a permissions failure) into a silent 15-minute hang.
    """
    a_directory = tmp_path / "not_a_socket"
    a_directory.mkdir()

    started = time.monotonic()
    with pytest.raises(OSError) as excinfo:
        _connect_when_ready(a_directory, ready_timeout=30.0, timeout=None)

    assert not isinstance(excinfo.value, TimeoutError), "should fail fast, not wait out the clock"
    assert time.monotonic() - started < 5.0, "a real fault must not be retried for 30s"


def test_ready_timeout_default_covers_observed_cold_loads():
    """The default must exceed real model-load times with margin.

    Measured cold loads: GroundingDINO ~47s, 4x UNet ~70s. A default anywhere near those would
    make the wait itself the flake.
    """
    assert DEFAULT_READY_TIMEOUT_S >= 300.0
