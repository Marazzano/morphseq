"""Tests for the generic harness/client/protocol/atomic-write parts.

Uses the trivial FakeAdapter (data_pipeline.model_servers.adapters.fake) — no GPU,
no torch, no real model needed. These tests exercise exactly the pieces the spec
calls out: harness dispatch, client round-trip, atomic write, and error
propagation from a failing request without killing the server.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from data_pipeline.model_servers.adapters.fake import FakeAdapter
from data_pipeline.model_servers.client import call_server
from data_pipeline.model_servers.harness import ModelServer


@pytest.fixture
def server_factory(tmp_path):
    """Start a ModelServer(FakeAdapter) in a background thread; yield (socket_path, adapter).

    Tears down via SIGTERM-equivalent (direct stop_event set + thread join) after
    the test, so tests don't leak background threads/sockets.
    """
    created = []

    def _make(load_delay_s: float = 0.0):
        socket_path = tmp_path / f"srv-{len(created)}.sock"
        adapter = FakeAdapter(load_delay_s=load_delay_s)
        server = ModelServer(socket_path=socket_path, adapter=adapter)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # Wait for readiness: socket file existing is the contract.
        deadline = time.monotonic() + 10.0
        while not socket_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("server did not create socket file in time")
            time.sleep(0.02)
        created.append((server, thread))
        return socket_path, adapter, server

    yield _make

    for server, thread in created:
        server._stop_event.set()
        thread.join(timeout=5.0)


def test_readiness_socket_created_only_after_load(tmp_path):
    """Socket file must not exist until adapter.load() has returned."""
    socket_path = tmp_path / "readiness.sock"
    adapter = FakeAdapter(load_delay_s=0.3)
    server = ModelServer(socket_path=socket_path, adapter=adapter)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Immediately after starting the thread, load() is still sleeping (0.3s),
        # so the socket file must not exist yet.
        time.sleep(0.05)
        assert not socket_path.exists(), "socket file appeared before load() finished"
        assert not adapter.loaded

        deadline = time.monotonic() + 5.0
        while not socket_path.exists():
            assert time.monotonic() < deadline, "socket file never appeared"
            time.sleep(0.02)
        assert adapter.loaded, "socket file exists but adapter.loaded is False"
    finally:
        server._stop_event.set()
        thread.join(timeout=5.0)


def test_client_round_trip_success(server_factory, tmp_path):
    socket_path, adapter, _server = server_factory()

    input_path = tmp_path / "in.txt"
    output_path = tmp_path / "out.txt"
    input_path.write_bytes(b"hello world")

    response = call_server(
        socket_path,
        {"input_path": str(input_path), "output_path": str(output_path)},
        timeout=10.0,
    )

    assert response.ok is True
    assert response.error is None
    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"HELLO WORLD")


def test_atomic_write_no_partial_file_visible(server_factory, tmp_path):
    """The output path must never be observable in a half-written state.

    We can't literally catch the write mid-flight from another thread deterministically,
    but we CAN assert: (a) no leftover .tmp-* file remains after a successful write,
    and (b) the final file's content is exactly the expected full payload (never
    truncated), which is what atomic rename guarantees vs. write-in-place.
    """
    socket_path, adapter, _server = server_factory()

    input_path = tmp_path / "in2.txt"
    output_path = tmp_path / "out2.txt"
    payload_bytes = b"x" * 100_000  # large enough that a naive write-in-place would be observable mid-write
    input_path.write_bytes(payload_bytes)

    response = call_server(
        socket_path,
        {"input_path": str(input_path), "output_path": str(output_path)},
        timeout=10.0,
    )

    assert response.ok is True
    leftover_tmp_files = list(tmp_path.glob(".out2.txt.tmp-*"))
    assert leftover_tmp_files == [], f"leftover temp file(s) after atomic write: {leftover_tmp_files}"
    content = output_path.read_bytes()
    assert content.startswith(b"X" * 100_000)


def test_error_propagation_does_not_corrupt_subsequent_requests(server_factory, tmp_path):
    """One malformed/failing request must return ok=False but leave the server usable."""
    socket_path, adapter, _server = server_factory()

    # 1. A request engineered to fail.
    bad_response = call_server(socket_path, {"fail": True}, timeout=10.0)
    assert bad_response.ok is False
    assert "forced failure" in (bad_response.error or "")

    # 2. A request missing required keys entirely (adapter.handle raises KeyError).
    malformed_response = call_server(socket_path, {}, timeout=10.0)
    assert malformed_response.ok is False

    # 3. The server must still work normally afterward.
    input_path = tmp_path / "in3.txt"
    output_path = tmp_path / "out3.txt"
    input_path.write_bytes(b"still alive")
    good_response = call_server(
        socket_path,
        {"input_path": str(input_path), "output_path": str(output_path)},
        timeout=10.0,
    )
    assert good_response.ok is True
    assert output_path.read_bytes().startswith(b"STILL ALIVE")


def test_per_request_isolation_state_does_not_leak_incorrectly(server_factory, tmp_path):
    """handled_count increments per request but each request's OWN output is self-consistent.

    This stands in for the SAM2 "no state bleed between wells" requirement: the
    fake adapter's counter is allowed to increment (that's expected/benign shared
    state, analogous to memory allocators warming up) but each request's output
    must reflect only ITS OWN input content, never another request's input.
    """
    socket_path, adapter, _server = server_factory()

    results = {}
    for i in range(5):
        input_path = tmp_path / f"in_multi_{i}.txt"
        output_path = tmp_path / f"out_multi_{i}.txt"
        marker = f"request-{i}-unique-payload".encode()
        input_path.write_bytes(marker)
        response = call_server(
            socket_path,
            {"input_path": str(input_path), "output_path": str(output_path)},
            timeout=10.0,
        )
        assert response.ok is True
        results[i] = output_path.read_bytes()

    for i in range(5):
        expected_marker = f"REQUEST-{i}-UNIQUE-PAYLOAD".encode()
        assert results[i].startswith(expected_marker), (
            f"request {i} output does not contain its own input content: {results[i][:60]!r}"
        )
        # And make sure it did NOT pick up another request's marker instead.
        for j in range(5):
            if j == i:
                continue
            other_marker = f"REQUEST-{j}-UNIQUE-PAYLOAD".encode()
            assert not results[i].startswith(other_marker)


def test_concurrent_requests_all_succeed_and_stay_isolated(server_factory, tmp_path):
    """Multiple clients hitting the server concurrently should each get correct, isolated results."""
    socket_path, adapter, _server = server_factory()

    n = 8
    inputs = []
    for i in range(n):
        input_path = tmp_path / f"cin_{i}.txt"
        output_path = tmp_path / f"cout_{i}.txt"
        input_path.write_bytes(f"concurrent-{i}".encode())
        inputs.append((input_path, output_path, i))

    results = {}
    lock = threading.Lock()

    def _worker(input_path, output_path, i):
        response = call_server(
            socket_path,
            {"input_path": str(input_path), "output_path": str(output_path)},
            timeout=15.0,
        )
        with lock:
            results[i] = (response, output_path)

    threads = [threading.Thread(target=_worker, args=args) for args in inputs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert len(results) == n
    for i, (response, output_path) in results.items():
        assert response.ok is True
        content = output_path.read_bytes()
        assert content.startswith(f"CONCURRENT-{i}".encode())


def test_client_reports_nonzero_exit_on_server_failure(server_factory, tmp_path):
    """The client CLI's main() should exit nonzero when the server reports failure."""
    from data_pipeline.model_servers.client import main as client_main

    socket_path, adapter, _server = server_factory()

    rc = client_main(["--socket-path", str(socket_path), "--payload-json", '{"fail": true}', "--timeout", "10"])
    assert rc != 0


def test_client_reports_zero_exit_on_success(server_factory, tmp_path):
    from data_pipeline.model_servers.client import main as client_main

    socket_path, adapter, _server = server_factory()
    input_path = tmp_path / "cli_in.txt"
    output_path = tmp_path / "cli_out.txt"
    input_path.write_bytes(b"cli test")

    rc = client_main(
        [
            "--socket-path", str(socket_path),
            "--payload-json", f'{{"input_path": "{input_path}", "output_path": "{output_path}"}}',
            "--timeout", "10",
        ]
    )
    assert rc == 0
    assert output_path.exists()


def test_adapter_registry_lookup():
    from data_pipeline.model_servers.adapter_base import get_adapter_class, registered_adapter_names

    assert "fake" in registered_adapter_names()
    assert get_adapter_class("fake") is FakeAdapter

    with pytest.raises(KeyError):
        get_adapter_class("does-not-exist")


def test_atomic_write_via_helper(tmp_path):
    from data_pipeline.model_servers.atomic_write import atomic_write_via

    final_path = tmp_path / "via.csv"

    def _writer(p: Path) -> None:
        p.write_text("a,b\n1,2\n")

    atomic_write_via(final_path, _writer)
    assert final_path.read_text() == "a,b\n1,2\n"
    leftover = list(tmp_path.glob(".via.csv.tmp-*"))
    assert leftover == []


def test_atomic_write_cleans_up_tmp_on_writer_failure(tmp_path):
    from data_pipeline.model_servers.atomic_write import atomic_write_via

    final_path = tmp_path / "fails.csv"

    def _writer(p: Path) -> None:
        p.write_text("partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_write_via(final_path, _writer)

    assert not final_path.exists(), "final path must not exist after a failed write"
    leftover = list(tmp_path.glob(".fails.csv.tmp-*"))
    assert leftover == [], "temp file must be cleaned up after writer failure"


def test_bind_rejects_overlong_socket_path(tmp_path):
    """A too-long AF_UNIX path must fail with an explanation, not a bare OSError.

    Regression test: the first wiring of frame_detections put the socket under DATA_ROOT, which
    on the shared tree produced a 157-byte path -- over the ~108-byte sockaddr_un.sun_path cap.
    The DAG built fine and dry-runs passed; the service only died at bind(), which makes this a
    trap that static checks cannot catch. The guard turns it into a legible startup error.
    """
    long_dir = tmp_path / ("d" * 120)
    server = ModelServer(socket_path=long_dir / "x.sock", adapter=FakeAdapter())

    with pytest.raises(ValueError, match="over the AF_UNIX limit"):
        server._bind()


def test_bind_accepts_short_socket_path(tmp_path):
    """The complement: a normal short path binds cleanly and creates the socket file."""
    server = ModelServer(socket_path=tmp_path / "ok.sock", adapter=FakeAdapter())
    try:
        server._bind()
        assert (tmp_path / "ok.sock").exists()
    finally:
        server._cleanup()
