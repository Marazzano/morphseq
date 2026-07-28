"""Concurrent requests must QUEUE at the model, not pile onto it.

The harness handles each connection in its own thread. Without a queue, N clients means N
concurrent `adapter.handle()` calls contending for one GPU -- which thrashes or OOMs, drops the
connections, and gives every client "connection closed while reading frame".

That is not hypothetical: job 22831848 logged 95 "dispatching" lines inside a single minute, zero
"ok", and every client failed exactly that way.

Snakemake cannot throttle this from the rule side. A `service()` group must be schedulable all at
once, so `threads:` on the client is forced to 0 (otherwise the group's summed `_cores` exceeds any
sane budget) and a custom `resources:` on the client deadlocks the group outright -- both verified.
The limit therefore has to live in the server, which is what these tests pin.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from data_pipeline.model_servers.client import call_server
from data_pipeline.model_servers.harness import ModelServer


class _ConcurrencyProbe:
    """Adapter that records the peak number of simultaneous handle() calls."""

    def __init__(self, work_seconds: float = 0.2) -> None:
        self.work_seconds = work_seconds
        self.peak = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def load(self) -> None:  # noqa: D102 - adapter contract
        pass

    def handle(self, payload: dict) -> None:  # noqa: D102 - adapter contract
        with self._lock:
            self._in_flight += 1
            self.peak = max(self.peak, self._in_flight)
        time.sleep(self.work_seconds)
        with self._lock:
            self._in_flight -= 1


@pytest.fixture
def running_server(tmp_path):
    """A started ModelServer plus its probe adapter; torn down after the test."""
    probe = _ConcurrencyProbe()
    socket_path = Path(tmp_path) / "q.sock"
    server = ModelServer(socket_path, probe)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    deadline = time.monotonic() + 10
    while not socket_path.exists():
        if time.monotonic() > deadline:
            pytest.fail("server never bound its socket")
        time.sleep(0.02)

    yield server, probe, socket_path
    server._stop_event.set()


def test_concurrent_clients_are_queued_one_at_a_time(running_server):
    """20 clients firing at once must reach the model ONE at a time, and none may be dropped."""
    _server, probe, socket_path = running_server

    errors: list[str] = []

    def fire(index: int) -> None:
        try:
            response = call_server(socket_path, {"i": index})
            if not response.ok:
                errors.append(f"request {index}: {response.error}")
        except Exception as e:  # noqa: BLE001 - the failure mode under test is a dropped connection
            errors.append(f"request {index}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"clients failed instead of queueing: {errors[:3]}"
    assert probe.peak == 1, (
        f"{probe.peak} requests were inside the model at once; one model on one GPU means the "
        f"dispatch lock must admit exactly one"
    )


def test_queueing_does_not_drop_or_reorder_work(running_server):
    """Every queued request still completes -- queueing delays work, it never discards it."""
    _server, probe, socket_path = running_server

    completed: list[int] = []
    lock = threading.Lock()

    def fire(index: int) -> None:
        response = call_server(socket_path, {"i": index})
        assert response.ok
        with lock:
            completed.append(index)

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert sorted(completed) == list(range(8))
