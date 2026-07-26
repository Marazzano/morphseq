"""Generic resident model-server harness.

This module owns everything that is NOT model-specific: socket bind, accept loop,
request decode, dispatch to the adapter, reply encode, signal handling, structured
logging. It must NEVER import torch and must know nothing about tensors, CUDA,
wells, or biology — see `adapter_base.py` for where model-specific code belongs.

Readiness contract: the server creates the Unix domain socket file ONLY AFTER
`adapter.load()` returns successfully. A client that manages to connect therefore
implies the model is loaded and ready — there is no separate "ready" handshake.

Isolation contract: each accepted connection is handled in its own thread, and each
request's `adapter.handle()` call is wrapped in try/except. An exception in one
request is converted to an error Response and logged; it never propagates out of
the request loop and never kills the process or corrupts state for the next well.

This is a PROTOTYPE. It is not wired into the Snakemake DAG.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from data_pipeline.model_servers.adapter_base import get_adapter_class, registered_adapter_names
from data_pipeline.model_servers.protocol import ProtocolError, Request, Response, recv_frame, send_frame

logger = logging.getLogger("model_servers.harness")


class ModelServer:
    """Binds a Unix domain socket, loads an adapter, and serves requests until stopped."""

    def __init__(self, socket_path: Path, adapter) -> None:
        self.socket_path = Path(socket_path)
        self.adapter = adapter
        self._server_sock: socket.socket | None = None
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------------

    def serve_forever(self) -> None:
        logger.info("loading adapter %r ...", type(self.adapter).__name__)
        t0 = time.monotonic()
        self.adapter.load()
        logger.info("adapter loaded in %.2fs", time.monotonic() - t0)

        self._bind()
        self._install_signal_handlers()  # no-op with a warning if not on the main thread
        logger.info("listening on %s (pid=%d)", self.socket_path, os.getpid())

        try:
            self._accept_loop()
        finally:
            self._cleanup()

    # AF_UNIX caps sockaddr_un.sun_path at 108 bytes on Linux (104 on macOS/BSD), including the
    # NUL terminator -- so 107 usable bytes. We check the Linux bound rather than the portable
    # one: this pipeline runs on a Linux cluster, and being stricter than the kernel would reject
    # paths that actually work (pytest's own tmp_path lands at ~107 bytes here).
    MAX_SOCKET_PATH_BYTES = 107

    def _bind(self) -> None:
        # Fail with an explanation rather than a bare OSError from bind(). A too-long path is a
        # runtime-only trap: any DAG or dry-run that references it builds perfectly happily, and
        # the failure only appears when the server actually starts. Deep data-tree paths blow the
        # limit easily -- {DATA_ROOT}/object_extraction/... was 157 bytes on the shared tree.
        # Sockets are transient IPC endpoints, not artifacts: put them somewhere short (/tmp).
        encoded = str(self.socket_path).encode()
        if len(encoded) > self.MAX_SOCKET_PATH_BYTES:
            raise ValueError(
                f"socket path is {len(encoded)} bytes, over the AF_UNIX limit of "
                f"{self.MAX_SOCKET_PATH_BYTES}: {self.socket_path}\n"
                "Use a short path (e.g. under /tmp) -- the socket is a transient IPC endpoint, "
                "not a data artifact, so it does not belong under the data root."
            )

        # Remove a stale socket file from a previous crashed run, if present.
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        sock.listen(socket.SOMAXCONN)
        sock.settimeout(0.5)  # so the accept loop can notice _stop_event
        self._server_sock = sock
        # Readiness signal: the socket file now exists AND the adapter is loaded
        # (load() already returned above). A client connect implies ready.

    def _install_signal_handlers(self) -> None:
        def _handle_sigterm(signum, frame):  # noqa: ARG001
            logger.info("received signal %s, shutting down", signum)
            self._stop_event.set()

        # signal.signal only works when called from the main thread of the main
        # interpreter. Real deployment runs the server as the process's main
        # thread (see __main__ / harness.main below), so this is always safe
        # there. Tests that run ModelServer inside a background thread (to keep
        # the test process itself in control) skip signal wiring instead of
        # crashing; those tests use server._stop_event.set() directly for
        # shutdown, which is the same mechanism the signal handler triggers.
        if threading.current_thread() is not threading.main_thread():
            logger.debug("not running on main thread; skipping SIGTERM/SIGINT handler installation")
            return
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        while not self._stop_event.is_set():
            try:
                conn, _addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                raise
            t = threading.Thread(target=self._handle_connection, args=(conn,), daemon=True)
            with self._threads_lock:
                self._threads.append(t)
            t.start()

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            with conn:
                self._serve_one_request(conn)
        except ProtocolError as e:
            logger.warning("protocol error on connection: %s", e)
        except Exception:
            logger.exception("unexpected error handling connection")

    def _serve_one_request(self, conn: socket.socket) -> None:
        frame = recv_frame(conn)
        request = Request.from_json(frame)
        logger.info("request %s: dispatching", request.request_id)
        t0 = time.monotonic()
        try:
            self.adapter.handle(request.payload)
        except Exception as e:  # noqa: BLE001 - isolation contract: never let a bad request kill the server
            logger.exception("request %s: adapter.handle raised", request.request_id)
            response = Response(request_id=request.request_id, ok=False, error=f"{type(e).__name__}: {e}")
        else:
            response = Response(request_id=request.request_id, ok=True, error=None)
        dt = time.monotonic() - t0
        logger.info("request %s: %s (%.2fs)", request.request_id, "ok" if response.ok else "FAILED", dt)
        send_frame(conn, response.to_json())

    def _cleanup(self) -> None:
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        # Best-effort join of in-flight handler threads; they're daemon threads so
        # process exit isn't blocked on them regardless.
        with self._threads_lock:
            threads = list(self._threads)
        for t in threads:
            t.join(timeout=5.0)
        logger.info("shut down cleanly")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a resident model server (prototype; not wired into Snakemake).",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help=f"registered adapter name. Available: {', '.join(registered_adapter_names()) or '(import adapters first)'}",
    )
    parser.add_argument("--socket-path", required=True, type=Path, help="Unix domain socket path to bind")
    parser.add_argument(
        "--adapter-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="adapter constructor kwarg, repeatable, e.g. --adapter-arg device=cuda",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _parse_adapter_kwargs(pairs: list[str]) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--adapter-arg must be KEY=VALUE, got: {pair!r}")
        key, value = pair.split("=", 1)
        kwargs[key] = value
    return kwargs


def main(argv: list[str] | None = None) -> None:
    # Importing adapters registers them into the adapter_base registry as a side effect.
    import data_pipeline.model_servers.adapters  # noqa: F401

    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    adapter_cls = get_adapter_class(args.adapter)
    adapter_kwargs = _parse_adapter_kwargs(args.adapter_arg)
    adapter = adapter_cls(**adapter_kwargs)

    server = ModelServer(socket_path=args.socket_path, adapter=adapter)
    server.serve_forever()


if __name__ == "__main__":
    main()
