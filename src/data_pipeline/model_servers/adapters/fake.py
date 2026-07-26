"""Trivial fake adapter used to test the generic harness/client without a GPU.

Not a real model. `payload` contract:
    {"input_path": str, "output_path": str}
    or {"fail": true} to force handle() to raise (for testing error propagation).

`load()` just flips a flag and sleeps briefly to simulate model-load cost, so tests
can assert load happens once and before the socket is ready.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from data_pipeline.model_servers.adapter_base import register_adapter
from data_pipeline.model_servers.atomic_write import atomic_write_bytes


@register_adapter("fake")
class FakeAdapter:
    def __init__(self, load_delay_s: str | float = 0.0) -> None:
        self.load_delay_s = float(load_delay_s)
        self.loaded = False
        self.handled_count = 0

    def load(self) -> None:
        time.sleep(self.load_delay_s)
        self.loaded = True

    def handle(self, payload: dict[str, Any]) -> None:
        if not self.loaded:
            raise RuntimeError("handle() called before load()")
        if payload.get("fail"):
            raise ValueError("forced failure for testing")

        input_path = Path(payload["input_path"])
        output_path = Path(payload["output_path"])
        content = input_path.read_bytes()
        # Trivial "processing": uppercase the bytes and tag with the request count,
        # so tests can verify per-request isolation (state does not leak across calls).
        self.handled_count += 1
        result = content.upper() + f"\n# handled_count={self.handled_count}\n".encode()
        atomic_write_bytes(output_path, result)
