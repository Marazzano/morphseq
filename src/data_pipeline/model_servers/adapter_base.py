"""Adapter interface + name->class registry.

This module defines the ONLY seam where model-specific code is allowed to exist.
It must stay import-light itself (no torch here), but concrete adapter subclasses
(e.g. `adapters/sam2.py`) are expected to import torch/model libraries freely —
that is the whole point of isolating them behind this interface.

Interface, deliberately narrow:

    class ModelAdapter:
        def load(self) -> None: ...
        def handle(self, payload: dict) -> None: ...

- `load()` is called exactly once, at server startup, before the socket file is
  created. It should load weights and do whatever one-time setup the model needs.
  Raise on failure — the harness will refuse to start serving.
- `handle(payload)` is called once per client request. `payload` is the adapter's
  own request shape (paths in, paths out) — the harness does not interpret it.
  Return value is ignored; signal failure by raising. The harness catches any
  exception from `handle()` and turns it into a Response(ok=False, error=str(e))
  WITHOUT killing the process, so one bad well cannot corrupt the model for
  subsequent wells (per-request isolation requirement).
- Adapters are responsible for their own output atomicity (temp file + rename)
  and for their own per-request state isolation (e.g. SAM2 must build a brand
  new inference_state per request and must never reuse a previous well's state).

RESIST ADDING MORE METHODS HERE. If an adapter needs a new hook, that's a signal
the seam is in the wrong place for that model family — prefer solving it inside
that adapter's `handle()` rather than growing this interface.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    """Structural interface every adapter must satisfy. See module docstring."""

    def load(self) -> None:
        """Load model weights / one-time setup. Called once, before the socket opens."""
        ...

    def handle(self, payload: dict[str, Any]) -> None:
        """Do the per-request work. Raise on failure. Return value is ignored."""
        ...


_ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(name: str):
    """Class decorator: register an adapter class under `name` for CLI selection.

    Usage:
        @register_adapter("sam2")
        class Sam2Adapter:
            def load(self) -> None: ...
            def handle(self, payload: dict) -> None: ...
    """

    def _decorator(cls: type) -> type:
        if name in _ADAPTER_REGISTRY and _ADAPTER_REGISTRY[name] is not cls:
            raise ValueError(f"adapter name {name!r} already registered to {_ADAPTER_REGISTRY[name]!r}")
        _ADAPTER_REGISTRY[name] = cls
        return cls

    return _decorator


def get_adapter_class(name: str) -> type:
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY)) or "(none registered)"
        raise KeyError(f"no adapter registered under {name!r}. Available: {available}")
    return _ADAPTER_REGISTRY[name]


def registered_adapter_names() -> list[str]:
    return sorted(_ADAPTER_REGISTRY)
