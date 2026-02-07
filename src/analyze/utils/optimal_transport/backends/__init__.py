"""UOT solver backends."""

from analyze.utils.optimal_transport.backends.base import UOTBackend, BackendResult
from analyze.utils.optimal_transport.backends.pot_backend import POTBackend

try:
    from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend
except ImportError:
    OTTBackend = None

__all__ = ["UOTBackend", "BackendResult", "POTBackend", "OTTBackend"]
