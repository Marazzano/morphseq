"""Backend interface for UOT solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
from abc import ABC, abstractmethod

from ..config import Coupling, UOTSupport, UOTConfig


@dataclass
class BackendResult:
    coupling: Optional[Coupling]
    cost: float
    diagnostics: Dict


class UOTBackend(ABC):
    """Abstract base class for UOT solver backends."""

    @abstractmethod
    def solve(self, src: UOTSupport, tgt: UOTSupport, config: UOTConfig) -> BackendResult:
        """Solve UOT between src and tgt supports."""
        raise NotImplementedError
