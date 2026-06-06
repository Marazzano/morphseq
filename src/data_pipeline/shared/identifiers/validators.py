"""Canonical identifier validators.

Validators are the guard rails: they FAIL LOUDLY when an identifier does not
match the contract, so malformed/stale data cannot silently flow downstream.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.
"""

from __future__ import annotations


def validate_well_id(well_id: str) -> str:
    """Assert ``well_id`` is a well-formed GLOBAL well id; return it unchanged.

    Scope-2 only. Its whole purpose is to reject a bare local ``A01`` once
    ``well_id`` semantics flip to global ``{experiment_id}_{well}`` — i.e. to stop
    stale local-``well_id`` data from silently flowing. Under the CURRENT (Scope 1)
    convention ``well_id`` IS the local ``A01``, so enforcing the global form now
    would reject every legitimately-current value. Activated in Scope 2. See
    target/well_id_throughline_refactor_plan.md (Scopes 1-2).
    """
    raise NotImplementedError(
        "validate_well_id enforces GLOBAL well_id ({experiment_id}_{well}); "
        "activated in Scope 2. Today's well_id is the local label 'A01'."
    )
