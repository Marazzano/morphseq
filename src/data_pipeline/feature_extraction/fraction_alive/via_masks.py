"""VIA (viability) mask lookup for fraction_alive.

VIA masks are auxiliary masks scoped per snip. Their canonical name is the snip's mask_id built
via ``build_mask_id(snip_id, local_mask_index)`` plus the ``_via`` family suffix, e.g.
``{snip_id}_m0000_via.png``. This module builds the ``snip_id -> path`` lookup the compute uses.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.shared.identifiers import build_mask_id

VIA_FAMILY_SUFFIX = "via"


def via_mask_filename(snip_id: str, *, local_mask_index: int = 0) -> str:
    """Return the canonical VIA mask filename for a snip: ``{snip_id}_m{NNNN}_via.png``."""
    return f"{build_mask_id(snip_id, local_mask_index)}_{VIA_FAMILY_SUFFIX}.png"


def build_via_mask_lookup(snip_ids, via_mask_dir: Path) -> dict[str, Path]:
    """Return ``{snip_id -> Path}`` for each snip's canonical VIA mask under ``via_mask_dir``."""
    via_mask_dir = Path(via_mask_dir)
    return {str(s): via_mask_dir / via_mask_filename(str(s)) for s in snip_ids}
