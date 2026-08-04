"""Compatibility aliases for the pre-product-key flat snip layout.

THE AUTHORITY MOVED, THE OLD PATH DID NOT DISAPPEAR:

    authoritative   {snips_dir}/{physical_embryo_id}/{snip_product_key}/{snip_id}.png
    alias           {snips_dir}/{physical_embryo_id}/{snip_id}.png   -> relative symlink

The direction matters. Writing real bytes to the flat path and pointing the product directory back
at it would keep the OLD model as the authority and make the product hierarchy decorative. The
product path holds the bytes; the flat path is a view onto them.

EMBRYO-FIRST MAKES THE ALIAS LOCAL. Because the product directory is a CHILD of the embryo
directory, the link body is just ``{snip_product_key}/{snip_id}.png`` — a sibling reference, no ".."
segments climbing out of one subtree and back into another. Removal is correspondingly tidy: delete
the two flat aliases and the product children are already in place, untouched.

WHO NEEDS THIS. In-repo consumers do NOT construct snip paths -- they read ``processed_snip_path``
off the inventory and resolve it through one helper, so they follow the authority automatically. The
alias exists for readers OUTSIDE the inventory: notebooks, ad-hoc scripts, anything holding a path
string from before the migration. No such reader is known, which is exactly why the alias is cheap
insurance rather than a proven requirement.

KNOWN LIMIT: tools that do not preserve or follow symlinks -- some archive, copy, and object-store
upload workflows -- will either duplicate the bytes or drop the alias. A symlink is the right answer
for ordinary filesystem readers (including over NFS) and the wrong answer for those; if such a
workflow appears, it needs a deliberate export step rather than a link.

TODO(deprecate-legacy-snip-symlinks): this whole module, the ``legacy_flat_snip_path`` column, and
``select_default_snip_product`` retire together once consumers name their product explicitly. It is
deliberately ONE small module with one entrypoint so that removal is a deletion, not an excavation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class LegacySnipPathConflict(RuntimeError):
    """A legacy path already holds something that is not the alias we would create."""


def legacy_flat_snip_path(*, snips_dir: Path, physical_embryo_id: str, filename: str) -> Path:
    """Where this artifact USED to live, before product keys entered the path."""
    return Path(snips_dir) / physical_embryo_id / filename


def link_legacy_flat_path(
    *,
    canonical_path: Path,
    legacy_path: Path,
    log_manifest: list[dict[str, str]] | None = None,
    snip_product_key: str | None = None,
) -> bool:
    """Point ``legacy_path`` at ``canonical_path`` with a RELATIVE symlink.

    Returns True when a link was created, False when a correct one already existed (so a rerun is
    idempotent and does not churn the filesystem).

    RELATIVE, NOT ABSOLUTE: an experiment tree gets copied, moved, and mounted at different prefixes
    on this cluster. An absolute link survives none of that; a relative one stays internally
    consistent as long as the tree moves as a unit.

    NEVER CLOBBERS. Three cases, and the third is why this function exists rather than a bare
    ``symlink_to``:

      absent                        -> create the link
      symlink to the same target    -> accept, no-op (idempotent rerun)
      real file, or a link pointing
      somewhere else                -> RAISE

    A real file at the legacy path means a PRE-MIGRATION run wrote actual pixels there. Overwriting
    it would destroy the only copy of that data. A link pointing elsewhere means two products are
    fighting over one alias, which is a configuration error rather than something to silently
    resolve by last-write-wins.
    """
    canonical_path = Path(canonical_path)
    legacy_path = Path(legacy_path)

    # os.path.relpath, not Path.relative_to: the target is a SIBLING subtree, not a descendant, so
    # the link body needs ".." segments that relative_to refuses to produce.
    relative_target = os.path.relpath(canonical_path, legacy_path.parent)

    if legacy_path.is_symlink():
        existing = os.readlink(legacy_path)
        if existing == relative_target:
            return False
        raise LegacySnipPathConflict(
            f"legacy snip path {legacy_path} is a symlink to {existing!r}, but this run would point "
            f"it at {relative_target!r}. Two products are claiming one compatibility alias; resolve "
            "which product owns the legacy path rather than letting the last writer win."
        )

    if legacy_path.exists():
        raise LegacySnipPathConflict(
            f"legacy snip path {legacy_path} is a REAL FILE, not a compatibility alias. A "
            "pre-migration run wrote pixels there and replacing it would destroy the only copy. "
            f"Move or delete that tree deliberately, then rerun; the canonical bytes are at "
            f"{canonical_path}."
        )

    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.symlink_to(relative_target)

    # A migration log turns "an external reader broke" from archaeology into a lookup.
    entry = {
        "legacy_path": str(legacy_path),
        "canonical_path": str(canonical_path),
        "relative_target": relative_target,
        "snip_product_key": str(snip_product_key or ""),
    }
    if log_manifest is not None:
        log_manifest.append(entry)
    logger.debug("legacy snip alias created: %s -> %s", legacy_path, relative_target)
    return True
