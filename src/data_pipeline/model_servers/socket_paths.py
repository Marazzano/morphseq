"""Socket paths for resident model services — the ONE place a service/client pair agrees.

A served step is TWO rules that must name the byte-identical socket path: a `service(...)` rule
that binds it and a per-well client rule that connects to it. Snakemake links them by that string,
so any disagreement is a `MissingInputException` naming a socket nothing creates.

That is not hypothetical. Both served steps shipped with this bug (fixed 2026-07-26): the socket
filename was a SHA1 of the experiment, and the two rules hashed different things --

    service:  service(str(_socket("{experiment}")))   -> sha1("{experiment}")  -> 678971a1....sock
    client:   str(_socket(wc.experiment))             -> sha1("20260408_pbx")  -> 15ab2858....sock

A hash DESTROYS its input, so both sides had to independently compute the same value, and the
service could not: `output:` cannot be a function (Snakemake: "Only input files can be specified
as functions"), so it is fixed at parse time and never sees the resolved experiment. The client,
using `lambda wc:`, does.

The fix is to stop hashing and let the experiment ride through as a WILDCARD -- a path segment
Snakemake substitutes for both rules at once. Same mechanism as `{well_id}` everywhere else.

Length still matters: AF_UNIX caps sun_path at 108 bytes (harness enforces 107), and a socket
under DATA_ROOT is ~157 bytes on the shared tree. Short parent + short leaf keeps a typical
experiment id near 70 bytes. The socket is a transient IPC endpoint, not a data artifact -- server
and client always share a node, since AF_UNIX cannot cross one -- so /tmp is correct, not a hack.

WIRING A SERVED STEP -- the three traps, all of which fail SILENTLY:

1. THE CLIENT MUST NOT DECLARE `resources: gpu=1`. The client holds no GPU memory; it sends paths
   over a socket and blocks. The SERVICE holds the GPU. If both declare gpu=1 the service takes
   the only unit, no client is ever schedulable, and Snakemake blocks forever WITHOUT an error --
   it correctly concludes nothing is runnable.

2. SERVICES NEED >= 2 CORES. A service job occupies a core for the whole run, so `--cores 1` dies
   with "Excess Resources: _cores: 2/1" before the gpu resource is even consulted.

3. GATE THE TWO RULES BY DEFINITION, NOT `ruleorder`. ruleorder only breaks a tie Snakemake
   considers AMBIGUOUS, and a served/in-process pair is not ambiguous: the served rule declares an
   extra input (the socket), so Snakemake treats them as different jobs and picks the one it can
   satisfy WITHOUT starting a service. SGE job 22798948 ran the in-process rule with the toggle
   ON, silently, for exactly this reason. Wrap each rule in `if <STEP>_SERVED:` / `else:` so only
   one is ever DEFINED and there is no choice left to get wrong. (Cost: `snakemake --list` shows
   only the active variant.)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# Mirrors ModelServer.MAX_SOCKET_PATH_BYTES. Checked here so an over-long path fails at DAG build
# instead of at bind() -- the harness guard is the last line of defence, not the first.
MAX_SOCKET_PATH_BYTES = 107

# Kept short deliberately; every byte here is a byte an experiment id cannot use.
_LEAF = "s.sock"


def service_socket_pattern(service_name: str, *, wildcard: str = "experiment") -> str:
    """Return the socket path pattern for `service_name`, with `{wildcard}` left LIVE.

    Use this for BOTH the service rule's `output:` and the client rule's `input:` — passing the
    same pattern to both is what guarantees they agree. Do not resolve it yourself in either rule;
    Snakemake substitutes the wildcard for both at once.

    Args:
        service_name: short service id (e.g. "gdino", "unetaux"). Kept in the DIRECTORY name so
            two services for one experiment cannot collide.
        wildcard: the Snakemake wildcard to leave unresolved. Defaults to "experiment"; services
            are per-experiment because concurrent runs on different experiments must not share a
            resident model.

    Raises:
        ValueError: if the pattern is already over the AF_UNIX limit before the wildcard is even
            substituted -- a guaranteed runtime failure, worth catching at parse time.
    """
    pattern = str(Path(tempfile.gettempdir()) / f"{service_name}_{{{wildcard}}}" / _LEAF)

    # The wildcard token is a placeholder; the real path is longer by (len(id) - len(token)).
    # Check the fixed part so a too-long PREFIX is caught now rather than at bind().
    fixed_bytes = len(pattern.encode()) - len(f"{{{wildcard}}}")
    if fixed_bytes >= MAX_SOCKET_PATH_BYTES:
        raise ValueError(
            f"socket pattern for service {service_name!r} is {fixed_bytes} bytes before the "
            f"{wildcard} id is substituted, at/over the AF_UNIX limit of {MAX_SOCKET_PATH_BYTES}. "
            f"Shorten the service name or the temp dir: {pattern}"
        )
    return pattern
