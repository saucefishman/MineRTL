"""<out>.server.json: what MCHPRS needs at run time beyond the blocks (print probes, the server-driven clock).

Positions are relative to the schematic's minimum corner, across all its regions.
"""
from __future__ import annotations

import json
from pathlib import Path

from litemapy import Region

SERVER_FILE_VERSION = 2
SERVER_FILE_SUFFIX = ".server.json"


def schematic_bounds(regions: list[Region]) -> tuple[list[int], list[int]]:
    """The schematic's minimum corner and size."""
    low = [min(r.x + min(r.range_x()) for r in regions), min(r.y + min(r.range_y()) for r in regions),
           min(r.z + min(r.range_z()) for r in regions)]
    high = [max(r.x + max(r.range_x()) for r in regions), max(r.y + max(r.range_y()) for r in regions),
            max(r.z + max(r.range_z()) for r in regions)]
    return low, [hi - lo + 1 for lo, hi in zip(low, high)]


def write_server_file(path: Path, size: list[int], probes: list[dict], clock: dict | None,
                      levers: list[dict] = ()) -> None:
    """`clock` is {"pin", "pos" (a lever), "half_period_ticks" (redstone ticks)}; no key means no server clock.
    `levers` are [{"name", "pos"}], each input pin's lever, for a command that flips one by name."""
    data = {"version": SERVER_FILE_VERSION, "schematic_size": size, "probes": probes}
    if clock:
        data["clock"] = clock
    if levers:
        data["levers"] = list(levers)
    path.write_text(json.dumps(data, indent=1) + "\n")
