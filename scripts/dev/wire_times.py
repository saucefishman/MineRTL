"""Seconds in each part of routing, logged when the build exits (development).

Takes the same arguments as `python -m minecraft_v`:

    uv run python scripts/dev/wire_times.py --netlist build/artifacts/netlist.json --place-seed 2 ...
"""
from __future__ import annotations

import atexit
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minecraft_v.placement_engine import builder  # noqa: E402

times: dict[str, list] = {}


def timed(name, fn):
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            entry = times.setdefault(name, [0, 0.0])
            entry[0] += 1
            entry[1] += time.perf_counter() - t
    return wrapper


for name in ("_lay_redstone_path", "_place_repeaters_for_net", "_find_wire_path", "_signal_depths", "rip_up",
             "rollback"):
    setattr(builder, name, timed(name, getattr(builder, name)))
for cls, name in ((builder._GreedyRouter, "_check_net"), (builder._GreedyRouter, "_open_via"),
                  (builder._TreeCells, "nearest"), (builder._GreedyRouter, "route_job"),
                  (builder._GreedyRouter, "_cells_since"), (builder._GreedyRouter, "net_paths")):
    setattr(cls, name, timed(name, getattr(cls, name)))
atexit.register(lambda: builder.log("[wire times] " + ", ".join(f"{k} {v[1]:.1f}s/{v[0]}" for k, v in times.items())))

sys.argv[0] = "minecraft_v"
runpy.run_module("minecraft_v", run_name="__main__")
