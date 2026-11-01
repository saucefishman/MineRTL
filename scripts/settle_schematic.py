"""Rewrite a built schematic so it pastes already settled, with nothing left to transition.

Otherwise blocks paste in states that don't match their inputs, and the pulses while they settle
can clock flip-flops. Flip-flops go to their clock-low state holding their initial value, their
clock inputs are cut while the rest settles in MCHPRS's redstone code, and the result is saved.

Usage: settle_schematic.py <in.litematic> <component_layout.json> <out.litematic>
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from minecraft_v.progress import ProgressBar, log, phase, status

with phase("[settle] starting"):
    from litemapy import Schematic

def headless_binary() -> str:
    """MCHPRS's headless runner: $MCHPRS_HEADLESS, else under $MCHPRS_DIR (default ~/Projects/MCHPRS)."""
    if "MCHPRS_HEADLESS" in os.environ:
        return os.environ["MCHPRS_HEADLESS"]
    root = Path(os.environ.get("MCHPRS_DIR", Path.home() / "Projects/MCHPRS"))
    return str(root / "target/release/mchprs_headless")


# Template x, z -> properties in the clock-low state, for the blocks whose saved state differs.
DFF_CLOCK_LOW = {
    (3, 1): {"powered": "false"},                     # clock input
    (1, 1): {"powered": "false"},                     # master's lock
    (0, 1): {"locked": "false"},                      # master, follows D
    (2, 0): {"lit": "true"},
    (1, 0): {"powered": "true"},                      # output's lock
    (0, 0): {"locked": "true"},                       # output, holds Q
}


def component_blocks(comp: dict) -> tuple[tuple[int, int, int], callable]:
    """The clock input cell, and template (x, z) -> world position."""
    pins = {p["name"]: p["offset"] for p in comp["pins"]}
    ox, oy, oz = comp["origin"]
    width, depth = comp["footprint"]["width"], comp["footprint"]["depth"]
    mirrored = pins["C"][0] < pins["Q"][0]

    def world(x: int, z: int) -> tuple[int, int, int]:
        return (ox + (width - 1 - x if mirrored else x), oy, oz + z)

    return world(pins["C"][0] if not mirrored else width - 1 - pins["C"][0], depth - 1 - pins["C"][2]), world


def settle(in_path: str, layout_path: str, out_path: str) -> None:
    with phase("[settle] loading schematic"):
        schematic = Schematic.load(in_path)
    region = schematic.regions["main"]
    cuts = []
    for comp in json.load(open(layout_path)):
        if comp["type"] not in ("DFF", "DFFE"):
            continue
        clock, world = component_blocks(comp)
        cuts.append(clock)
        if comp["type"] == "DFF":
            for (x, z), props in DFF_CLOCK_LOW.items():
                region[world(x, z)] = region[world(x, z)].with_properties(**props)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with phase("[settle] simulating"):
            status("saving input")
            # Without any generated clock region: it would never settle, and the clock counts as low.
            Schematic(name="settle", regions={"main": region}).save(str(tmp / "in.litematic"))
            (tmp / "cuts.txt").write_text("".join(f"{x} {y} {z}\n" for x, y, z in cuts))
            status("running mchprs_headless")
            result = subprocess.run([headless_binary(), "settle", tmp / "in.litematic", tmp / "cuts.txt", tmp / "states.txt"],
                                    check=True, capture_output=True, text=True)
        log(result.stdout.strip())
        cut = set(cuts)
        lines = (tmp / "states.txt").read_text().splitlines()
        with ProgressBar("[settle] applying states", len(lines)) as bar:
            for line in lines:
                bar.update()
                x, y, z, kind, *values = line.split()
                pos = (int(x), int(y), int(z))
                if pos in cut:  # settled with its input cut, so back in its clock-low state
                    region[pos] = region[pos].with_properties(powered="false", locked="false")
                    continue
                flag = lambda v: "true" if v == "1" else "false"
                if kind == "powered":
                    props = {"powered": flag(values[0]), "locked": flag(values[1])}
                elif kind == "power":
                    props = {"power": values[0]}
                else:
                    props = {"lit": flag(values[0])}
                region[pos] = region[pos].with_properties(**props)
    with phase("[settle] saving litematic"):
        schematic.save(out_path)


if __name__ == "__main__":
    settle(*sys.argv[1:4])
