"""Add a generated clock to a built schematic without rebuilding it.

Usage: add_clock.py <in.litematic> <component_layout.json> <out.litematic> <clock pin> <critical path ticks>
"""
import json
import sys

from litemapy import Schematic

from minecraft_v.placement_engine.builder import _construct_clock_region


def add_clock(in_path: str, layout_path: str, out_path: str, pin: str, critical_path_ticks: int) -> None:
    schematic = Schematic.load(in_path)
    main = schematic.regions["main"]
    origin = next(c["origin"] for c in json.load(open(layout_path)) if c["id"] == pin)
    target = (origin[0], origin[1], origin[2] + 2)
    clock = _construct_clock_region(target, critical_path_ticks)
    # The same as the builder: the first clock row is inside main, and loaders paste regions in any order.
    for x in range(clock.width):
        for y in range(clock.height):
            wx, wy = clock.x + x, clock.y + y
            if wx < main.width and target[2] < main.length and clock[x, y, 0].id != "minecraft:air":
                main[wx, wy, target[2]] = clock[x, y, 0]
    regions = {**schematic.regions, "clock": clock}
    Schematic(name=schematic.name, author=schematic.author, regions=regions).save(out_path)


if __name__ == "__main__":
    add_clock(*sys.argv[1:5], int(sys.argv[5]))
