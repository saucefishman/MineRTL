"""Cosmetic post-processing of a finished build: dust shapes and I/O label signs.

Nothing here changes the circuit. Litematica pastes block states as stored, so dust
keeps whatever shape it was written with (a cross) until a block update reshapes it.
"""
from __future__ import annotations

from collections.abc import Collection
from json import dumps

import numpy as np
from litemapy import BlockState, Region
from litemapy.minecraft import TileEntity
from nbtlib.tag import Byte, Compound, List, String

from .block_utils import Box, cells_around_boxes, palette_writer
from .verify import _World

Pos = tuple[int, int, int]

_DIR_NAME = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}
# Standing sign `rotation` whose front faces each horizontal direction.
_SIGN_ROTATION = {(0, 1): "0", (-1, 0): "4", (0, -1): "8", (1, 0): "12"}


def apply_dust_shapes(region: Region, opaque: Collection[Box] = ()) -> int:
    """Set each dust's north/south/east/west to its in-game shape. Returns dust count.

    opaque: boxes already shaped alone. A dust's shape depends only on cells one step away, so inside one only dust
    touching something outside it can change."""
    palette = region._Region__palette
    blocks = region._Region__blocks
    wire_ids = [i for i, b in enumerate(palette) if b.id == "minecraft:redstone_wire"]
    if not wire_ids:
        return 0
    is_wire = np.isin(blocks, wire_ids)
    for lo, hi in opaque:
        is_wire[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False
    cells = list(map(tuple, np.argwhere(is_wire).tolist()))
    del is_wire
    wires = set(wire_ids)
    near = {(x + dx, y + dy, z + dz) for x, y, z in cells_around_boxes(region, opaque, 1)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)}
    cells += sorted(p for p in near if any(all(a <= c < b for a, c, b in zip(lo, p, hi)) for lo, hi in opaque)
                    and int(blocks[p]) in wires)
    world = _World(region, frozenset(), frozenset())
    shapes: list[tuple[Pos, tuple[str, ...]]] = []
    for p in cells:
        x, y, z = p
        ups = {(lx - x, lz - z) for lx, ly, lz in world.dust_links(p) if ly == y + 1}
        dirs = world.dust_directions(p)
        shapes.append((p, tuple(("up" if d in ups else "side") if d in dirs else "none" for d in _DIR_NAME)))
    # Write after computing all shapes; the world's block cache must not see half-updated dust.
    write = palette_writer(region)
    names = tuple(_DIR_NAME.values())
    shaped: dict[tuple[int, tuple[str, ...]], BlockState] = {}  # the few distinct results, instead of building each
    for p, sides in shapes:
        key = int(blocks[p]), sides
        block = shaped.get(key)
        if block is None:
            block = shaped[key] = palette[key[0]].with_properties(**dict(zip(names, sides)))
        write(p, block)
    return len(shapes)


def place_io_signs(region: Region, signs: list[tuple[Pos, tuple[int, int], str]]) -> None:
    """Oak sign with `label` on the cell above each I/O repeater, front facing `outward`.

    A standing sign can't really sit on a repeater; fine for a pasted, labelled build.
    """
    for (x, y, z), outward, label in signs:
        pos = (x, y + 1, z)
        if not (0 <= pos[1] < region.height) or region[pos].id != "minecraft:air":
            continue
        region[pos] = BlockState("minecraft:oak_sign", rotation=_SIGN_ROTATION[outward])
        lines = [label[i:i + 15] for i in range(0, len(label), 15)][:4]
        lines += [""] * (4 - len(lines))
        side = Compound({
            "messages": List[String]([String(dumps({"text": line})) for line in lines]),
            "color": String("black"),
            "has_glowing_text": Byte(0),
        })
        te = TileEntity(Compound({
            "id": String("minecraft:sign"),
            "front_text": side,
            "back_text": Compound({
                "messages": List[String]([String(dumps({"text": ""}))] * 4),
                "color": String("black"),
                "has_glowing_text": Byte(0),
            }),
            "is_waxed": Byte(0),
        }))
        te.position = pos
        region.tile_entities.append(te)


def place_io_fixtures(region: Region, fixtures: list[tuple[Pos, tuple[int, int], bool]]) -> int:
    """Redstone lamp past each output repeater, stone + lever behind each input repeater.

    `fixtures` is (repeater cell, outward direction, is_input). A fixture is skipped when its
    cells aren't free or any neighbour carries other redstone parts (it would power or be
    powered by them). Returns the number placed.
    """
    parts = ("redstone_wire", "repeater", "comparator", "torch", "lever", "redstone_block", "lamp")

    def free(p: Pos) -> bool:
        return all(0 <= c < n for c, n in zip(p, (region.width, region.height, region.length))) \
            and region[p].id == "minecraft:air"

    def quiet(p: Pos, ignore: set[Pos]) -> bool:
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            if q in ignore or not (0 <= q[0] < region.width and 0 <= q[1] < region.height
                                   and 0 <= q[2] < region.length):
                continue
            if any(m in region[q].id for m in parts):
                return False
        return True

    placed = 0
    for (x, y, z), (nx, nz), is_input in fixtures:
        if is_input:
            base = (x + nx, y, z + nz)
            lever = (base[0], y + 1, base[2])
            if not (free(base) and free(lever)):
                continue
            ignore = {(x, y, z), base, lever}
            if not (quiet(base, ignore) and quiet(lever, ignore)):
                continue
            region[base] = BlockState("minecraft:stone")
            region[lever] = BlockState(
                "minecraft:lever", face="floor", facing=_DIR_NAME[(nx, nz)], powered="false",
            )
        else:
            lamp = (x + nx, y, z + nz)
            if not free(lamp) or not quiet(lamp, {(x, y, z), lamp}):
                continue
            region[lamp] = BlockState("minecraft:redstone_lamp", lit="false")
        placed += 1
    return placed
