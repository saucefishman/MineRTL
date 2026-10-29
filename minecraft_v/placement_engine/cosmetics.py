"""Cosmetic post-processing of a finished build: dust shapes and I/O label signs.

Nothing here changes the circuit. Litematica pastes block states as stored, so dust
keeps whatever shape it was written with (a cross) until a block update reshapes it.
"""
from __future__ import annotations

from json import dumps

import numpy as np
from litemapy import BlockState, Region
from litemapy.minecraft import TileEntity
from nbtlib.tag import Byte, Compound, List, String

from .verify import _World

Pos = tuple[int, int, int]

_DIR_NAME = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}
# Standing sign `rotation` whose front faces each horizontal direction.
_SIGN_ROTATION = {(0, 1): "0", (-1, 0): "4", (0, -1): "8", (1, 0): "12"}


def apply_dust_shapes(region: Region) -> int:
    """Set each dust's north/south/east/west to its in-game shape. Returns dust count."""
    palette = region._Region__palette
    wire_ids = [i for i, b in enumerate(palette) if b.id == "minecraft:redstone_wire"]
    if not wire_ids:
        return 0
    cells = list(map(tuple, np.argwhere(np.isin(region._Region__blocks, wire_ids)).tolist()))
    world = _World(region, frozenset(), frozenset())
    shapes: list[tuple[Pos, dict[str, str]]] = []
    for p in cells:
        x, y, z = p
        ups = {(lx - x, lz - z) for lx, ly, lz in world.dust_links(p) if ly == y + 1}
        dirs = world.dust_directions(p)
        shapes.append((p, {
            name: ("up" if d in ups else "side") if d in dirs else "none"
            for d, name in _DIR_NAME.items()
        }))
    # Write after computing all shapes; the world's block cache must not see half-updated dust.
    for p, props in shapes:
        region[p] = region[p].with_properties(**props)
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
