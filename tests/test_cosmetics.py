from litemapy import BlockState, Region

from minecraft_v.placement_engine.cosmetics import apply_dust_shapes, place_io_signs

DUST = BlockState("minecraft:redstone_wire")
STONE = BlockState("minecraft:stone")


def _shape(region: Region, p) -> tuple[str, str, str, str]:
    b = region[p]
    return b["north"], b["south"], b["east"], b["west"]


def test_dust_line_corner_and_slope():
    r = Region(0, 0, 0, 6, 4, 6)
    for x in range(6):
        for z in range(6):
            r[x, 0, z] = STONE
    for x in (0, 1, 2):
        r[x, 1, 2] = DUST            # straight run along X ...
    r[2, 1, 3] = DUST                # ... turning south
    r[3, 1, 3] = STONE
    r[3, 2, 3] = DUST                # slope up east
    r[0, 1, 5] = DUST                # isolated

    apply_dust_shapes(r)
    assert _shape(r, (0, 1, 2)) == ("none", "none", "side", "side")  # one link -> line
    assert _shape(r, (1, 1, 2)) == ("none", "none", "side", "side")
    assert _shape(r, (2, 1, 2)) == ("none", "side", "none", "side")  # corner
    assert _shape(r, (2, 1, 3)) == ("side", "none", "up", "none")
    assert _shape(r, (3, 2, 3)) == ("none", "none", "side", "side")
    assert _shape(r, (0, 1, 5)) == ("side", "side", "side", "side")  # cross


def test_io_sign_above_repeater():
    r = Region(0, 0, 0, 3, 3, 3)
    r[1, 0, 1] = BlockState("minecraft:repeater", facing="north")
    place_io_signs(r, [((1, 0, 1), (0, -1), "count[0]")])
    assert r[1, 1, 1] == BlockState("minecraft:oak_sign", rotation="8")
    te, = r.tile_entities
    assert te.position == (1, 1, 1)
    assert '"count[0]"' in str(te.data["front_text"]["messages"][0])
