import pytest

from minecraft_v.placement_engine.ir import ComponentType


def test_mirror_component_x():
    from minecraft_v.cell_library import CELL_TYPE_MAP, SCHEMATIC_MAP
    from minecraft_v.placement_engine.ir import CardinalDirection, Component
    from minecraft_v.placement_engine.template import _mirror_component_x

    info = SCHEMATIC_MAP[ComponentType.DFF]
    dff = Component(id="d", type=ComponentType.DFF, pins=info.pins, footprint=info.footprint)
    m = _mirror_component_x(dff)
    pins = {p.name: p for p in m.pins}
    assert m.metadata["mirror_x"] is True
    assert pins["C"].offset == (0, 0, 0) and pins["C"].side == CardinalDirection.WEST
    assert pins["D"].offset == (3, 0, 0) and pins["D"].side == CardinalDirection.SOUTH
    back = _mirror_component_x(m)
    assert back.pins == dff.pins and back.metadata["mirror_x"] is False


@pytest.mark.parametrize("ctype", [ComponentType.DFF, ComponentType.MUX, ComponentType.XOR])
def test_mirrored_paste_reflects_template(ctype):
    from pathlib import Path
    from litemapy import Region
    from minecraft_v.cell_library import SCHEMATIC_MAP
    from minecraft_v.placement_engine.template import _load_template_region, _mirror_block_x, _paste_template

    info = SCHEMATIC_MAP[ctype]
    template = _load_template_region(Path(__file__).resolve().parents[1] / "schematics", info.file_prefix)
    w = info.footprint.width
    plain, mirrored = Region(0, 0, 0, 20, 20, 20), Region(0, 0, 0, 20, 20, 20)
    _paste_template(plain, template, (2, 2, 2), set())
    _paste_template(mirrored, template, (2, 2, 2), set(), mirror_width=w)
    for x in range(w):
        for y in range(2, 2 + info.footprint.height + 1):
            for z in range(2, 2 + info.footprint.depth):
                a, b = plain[2 + x, y, z], mirrored[2 + w - 1 - x, y, z]
                assert _mirror_block_x(a) == b, (x, y, z, a, b)
