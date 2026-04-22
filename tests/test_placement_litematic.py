from pathlib import Path
import pytest
from litemapy import Schematic
from minecraft_v.main import module_to_component_list
from minecraft_v.models import Cell, Module, Port
from minecraft_v.placement_engine import build_litematic_from_component_list

def _make_module(ports=None, cells=None):
    return Module(ports=ports or {}, cells=cells or {})

def _make_cell(type_, port_directions, connections, parameters=None):
    return Cell(
        hide_name=1,
        type=type_,
        parameters=parameters or {},
        port_directions=port_directions,
        connections=connections,
    )

def _inverter_module():
    return _make_module(
        ports={
            "a_in": Port(direction="input", bits=[2]),
            "y_out": Port(direction="output", bits=[3]),
        },
        cells={
            "inv": _make_cell(
                "$_NOT_",
                {"A": "input", "Y": "output"},
                {"A": [2], "Y": [3]},
            )
        },
    )

def test_build_litematic_from_netlist_smoke(tmp_path: Path):
    mod = _inverter_module()
    comp_list = module_to_component_list(mod)
    out = tmp_path / "out.litematic"
    build_litematic_from_component_list(
        comp_list,
        schematics_dir=Path("schematics"),
        out_path=out,
        schematic_name="test_inv",
        gutter=10,
    )
    assert out.is_file()
    loaded = Schematic.load(str(out))
    region = next(iter(loaded.regions.values()))
    non_air = sum(1 for pos in region.block_positions() if str(region[pos]) != "minecraft:air")
    assert non_air > 0
    has_repeater = any("repeater" in str(region[pos]) for pos in region.block_positions())
    assert has_repeater
