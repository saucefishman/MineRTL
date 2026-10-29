from pathlib import Path

import pytest
from litemapy import Schematic

from minecraft_v.main import module_to_component_list
from minecraft_v.models import Cell, Module, Port
from minecraft_v.placement_engine import build_litematic_from_component_list
from minecraft_v.placement_engine.ir import CardinalDirection, ComponentType
from minecraft_v.placement_engine.template import _io_repeater_facing

SCHEMATICS_DIR = Path("schematics")


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


def _single_gate_module(type_: str, *, has_b: bool = True, has_s: bool = False) -> Module:
    ports = {
        "a": Port(direction="input", bits=[2]),
        "y": Port(direction="output", bits=[10]),
    }
    conns = {"A": [2], "Y": [10]}
    pds = {"A": "input", "Y": "output"}
    if has_b:
        ports["b"] = Port(direction="input", bits=[3])
        conns["B"] = [3]
        pds["B"] = "input"
    if has_s:
        ports["s"] = Port(direction="input", bits=[4])
        conns["S"] = [4]
        pds["S"] = "input"
    return _make_module(
        ports=ports,
        cells={"g": _make_cell(type_, pds, conns)},
    )


def _inverter_module() -> Module:
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


def _build(module: Module, out: Path, capsys, **kwargs):
    """Build with routing failures raising; return the loaded main region.

    Also fails on anything the post-route checks report: [verify] opens / shorts /
    unsupported blocks, and ignored pin-target failures.
    """
    build_litematic_from_component_list(
        module_to_component_list(module),
        schematics_dir=SCHEMATICS_DIR,
        out_path=out,
        schematic_name=out.stem,
        allow_routing_failures=False,
        **kwargs,
    )
    printed = capsys.readouterr().out
    problems = [line for line in printed.splitlines() if line.startswith(("[verify]", "IGNORING"))]
    assert not problems, "\n".join(problems)
    assert out.is_file()
    return Schematic.load(str(out)).regions["main"]


def test_io_repeater_facing_rules():
    # facing = input direction: an input pin's repeater reads from the pin (its side),
    # an output pin's reads from the wire arriving on the far side.
    assert _io_repeater_facing(ComponentType.INPUT_PIN, CardinalDirection.SOUTH) == "south"
    assert _io_repeater_facing(ComponentType.INPUT_PIN, CardinalDirection.NORTH) == "north"
    assert _io_repeater_facing(ComponentType.OUTPUT_PIN, CardinalDirection.NORTH) == "south"
    assert _io_repeater_facing(ComponentType.OUTPUT_PIN, CardinalDirection.SOUTH) == "north"


def test_build_litematic_from_netlist_smoke(tmp_path: Path, capsys):
    region = _build(_inverter_module(), tmp_path / "test_inv.litematic", capsys)
    blocks = [str(region[pos]) for pos in region.block_positions()]
    assert any(b != "minecraft:air" for b in blocks)
    assert any("repeater" in b for b in blocks)


def test_output_pin_target_layer_wiring(tmp_path: Path, capsys):
    target = (2, 5)
    region = _build(_inverter_module(), tmp_path / "test_inv_targets.litematic", capsys,
                    output_pin_targets={"y_out": target})
    # Targets are x-offset to the middle of the output layer (see _route_output_pin_extensions).
    tx, ty = target
    etx = tx + (region.width - 1) // 2
    assert "redstone_wire" in str(region[etx, ty, 0])


@pytest.mark.parametrize("placer", ["legacy", "sa"])
@pytest.mark.parametrize(
    "gate_type, kwargs",
    [
        ("$_AND_", {"has_b": True}),
        ("$_OR_", {"has_b": True}),
        ("$_XOR_", {"has_b": True}),
        ("$_NOT_", {"has_b": False}),
        pytest.param("$_MUX_", {"has_b": True, "has_s": True}, marks=pytest.mark.xfail(
            raises=ValueError, strict=True,
            reason="known layout bug (also examples/mux): IO repeater cell lands outside the workspace")),
    ],
)
def test_single_gate_routes_clean(tmp_path: Path, capsys, gate_type, kwargs, placer):
    _build(_single_gate_module(gate_type, **kwargs), tmp_path / "gate.litematic", capsys, placer=placer)
