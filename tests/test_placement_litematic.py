from collections import deque
from pathlib import Path

import pytest
from litemapy import Schematic

from minecraft_v.main import module_to_component_list
from minecraft_v.models import Cell, Module, Port
from minecraft_v.placement_engine import (
    _io_repeater_facing,
    _plan_placement,
    build_litematic_from_component_list,
)
from minecraft_v.placement_engine.ir import CardinalDirection, ComponentType, Direction

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


def test_io_repeater_facing_rules():
    assert _io_repeater_facing(ComponentType.INPUT_PIN, CardinalDirection.SOUTH) == "north"
    assert _io_repeater_facing(ComponentType.INPUT_PIN, CardinalDirection.NORTH) == "south"
    assert _io_repeater_facing(ComponentType.OUTPUT_PIN, CardinalDirection.NORTH) == "north"
    assert _io_repeater_facing(ComponentType.OUTPUT_PIN, CardinalDirection.SOUTH) == "south"


def _plan(module: Module):
    comp = module_to_component_list(module)
    return _plan_placement(comp, schematics_dir=SCHEMATICS_DIR)


def _per_net_cells(plan) -> dict[str, set[tuple[int, int, int]]]:
    groups: dict[str, set[tuple[int, int, int]]] = {}
    for cell, nid in plan.dust_owner.items():
        groups.setdefault(nid, set()).add(cell)
    return groups


def _assert_net_connected(cells: set[tuple[int, int, int]], net_id: str) -> None:
    assert cells, f"net '{net_id}' has zero routed cells"
    start = next(iter(cells))
    seen: set[tuple[int, int, int]] = {start}
    queue: deque[tuple[int, int, int]] = deque([start])
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in (
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
        ):
            npos = (x + dx, y + dy, z + dz)
            if npos in cells and npos not in seen:
                seen.add(npos)
                queue.append(npos)
    missing = cells - seen
    assert not missing, f"net '{net_id}' disconnected: {len(missing)} stray cells, e.g. {next(iter(missing))}"


def _assert_no_cross_net_chebyshev(groups: dict[str, set[tuple[int, int, int]]]) -> None:
    cell_to_net = {cell: nid for nid, cells in groups.items() for cell in cells}
    for (x, y, z), nid in cell_to_net.items():
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                npos = (x + dx, y, z + dz)
                other = cell_to_net.get(npos)
                if other is not None and other != nid:
                    raise AssertionError(
                        f"cross-net adjacency: net '{nid}' at {(x, y, z)} "
                        f"touches net '{other}' at {npos}"
                    )


def _pin_reaches_io(plan, io_component_id: str, pin_name: str) -> None:
    io_cell = plan.io_repeater_cell[(io_component_id, pin_name)]
    pin_to_net = {}
    for net in plan.component_list.nets:
        pin_to_net[(net.source.component_id, net.source.pin_name)] = net.net_id
        for sink in net.sinks:
            pin_to_net[(sink.component_id, sink.pin_name)] = net.net_id
    net_id = pin_to_net[(io_component_id, pin_name)]
    x, y, z = io_cell
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        npos = (x + dx, y, z + dz)
        if plan.dust_owner.get(npos) == net_id:
            return
    raise AssertionError(
        f"IO pin {io_component_id}.{pin_name} repeater at {io_cell} has no "
        f"dust-adjacent cell of its own net '{net_id}'"
    )


def test_build_litematic_from_netlist_smoke(tmp_path: Path):
    comp_list = module_to_component_list(_inverter_module())
    out = tmp_path / "out.litematic"
    build_litematic_from_component_list(
        comp_list,
        schematics_dir=SCHEMATICS_DIR,
        out_path=out,
        schematic_name="test_inv",
        gutter=10,
    )
    assert out.is_file()
    loaded = Schematic.load(str(out))
    region = next(iter(loaded.regions.values()))
    non_air = sum(
        1 for pos in region.block_positions() if str(region[pos]) != "minecraft:air"
    )
    assert non_air > 0
    has_repeater = any(
        "repeater" in str(region[pos]) for pos in region.block_positions()
    )
    assert has_repeater


def test_output_pin_target_layer_wiring(tmp_path: Path):
    comp_list = module_to_component_list(_inverter_module())
    out = tmp_path / "out_targets.litematic"
    target = (2, 5)
    build_litematic_from_component_list(
        comp_list,
        schematics_dir=SCHEMATICS_DIR,
        out_path=out,
        schematic_name="test_inv_targets",
        output_pin_targets={"y_out": target},
    )
    loaded = Schematic.load(str(out))
    region = next(iter(loaded.regions.values()))
    tx, ty = target
    assert "redstone_wire" in str(region[tx, ty, region.min_z()])


@pytest.mark.parametrize(
    "gate_type, kwargs",
    [
        ("$_AND_", {"has_b": True}),
        ("$_OR_", {"has_b": True}),
        ("$_XOR_", {"has_b": True}),
        ("$_NOT_", {"has_b": False}),
    ],
)
def test_single_gate_routes_and_connects(gate_type, kwargs):
    plan = _plan(_single_gate_module(gate_type, **kwargs))
    groups = _per_net_cells(plan)
    for net in plan.component_list.nets:
        assert net.net_id in groups, f"net {net.net_id} for {gate_type} not routed"
    for net_id, cells in groups.items():
        _assert_net_connected(cells, net_id)
    _assert_no_cross_net_chebyshev(groups)
    for comp in plan.component_list.components:
        if comp.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
            for pin in comp.pins:
                _pin_reaches_io(plan, comp.id, pin.name)


def test_mux_routes_and_connects():
    plan = _plan(_single_gate_module("$_MUX_", has_b=True, has_s=True))
    groups = _per_net_cells(plan)
    for net in plan.component_list.nets:
        assert net.net_id in groups
    for net_id, cells in groups.items():
        _assert_net_connected(cells, net_id)
    _assert_no_cross_net_chebyshev(groups)


_TRUTH = {
    ComponentType.AND: lambda v: {"Y": v["A"] & v["B"]},
    ComponentType.OR: lambda v: {"Y": v["A"] | v["B"]},
    ComponentType.XOR: lambda v: {"Y": v["A"] ^ v["B"]},
    ComponentType.NOT: lambda v: {"Y": 1 - v["A"]},
    ComponentType.MUX: lambda v: {"Y": v["B"] if v["S"] else v["A"]},
}


def _simulate(plan, inputs: dict[str, int]) -> dict[str, int]:
    comp_by_id = {c.id: c for c in plan.component_list.components}
    pin_to_net: dict[tuple[str, str], str] = {}
    for net in plan.component_list.nets:
        pin_to_net[(net.source.component_id, net.source.pin_name)] = net.net_id
        for sink in net.sinks:
            pin_to_net[(sink.component_id, sink.pin_name)] = net.net_id

    net_value: dict[str, int] = {}
    for net in plan.component_list.nets:
        src = comp_by_id[net.source.component_id]
        if src.type == ComponentType.INPUT_PIN:
            net_value[net.net_id] = inputs[src.id]

    pending = [
        c
        for c in plan.component_list.components
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
    ]
    while pending:
        progressed = False
        for gate in list(pending):
            vals: dict[str, int] = {}
            ok = True
            for pin in gate.pins:
                if pin.direction != Direction.IN:
                    continue
                nid = pin_to_net.get((gate.id, pin.name))
                if nid is None or nid not in net_value:
                    ok = False
                    break
                vals[pin.name] = net_value[nid]
            if not ok:
                continue
            if gate.type not in _TRUTH:
                raise AssertionError(f"No truth table for {gate.type}")
            out = _TRUTH[gate.type](vals)
            for pin in gate.pins:
                if pin.direction != Direction.OUT:
                    continue
                nid = pin_to_net.get((gate.id, pin.name))
                if nid is not None and pin.name in out:
                    net_value[nid] = out[pin.name]
            pending.remove(gate)
            progressed = True
        if not progressed:
            raise AssertionError("simulation stalled (missing input or cycle)")

    results: dict[str, int] = {}
    for comp in plan.component_list.components:
        if comp.type == ComponentType.OUTPUT_PIN:
            for pin in comp.pins:
                nid = pin_to_net.get((comp.id, pin.name))
                if nid is not None and nid in net_value:
                    results[comp.id] = net_value[nid]
    return results


@pytest.mark.parametrize(
    "gate_type, f",
    [
        ("$_AND_", lambda a, b: a & b),
        ("$_OR_", lambda a, b: a | b),
        ("$_XOR_", lambda a, b: a ^ b),
    ],
)
def test_two_input_gate_truth_table(gate_type, f):
    plan = _plan(_single_gate_module(gate_type, has_b=True))
    groups = _per_net_cells(plan)
    for net_id, cells in groups.items():
        _assert_net_connected(cells, net_id)
    _assert_no_cross_net_chebyshev(groups)
    for a in (0, 1):
        for b in (0, 1):
            got = _simulate(plan, {"a": a, "b": b})
            assert got == {"y": f(a, b)}, f"{gate_type}({a},{b}) -> {got}"


def test_not_truth_table():
    plan = _plan(_inverter_module())
    groups = _per_net_cells(plan)
    for net_id, cells in groups.items():
        _assert_net_connected(cells, net_id)
    _assert_no_cross_net_chebyshev(groups)
    for a in (0, 1):
        got = _simulate(plan, {"a_in": a})
        assert got == {"y_out": 1 - a}


def test_mux_truth_table():
    plan = _plan(_single_gate_module("$_MUX_", has_b=True, has_s=True))
    groups = _per_net_cells(plan)
    for net_id, cells in groups.items():
        _assert_net_connected(cells, net_id)
    _assert_no_cross_net_chebyshev(groups)
    for a in (0, 1):
        for b in (0, 1):
            for s in (0, 1):
                got = _simulate(plan, {"a": a, "b": b, "s": s})
                expected = b if s else a
                assert got == {"y": expected}, f"MUX(a={a},b={b},s={s}) -> {got}"
