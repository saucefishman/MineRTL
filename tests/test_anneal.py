import random
from collections import defaultdict

import pytest

from minecraft_v.main import module_to_component_list
from minecraft_v.models import Cell, Module, Port
import minecraft_v.placement_engine.anneal as anneal
from minecraft_v.placement_engine.ir import ComponentType
from minecraft_v.placement_engine.layout import (
    _assign_component_y_levels,
    _assign_gate_layers,
    _expand_multibit_io,
    _feedforward_nets,
    _io_base_name,
)

_IO = (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
_KW = dict(base_y=5, gutter=10, io_margin=7, routing_gutter=6)


def _cell(type_, conns):
    pds = {p: ("output" if p == "Y" else "input") for p in conns}
    return Cell(hide_name=1, type=type_, parameters={}, port_directions=pds, connections=conns)


def _adder_like_module(width: int = 4) -> Module:
    """Per bit: s = (a ^ b) ^ c, carry = (a & b) | (c & (a ^ b)) — enough fan-in/out to give SA work."""
    a = list(range(2, 2 + width))
    b = list(range(10, 10 + width))
    nid = 100
    cells = {}
    s_bits, carry = [], None
    for i in range(width):
        x, g = nid, nid + 1
        nid += 2
        cells[f"x{i}"] = _cell("$_XOR_", {"A": [a[i]], "B": [b[i]], "Y": [x]})
        cells[f"g{i}"] = _cell("$_AND_", {"A": [a[i]], "B": [b[i]], "Y": [g]})
        if carry is None:
            s_bits.append(x)
            carry = g
            continue
        s, p, c = nid, nid + 1, nid + 2
        nid += 3
        cells[f"s{i}"] = _cell("$_XOR_", {"A": [x], "B": [carry], "Y": [s]})
        cells[f"p{i}"] = _cell("$_AND_", {"A": [x], "B": [carry], "Y": [p]})
        cells[f"c{i}"] = _cell("$_OR_", {"A": [g], "B": [p], "Y": [c]})
        s_bits.append(s)
        carry = c
    return Module(
        ports={
            "a": Port(direction="input", bits=a),
            "b": Port(direction="input", bits=b),
            "s": Port(direction="output", bits=s_bits),
            "cout": Port(direction="output", bits=[carry]),
        },
        cells=cells,
    )


def _setup():
    comp = _expand_multibit_io(module_to_component_list(_adder_like_module()))
    stride = max(c.footprint.height for c in comp.components) + 4
    y_level = _assign_component_y_levels(comp.components, comp.nets, y_stride=stride)
    return comp, y_level, stride


def _state(comp, y_level):
    gates = [c for c in comp.components if c.type not in _IO]
    layers = _assign_gate_layers(gates, comp.nets, {c.id: y_level[c.id] for c in gates})
    return anneal._PlaceState(
        comp.components, comp.nets, layers, {c.id: y_level.get(c.id, 0) for c in comp.components},
        base_y=5, gutter=10, routing_gutter=6, y_moves=True,
    )


def _origins(placed):
    return [(p.component.id, p.origin) for p in placed]


@pytest.mark.parametrize("placer", ["bary", "sa"])
def test_placement_is_legal(placer):
    comp, y_level, stride = _setup()
    placed = anneal._optimize_placement(comp.components, comp.nets, y_level, placer=placer, **_KW)

    ids = [p.component.id for p in placed]
    assert sorted(ids) == sorted(c.id for c in comp.components)

    # No footprint overlap among gates (gutter is kept between row neighbours).
    gates = [p for p in placed if p.component.type not in _IO]
    rows = defaultdict(list)
    for p in gates:
        rows[(p.origin[1], p.origin[2])].append(p)
    for row in rows.values():
        row.sort(key=lambda p: p.origin[0])
        for left, right in zip(row, row[1:]):
            assert left.origin[0] + left.component.footprint.width + 10 <= right.origin[0]
    for p in gates:
        for q in gates:
            if p is q or p.origin[1] != q.origin[1] or (p.origin[1], p.origin[2]) == (q.origin[1], q.origin[2]):
                continue
            pz, qz = p.origin[2], q.origin[2]
            assert pz + p.component.footprint.depth <= qz or qz + q.component.footprint.depth <= pz

    # Same-level feed-forward sinks sit strictly nearer the outputs (lower Z) than their drivers.
    origin = {p.component.id: p.origin for p in placed}
    gate_ids = {p.component.id for p in gates}
    for net in _feedforward_nets([p.component for p in gates], comp.nets):
        src = net.source.component_id
        for sink in net.sinks:
            dst = sink.component_id
            if dst in gate_ids and dst != src and origin[src][1] == origin[dst][1]:
                assert origin[dst][2] < origin[src][2], (src, dst)


def test_io_organisation_kept():
    comp, y_level, stride = _setup()
    placed = anneal._optimize_placement(comp.components, comp.nets, y_level, placer="sa", **_KW)
    gate_z = [p.origin[2] for p in placed if p.component.type not in _IO]
    group_x = defaultdict(set)
    for p in placed:
        t = p.component.type
        if t == ComponentType.OUTPUT_PIN:
            assert p.origin[2] == 1
        if t == ComponentType.INPUT_PIN:
            assert p.origin[2] > max(gate_z)
        if t in _IO:
            group_x[_io_base_name(p.component.id)].add(p.origin[0])
            bit = p.component.id.rsplit("[", 1)
            if len(bit) == 2:
                assert p.origin[1] == 5 + int(bit[1].rstrip("]")) * stride
    assert all(len(xs) == 1 for xs in group_x.values())


def test_sa_deterministic():
    comp, y_level, _ = _setup()
    runs = [_origins(anneal._optimize_placement(comp.components, comp.nets, y_level, placer="sa", **_KW))
            for _ in range(2)]
    assert runs[0] == runs[1]


def test_cost_never_worse_than_seed():
    comp, y_level, _ = _setup()
    st = _state(comp, y_level)
    legacy = st.cost
    anneal._barycenter(st)
    bary = st.cost
    anneal._anneal(st, random.Random(0), y_moves=True, z_moves=True)
    assert st.cost <= bary <= legacy
    assert st.cost < legacy


@pytest.mark.parametrize("w_cong", [0.0, 1.0])
def test_incremental_cost_matches_full_recompute(monkeypatch, w_cong):
    monkeypatch.setattr(anneal, "_PLACE_W_CONG", w_cong)
    comp, y_level, _ = _setup()
    st = _state(comp, y_level)
    rng = random.Random(1)
    kinds = list(anneal._PLACE_MOVE_WEIGHTS)
    weights = list(anneal._PLACE_MOVE_WEIGHTS.values())
    for _ in range(2000):
        mv = st.propose(rng, kinds, weights, 4)
        if mv is None:
            continue
        st.apply(*mv)
        st.commit() if rng.random() < 0.5 else st.undo()
    incremental = st.cost
    assert incremental == pytest.approx(st.full_recompute(restack=False))
