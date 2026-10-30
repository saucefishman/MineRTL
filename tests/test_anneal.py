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


def _module_with_scalar_inputs() -> Module:
    """The adder plus scalar inputs: ``cin`` into bit 0, ``clk`` gating the top sum bit."""
    m = _adder_like_module()
    top = m.ports["s"].bits[-1]
    m.cells["cin_and"] = _cell("$_AND_", {"A": [m.ports["a"].bits[0]], "B": [50], "Y": [60]})
    m.cells["clk_and"] = _cell("$_AND_", {"A": [top], "B": [51], "Y": [61]})
    m.ports.update({
        "cin": Port(direction="input", bits=[50]),
        "clk": Port(direction="input", bits=[51]),
        "q0": Port(direction="output", bits=[60]),
        "q3": Port(direction="output", bits=[61]),
    })
    return m


@pytest.mark.parametrize("clock_pin", [None, "clk"])
def test_input_pins_align_by_bit_and_name(clock_pin):
    comp = _expand_multibit_io(module_to_component_list(_module_with_scalar_inputs()))
    stride = max(c.footprint.height for c in comp.components) + 4
    y_level = _assign_component_y_levels(comp.components, comp.nets, y_stride=stride, clock_pin=clock_pin)
    placed = anneal._optimize_placement(
        comp.components, comp.nets, y_level, placer="sa", clock_pin=clock_pin, **_KW)
    origin = {p.component.id: p.origin for p in placed}

    # Bit k of every input on one Y level (scalar inputs count as bit 0); bits of one input in one X column.
    by_name = defaultdict(set)
    for k in range(4):
        assert origin[f"a[{k}]"][1] == origin[f"b[{k}]"][1]
    assert origin["cin"][1] == origin["a[0]"][1]
    for c in comp.components:
        if c.type == ComponentType.INPUT_PIN:
            by_name[_io_base_name(c.id)].add(origin[c.id][0])
    assert all(len(xs) == 1 for xs in by_name.values())

    clk_gate_y = origin["clk_and"][1]
    assert origin["clk"][1] == (clk_gate_y if clock_pin else origin["a[0]"][1])


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


def test_sa_mirror_moves_keep_cost_consistent():
    comp, y_level, _ = _setup()
    st = _state(comp, y_level)
    gid = next(g for g in st.gates if st.comps[g].footprint.width > 1)
    st.apply({}, {}, gid)
    st.commit()
    assert gid in st.mirrored
    assert st.cost == pytest.approx(st.full_recompute(restack=False))
    st.apply({}, {}, gid)
    st.undo()
    assert gid in st.mirrored
    snap = st.snapshot()
    st.apply({}, {}, gid)
    st.commit()
    st.restore(snap)
    assert gid in st.mirrored
    assert st.cost == pytest.approx(st.full_recompute(refresh_io=True))


def test_undo_restores_state_exactly():
    comp, y_level, _ = _setup()
    st = _state(comp, y_level)
    rng = random.Random(5)
    kinds = list(anneal._PLACE_MOVE_WEIGHTS)
    weights = list(anneal._PLACE_MOVE_WEIGHTS.values())
    for _ in range(500):
        mv = st.propose(rng, kinds, weights, 4)
        if mv is None:
            continue
        before = (st.snapshot(), dict(st.pos), list(st.net_cost), st.cost)
        st.apply(*mv)
        st.undo()
        assert (st.snapshot(), dict(st.pos), list(st.net_cost), st.cost) == before
        st.apply(*mv)
        st.commit()


@pytest.mark.parametrize("seed", range(4))
def test_anneal_cost_is_true_cost(seed):
    """Final state's cost is its recomputed (fresh IO/Z) cost and no worse than the seed's."""
    comp, y_level, _ = _setup()
    st = _state(comp, y_level)
    anneal._barycenter(st)
    anneal._greedy_mirror(st)
    seed_cost = st.cost
    anneal._anneal(st, random.Random(seed), y_moves=True, z_moves=True)
    final = st.cost
    assert final <= seed_cost + 1e-9
    assert final == pytest.approx(st.full_recompute(refresh_io=True))


def test_parallel_anneal_cost_is_true_cost_and_deterministic():
    """Forked chains: final cost is the recomputed cost, no worse than the seed, repeatable."""
    comp, y_level, _ = _setup()
    finals = []
    for _ in range(2):
        st = _state(comp, y_level)
        anneal._barycenter(st)
        seed_cost = st.cost
        anneal._anneal(st, random.Random(0), y_moves=True, z_moves=True, workers=2)
        assert st.cost <= seed_cost + 1e-9
        assert st.cost == pytest.approx(st.full_recompute(refresh_io=True))
        finals.append(st.snapshot())
    assert finals[0] == finals[1]
