import numpy as np
from scipy import fft

import minecraft_v.placement_engine.poisson as poisson
from minecraft_v.main import module_to_component_list
from minecraft_v.models import Cell, Module, Port
from minecraft_v.placement_engine.builder import _io_barrier
from minecraft_v.placement_engine.ir import Component, ComponentType, Footprint
from minecraft_v.placement_engine.layout import _expand_multibit_io, _io_base_name

_KW = dict(base_y=5, gutter=10, io_margin=7, routing_gutter=6)


def _cell(type_, conns):
    pds = {p: ("output" if p == "Y" else "input") for p in conns}
    return Cell(hide_name=1, type=type_, parameters={}, port_directions=pds, connections=conns)


def _adder_like_module(width: int = 4) -> Module:
    """Per bit: s = (a ^ b) ^ c, carry = (a & b) | (c & (a ^ b)) — enough fan-in/out to give the placer work."""
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
    return _expand_multibit_io(module_to_component_list(_adder_like_module()))

_IO = (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)


def _place(seed=0, **kw):
    comp = _setup()
    args = dict(_KW, y_spacing=3, seed=seed, **kw)
    return comp, poisson._poisson_placement(comp.components, comp.nets, **args)


def test_every_component_placed_once():
    comp, placed = _place()
    assert sorted(p.component.id for p in placed) == sorted(c.id for c in comp.components)


def test_gates_keep_their_spacing():
    _, placed = _place()
    gates = [p for p in placed if p.component.type not in _IO]
    for i, a in enumerate(gates):
        fa = a.component.footprint
        for b in gates[i + 1:]:
            fb = b.component.footprint
            apart = (a.origin[0] + fa.width + _KW["gutter"] <= b.origin[0]
                     or b.origin[0] + fb.width + _KW["gutter"] <= a.origin[0]
                     or a.origin[1] + fa.height + 3 <= b.origin[1]
                     or b.origin[1] + fb.height + 3 <= a.origin[1]
                     or a.origin[2] + fa.depth + _KW["routing_gutter"] <= b.origin[2]
                     or b.origin[2] + fb.depth + _KW["routing_gutter"] <= a.origin[2])
            assert apart, (a.component.id, b.component.id)


def test_no_gate_behind_input_or_ahead_of_output():
    _, placed = _place()
    gates = [p for p in placed if p.component.type not in _IO]
    for pin in (p for p in placed if p.component.type in _IO):
        x, _, z = pin.origin
        for g in gates:
            clear = poisson._IO_COLUMN_CLEAR
            if g.origin[0] - clear <= x <= g.origin[0] + g.component.footprint.width - 1 + clear:
                if pin.component.type == ComponentType.INPUT_PIN:
                    assert g.origin[2] + g.component.footprint.depth <= z, (pin.component.id, g.component.id)
                else:
                    assert z < g.origin[2], (pin.component.id, g.component.id)


def test_io_alignment_rules():
    _, placed = _place()
    pins = [p for p in placed if p.component.type in _IO]
    for kind in _IO:
        assert len({p.origin[2] for p in pins if p.component.type == kind}) == 1
    columns = {}
    for p in pins:
        columns.setdefault((p.component.type, _io_base_name(p.component.id)), []).append(p.origin)
    for origins in columns.values():
        assert len({o[0] for o in origins}) == 1
        ys = sorted(o[1] for o in origins)
        assert all(b - a >= 4 for a, b in zip(ys, ys[1:]))
    for kind in _IO:
        xs = sorted({o[0][0] for (k, _), o in columns.items() if k == kind})
        assert all(b - a >= _KW["gutter"] + 1 for a, b in zip(xs, xs[1:]))


def test_io_y_rises_with_bit_index():
    # Gates that want the bits reversed must not reverse them; a[10] sorts numerically, after a[9].
    pins = [Component(id=f"a[{i}]", type=ComponentType.INPUT_PIN, pins=[]) for i in (10, 9, 0)]
    gate = {f"a[{i}]": [("g", str(i))] for i in (10, 9, 0)}
    io = poisson._IoPlacer(pins, [], [], 1, 1, 4, False, None)
    io.neighbours.update(gate)
    out = io._side(pins, {("g", "10"): (0.0, 0.0), ("g", "9"): (0.0, 20.0), ("g", "0"): (0.0, 40.0)}, 0.0, {})
    assert out["a[0]"][1] < out["a[9]"][1] < out["a[10]"][1]

    _, placed = _place()
    columns = {}
    for p in placed:
        if p.component.type in _IO:
            columns.setdefault((p.component.type, _io_base_name(p.component.id)), []).append(p)
    for bits in columns.values():
        bits.sort(key=lambda p: poisson._bit_index(p.component.id))
        ys = [p.origin[1] for p in bits]
        assert ys == sorted(set(ys))


def test_spread_keeps_order_and_gap():
    assert poisson._spread([10.0, 10.0, 10.0], 4) == [6, 10, 14]
    assert poisson._spread([0.0, 50.0], 4) == [0, 50]
    assert poisson._spread([], 4) == []


def test_outputs_stay_in_row_with_extension_targets():
    comp = _setup()
    order = {c.id: float(i) for i, c in enumerate(comp.components) if c.type == ComponentType.OUTPUT_PIN}
    placed = poisson._poisson_placement(comp.components, comp.nets, **dict(_KW, y_spacing=3),
                                        output_x_order=order)
    assert len({p.origin[2] for p in placed if p.component.type == ComponentType.OUTPUT_PIN}) == 1


def test_io_barrier_closes_pockets_and_clears_gates():
    cells = set(_io_barrier({5: 20, 12: 30}, [(14, 18, 40)], 25))
    for z in range(21, 24):
        assert (5, z) in cells  # right behind the shallow pin, and beside it
        assert (6, z) in cells
    assert (8, 31) in cells and (8, 30) not in cells  # between columns: limit of the deeper one
    assert (12, 31) in cells and (12, 30) not in cells
    assert (16, 46) not in cells and (16, 47) in cells  # past a gate, its pins keep room


def test_deterministic():
    runs = [{p.component.id: p.origin for p in _place(seed=3)[1]} for _ in range(2)]
    assert runs[0] == runs[1]


def test_flat_uses_one_level():
    comp = _setup()
    placed = poisson._poisson_placement(comp.components, comp.nets, **dict(_KW, y_spacing=3), flat=True)
    assert len({p.origin[1] for p in placed}) == 1


def test_flat_io_rows_clear_every_gate():
    comp = _setup()
    placed = poisson._poisson_placement(comp.components, comp.nets, **dict(_KW, y_spacing=3), flat=True)
    gates = [p for p in placed if p.component.type not in _IO]
    far = max(p.origin[2] + p.component.footprint.depth for p in gates)
    near = min(p.origin[2] for p in gates)
    for p in placed:
        if p.component.type == ComponentType.INPUT_PIN:
            assert p.origin[2] >= far
        elif p.component.type == ComponentType.OUTPUT_PIN:
            assert p.origin[2] < near


def test_field_solves_discrete_poisson():
    sizes = np.full((6, 3), 8.0)
    field = poisson._Field(sizes, np.array([32.0, 24.0, 32.0]))
    rho = np.random.default_rng(0).random(field.shape)
    rho -= rho.mean()
    coeff = fft.dctn(rho, type=2, norm="ortho") / field.laplace
    coeff[0, 0, 0] = 0.0
    phi = fft.idctn(coeff, type=2, norm="ortho")
    lap = np.zeros_like(phi)
    for a, h in enumerate(field.h):
        flux = np.diff(phi, axis=a) / h ** 2
        s = [slice(None)] * 3
        s[a] = slice(0, -1)
        lap[tuple(s)] -= flux
        s[a] = slice(1, None)
        lap[tuple(s)] += flux
    assert np.allclose(lap, rho, atol=1e-10)


def test_forces_push_apart():
    sizes = np.full((2, 3), 8.0)
    field = poisson._Field(sizes, np.array([64.0, 64.0, 64.0]))
    force, _ = field.forces(np.array([[28.0, 32.0, 32.0], [36.0, 32.0, 32.0]]))
    assert force[0, 0] < 0 < force[1, 0]


def test_stretch_above_leaves_lower_heights():
    out = poisson._stretch_above(np.array([0.0, 80.0, 90.0, 180.0]), 80.0, 1.4)
    assert out.tolist() == [0.0, 80.0, 94.0, 220.0]


def test_legalize_separates_stacked_boxes():
    sizes = np.tile([10.0, 4.0, 8.0], (20, 1))
    out = poisson._legalize(np.zeros((20, 3)) + np.arange(20)[:, None] * 0.01, sizes, flat=False)
    for i in range(20):
        for j in range(i + 1, 20):
            assert any(out[i, a] + sizes[i, a] <= out[j, a] or out[j, a] + sizes[j, a] <= out[i, a] for a in range(3))


def test_all_pairs_legalize_by_blocks_of_rows_equals_all_pairs_at_once(monkeypatch):
    rng = np.random.default_rng(3)
    pos = rng.integers(0, 60, (150, 3)).astype(float)
    sizes = rng.integers(2, 14, (150, 3)).astype(float)
    for flat in (False, True):
        monkeypatch.setattr(poisson, "_LEGAL_BLOCK_PAIRS", 1 << 30)
        whole = poisson._legalize_all_pairs(pos, sizes, flat)
        monkeypatch.setattr(poisson, "_LEGAL_BLOCK_PAIRS", 400)  # 2 rows at a time
        assert (poisson._legalize_all_pairs(pos, sizes, flat) == whole).all()


def _overlapping_pairs(pos, sizes):
    return [(i, j) for i in range(len(pos)) for j in range(i + 1, len(pos))
            if ((pos[i] < pos[j] + sizes[j]) & (pos[j] < pos[i] + sizes[i])).all()]


def test_near_pairs_are_the_pairs_closer_than_the_margin():
    rng = np.random.default_rng(4)
    lo = rng.integers(0, 100, (250, 3)).astype(float)
    size = rng.integers(1, 30, (250, 3)).astype(float)
    for margin in (np.zeros(3), np.array([5.0, 0.0, 12.0])):
        i, j = poisson._near_pairs(lo, lo + size, margin)
        want = {(a, b) for a in range(250) for b in range(a + 1, 250)
                if ((lo[a] - margin / 2 < lo[b] + size[b] + margin / 2) & (lo[b] - margin / 2 < lo[a] + size[a] + margin / 2)).all()}
        assert len(i) == len(want) and set(zip(i.tolist(), j.tolist())) == want and (i < j).all()


def test_compact_matches_every_pair_scan():
    rng = np.random.default_rng(11)
    for n, span, big in ((1, 10, 5), (200, 60, 14), (300, 400, 30), (250, 25, 40)):
        pos = rng.integers(0, span, (n, 3)).astype(float)
        sizes = rng.integers(1, big, (n, 3)).astype(float)
        for axis in range(3):
            fast, slow = pos.copy(), pos.copy()
            poisson._compact(fast, sizes, axis)
            poisson._compact_all_pairs(slow, sizes, axis)
            assert (fast == slow).all()


def test_legalize_leaves_no_overlap_and_flat_keeps_y():
    rng = np.random.default_rng(7)
    for n, span in ((300, 150), (500, 60)):
        for flat in (False, True):
            pos = rng.integers(0, span, (n, 3)).astype(float)
            sizes = rng.integers(2, 20, (n, 3)).astype(float)
            if flat:
                sizes[:, 1] = 1
            out = poisson._legalize(pos, sizes, flat)
            assert _overlapping_pairs(out, sizes) == []
            assert (out >= 0).all()
            if flat:
                assert (out[:, 1] == pos[:, 1]).all()


def test_legalize_is_deterministic():
    rng = np.random.default_rng(9)
    pos = rng.integers(0, 80, (300, 3)).astype(float)
    sizes = rng.integers(2, 20, (300, 3)).astype(float)
    assert (poisson._legalize(pos, sizes, False) == poisson._legalize(pos, sizes, False)).all()


def test_legalize_settles_boxes_stacked_on_one_spot():
    sizes = np.tile([6.0, 3.0, 5.0], (400, 1))
    for flat in (False, True):
        out = poisson._legalize(np.zeros((400, 3)), sizes, flat)
        assert _overlapping_pairs(out, sizes) == []


def test_legalize_without_boxes():
    assert poisson._legalize(np.zeros((0, 3)), np.zeros((0, 3)), False).shape == (0, 3)
    assert poisson._legalize(np.zeros((1, 3)), np.ones((1, 3)), False).tolist() == [[0.0, 0.0, 0.0]]


def _bank(height: int, **metadata) -> Component:
    return Component(id=f"bank{height}", type=ComponentType.CUSTOM, pins=[], footprint=Footprint(width=300, height=height, depth=100),
                     metadata={"bank": True, **metadata})


def test_routed_banks_stay_in_one_stack():
    """Routed banks: a second stack doubled the width and walled nets off from pins on the banks' faces."""
    gates = [_bank(h) for h in (67, 67, 41, 41)]
    fixed, width, height = poisson._stack_banks(gates, list(range(4)), 24)
    assert {origin[0] for origin in fixed.values()} == {0.0}
    assert width == 300 and height == 67 + 67 + 41 + 41 + 4 * poisson._BANK_Y_GAP


def test_ram_blocks_start_a_new_stack():
    gates = [_bank(150, south_pins=True), _bank(151, south_pins=True)]
    fixed, width, _ = poisson._stack_banks(gates, [0, 1], 24)
    assert sorted(origin[0] for origin in fixed.values()) == [0.0, 300.0 + poisson._BANK_COLUMN_GAP]
    assert width == 600 + poisson._BANK_COLUMN_GAP


def test_forces_with_boxes_of_many_spans():
    """Boxes are deposited in groups by how many bins they span; a big box doesn't stop small ones pushing apart."""
    dims = np.array([96.0, 64.0, 96.0])
    sizes = np.vstack([np.full((2, 3), 8.0), [[40.0, 8.0, 40.0]]])
    field = poisson._Field(sizes, dims)
    members = np.sort(np.concatenate([m for m, _ in field.groups]))
    assert members.tolist() == [0, 1, 2] and len(field.groups) == 2
    force, _ = field.forces(np.array([[28.0, 32.0, 32.0], [36.0, 32.0, 32.0], [70.0, 32.0, 70.0]]))
    assert force[0, 0] < force[1, 0]  # the small two push apart


def test_net_cost_matches_hpwl():
    comp = _setup()
    gates = [c for c in comp.components if c.type not in _IO]
    io = [c for c in comp.components if c.type in _IO]
    netlist = poisson._Netlist(gates, io, comp.nets)
    pos = np.random.default_rng(3).integers(0, 40, (len(netlist.comps), 3)).astype(float)
    cost = poisson._NetCost(netlist, pos)
    p = pos[netlist.pin_comp] + netlist.pin_offset
    for i in range(len(netlist.comps)):
        want = 0.0
        for ni in set(netlist.pin_net[netlist.pin_comp == i].tolist()):
            q = p[netlist.pin_net == ni]
            want += float(((q.max(axis=0) - q.min(axis=0)) * poisson._WEIGHTS).sum())
        assert abs(cost.of(i) - want) < 1e-9


def _kernel():
    import pytest
    if poisson._place is None:
        pytest.skip("C++ placement kernel not built")
    return poisson._place


def test_kernel_occupancy_matches_python():
    kernel = _kernel()
    rng = np.random.default_rng(5)
    sizes = rng.integers(1, 20, (300, 3)).astype(float)
    py, cc = poisson._Occupancy(sizes), kernel.Occupancy(sizes)
    for i in range(0, 300, 2):
        p = tuple(int(v) for v in rng.integers(-40, 200, 3))
        py.add(i, p)
        cc.add(i, p)
    for i in range(0, 100, 4):
        py.remove(i)
        cc.remove(i)
    for i in range(1, 300, 2):
        target = tuple(int(v) for v in rng.integers(-20, 180, 3))
        for flat in (False, True):
            for pops in (5, 60, 2000):
                assert py.nearest_free(i, target, flat, pops) == cc.nearest_free(i, target, flat, pops)
        s = tuple(int(v) for v in sizes[i])
        assert py.conflicts(target, s, skip=i) == cc.conflicts(target, s, skip=i)


def test_kernel_refine_matches_python():
    kernel = _kernel()
    comp = _setup()
    gates = [c for c in comp.components if c.type not in _IO]
    io = [c for c in comp.components if c.type in _IO]
    netlist = poisson._Netlist(gates, io, comp.nets)
    sizes = np.array([[c.footprint.width + 4, c.footprint.height + 1, c.footprint.depth + 4] for c in gates], float)
    rng = np.random.default_rng(2)
    start = poisson._legalize(rng.integers(0, 30, (len(gates), 3)).astype(float), sizes, False)
    pos_all = np.vstack([start, rng.integers(-10, 40, (len(io), 3)).astype(float)])
    want = pos_all.copy()
    saved, poisson._place = poisson._place, None
    try:
        poisson._refine(netlist, want, sizes, False)
    finally:
        poisson._place = saved
    got = pos_all.copy()
    poisson._refine(netlist, got, sizes, False)
    assert np.array_equal(got, want) and not np.array_equal(got, pos_all)


def test_kernel_forces_match_numpy():
    _kernel()
    rng = np.random.default_rng(0)
    sizes = np.vstack([rng.integers(3, 12, (400, 3)).astype(float), [[60.0, 20.0, 40.0]]])
    dims = np.array([120.0, 60.0, 200.0])
    field = poisson._Field(sizes, dims)
    field.fix_box(np.zeros(3), np.array([30.0, 20.0, 30.0]))
    centres = rng.uniform(0, 1, (len(sizes), 3)) * dims
    got, got_overflow = field.forces(centres)
    field.kernel = None
    want, want_overflow = field.forces(centres)
    assert got_overflow == want_overflow and np.array_equal(got, want)  # the same floats, so the same placement


def test_kernel_wirelength_matches_numpy():
    kernel = _kernel()
    comp = _setup()
    gates = [c for c in comp.components if c.type not in _IO]
    io = [c for c in comp.components if c.type in _IO]
    netlist = poisson._Netlist(gates, io, comp.nets)
    wire = kernel.Wire(netlist.pin_comp, netlist.pin_offset, np.r_[netlist.net_start, len(netlist.pin_net)],
                       poisson._WEIGHTS, 4)
    pos = np.random.default_rng(4).uniform(0, 50, (len(netlist.comps), 3))
    gamma = np.array([2.0, 1.5, 3.0])
    assert np.array_equal(wire.gradient(pos, gamma), netlist.wa_gradient(pos, gamma))


def test_kernel_mirror_matches_python():
    _kernel()
    comp = _setup()
    gates = [c for c in comp.components if c.type not in _IO]
    io = [c for c in comp.components if c.type in _IO]
    netlist = poisson._Netlist(gates, io, comp.nets)
    pos = np.random.default_rng(6).integers(0, 40, (len(netlist.comps), 3)).astype(float)
    got = poisson._mirror_pass(netlist, pos, gates)
    saved, poisson._place = poisson._place, None
    try:
        want = poisson._mirror_pass(netlist, pos, gates)
    finally:
        poisson._place = saved
    assert [c.model_dump() for c in got] == [c.model_dump() for c in want] and got != gates
