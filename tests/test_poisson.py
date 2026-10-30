import numpy as np
from scipy import fft

import minecraft_v.placement_engine.poisson as poisson
from minecraft_v.placement_engine.builder import _io_barrier
from minecraft_v.placement_engine.ir import ComponentType
from tests.test_anneal import _KW, _setup

_IO = (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)


def _place(seed=0, **kw):
    comp, y_level, _ = _setup()
    args = dict(_KW, y_spacing=3, seed=seed, **kw)
    return comp, poisson._poisson_placement(comp.components, comp.nets, y_level, **args)


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
            if g.origin[0] - 1 <= x <= g.origin[0] + g.component.footprint.width:
                if pin.component.type == ComponentType.INPUT_PIN:
                    assert g.origin[2] + g.component.footprint.depth <= z, (pin.component.id, g.component.id)
                else:
                    assert z < g.origin[2], (pin.component.id, g.component.id)


def test_outputs_stay_in_row_with_extension_targets():
    comp, y_level, _ = _setup()
    order = {c.id: float(i) for i, c in enumerate(comp.components) if c.type == ComponentType.OUTPUT_PIN}
    placed = poisson._poisson_placement(comp.components, comp.nets, y_level, **dict(_KW, y_spacing=3),
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
    comp, y_level, _ = _setup()
    y_level = {k: 0 for k in y_level}
    placed = poisson._poisson_placement(comp.components, comp.nets, y_level, **dict(_KW, y_spacing=3), flat=True)
    assert len({p.origin[1] for p in placed}) == 1


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


def test_legalize_separates_stacked_boxes():
    sizes = np.tile([10.0, 4.0, 8.0], (20, 1))
    out = poisson._legalize(np.zeros((20, 3)) + np.arange(20)[:, None] * 0.01, sizes, flat=False)
    for i in range(20):
        for j in range(i + 1, 20):
            assert any(out[i, a] + sizes[i, a] <= out[j, a] or out[j, a] + sizes[j, a] <= out[i, a] for a in range(3))
