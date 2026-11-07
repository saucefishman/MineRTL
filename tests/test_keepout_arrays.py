"""The array-built _KeepoutMap against the per-terminal Python functions it replaces."""
import random

from minecraft_v.placement_engine.builder import (
    _APPROACH_CORRIDOR, _HALO_STENCIL, _HALO_STEP_STENCIL, _KEEPOUT_STENCIL, _ForeignKeepout, _KeepoutMap,
    _count_cells, _terminal_halo, _terminal_keepout,
)

_APPROACHES = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _random_design(rng: random.Random, nets: int, box: int, with_approach: bool):
    terminals = {f"n{i}": {(rng.randrange(box), rng.randrange(4), rng.randrange(box)) for _ in range(rng.randrange(1, 4))}
                 for i in range(nets)}
    approach = {}
    if with_approach:
        for ts in terminals.values():
            for t in ts:
                if rng.random() < 0.6:
                    approach[t] = rng.choice(_APPROACHES)
    return terminals, approach


def _offsets(cells):
    return sorted(map(tuple, cells.tolist()))


def test_stencils_are_the_functions_cells():
    origin = {(0, 0, 0)}
    assert set(_offsets(_KEEPOUT_STENCIL)) == _terminal_keepout(origin)
    assert set(_offsets(_HALO_STENCIL)) == _terminal_halo(origin, {})
    for ad in _APPROACHES:
        want = _terminal_halo(origin, {(0, 0, 0): ad})
        got = set(_offsets(_HALO_STENCIL))
        for k in range(1, _APPROACH_CORRIDOR + 1):
            got |= {(x + k * ad[0], y, z + k * ad[1]) for x, y, z in _offsets(_HALO_STEP_STENCIL)}
        assert got == want


def _check_same(terminals, approach):
    halo = {n: _terminal_halo(ts, approach) for n, ts in terminals.items()}
    own = {n: _terminal_keepout(ts) | halo[n] for n, ts in terminals.items()}
    old = {n: _ForeignKeepout(_count_cells(own), own[n], halo=_count_cells(halo), own_halo=halo[n]) for n in terminals}
    keepouts = _KeepoutMap(terminals, approach)
    count, halo_count = _count_cells(own), _count_cells(halo)
    allc = set(count)
    for n in terminals:
        new = keepouts.for_net(n)
        assert set(new._own) == own[n]
        assert set(new._own_halo) == halo[n]
    for cell in allc:
        assert keepouts._count.get(cell, 0) == min(count[cell], 2)
        assert keepouts._halo.get(cell, 0) == min(halo_count.get(cell, 0), 2)
    assert set(keepouts._count) == allc and set(keepouts._halo) == set(halo_count)
    # __contains__, plain and with goal relief (excluded cells), over a box around everything
    rng = random.Random(7)
    probes = list(allc)
    xs = [c[0] for c in allc]
    zs = [c[2] for c in allc]
    probes += [(rng.randint(min(xs) - 3, max(xs) + 3), rng.randint(-3, 7), rng.randint(min(zs) - 3, max(zs) + 3))
               for _ in range(2000)]
    for n in rng.sample(sorted(terminals), min(6, len(terminals))):
        relief = frozenset(rng.sample(probes, 60))
        for a, b in ((keepouts.for_net(n), old[n]), (keepouts.for_net(n) - relief, old[n] - relief),
                     (keepouts.for_net(n).with_extra(list(relief)[:5]) - relief,
                      old[n].with_extra(list(relief)[:5]) - relief)):
            for p in probes:
                assert (p in a) == (p in b), (n, p)


def test_matches_functions_without_approach():
    for seed in range(4):
        _check_same(*_random_design(random.Random(seed), 14, 18, False))


def test_matches_functions_with_approach():
    for seed in range(4):
        _check_same(*_random_design(random.Random(100 + seed), 14, 18, True))


def test_negative_and_empty():
    _check_same({"a": {(0, 0, 0)}, "b": {(-5, 2, 3)}}, {(0, 0, 0): (-1, 0)})
    keepouts = _KeepoutMap({})
    assert (1, 2, 3) not in keepouts.for_net("zz")
    assert (1, 2, 3) not in _KeepoutMap({"a": set()}).for_net("a")
