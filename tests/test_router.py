import pytest
from litemapy import Region

from minecraft_v.placement_engine.builder import _GreedyRouter, _KeepoutMap, _RouteJob, _terminal_halo
from minecraft_v.placement_engine.constants import GLASS, STONE
from minecraft_v.placement_engine.ripup import Recorder, JournaledDict, JournaledSet
from minecraft_v.placement_engine.verify import verify_signals
from minecraft_v.placement_engine.wire import LayConflict, _check_slope_notch

BOUNDS = (0, 0, 0, 23, 7, 23)


def _state(region: Region, solid, dust_owner):
    blocks = {
        (x, y, z): str(region[x, y, z])
        for x in range(region.width) for y in range(region.height) for z in range(region.length)
    }
    return blocks, set(solid), dict(dust_owner)


def _router():
    rec = Recorder()
    region = Region(0, 0, 0, BOUNDS[3] + 1, BOUNDS[4] + 1, BOUNDS[5] + 1)
    solid = JournaledSet(rec, name="solid")
    dust_owner = JournaledDict(rec)
    inverted = JournaledSet(rec, name="inverted")
    router = _GreedyRouter(region, rec, solid, dust_owner, inverted, frozenset(), BOUNDS[4], frozenset())
    return router, region, solid, dust_owner


def _job(key, src, sinks, **kwargs):
    return _RouteJob(key=key, net_id=key, src=src, sinks=sinks, bounds=BOUNDS,
                     protected=frozenset(), terminals={src, *sinks}, **kwargs)


# --- keepout ---------------------------------------------------------------------------------

def test_goal_relief_keeps_other_terminals_halo():
    a, b = (10, 5, 10), (14, 5, 10)
    keepouts = _KeepoutMap({"a": {a}, "b": {b}})
    prot_b = keepouts.for_net("b")
    halo_cell = (11, 4, 10)     # cardinal of a at dy -1: wire here makes a unwalkable
    plain_cell = (11, 7, 10)    # in a's keepout (dy +2) but not its halo
    assert halo_cell in prot_b and plain_cell in prot_b
    relieved = prot_b - {halo_cell, plain_cell}  # what b's goal relief does
    assert halo_cell in relieved
    assert plain_cell not in relieved
    # a's own halo never blocks a.
    assert halo_cell not in keepouts.for_net("a")


def test_approach_corridor_reserved_and_survives_relief():
    a = (10, 5, 10)
    approach: dict[tuple[int, int, int], tuple[int, int]] = {a: (0, 1)}
    corridor = {(10, 5, 11), (10, 5, 12)}
    assert corridor <= _terminal_halo({a}, approach)
    prot_b = _KeepoutMap({"a": {a}, "b": {(16, 5, 16)}}, approach).for_net("b")
    beside_corridor = (11, 5, 12)
    assert all(c in prot_b - corridor - {beside_corridor} for c in corridor | {beside_corridor})


def test_with_extra_survives_subtraction():
    prot = _KeepoutMap({"a": {(10, 5, 10)}, "b": {(14, 5, 10)}}).for_net("b")
    cell = (3, 3, 3)
    assert cell not in prot
    assert cell in (prot.with_extra({cell}) - {cell})


# --- laying ----------------------------------------------------------------------------------

@pytest.mark.parametrize("prev, cell, notch", [
    ((1, 3, 1), (2, 2, 1), (2, 3, 1)),  # slope down: above the lower (dest) end
    ((1, 2, 1), (2, 3, 1), (1, 3, 1)),  # slope up: above the lower (source) end
])
def test_slope_notch_check(prev, cell, notch):
    region = Region(0, 0, 0, 5, 5, 5)
    solid = {notch}
    region[notch] = GLASS
    _check_slope_notch(region, solid, prev, cell)  # glass doesn't cut the link
    region[notch] = STONE
    with pytest.raises(LayConflict) as e:
        _check_slope_notch(region, solid, prev, cell)
    assert e.value.cells == (cell, prev)
    _check_slope_notch(region, solid, prev, (cell[0], prev[1], cell[2]))  # flat: no notch


# --- greedy router ---------------------------------------------------------------------------

def test_router_routes_tree_and_records_paths():
    router, region, _, dust_owner = _router()
    src, sinks = (2, 1, 2), [(14, 1, 2), (2, 1, 14)]
    assert router.run([_job("n", src, sinks)], "[test]") == []
    assert dust_owner[src] == "n" and all(dust_owner[s] == "n" for s in sinks)
    net_id, paths = router.paths["n"]
    assert net_id == "n" and len(paths) == 2
    assert not verify_signals(region, {"n": (src, sinks)}, dust_owner)


def test_router_rips_up_failed_job():
    router, region, solid, dust_owner = _router()
    walled = (14, 1, 14)
    region[walled] = STONE  # goal itself is solid: unreachable
    solid.add(walled)
    before = _state(region, solid, dust_owner)
    # The first sink routes and is laid before the second fails.
    failures = router.run([_job("n", (2, 1, 2), [(8, 1, 2), walled])], "[test]")
    assert [key for key, _ in failures] == ["n"]
    assert _state(region, solid, dust_owner) == before
    assert "n" not in router.paths and "n" not in router.rec.journals


def test_blockers_finds_jobs_near_a_goal():
    router = _router()[0]
    assert router.run([_job("a", (2, 1, 6), [(14, 1, 6)])], "[test]") == []
    goal = (8, 1, 8)  # two cells from a's wire
    assert router.blockers(goal, "b", {"a", "b"}) == ["a"]
    assert router.blockers(goal, "b", {"b"}) == []
