import pytest
from litemapy import Region

from minecraft_v.placement_engine.builder import _GreedyRouter, _KeepoutMap, _RouteJob, _terminal_halo
from minecraft_v.placement_engine.constants import GLASS, STONE, _REPEATER_INTERVAL
from minecraft_v.placement_engine.pathfinding import _signal_depths
from minecraft_v.placement_engine.ripup import Recorder, JournaledDict, JournaledSet
from minecraft_v.placement_engine.verify import verify_signals
from minecraft_v.placement_engine.wire import LayConflict, _check_slope_notch, _place_repeaters_for_net

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


# --- signal depth (repeater slots) -----------------------------------------------------------

# A stretch net_859 (video) was routed on: every move steps down, so no cell can hold a repeater.
_ALL_DOWN = [(109, 51, 127), (109, 50, 128), (111, 48, 128), (113, 46, 128), (113, 45, 127),
             (111, 43, 127), (109, 41, 127), (108, 40, 127), (107, 39, 127), (107, 38, 126)]


def test_staircase_has_no_repeater_slot():
    path = [(109, 51, 126)] + _ALL_DOWN + [(107, 37, 125)]  # from a fresh source, 15 deep at the end
    region = Region(0, 0, 0, 120, 60, 140)
    with pytest.raises(LayConflict, match="cannot place repeater"):
        _place_repeaters_for_net(region, {}, "n", [path])
    assert max(_signal_depths([path]).values()) >= _REPEATER_INTERVAL  # the search would prune it


def test_signal_depth_resets_on_flat_straight_step():
    # Each cell entered and left flat along +x is a repeater slot; 2x slopes are never slots.
    depths = _signal_depths([[(0, 5, 0), (1, 5, 0), (2, 5, 0), (3, 5, 0), (5, 3, 0), (7, 1, 0)]])
    assert [depths[p] for p in ((1, 5, 0), (2, 5, 0), (3, 5, 0), (5, 3, 0), (7, 1, 0))] == [1, 1, 1, 3, 5]
    # A tower top restarts the count; a branch start keeps the depth of the cell it leaves from.
    depths = _signal_depths([[(0, 1, 0), (1, 1, 0), (3, 5, 0)], [(1, 1, 0), (1, 1, 1)]])
    assert depths[(3, 5, 0)] == 1 and depths[(1, 1, 1)] == 2


@pytest.mark.parametrize("drop, horiz", [(16, 18), (20, 22)])
def test_router_breaks_long_descent_for_repeaters(drop, horiz):
    # A one-cell-wide corridor (keepout planes either side) rules out powered -4 drops, so the
    # descent has to be slopes; without signal depth in the search the cheapest path was all
    # 2x slopes with no repeater slot.
    bounds = (0, 0, 0, horiz + 8, drop + 8, 8)
    rec = Recorder()
    region = Region(0, 0, 0, bounds[3] + 1, bounds[4] + 1, bounds[5] + 1)
    solid, dust_owner = JournaledSet(rec, name="solid"), JournaledDict(rec)
    router = _GreedyRouter(region, rec, solid, dust_owner, JournaledSet(rec, name="inverted"),
                           frozenset(), bounds[4], frozenset())
    walls = frozenset((x, y, z) for x in range(bounds[3] + 1) for y in range(bounds[4] + 1) for z in (3, 5))
    src, sink = (2, 1 + drop, 4), (2 + horiz, 1, 4)
    job = _RouteJob(key="n", net_id="n", src=src, sinks=[sink], bounds=bounds, protected=walls,
                    terminals={src, sink})
    assert router.run([job], "[test]") == []
    assert not verify_signals(region, {"n": (src, [sink])}, dust_owner)
