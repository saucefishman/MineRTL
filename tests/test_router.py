import pytest
from litemapy import BlockState, Region

from minecraft_v.placement_engine import _astar
from minecraft_v.placement_engine.builder import (
    _GreedyRouter, _KeepoutMap, _RouteJob, _repair_faulty_nets, _terminal_halo,
)
from minecraft_v.placement_engine.constants import GLASS, STONE, _DELTA_TO_FACING, _HORIZ_DIRS, _REPEATER_INTERVAL
from minecraft_v.placement_engine.pathfinding import _signal_depths
from minecraft_v.placement_engine.ripup import Recorder, JournaledDict, JournaledSet
from minecraft_v.placement_engine.verify import verify_signals
from minecraft_v.placement_engine.wire import (
    LayConflict, _check_slope_notch, _lay_redstone_path, _place_repeaters_for_net, _signal_returns,
)

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


@pytest.mark.parametrize("path, notch", [
    ([(1, 5, 1), (3, 3, 1)], (2, 5, 1)),  # 2x down: above the middle cell
    ([(1, 5, 1), (3, 3, 1)], (3, 4, 1)),  # 2x down: above the dest
    ([(1, 3, 1), (3, 5, 1)], (1, 4, 1)),  # 2x up: above the launch
    ([(1, 3, 1), (3, 5, 1)], (2, 5, 1)),  # 2x up: above the middle cell
])
def test_slope2_lay_rejects_block_cutting_a_step(path, notch):
    rec = Recorder()
    region = Region(0, 0, 0, 7, 8, 7)
    solid = JournaledSet(rec, name="solid")
    dust_owner = JournaledDict(rec)
    region[notch] = STONE
    solid.add(notch)
    with pytest.raises(LayConflict):
        _lay_redstone_path(region, solid, dust_owner, path, "a")


def test_signal_returns_around_a_repeater_cell():
    region = Region(0, 0, 0, 8, 4, 6)
    owner = {}
    from minecraft_v.placement_engine.constants import REDSTONE
    def dust(*cells, net="a"):
        for c in cells:
            region[c] = REDSTONE
            owner[c] = net
    dust((1, 1, 1), (2, 1, 1), (3, 1, 1))  # 2 is the candidate repeater cell
    assert not _signal_returns(region, owner, "a", (3, 1, 1), (1, 1, 1), (2, 1, 1))
    dust((3, 1, 2), (2, 1, 2), (1, 1, 2))  # bypass from past it back to its input
    assert _signal_returns(region, owner, "a", (3, 1, 1), (1, 1, 1), (2, 1, 1))
    for c in ((3, 1, 2), (2, 1, 2), (1, 1, 2)):
        owner[c] = "b"  # another net's wire is not this net's signal
    assert not _signal_returns(region, owner, "a", (3, 1, 1), (1, 1, 1), (2, 1, 1))


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


def test_slope_notch_checked_when_cell_starts_a_longer_move():
    """A cell that a 1-step slope lands on can also start a 2x slope; the notch check still applies."""
    workspace = Region(0, 0, 0, 12, 16, 12)
    notch = (2, 11, 2)
    workspace[notch] = STONE
    path = [(1, 11, 2), (2, 10, 2), (4, 8, 2)]
    with pytest.raises(LayConflict, match="would be cut"):
        _lay_redstone_path(workspace, {notch}, {}, path, "n")


def test_bank_leads_only_for_the_bank_pins_and_channel():
    from minecraft_v.placement_engine.builder import _bank_pin_leads
    from minecraft_v.placement_engine.layout import _Placed
    from minecraft_v.placement_engine.ir import (
        Component, ComponentType, Footprint, NetConnection, NetEndpoint,
    )
    bank = _Placed(component=Component(id="bank", type=ComponentType.CUSTOM, pins=[], footprint=Footprint(width=100, height=20, depth=50),
                                       metadata={"bank": True}), origin=(0, 0, 100))
    terminals = {("s", "y"): (200, 5, 120), ("b", "east"): (100, 4, 120), ("g", "a"): (150, 4, 120),
                 ("b", "back"): (30, 6, 99)}
    net = NetConnection(net_id="n", source=NetEndpoint(component_id="s", pin_name="y"), sinks=[
        NetEndpoint(component_id=c, pin_name=p) for c, p in (("b", "east"), ("g", "a"), ("b", "back"))])
    leads = _bank_pin_leads(net, terminals, [bank], [bank])
    assert len(leads) == 2
    east, back = leads
    assert east[1:] == (4, 120) and east[0] > 100
    assert back[0] > 100 and back[1] == 6 and back[2] < 99  # the channel's mouth, away from the face


def test_net_ticks_count_the_deepest_branch_not_every_repeater():
    from minecraft_v.placement_engine.builder import _net_wire_ticks
    from minecraft_v.placement_engine.ir import (
        Component, ComponentList, ComponentType, CURRENT_SCHEMA_VERSION, NetConnection, NetEndpoint,
    )
    ws = Region(0, 0, 0, 21, 3, 3)
    dust = BlockState("minecraft:redstone_wire")
    rep = BlockState("minecraft:repeater", facing="west", delay="1")
    owner = {}
    for x in range(21):  # a line from x=10 both ways, 2 repeaters on each arm
        ws[x, 0, 1] = rep if x in (2, 6, 14, 18) else dust
        owner[(x, 0, 1)] = "n"
    net = NetConnection(net_id="n", source=NetEndpoint(component_id="s", pin_name="y"),
                        sinks=[NetEndpoint(component_id="t", pin_name="a")])
    comp = ComponentList.model_construct(schema_version=CURRENT_SCHEMA_VERSION, components=[], nets=[net])
    assert _net_wire_ticks(comp, owner, ws, {("s", "y"): (10, 0, 1)}) == {"n": 2}


def test_tower_starting_where_a_powered_drop_lands_is_laid():
    ws = Region(0, 0, 0, 14, 14, 14)
    solid, owner = set(), {}
    _lay_redstone_path(ws, solid, owner, [(6, 8, 6), (6, 4, 6), (6, 8, 8)], "n")
    assert "repeater" in ws[6, 4, 7].id  # the tower's repeater
    assert (6, 7, 8) in owner  # and its column


def test_powered_drop_from_a_tower_top_cant_land_in_its_own_column():
    with pytest.raises(ValueError):
        _lay_redstone_path(Region(0, 0, 0, 14, 14, 14), set(), {}, [(6, 4, 6), (6, 8, 8), (6, 4, 8)], "n")


def test_slope_notch_checked_on_both_steps_of_a_2x_slope():
    ws = Region(0, 0, 0, 14, 14, 14)
    ws[6, 7, 7] = STONE
    with pytest.raises(LayConflict):
        _check_slope_notch(ws, {(6, 7, 7)}, (6, 7, 6), (6, 5, 8))



def _tower_moves(pending):
    """Where the kernel will build a tower from (5, 1, 5) in an empty world."""
    pos_ids, pos_list, h_list, cache = {}, [], [], []
    empty = frozenset()
    world = _astar.World(empty, empty, empty, empty, {(x, 0, z) for x in range(24) for z in range(24)}, {}, "n",
                         BOUNDS, BOUNDS[4], (20, 1, 20), 1, ("set", empty), ("set", empty), empty, empty, empty,
                         empty, Region(0, 0, 0, 24, 8, 24), GLASS, [_DELTA_TO_FACING[d] for d in _HORIZ_DIRS],
                         pos_ids, pos_list, h_list, cache, frozenset(pending))
    world_id = len(pos_list)
    pos_ids[(5, 1, 5)] = world_id
    pos_list.append((5, 1, 5))
    h_list.append(0)
    cache.append(None)
    moves, _ = world.static_moves(world_id)
    return {pos_list[m[0]] for m in moves if pos_list[m[0]][1] == 5}


def test_tower_keeps_clear_of_a_later_sinks_approach():
    assert (7, 5, 5) in _tower_moves(())
    # Diagonal to the inverted blocks at (7, 2..3, 5): every approach to this sink would touch them.
    assert (7, 5, 5) not in _tower_moves([(6, 3, 4)])
    # Straight along a line two cells off and two up or down from them, too.
    assert (7, 5, 5) not in _tower_moves([(9, 1, 5)])
    assert (7, 5, 5) in _tower_moves([(10, 3, 5)])


def test_sink_a_job_cant_reach_is_retried_first():
    router, *_ = _router()
    job = _job("n", (1, 1, 1), [(3, 1, 1), (5, 1, 1)])
    orders = []

    def route_job(job, blocked, first=()):
        orders.append(first)
        if not first:
            router.failed_goal[job.key] = (5, 1, 1)
            raise ValueError("No route")

    router.route_job = route_job
    router.route_job_retrying(job)
    assert orders == [(), ((5, 1, 1),)]


def test_sink_failing_again_while_first_is_final():
    router, *_ = _router()
    job = _job("n", (1, 1, 1), [(3, 1, 1)])

    def route_job(job, blocked, first=()):
        router.failed_goal[job.key] = (3, 1, 1)
        raise ValueError("No route")

    router.route_job = route_job
    with pytest.raises(ValueError):
        router.route_job_retrying(job)


def test_repair_reroutes_a_net_verify_finds_fault_with():
    router, region, solid, dust_owner = _router()
    job = _job("n", (2, 1, 2), [(8, 1, 2)])
    assert router.run([job], "[test]") == []
    assert verify_signals(region, {"n": (job.src, job.sinks)}, dust_owner) == []
    # Something the search couldn't have known cut the wire: verify sees an open, repair re-routes.
    cut = router.paths["n"][1][0][3]
    region[cut] = STONE
    dust_owner.pop(cut)
    assert _repair_faulty_nets(router, [job], set(), frozenset(), None) == []
    assert verify_signals(region, {"n": (job.src, job.sinks)}, dust_owner) == []
