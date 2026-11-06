import pytest
from litemapy import BlockState, Region

from minecraft_v.placement_engine import _astar, builder, pathfinding
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


def test_cell_grid_gives_the_moves_the_containers_give(monkeypatch):
    # check mode works out every cell's moves from both, raising on a difference; the second net
    # is searched over a grid the first net's laying and rip-up have written to.
    monkeypatch.setattr(pathfinding, "_ASTAR_GRID", "check")
    router, region, solid, dust_owner = _router()
    assert router.indexed.block_index().grid is not None
    walled = (14, 1, 14)
    terminals = {"a": {(2, 1, 2), (8, 1, 2), walled}, "b": {(2, 1, 6), (14, 1, 6)}}
    keepouts = _KeepoutMap(terminals)
    region[walled] = STONE
    solid.add(walled)
    jobs = [_RouteJob(key=net, net_id=net, src=src, sinks=sinks, bounds=BOUNDS, protected=keepouts.for_net(net),
                      terminals=terminals[net])
            for net, src, sinks in (("a", (2, 1, 2), [(8, 1, 2), walled]), ("b", (2, 1, 6), [(14, 1, 6)]))]
    failures = router.run(jobs, "[test]")
    assert [key for key, _ in failures] == ["a"]
    assert dust_owner[(14, 1, 6)] == "b"


def test_blockers_finds_jobs_near_a_goal():
    router = _router()[0]
    assert router.run([_job("a", (2, 1, 6), [(14, 1, 6)])], "[test]") == []
    goal = (8, 1, 8)  # two cells from a's wire
    assert router.blockers(goal, "b", {"a", "b"}) == ["a"]
    assert router.blockers(goal, "b", {"b"}) == []


# --- signal depth (repeater slots) -----------------------------------------------------------

# A stretch a net was routed on: every move steps down, so no cell can hold a repeater.
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


def test_router_gets_past_a_wall_on_a_wider_budget():
    # Source and sink at either X end of a memory bank. A clear route exists, but the search
    # spends a normal budget on states along the near face.
    bounds = (0, 0, 0, 29, 44, 104)
    rec = Recorder()
    region = Region(0, 0, 0, bounds[3] + 1, bounds[4] + 1, bounds[5] + 1)
    solid, dust_owner = JournaledSet(rec, name="solid"), JournaledDict(rec)
    wall = frozenset((x, y, z) for x in (14, 15) for y in range(23) for z in range(4, 101))
    router = _GreedyRouter(region, rec, solid, dust_owner, JournaledSet(rec, name="inverted"),
                           wall, bounds[4], frozenset())
    src, sink = (8, 17, 76), (21, 11, 76)
    job = _RouteJob(key="n", net_id="n", src=src, sinks=[sink], bounds=bounds, protected=frozenset(),
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


def test_no_repeater_slot_reroutes_only_the_branch_at_the_fault(monkeypatch):
    router, *_ = _router()
    job = _job("n", (1, 1, 1), [(3, 1, 3), (9, 1, 1)])
    searches = []
    find = builder._find_wire_path

    def counting_find(ws, solid, owner, start, dst, *args, **kwargs):
        searches.append(dst)
        return find(ws, solid, owner, start, dst, *args, **kwargs)

    place = builder._place_repeaters_for_net
    calls = []

    def failing_once(ws, dust_owner, net_id, paths, *args):
        calls.append(paths)
        if len(calls) == 1:
            raise LayConflict("Net 'n': cannot place repeater to extend signal", tuple(paths[-1][1:]))
        place(ws, dust_owner, net_id, paths, *args)

    monkeypatch.setattr(builder, "_find_wire_path", counting_find)
    monkeypatch.setattr(builder, "_place_repeaters_for_net", failing_once)
    with router.rec.recording(job.key):
        router.route_job_retrying(job)
    assert searches == [(3, 1, 3), (9, 1, 1), (9, 1, 1)]
    assert [path[-1] for path in router.paths["n"][1]] == [(3, 1, 3), (9, 1, 1)]
    assert "n" not in router.has_kept_wire


def test_failed_sink_keeps_the_jobs_branches_and_rips_only_blockers(monkeypatch):
    router, *_ = _router()
    blocker = _job("blocker", (1, 1, 10), [(9, 1, 10)])
    net = _job("n", (1, 1, 1), [(3, 1, 1), (5, 1, 1), (9, 1, 7)])
    find = builder._find_wire_path
    searches = []

    def failing_once(ws, solid, owner, start, dst, net_id, *args, **kwargs):
        searches.append((net_id, dst))
        if (net_id, dst) == ("n", (9, 1, 7)) and searches.count(("n", (9, 1, 7))) == 1:
            raise ValueError("No route")
        return find(ws, solid, owner, start, dst, net_id, *args, **kwargs)

    monkeypatch.setattr(builder, "_find_wire_path", failing_once)
    assert router.run([blocker, net], "[test]") == []
    assert searches.count(("n", (3, 1, 1))) == 1 and searches.count(("n", (5, 1, 1))) == 1  # not routed again
    assert searches.count(("n", (9, 1, 7))) == 2
    assert searches.count(("blocker", (9, 1, 10))) == 2  # ripped, then routed after n
    assert not router.kept and not router.has_kept_wire
    assert len(router.paths["n"][1]) == 3


def test_rip_branch_keeps_the_other_branches_and_reroutes_only_its_sink(monkeypatch):
    router, region, solid, dust_owner = _router()
    job = _job("n", (2, 1, 2), [(8, 1, 2), (2, 1, 14)])
    assert router.run([job], "[test]") == []
    far, near = job.sinks[1], job.sinks[0]
    router.rip_branch(job, [far])
    assert dust_owner.get(near) == "n" and far not in dust_owner  # one branch gone, the other still there
    assert "n" not in router.paths

    find = builder._find_wire_path
    searched = []

    def recording(ws, solid, owner, start, dst, net_id, *args, **kwargs):
        searched.append(dst)
        return find(ws, solid, owner, start, dst, net_id, *args, **kwargs)

    monkeypatch.setattr(builder, "_find_wire_path", recording)
    assert router.run([job], "[test]") == []
    assert searched == [far]
    assert len(router.paths["n"][1]) == 2 and not router.kept and not router.has_kept_wire
    assert not verify_signals(region, {"n": (job.src, job.sinks)}, dust_owner)


def test_rip_branch_takes_the_branches_hanging_off_it():
    router, _, _, dust_owner = _router()
    # (14, 1, 2) has to branch off the wire to (8, 1, 2), which is on its way there.
    job = _job("n", (2, 1, 2), [(8, 1, 2), (14, 1, 2)])
    assert router.run([job], "[test]") == []
    parent, child = router.laid["n"].paths
    assert child[0] in parent
    router.rip_branch(job, [parent[len(parent) // 2]])
    assert not dust_owner  # nothing left to carry on from
    assert "n" not in router.kept


def test_rip_branch_leaves_later_nets_alone_and_rip_up_clean():
    router, region, solid, dust_owner = _router()
    empty = _state(region, solid, dust_owner)
    a = _job("a", (2, 1, 2), [(8, 1, 2), (2, 1, 14)])
    b = _job("b", (14, 1, 14), [(14, 1, 8)])
    assert router.run([a, b], "[test]") == []
    router.rip_branch(a, [(2, 1, 14)])
    assert dust_owner[(14, 1, 8)] == "b"
    router.rip("a")
    router.rip("b")
    assert _state(region, solid, dust_owner) == empty


def _searches_with_limits(monkeypatch, fail_tight_at):
    find = builder._find_wire_path
    limits = []

    def recording(ws, solid, owner, start, dst, net_id, *args, stagnation=-1, **kwargs):
        limits.append((dst, stagnation))
        if dst == fail_tight_at and stagnation >= 0:
            ws[(dst[0], dst[1] + 3, dst[2])] = BlockState("minecraft:redstone_lamp")  # as a failed search does
            raise ValueError("No route")
        return find(ws, solid, owner, start, dst, net_id, *args, stagnation=stagnation, **kwargs)

    monkeypatch.setattr(builder, "_find_wire_path", recording)
    return limits


def test_sink_gets_the_full_limit_after_its_rip(monkeypatch):
    router, region, *_ = _router()
    blocker = _job("blocker", (1, 1, 10), [(9, 1, 10)])
    sink = (9, 1, 7)
    limits = _searches_with_limits(monkeypatch, sink)
    assert router.run([blocker, _job("n", (1, 1, 1), [sink])], "[test]") == []
    tight, full = builder._ROUTE_STAGNATION_FIRST, -1
    assert [limit for dst, limit in limits if dst == sink] == [tight, tight, full]  # tight, rip, tight, full
    assert region[(9, 4, 7)].id == "minecraft:air"  # the failed searches' marks went


def test_sink_with_no_rip_to_come_gets_the_full_limit_at_once(monkeypatch):
    router, region, *_ = _router()
    sink = (9, 1, 7)
    limits = _searches_with_limits(monkeypatch, sink)
    assert router.run([_job("n", (1, 1, 1), [sink])], "[test]", rip_blockers=False) == []
    assert [limit for dst, limit in limits if dst == sink] == [builder._ROUTE_STAGNATION_FIRST, -1]
    assert region[(9, 4, 7)].id == "minecraft:air"


def test_failed_sink_with_nothing_to_change_is_not_searched_again(monkeypatch):
    router, *_ = _router()
    sink = (9, 1, 7)
    searches = []

    def failing(ws, solid, owner, start, dst, net_id, *args, stagnation=-1, **kwargs):
        searches.append(stagnation)
        raise ValueError("No route")

    monkeypatch.setattr(builder, "_find_wire_path", failing)
    failures = router.run([_job("n", (1, 1, 1), [sink])], "[test]", rip_blockers=False)
    assert [key for key, _ in failures] == ["n"]
    assert searches == [builder._ROUTE_STAGNATION_FIRST, -1]


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


def test_repair_runs_on_a_layout_with_a_macro(monkeypatch):
    # One RAM block turned repair off for the whole design, so faults only surfaced at the flat verify.
    from types import SimpleNamespace

    from minecraft_v.placement_engine.ir import ComponentType, NetConnection, NetEndpoint

    router, region, solid, dust_owner = _router()
    macro = SimpleNamespace(origin=(20, 1, 20), component=SimpleNamespace(
        id="ram", type=ComponentType.CUSTOM, metadata={}, pins=[], footprint=SimpleNamespace(width=4, height=4, depth=4)))
    region[21, 2, 21] = STONE  # inside the macro: hidden from verify
    terminals = {("a", "y"): (2, 1, 2), ("b", "a"): (8, 1, 2)}
    net = NetConnection(net_id="n", source=NetEndpoint(component_id="a", pin_name="y"),
                        sinks=[NetEndpoint(component_id="b", pin_name="a")])
    seen = []
    real = builder._repair_faulty_nets

    def spy(router, jobs, failed, cells, deadline):
        seen.append((21, 2, 21) in cells)
        return real(router, jobs, failed, cells, deadline)

    monkeypatch.setattr(builder, "_repair_faulty_nets", spy)
    failures, _, _ = builder._route_all_nets(region, router.rec, solid, dust_owner, [net], terminals, BOUNDS, BOUNDS[4],
                                             frozenset(), {"n": set(terminals.values())}, placed=[macro])
    assert failures == [] and seen == [True]


def test_no_repeater_at_a_slopes_foot():
    # A repeater there reads the block under the slope, which can be glass or powered by other wire.
    assert pathfinding._signal_step(3, 0, (1, -1, 0)) == (4, 0)
    region = Region(0, 0, 0, 40, 6, 3)
    path = [(x, 3, 1) for x in range(1, 3)] + [(x, 2, 1) for x in range(3, 17)]
    for x, y, z in path:
        region[x, y, z] = BlockState("minecraft:redstone_wire")
        region[x, y - 1, z] = STONE
    _place_repeaters_for_net(region, {p: "n" for p in path}, "n", [path])
    assert region[3, 2, 1].id == "minecraft:redstone_wire"
    assert any(region[p].id == "minecraft:repeater" for p in path)


def test_repeater_at_a_slopes_foot_when_allowed():
    # Routed banks without RAM blocks fail routing without these; main.py turns them on for such designs.
    pathfinding.set_foot_repeaters(True)
    try:
        assert pathfinding._signal_step(3, 0, (1, -1, 0)) == (4, 1)
        region = Region(0, 0, 0, 40, 6, 3)
        path = [(x, 3, 1) for x in range(1, 3)] + [(x, 2, 1) for x in range(3, 17)]
        for x, y, z in path:
            region[x, y, z] = BlockState("minecraft:redstone_wire")
            region[x, y - 1, z] = STONE
        _place_repeaters_for_net(region, {p: "n" for p in path}, "n", [path])
        assert any(region[p].id == "minecraft:repeater" for p in path)
    finally:
        pathfinding.set_foot_repeaters(False)


def test_no_repeater_at_a_sinks_pin_cell():
    # A later branch leaves from a sink's pin cell and runs straight on, and the repeater search
    # picked that cell, so the gate read the repeater's output side and never switched.
    sink = (_REPEATER_INTERVAL - 1, 2, 1)
    first = [(x, 2, 1) for x in range(0, sink[0] + 1)]
    second = [(x, 2, 1) for x in range(sink[0], 2 * _REPEATER_INTERVAL + 4)]
    region = Region(0, 0, 0, 60, 6, 3)
    for x, y, z in first + second:
        region[x, y, z] = BlockState("minecraft:redstone_wire")
        region[x, y - 1, z] = STONE
    _place_repeaters_for_net(region, {p: "n" for p in first + second}, "n", [first, second])
    assert region[sink].id == "minecraft:redstone_wire"
    assert any(region[p].id == "minecraft:repeater" for p in second)


def test_long_edge_leads_split_only_long_tree_edges():
    # Two sinks near the source and one far off: only the far edge of the spanning tree is split.
    leads = builder._long_edge_leads((0, 10, 0), [(5, 10, 0), (0, 10, 6), (305, 10, 0)], 100)
    assert leads == [(80, 10, 0), (155, 10, 0), (230, 10, 0)]
    assert builder._long_edge_leads((0, 10, 0), [(90, 10, 0)], 100) == []


def test_long_edge_split_only_on_large_designs(monkeypatch):
    # Mid-size designs (hundreds to a few thousand gates) route slower split, large ones faster.
    monkeypatch.setattr(builder, "_long_edge_split", [0])
    builder.set_long_edge_split(4509)
    assert builder._long_edge_split[0] == 0
    builder.set_long_edge_split(12448)
    assert builder._long_edge_split[0] == 100
