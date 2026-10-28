from litemapy import Region

from minecraft_v.placement_engine.constants import STONE
from minecraft_v.placement_engine.negotiate import (
    Congestion, OverlayDict, OverlaySet, NetRoute, OverlayWorkspace,
)
from minecraft_v.placement_engine.pathfinding import _BlockIndex
from minecraft_v.placement_engine.wire import _lay_redstone_path


def _virtual_route(region, base_index, solid, dust_owner, net_id, path) -> NetRoute:
    view = OverlayWorkspace(region, base_index)
    solid_view = OverlaySet(solid)
    owner_view = OverlayDict(dust_owner)
    inverted: set = set()
    _lay_redstone_path(view, solid_view, owner_view, path, net_id, inverted_cells=inverted)
    return NetRoute.from_view(view, solid_view, owner_view, inverted)


def test_virtual_routes_leave_base_untouched_and_may_overlap():
    region = Region(0, 0, 0, 12, 8, 12)
    region[0, 0, 0] = STONE
    solid = {(0, 0, 0)}
    dust_owner: dict = {}
    base_index = _BlockIndex(region)

    a = _virtual_route(region, base_index, solid, dust_owner, "a", [(1, 2, 5), (2, 2, 5), (3, 2, 5)])
    # b crosses a's wire at (2, 2, 5): allowed virtually, it's a short to be negotiated away.
    b = _virtual_route(region, base_index, solid, dust_owner, "b", [(2, 2, 4), (2, 2, 5), (2, 2, 6)])

    assert solid == {(0, 0, 0)} and not dust_owner
    assert str(region[2, 2, 5]) == "minecraft:air"
    assert (2, 2, 5) in a.signal and (2, 1, 5) in a.support

    cong = Congestion()
    cong.add("a", a)
    cong.add("b", b)
    over = cong.overused()
    assert (2, 2, 5) in over
    assert cong.nets_at(over) == {"a", "b"}
    assert cong.users((2, 2, 5)) >= 2

    cong.remove("b", b)
    assert not cong.overused()
    assert cong.users((2, 2, 5)) == 1  # a's own wire (a's halo excludes its own signal)


def test_parallel_wires_one_apart_conflict_via_halo():
    region = Region(0, 0, 0, 12, 8, 12)
    base_index = _BlockIndex(region)
    a = _virtual_route(region, base_index, set(), {}, "a", [(1, 2, 5), (2, 2, 5), (3, 2, 5)])
    b = _virtual_route(region, base_index, set(), {}, "b", [(1, 2, 6), (2, 2, 6), (3, 2, 6)])
    far = _virtual_route(region, base_index, set(), {}, "c", [(1, 2, 8), (2, 2, 8), (3, 2, 8)])
    cong = Congestion()
    for net_id, route in (("a", a), ("b", b), ("c", far)):
        cong.add(net_id, route)
    assert cong.nets_at(cong.overused()) == {"a", "b"}


def test_congestion_cost_grows_with_history_and_pres():
    cong = Congestion()
    pos = (1, 1, 1)
    assert cong.cost(pos, 1.0, 4.0) == 0
    cong.signal[pos].add("x")
    assert cong.cost(pos, 2.0, 4.0) > cong.cost(pos, 1.0, 4.0) > 0
    before = cong.cost(pos, 1.0, 4.0)
    cong.hist[pos] += 1
    assert cong.cost(pos, 1.0, 4.0) > before
