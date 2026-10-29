from litemapy import Region

from minecraft_v.placement_engine.constants import STONE
from minecraft_v.placement_engine.ripup import (
    Recorder, JournaledSet, JournaledDict, JournalingWorkspace, rip_up, rollback, invalid_supports,
)
from minecraft_v.placement_engine.wire import LayConflict, _lay_redstone_path, _lay_dust_cell


def _state(region: Region, solid, dust_owner, inverted):
    blocks = {
        (x, y, z): str(region[x, y, z])
        for x in range(region.width) for y in range(region.height) for z in range(region.length)
    }
    return blocks, set(solid), dict(dust_owner), set(inverted)


def _setup():
    rec = Recorder()
    region = Region(0, 0, 0, 12, 10, 12)
    solid = JournaledSet(rec, name="solid")
    dust_owner = JournaledDict(rec)
    inverted = JournaledSet(rec, name="inverted")
    return rec, region, JournalingWorkspace(region, rec), solid, dust_owner, inverted


def test_rip_up_restores_pre_state():
    rec, region, ws, solid, dust_owner, inverted = _setup()
    # Pre-existing (unjournaled) content must survive.
    region[0, 0, 0] = STONE
    solid.add((0, 0, 0))
    before = _state(region, solid, dust_owner, inverted)

    # Flat run, slope up, tower move (+4Y over ±2X), then flat.
    path = [(1, 2, 1), (2, 2, 1), (3, 3, 1), (4, 3, 1), (6, 7, 1), (6, 8, 1)]
    with rec.recording("a"):
        _lay_redstone_path(ws, solid, dust_owner, path, "a", inverted_cells=inverted)
    assert _state(region, solid, dust_owner, inverted) != before
    assert not invalid_supports(region, solid, dust_owner)

    rip_up(rec, "a", region, solid, dust_owner)
    assert _state(region, solid, dust_owner, inverted) == before
    assert "a" not in rec.journals


def test_rip_up_keeps_support_used_by_other_net():
    rec, region, ws, solid, dust_owner, inverted = _setup()
    before = _state(region, solid, dust_owner, inverted)
    support = (5, 2, 5)
    with rec.recording("a"):
        ws[support] = STONE
        solid.add(support)
    with rec.recording("b"):
        # Lands on a's block — no new support placed for b.
        _lay_dust_cell(ws, solid, dust_owner, "b", (5, 3, 5), STONE)

    rip_up(rec, "a", region, solid, dust_owner)
    assert support in solid
    assert not invalid_supports(region, solid, dust_owner)

    # The kept support now belongs to b and goes away with it.
    rip_up(rec, "b", region, solid, dust_owner)
    assert _state(region, solid, dust_owner, inverted) == before


def test_rollback_undoes_only_failed_attempt():
    rec, region, ws, solid, dust_owner, inverted = _setup()
    with rec.recording("a"):
        _lay_redstone_path(ws, solid, dust_owner, [(1, 2, 1), (2, 2, 1)], "a", inverted_cells=inverted)
    kept = _state(region, solid, dust_owner, inverted)
    mark = len(rec.journals["a"])
    # Path that comes back over its own wire: the last cell's support would be (3, 2, 1).
    bad = [(2, 2, 1), (3, 2, 1), (4, 3, 1), (4, 3, 2), (3, 3, 2), (3, 3, 1)]
    with rec.recording("a"):
        try:
            _lay_redstone_path(ws, solid, dust_owner, bad, "a", inverted_cells=inverted)
        except LayConflict as e:
            assert e.cells == ((3, 3, 1), (3, 2, 1))
        else:
            raise AssertionError("expected LayConflict")
        rollback(rec, "a", mark)
    assert _state(region, solid, dust_owner, inverted) == kept
    assert len(rec.journals["a"]) == mark
