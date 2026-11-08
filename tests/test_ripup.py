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


def test_a_finished_nets_journal_is_packed_and_reads_back_the_same():
    from minecraft_v.placement_engine.ripup import PackedJournal

    rec, region, ws, solid, dust_owner, inverted = _setup()
    path = [(1, 2, 1), (2, 2, 1), (3, 3, 1), (4, 3, 1), (6, 7, 1), (6, 8, 1)]
    with rec.recording("a"):
        _lay_redstone_path(ws, solid, dust_owner, path, "a", inverted_cells=inverted)
        written = list(rec.journals["a"])
        assert type(rec.journals["a"]) is list  # while recording it is the list the compiled layer appends to
    assert type(dict.get(rec.journals, "a")) is PackedJournal and len(rec.journals) == 1
    assert rec.journals["a"] == written  # the same entries, targets and previous values included
    assert type(dict.get(rec.journals, "a")) is list  # read: thawed, to be packed again when it's next recorded
    with rec.recording("a"):
        pass
    assert type(dict.get(rec.journals, "a")) is PackedJournal
    before = _state(region, solid, dust_owner, inverted)
    rip_up(rec, "a", region, solid, dust_owner)
    assert "a" not in rec.journals and _state(region, solid, dust_owner, inverted) != before
    assert not dust_owner and not solid


def test_cell_set_matches_the_cells_of_its_boxes():
    from minecraft_v.placement_engine.cellset import CellSet

    boxes = [((0, 0, 0), (2, 3, 1)), ((5, 1, 7), (1, 1, 1)), ((1, 0, 0), (2, 2, 2)), ((9, 9, 9), (0, 4, 4))]
    expected = {(ox + dx, oy + dy, oz + dz) for (ox, oy, oz), (w, h, d) in boxes
                for dx in range(w) for dy in range(h) for dz in range(d)}
    cells = CellSet.from_boxes(boxes)
    assert set(cells) == expected and len(cells) == len(expected) and bool(cells)
    assert all(c in cells for c in expected) and (9, 9, 9) not in cells and (6, 1, 7) not in cells
    assert not CellSet.from_boxes([])


def test_containers_keep_their_cells_in_the_grid_and_the_rest_in_themselves():
    """Once a CellGrid mirrors them, cells it covers live only in its words; cells beyond it stay in the container."""
    from litemapy import Region
    from minecraft_v.placement_engine.pathfinding import IndexedWorkspace

    rec = Recorder()
    region = Region(0, 0, 0, 6, 6, 6)
    solid, dust_owner, inverted = JournaledSet(rec, name="solid"), JournaledDict(rec), JournaledSet(rec, name="inverted")
    inside, outside = (1, 1, 1), (900, 1, 1)
    solid.add(inside), solid.add(outside), dust_owner.__setitem__(inside, "a"), dust_owner.__setitem__(outside, "b")
    ws = IndexedWorkspace(region)
    ws.attach_grid(solid, dust_owner, inverted, frozenset(), frozenset())
    assert solid._grid is not None and dust_owner._grid is not None
    assert solid._back == {outside} and dust_owner._back == {outside: "b"}  # the cells in the grid moved there
    assert inside in solid and outside in solid and (2, 2, 2) not in solid and len(solid) == 2
    assert set(solid) == {inside, outside}
    assert dust_owner.get(inside) == "a" and dust_owner[outside] == "b" and dust_owner.get((2, 2, 2)) is None
    assert dict(dust_owner.items()) == {inside: "a", outside: "b"} and len(dust_owner) == 2 and inside in dust_owner
    with rec.recording("n"):
        dust_owner[inside] = "n"
        solid.discard(inside)
        assert dust_owner.pop(outside) == "b" and outside not in dust_owner
    rip_up(rec, "n", region, solid, dust_owner)
    assert dict(dust_owner.items()) == {inside: "a", outside: "b"} and set(solid) == {inside, outside}
    assert dust_owner.pop(inside) == "a" and inside not in dust_owner and dust_owner.pop(inside, None) is None


def test_pos_map_is_a_dict_of_cells():
    from minecraft_v.placement_engine.pathfinding import _astar

    m = _astar.PosMap()
    m[(1, 2, 3)] = "a"
    m[(-5, 0, 7)] = "b"
    m[(1, 2, 3)] = "c"
    assert m.get((1, 2, 3)) == "c" and m[(-5, 0, 7)] == "b" and m.get((9, 9, 9), "x") == "x" and len(m) == 2
    assert dict(m.items()) == {(1, 2, 3): "c", (-5, 0, 7): "b"} and (1, 2, 3) in m and (0, 0, 0) not in m
    assert m.pop((1, 2, 3)) == "c" and (1, 2, 3) not in m and m.pop((1, 2, 3), None) is None and len(m) == 1
    m[(1, 2, 3)] = "d"  # a removed cell comes back
    assert dict(m.items()) == {(1, 2, 3): "d", (-5, 0, 7): "b"} and dict(m.copy().items()) == dict(m.items())


def test_cell_set_as_the_hidden_cells_of_signal_bytes_matches_a_frozenset():
    import numpy as np
    from litemapy import BlockState
    from minecraft_v.placement_engine.cellset import CellSet, pack
    from minecraft_v.placement_engine.constants import REDSTONE
    from minecraft_v.placement_engine.verify import HiddenBoxes, signal_bytes

    region = Region(0, 0, 0, 8, 4, 8)
    for x in range(8):
        region[x, 1, 3] = REDSTONE
    cells = {(2, 1, 3), (3, 1, 3), (5, 1, 3), (20, 1, 3)}  # the last is outside the region
    box = [((6, 0, 0), (8, 4, 8))]
    plain = signal_bytes(region, HiddenBoxes(frozenset(cells), box), frozenset())
    packed = signal_bytes(region, HiddenBoxes(CellSet.from_cells(cells), box), frozenset())
    assert plain is not None and np.array_equal(plain, packed)
    only = CellSet.from_boxes([((2, 1, 3), (2, 1, 1))])
    assert np.array_equal(signal_bytes(region, frozenset({(2, 1, 3), (3, 1, 3)}), frozenset()), signal_bytes(region, only, frozenset()))
    both = CellSet.from_cells({(1, 2, 3), (9, 9, 9)}).union(CellSet.from_cells({(4, 5, 6), (1, 2, 3)}))
    assert sorted(both) == [(1, 2, 3), (4, 5, 6), (9, 9, 9)]
    xs, ys, zs = next(both.chunks(2))
    assert len(xs) == 2
    assert CellSet.from_cells({(1, 2, 3)}).with_keys([pack(4, 5, 6), pack(1, 2, 3)]).cells_array().tolist() == [[1, 2, 3], [4, 5, 6]]


def test_cells_in_boxes_near_gives_the_same_cells_however_it_is_chunked(monkeypatch):
    import numpy as np
    from litemapy import BlockState
    from minecraft_v.placement_engine import block_utils

    rng = np.random.default_rng(3)
    region = Region(0, 0, 0, 14, 8, 14)
    for _ in range(300):
        region[int(rng.integers(14)), int(rng.integers(8)), int(rng.integers(14))] = STONE
    boxes = [((3, 1, 3), (7, 5, 7)), ((9, 2, 8), (13, 6, 12))]
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 0, 2), (0, -1, 1), (2, 1, -2)]
    whole = block_utils.cells_in_boxes_near(region, boxes, 2, offsets)
    monkeypatch.setattr(block_utils, "_NEAR_CHUNK", 7)
    assert np.array_equal(block_utils.cells_in_boxes_near(region, boxes, 2, offsets), whole) and len(whole)


def test_owner_overlay_packs_blocks_and_reads_like_the_dict_it_stands_for():
    import numpy as np
    from minecraft_v.placement_engine.verify import OwnerOverlay

    base = {(1, 1, 1): "a", (2, 2, 2): "b", (5, 5, 5): "z"}
    owner = OwnerOverlay(base)
    owner[(9, 9, 9)] = "x"
    owner.add_block(np.array([[2, 2, 2], [3, 3, 3], [3, 3, 3], [4, 4, 4]]), ["n1", "n2"], np.array([0, 0, 1, 1]))
    expected = {(1, 1, 1): "a", (2, 2, 2): "n1", (3, 3, 3): "n2", (4, 4, 4): "n2", (5, 5, 5): "z", (9, 9, 9): "x"}
    assert dict(owner.items()) == expected and len(owner) == len(expected)
    assert owner.get((3, 3, 3)) == "n2" and owner.get((7, 7, 7)) is None and (4, 4, 4) in owner and (8, 8, 8) not in owner
    assert sorted(owner.values()) == sorted(expected.values()) and set(owner) == set(expected)


def test_fork_batches_splits_in_runs_and_runs_few_items_here():
    from minecraft_v.placement_engine.forks import fork_batches, fork_map

    items = list(range(300))
    runs = fork_batches(lambda run: (len(run), sum(run)), items, 3)
    assert len(runs) == 3 and sum(n for n, _ in runs) == 300 and sum(s for _, s in runs) == sum(items)
    assert fork_batches(lambda run: len(run), items[:5], 3) == [5]
    assert fork_map(lambda x: x * x, items, 3) == [x * x for x in items]
