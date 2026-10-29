"""Tests for verify.verify_signals — per-net signal connectivity and short detection.

Model (Java redstone, simplified to what the router builds):
- Dust connects to horizontally adjacent dust, and diagonally up/down one Y unless an
  opaque block sits directly above the lower dust (glass/air don't cut).
- Dust weakly powers the block below it and blocks it points into (isolated dust = cross,
  one connection = line pointing both ways, 2+ connections = only those directions).
- Weakly powered blocks only feed torches/repeaters; strongly powered blocks (repeater
  output, torch underneath) also feed dust on all six faces.
- Repeater (facing = input direction): input from behind, output in front; sides isolated.
- Torch: input from its attachment block; strongly powers the block above, feeds adjacent dust.
- Short: a cell reached from two nets' sources, or another net's owned cell reached.
  Open: a sink its own source can't reach.
"""
from litemapy import BlockState, Region

from minecraft_v.placement_engine.constants import GLASS, REDSTONE, REDSTONE_TORCH, STONE, _DELTA_TO_FACING
from minecraft_v.placement_engine.verify import Violation, verify_signals, verify_support
from minecraft_v.placement_engine.wire import _lay_redstone_path


def _region() -> Region:
    return Region(0, 0, 0, 16, 14, 16)


def _wire(region: Region, owner: dict, net: str, cells) -> list:
    """Dust on each cell with stone under it (unless something is already there)."""
    cells = list(cells)
    for x, y, z in cells:
        region[x, y, z] = REDSTONE
        owner[(x, y, z)] = net
        if str(region[x, y - 1, z]) == "minecraft:air":
            region[x, y - 1, z] = STONE
    return cells


def _repeater(input_delta: tuple[int, int]) -> BlockState:
    return BlockState("minecraft:repeater", facing=_DELTA_TO_FACING[input_delta], delay="1")


def _line_x(x0: int, x1: int, y: int, z: int):
    return [(x, y, z) for x in range(x0, x1 + 1)]


def _line_z(z0: int, z1: int, y: int, x: int):
    return [(x, y, z) for z in range(z0, z1 + 1)]


def _kinds(violations: list[Violation]) -> set[str]:
    return {v.kind for v in violations}


def _shorts(violations: list[Violation]) -> set[frozenset[str]]:
    return {v.nets for v in violations if v.kind == "short"}


# ---------------------------------------------------------------- connectivity

def test_straight_wire_is_clean():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", _line_x(1, 8, 2, 5))
    assert verify_signals(region, {"a": (a[0], [a[-1]])}, owner) == []


def test_gap_in_wire_is_open():
    region, owner = _region(), {}
    left = _wire(region, owner, "a", _line_x(1, 3, 2, 5))
    right = _wire(region, owner, "a", _line_x(5, 8, 2, 5))
    violations = verify_signals(region, {"a": (left[0], [right[-1]])}, owner)
    assert _kinds(violations) == {"open"}
    assert violations[0].nets == frozenset({"a"})
    assert violations[0].pos == right[-1]


def test_multiple_sinks_each_checked():
    region, owner = _region(), {}
    trunk = _wire(region, owner, "a", _line_x(1, 6, 2, 5))
    branch = _wire(region, owner, "a", _line_z(6, 9, 2, 4))  # branches off the trunk at (4,2,5)
    detached = _wire(region, owner, "a", _line_z(8, 10, 2, 10))
    violations = verify_signals(region, {"a": (trunk[0], [trunk[-1], branch[-1], detached[-1]])}, owner)
    assert {v.pos for v in violations if v.kind == "open"} == {detached[-1]}


def test_slope_up_connects():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", [(1, 2, 5), (2, 2, 5), (3, 3, 5), (4, 3, 5)])
    assert verify_signals(region, {"a": (a[0], [a[-1]])}, owner) == []


def test_opaque_block_above_lower_dust_cuts_slope():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", [(1, 2, 5), (2, 2, 5), (3, 3, 5), (4, 3, 5)])
    region[2, 3, 5] = STONE  # directly above the lower dust of the slope
    violations = verify_signals(region, {"a": (a[0], [a[-1]])}, owner)
    assert _kinds(violations) == {"open"}


def test_glass_above_lower_dust_does_not_cut_slope():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", [(1, 2, 5), (2, 2, 5), (3, 3, 5), (4, 3, 5)])
    region[2, 3, 5] = GLASS
    assert verify_signals(region, {"a": (a[0], [a[-1]])}, owner) == []


def test_repeater_passes_signal_forward_only():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", [(1, 2, 5), (2, 2, 5)])
    region[3, 2, 5] = _repeater((-1, 0))  # input from -x, output to +x
    region[3, 1, 5] = STONE
    tail = _wire(region, owner, "a", [(4, 2, 5), (5, 2, 5)])
    assert verify_signals(region, {"a": (a[0], [tail[-1]])}, owner) == []
    # Driven from the output side, the repeater must not pass signal backwards.
    backwards = verify_signals(region, {"a": (tail[-1], [a[0]])}, owner)
    assert _kinds(backwards) == {"open"}


# ---------------------------------------------------------------- shorts

def test_adjacent_parallel_wires_short():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", _line_x(1, 8, 2, 5))
    b = _wire(region, owner, "b", _line_x(1, 8, 2, 6))
    violations = verify_signals(region, {"a": (a[0], [a[-1]]), "b": (b[0], [b[-1]])}, owner)
    assert _shorts(violations) == {frozenset({"a", "b"})}


def test_parallel_wires_with_gap_are_clean():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", _line_x(1, 8, 2, 5))
    b = _wire(region, owner, "b", _line_x(1, 8, 2, 7))
    assert verify_signals(region, {"a": (a[0], [a[-1]]), "b": (b[0], [b[-1]])}, owner) == []


def test_diagonal_slope_into_other_net_shorts():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", _line_x(1, 8, 2, 5))
    b = _wire(region, owner, "b", _line_x(1, 8, 3, 6))  # one up, one over: slope link
    violations = verify_signals(region, {"a": (a[0], [a[-1]]), "b": (b[0], [b[-1]])}, owner)
    assert _shorts(violations) == {frozenset({"a", "b"})}


def test_wire_crossing_over_on_support_is_clean():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", _line_x(1, 8, 2, 5))
    # b crosses above a at y=4; its support at (4,3,5) sits on top of a's dust at (4,2,5).
    b = _wire(region, owner, "b", _line_z(2, 8, 4, 4))
    assert verify_signals(region, {"a": (a[0], [a[-1]]), "b": (b[0], [b[-1]])}, owner) == []


def test_torch_under_other_nets_support_shorts():
    region, owner = _region(), {}
    b = _wire(region, owner, "b", _line_z(3, 7, 3, 5))  # support at (5,2,5)
    region[5, 1, 5] = REDSTONE_TORCH  # strongly powers (5,2,5) → b's dust on top
    region[5, 0, 5] = STONE
    owner[(5, 1, 5)] = "a"
    violations = verify_signals(
        region, {"a": ((5, 1, 5), []), "b": (b[0], [b[-1]])}, owner)
    assert _shorts(violations) == {frozenset({"a", "b"})}


def test_repeater_into_block_under_other_net_shorts():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", [(1, 3, 5), (2, 3, 5)])
    region[3, 3, 5] = _repeater((-1, 0))  # output into (4,3,5)
    region[3, 2, 5] = STONE
    owner[(3, 3, 5)] = "a"
    b = _wire(region, owner, "b", _line_z(3, 7, 4, 4))  # support (4,3,5) is a's repeater target
    violations = verify_signals(region, {"a": (a[0], []), "b": (b[0], [b[-1]])}, owner)
    assert _shorts(violations) == {frozenset({"a", "b"})}


def test_repeater_side_does_not_short():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", [(1, 2, 5), (2, 2, 5)])
    region[3, 2, 5] = _repeater((-1, 0))
    region[3, 1, 5] = STONE
    a_tail = _wire(region, owner, "a", [(4, 2, 5), (5, 2, 5)])
    b = _wire(region, owner, "b", _line_z(6, 9, 2, 3))  # b touches the repeater's side at (3,2,6)
    violations = verify_signals(region, {"a": (a[0], [a_tail[-1]]), "b": (b[0], [b[-1]])}, owner)
    assert violations == []


def test_wire_running_past_block_does_not_power_it():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", _line_x(1, 8, 2, 5))
    # b's block beside a's straight run, with b's wall torch on the far side reading it.
    region[4, 2, 6] = STONE
    region[4, 2, 7] = BlockState("minecraft:redstone_wall_torch", facing=_DELTA_TO_FACING[(0, 1)])
    owner[(4, 2, 7)] = "b"
    b = _wire(region, owner, "b", _line_x(4, 7, 2, 8))  # torch feeds b's dust at (4,2,8)
    violations = verify_signals(region, {"a": (a[0], [a[-1]]), "b": ((4, 2, 7), [b[-1]])}, owner)
    assert violations == []


def test_wire_ending_into_block_powers_it():
    region, owner = _region(), {}
    a = _wire(region, owner, "a", _line_z(1, 5, 2, 4))  # line ends pointing into (4,2,6)
    region[4, 2, 6] = STONE
    region[4, 2, 7] = BlockState("minecraft:redstone_wall_torch", facing=_DELTA_TO_FACING[(0, 1)])
    owner[(4, 2, 7)] = "b"
    b = _wire(region, owner, "b", _line_x(4, 7, 2, 8))
    violations = verify_signals(region, {"a": (a[0], [a[-1]]), "b": ((4, 2, 7), [b[-1]])}, owner)
    assert _shorts(violations) == {frozenset({"a", "b"})}


def test_reaching_unsourced_owned_cell_is_short():
    """Another net's wire counts even if that net isn't in `nets` (e.g. it failed to route)."""
    region, owner = _region(), {}
    a = _wire(region, owner, "a", _line_x(1, 8, 2, 5))
    _wire(region, owner, "b", _line_x(1, 3, 2, 6))
    violations = verify_signals(region, {"a": (a[0], [a[-1]])}, owner)
    assert _shorts(violations) == {frozenset({"a", "b"})}


def test_blocked_cells_are_ignored():
    """Component internals (blocked) aren't traversed or reported, except terminals."""
    region, owner = _region(), {}
    a = _wire(region, owner, "a", _line_x(1, 8, 2, 5))
    region[4, 2, 6] = REDSTONE  # component-internal dust beside a's wire
    owner[(4, 2, 6)] = "component_internal"
    violations = verify_signals(region, {"a": (a[0], [a[-1]])}, owner, blocked=frozenset({(4, 2, 6)}))
    assert violations == []


# ---------------------------------------------------------------- router move geometry

def _laid(path, net="a"):
    region, owner, solid, inverted = _region(), {}, set(), set()
    _lay_redstone_path(region, solid, owner, path, net, inverted_cells=inverted)
    return region, owner


def test_tower_move_connects():
    path = [(1, 2, 5), (2, 2, 5), (4, 6, 5), (4, 7, 5), (5, 7, 5)]
    region, owner = _laid(path)
    assert verify_signals(region, {"a": (path[0], [path[-1]])}, owner) == []


def test_double_slope_move_connects():
    path = [(1, 2, 5), (2, 2, 5), (4, 4, 5), (5, 4, 5)]
    region, owner = _laid(path)
    assert verify_signals(region, {"a": (path[0], [path[-1]])}, owner) == []


def test_powered_minus4_move_connects():
    path = [(4, 9, 5), (5, 9, 5), (5, 5, 5), (6, 5, 5)]
    region, owner = _laid(path)
    assert verify_signals(region, {"a": (path[0], [path[-1]])}, owner) == []


def test_laid_moves_are_supported():
    for path in ([(1, 2, 5), (2, 2, 5), (4, 6, 5), (4, 7, 5), (5, 7, 5)],
                 [(1, 2, 5), (2, 2, 5), (4, 4, 5), (5, 4, 5)],
                 [(4, 9, 5), (5, 9, 5), (5, 5, 5), (6, 5, 5)]):
        region, owner = _laid(path)
        assert verify_support(region, owner) == []


# ---------------------------------------------------------------- support

def test_dust_on_stone_or_glass_is_supported():
    region, owner = _region(), {}
    _wire(region, owner, "a", _line_x(1, 3, 2, 5))
    region[2, 1, 5] = GLASS
    assert verify_support(region, owner) == []


def test_floating_dust_is_unsupported():
    region, owner = _region(), {}
    region[3, 4, 5] = REDSTONE
    owner[(3, 4, 5)] = "a"
    assert verify_support(region, owner) == [Violation("unsupported", frozenset({"a"}), (3, 4, 5))]


def test_dust_on_repeater_is_unsupported():
    region, owner = _region(), {}
    region[3, 1, 5] = STONE
    region[3, 2, 5] = _repeater((1, 0))
    region[3, 3, 5] = REDSTONE
    owner[(3, 3, 5)] = "a"
    assert verify_support(region, owner) == [Violation("unsupported", frozenset({"a"}), (3, 3, 5))]


def test_wall_torch_needs_block_behind():
    region, owner = _region(), {}
    region[3, 2, 5] = BlockState("minecraft:redstone_wall_torch", facing="east", lit="true")
    assert [v.pos for v in verify_support(region, owner)] == [(3, 2, 5)]
    region[2, 2, 5] = STONE
    assert verify_support(region, owner) == []


def test_blocked_cells_skip_support_check():
    region, owner = _region(), {}
    region[3, 4, 5] = REDSTONE
    assert verify_support(region, owner, blocked=frozenset({(3, 4, 5)})) == []
