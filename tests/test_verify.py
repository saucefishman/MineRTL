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
from minecraft_v.placement_engine.verify import Violation, find_inverted_contacts, verify_signals, verify_support
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


def test_repeater_feeding_itself_is_a_latch():
    region, owner = _region(), {}
    _wire(region, owner, "a", [(1, 1, 1), (2, 1, 1)])
    region[3, 1, 1] = _repeater((-1, 0))
    _wire(region, owner, "a", [(4, 1, 1), (4, 1, 2), (3, 1, 2), (2, 1, 2), (1, 1, 2)])
    violations = verify_signals(region, {"a": ((1, 1, 1), [(4, 1, 2)])}, owner)
    assert any(v.kind == "latch" for v in violations)


def test_signal_running_out_is_weak():
    region, owner = Region(0, 0, 0, 20, 4, 4), {}
    wire = _line_x(0, 18, 1, 1)
    _wire(region, owner, "a", wire)
    assert "weak" in _kinds(verify_signals(region, {"a": (wire[0], [wire[-1]])}, owner))
    assert "weak" not in _kinds(verify_signals(region, {"a": (wire[0], [wire[14]])}, owner))


# ---------------------------------------------------------------- polarity

def _tower_net(region: Region, owner: dict) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Source wire, a tower going +x and wire off its top; returns (source, sink)."""
    solid: set = set()
    _wire(region, owner, "n", _line_x(1, 2, 1, 5))
    _lay_redstone_path(region, solid, owner, [(2, 1, 5), (4, 5, 5), (5, 5, 5), (6, 5, 5)], "n")
    return (1, 1, 5), (6, 5, 5)


def test_tower_alone_has_no_inverted_contact():
    region, owner = _region(), {}
    src, sink = _tower_net(region, owner)
    assert verify_signals(region, {"n": (src, [sink])}, owner) == []


def test_wire_beside_a_towers_inverted_block_is_flagged():
    region, owner = _region(), {}
    src, sink = _tower_net(region, owner)
    # Own wire back down from the tower top, ending against the stone at y=3.
    side = _wire(region, owner, "n", [(7, 4, 5), (6, 3, 5), (5, 3, 5)])
    region[6, 4, 5] = GLASS
    violations = verify_signals(region, {"n": (src, [sink])}, owner)
    assert "inverted" in _kinds(violations)
    assert side[-1] in find_inverted_contacts(region, src, [sink], frozenset())


def test_sink_fed_only_through_the_inverted_block_is_flagged():
    region, owner = _region(), {}
    src, _ = _tower_net(region, owner)
    sink = _wire(region, owner, "n", [(5, 3, 5)])[0]
    region[5, 4, 5] = GLASS
    assert sink in find_inverted_contacts(region, src, [sink], frozenset())


def test_gate_contact_finds_dust_pointing_into_a_gate_block():
    from litemapy import BlockState, Region

    from minecraft_v.placement_engine.verify import find_gate_contacts

    region = Region(0, 0, 0, 8, 4, 4)
    region[3, 1, 1] = BlockState("minecraft:smooth_quartz")
    for x in (1, 2):  # runs east and ends against the block
        region[x, 1, 1] = BlockState("minecraft:redstone_wire")
    region[1, 1, 3] = BlockState("minecraft:redstone_wire")  # far from it
    assert find_gate_contacts(region, frozenset({(1, 1, 1), (2, 1, 1), (1, 1, 3)}), frozenset({(3, 1, 1)})) == [(2, 1, 1)]
    assert find_gate_contacts(region, frozenset({(1, 1, 1), (2, 1, 1)}), frozenset()) == []


def test_hold_arrival_counts_a_repeaters_delay():
    from litemapy import BlockState, Region

    from minecraft_v.placement_engine.hold import wire_arrivals
    from minecraft_v.placement_engine.verify import _World

    region = Region(0, 0, 0, 8, 4, 4)
    for x in (1, 3):
        region[x, 1, 1] = BlockState("minecraft:redstone_wire")
    region[2, 1, 1] = BlockState("minecraft:repeater", facing="west", delay="3")
    arrival, _ = wire_arrivals(_World(region, frozenset(), frozenset()), (1, 1, 1), frozenset())
    assert arrival[(3, 1, 1)] == 3


def test_hold_finds_clock_inputs_through_a_gate():
    # examples/memread: _clk & switch drives every clock pin, and hold fixing used to skip the design without a word.
    from types import SimpleNamespace as NS

    from litemapy import Region

    from minecraft_v.placement_engine.hold import _HoldTiming
    from minecraft_v.placement_engine.ir import ComponentType as T
    from minecraft_v.placement_engine.verify import _World

    def end(cid, pin):
        return NS(component_id=cid, pin_name=pin)

    kinds = {"clk": T.INPUT_PIN, "en": T.INPUT_PIN, "data": T.INPUT_PIN, "ff": T.DFF, "gate": T.AND, "div": T.DFF, "ff2": T.DFF}
    comp = NS(components=[NS(id=c, type=k) for c, k in kinds.items()], nets=[
        NS(net_id="a", source=end("clk", "Y"), sinks=[end("gate", "A"), end("div", "C")]),
        NS(net_id="b", source=end("en", "Y"), sinks=[end("gate", "B")]),
        NS(net_id="c", source=end("gate", "Y"), sinks=[end("ff", "C")]),
        NS(net_id="d", source=end("data", "Y"), sinks=[end("ff", "D")]),
        NS(net_id="e", source=end("div", "Q"), sinks=[end("ff2", "C")]),  # a flip-flop stops the walk back
    ])
    timing = _HoldTiming(_World(Region(0, 0, 0, 2, 2, 2), frozenset(), frozenset()), comp, {}, set())
    assert timing.roots == {"clk", "en"}


def test_wire_on_a_live_pin_with_no_net_is_a_violation():
    from litemapy import BlockState, Region

    from minecraft_v.placement_engine.verify import find_live_pin_wire

    region = Region(0, 0, 0, 8, 4, 4)
    region[1, 1, 1] = BlockState("minecraft:redstone_wire")
    region[2, 1, 1] = BlockState("minecraft:red_wool")  # bpmem, t1: a repeater strongly powers a wire's support
    region[3, 1, 1] = BlockState("minecraft:glass")
    pins = {"dust": (1, 1, 1), "wool": (2, 1, 1), "glass": (3, 1, 1), "clear": (4, 1, 1), "outside": (9, 1, 1)}
    found = find_live_pin_wire(region, pins)
    assert [(v.kind, sorted(v.nets), v.pos) for v in found] == [("live_pin", ["dust"], (1, 1, 1)), ("live_pin", ["wool"], (2, 1, 1))]


def test_hold_only_converts_dust_on_a_straight_run():
    from litemapy import BlockState, Region

    from minecraft_v.placement_engine.hold import _straight_run
    from minecraft_v.placement_engine.verify import _World

    region = Region(0, 0, 0, 8, 4, 6)
    for x in (1, 2, 3):
        region[x, 1, 1] = BlockState("minecraft:redstone_wire")
    region[2, 1, 2] = BlockState("minecraft:redstone_wire")  # a branch at the middle
    for x in (1, 2, 3):
        region[x, 1, 4] = BlockState("minecraft:redstone_wire")
    world = _World(region, frozenset(), frozenset())
    assert _straight_run(world, (2, 1, 1)) is None
    assert set(_straight_run(world, (2, 1, 4))) == {(1, 1, 4), (3, 1, 4)}


def test_hold_keeps_dust_that_powers_a_torch_below():
    # cpu net_118: a repeater on a powered -4 drop's launch cell leaves the drop's torch stuck.
    from litemapy import BlockState, Region

    from minecraft_v.placement_engine.hold import _straight_run
    from minecraft_v.placement_engine.verify import _World

    region = Region(0, 0, 0, 8, 6, 6)
    for z in (1, 4):
        for x in (1, 2, 3):
            region[x, 3, z] = BlockState("minecraft:redstone_wire")
        region[2, 2, z] = BlockState("minecraft:stone")
    region[2, 2, 2] = BlockState("minecraft:redstone_wall_torch", facing="south")  # on the block under (2, 3, 1)
    region[2, 2, 5] = BlockState("minecraft:redstone_wall_torch", facing="north")  # on another block's far side
    region[2, 1, 4] = BlockState("minecraft:stone")
    world = _World(region, frozenset(), frozenset())
    assert _straight_run(world, (2, 3, 1)) is None
    assert set(_straight_run(world, (2, 3, 4))) == {(1, 3, 4), (3, 3, 4)}


def test_world_successors_follow_a_changed_block():
    from minecraft_v.placement_engine.verify import _World
    region = Region(0, 0, 0, 5, 3, 3)
    for x in range(5):
        region[x, 0, 1] = BlockState("minecraft:stone")
        region[x, 1, 1] = BlockState("minecraft:redstone_wire")
    world = _World(region, frozenset(), frozenset())
    assert ((3, 1, 1), "") in world.successors(((2, 1, 1), ""))
    assert ((1, 1, 1), "") in world.successors(((2, 1, 1), ""))

    region[2, 1, 1] = BlockState("minecraft:repeater", facing=_DELTA_TO_FACING[(-1, 0)], delay="1")
    world.changed((2, 1, 1))
    assert world.successors(((2, 1, 1), "")) == [((3, 1, 1), "")]
    assert ((2, 1, 1), "") not in world.successors(((3, 1, 1), ""))


def _torch_drop(region: Region, owner: dict) -> None:
    """Wire into a block, torch, wire into a block, torch: same polarity out as in. Output cell (7, 3, 8)."""
    _wire(region, owner, "a", _line_x(2, 5, 3, 5))
    region[6, 3, 5] = STONE
    region[7, 3, 5] = BlockState("minecraft:redstone_wall_torch", facing="east", lit="true")
    region[7, 2, 5] = STONE
    _wire(region, owner, "a", _line_z(5, 6, 3, 8))
    region[8, 3, 7] = STONE
    region[8, 3, 8] = BlockState("minecraft:redstone_wall_torch", facing="south", lit="false")
    region[8, 2, 8] = STONE


def test_torch_drop_feeding_its_own_input_is_a_latch():
    """hf3's net_1820: the wire after a drop joined the wire before it, and stuck high from power-up."""
    region, owner = _region(), {}
    _torch_drop(region, owner)
    _wire(region, owner, "a", _line_x(2, 7, 3, 8) + _line_z(6, 7, 3, 2))
    violations = verify_signals(region, {"a": ((2, 3, 5), [(4, 3, 5)])}, owner)
    assert any(v.kind == "latch" for v in violations)


def test_torch_drop_going_on_is_not_a_latch():
    region, owner = _region(), {}
    _torch_drop(region, owner)
    _wire(region, owner, "a", _line_x(4, 7, 3, 8))
    violations = verify_signals(region, {"a": ((2, 3, 5), [(4, 3, 8)])}, owner)
    assert not any(v.kind == "latch" for v in violations)


def test_repeater_across_a_pin_cell_is_open():
    """hf3's net_2367 and three nets on h1: wire ran past a gate's pin with a repeater on the pin's cell, and the gate
    never switched."""
    region, owner = _region(), {}
    _wire(region, owner, "a", _line_x(2, 3, 3, 5) + _line_x(5, 6, 3, 5))
    region[4, 3, 5] = _repeater((-1, 0))
    region[4, 2, 5] = STONE
    region[4, 3, 4] = _repeater((0, 1))  # the gate's input, hidden from verify
    violations = verify_signals(region, {"a": ((2, 3, 5), [(4, 3, 5), (6, 3, 5)])}, owner, blocked=frozenset({(4, 3, 4)}))
    assert [v.pos for v in violations if v.kind == "open"] == [(4, 3, 5)]


def test_repeater_across_a_pin_cell_is_open_with_the_input_in_view():
    """Compact gates leave their input repeater outside the footprint, so verify sees it, unfed."""
    region, owner = _region(), {}
    _wire(region, owner, "a", _line_x(2, 3, 3, 5) + _line_x(5, 6, 3, 5))
    region[4, 3, 5] = _repeater((-1, 0))
    region[4, 2, 5] = STONE
    region[4, 3, 4] = _repeater((0, 1))
    region[4, 2, 4] = STONE
    violations = verify_signals(region, {"a": ((2, 3, 5), [(4, 3, 5), (6, 3, 5)])}, owner)
    assert [v.pos for v in violations if v.kind == "open"] == [(4, 3, 5)]


def test_repeater_into_a_pin_is_not_open():
    region, owner = _region(), {}
    _wire(region, owner, "a", _line_z(6, 7, 3, 4))
    region[4, 3, 5] = _repeater((0, 1))
    region[4, 2, 5] = STONE
    region[4, 3, 4] = _repeater((0, 1))
    violations = verify_signals(region, {"a": ((4, 3, 7), [(4, 3, 5)])}, owner, blocked=frozenset({(4, 3, 4)}))
    assert not violations
