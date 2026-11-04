"""Hold fixing: what a flip-flop really needs, and paths with no room for padding."""
import heapq
from types import SimpleNamespace as NS

import pytest
from litemapy import BlockState, Region, Schematic

from minecraft_v.cell_library import SCHEMATICS_DIR, SCHEMATIC_MAP
from minecraft_v.placement_engine import hold
from minecraft_v.placement_engine.block_utils import BulkRegion
from minecraft_v.placement_engine.builder import _GreedyRouter, _hold_with_detours, _RouteJob
from minecraft_v.placement_engine.constants import STONE
from minecraft_v.placement_engine.ir import ComponentType
from minecraft_v.placement_engine.ripup import JournaledDict, JournaledSet, Recorder
from minecraft_v.placement_engine.template import _non_air_bounds
from minecraft_v.placement_engine.tick_sim import TickSim

WIRE = BlockState("minecraft:redstone_wire", power="0")
_ORIGIN = (8, 4, 8)


def _flip_flop_bench(kind: ComponentType):
    """One flip-flop, repeaters driving its pins' cells from outside and dust on Q. Repeaters here face their input."""
    info = SCHEMATIC_MAP[kind]
    template = next(iter(Schematic.load(str(SCHEMATICS_DIR / f"{info.file_prefix}.litematic")).regions.values()))
    x0, y0, z0, x1, y1, z1 = _non_air_bounds(template)
    region = BulkRegion(0, 0, 0, 30, 10, 30)
    for x in range(30):
        for z in range(30):
            region[x, 2, z] = STONE
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                if template[x, y, z].id != "minecraft:air":
                    region[_ORIGIN[0] + x - x0, _ORIGIN[1] + y - y0, _ORIGIN[2] + z - z0] = template[x, y, z]
    at = {p.name: (_ORIGIN[0] + p.offset[0], _ORIGIN[1] + p.offset[1], _ORIGIN[2] + info.footprint.depth - 1 - p.offset[2])
          for p in info.pins}
    drivers = {}
    for pin, (x, y, z) in at.items():
        if pin == "Q":
            region[x, y, z - 1] = WIRE
        elif pin == "D":
            region[x, y, z + 1] = WIRE
            region[x, y, z + 2] = BlockState("minecraft:repeater", facing="south", delay="1")
            drivers[pin] = (x, y, z + 2)
        else:
            region[x + 1, y, z] = WIRE
            region[x + 2, y, z] = BlockState("minecraft:repeater", facing="east", delay="1")
            drivers[pin] = (x + 2, y, z)
    x, y, z = at["Q"]
    return region, drivers, (x, y, z - 1)


def _run_to(sim: TickSim, when: int) -> None:
    while sim.heap and sim.heap[0][0] < when:
        t, _, _, i = heapq.heappop(sim.heap)
        sim.now = t
        sim._tick(i)
    sim.now = when


def _latched_after_change(kind: ComponentType, pin: str, ticks: int, value: bool) -> tuple[bool, int]:
    """Q wanted `value`, then `pin` toggles `ticks` after C rises. Whether Q held it, and ticks from C to Q."""
    region, drivers, q = _flip_flop_bench(kind)
    sim = TickSim(region)
    cells = {name: sim.cell(p) for name, p in drivers.items()}
    out = sim.cell(q)
    sim.drive(cells["C"], False)
    if "E" in cells:
        sim.drive(cells["E"], True)
    sim.run()
    clock_to_q = 0
    for stored in (not value, value):
        sim.drive(cells["D"], stored)
        sim.run()
        start = sim.now
        sim.watching[out] = "q"
        sim.seen.clear()
        sim.drive(cells["C"], True)
        if stored == value:
            _run_to(sim, start + ticks)
            sim.drive(cells[pin], not stored if pin == "D" else False)
        sim.run()
        if ("q", stored) in sim.seen:
            clock_to_q = sim.seen["q", stored] - start
        sim.drive(cells["C"], False)
        sim.run()
    return (sim.power[out] > 0) == value, clock_to_q


@pytest.mark.parametrize("kind", [ComponentType.DFF, ComponentType.DFFE])
@pytest.mark.parametrize("value", [False, True])
def test_measured_flip_flop_timing(kind, value):
    need = hold._HOLD_TICKS[kind]
    assert not _latched_after_change(kind, "D", need - 1, value)[0]
    for ticks in range(need, need + 12):
        held, clock_to_q = _latched_after_change(kind, "D", ticks, value)
        assert held
    if clock_to_q:  # one direction of a plain DFF's Q changes only with the opposite stored
        assert clock_to_q == hold._CLOCK_TO_Q[kind]


def _end(cid, pin, terminals, pos):
    terminals[cid, pin] = pos
    return NS(component_id=cid, pin_name=pin)


def _skewed_world(region):
    """Two flip-flops with ff1's clock 20 ticks behind ff0's. The clock line runs along z = 50."""
    for x in range(2, 21):
        region[x, 1, 50] = STONE
        region[x, 2, 50] = BlockState("minecraft:repeater", facing="west", delay="4") if x in (5, 7, 9, 11, 13) else WIRE
    terminals = {}
    clock = NS(net_id="clock", source=_end("clk", "Y", terminals, (2, 2, 50)),
               sinks=[_end("ff0", "C", terminals, (3, 2, 50)), _end("ff1", "C", terminals, (20, 2, 50))])
    components = [NS(id="clk", type=ComponentType.INPUT_PIN), NS(id="ff0", type=ComponentType.DFF), NS(id="ff1", type=ComponentType.DFF)]
    return components, clock, terminals


def test_short_path_with_no_room_is_a_hold_violation():
    # A few sloped cells between flip-flops 24 ticks apart in clock, nowhere to put a repeater.
    for bulk in (True, False):
        region = (BulkRegion if bulk else Region)(0, 0, 0, 40, 8, 60)
        components, clock, terminals = _skewed_world(region)
        for i, (x, y, z) in enumerate([(30, 2, 30), (31, 3, 30), (32, 4, 30)]):
            region[x, y - 1, z] = STONE
            region[x, y, z] = WIRE
        data = NS(net_id="d", source=_end("ff0", "Q", terminals, (30, 2, 30)), sinks=[_end("ff1", "D", terminals, (32, 4, 30))])
        shorts: list = []
        hold.fix_hold(region, NS(components=components, nets=[clock, data]), terminals, set(), shorts=shorts)
        assert [s.pin for s in shorts] == [("ff1", "D")]
        assert shorts[0].margin == 1 - 20 and shorts[0].slack == 4 - 20 - 2
        assert [(v.kind, v.pos) for v in hold.hold_violations(shorts)] == [("hold", (32, 4, 30))]
        assert shorts[0].path == [("d", (32, 4, 30))]


def test_a_margin_under_target_can_still_be_safe():
    assert hold.hold_violations([hold.HoldShort(("f", "D"), (0, 0, 0), 0, 1, [])]) == []


def test_detour_makes_room_and_padding_finishes():
    bounds = (0, 0, 0, 59, 7, 59)
    region = Region(0, 0, 0, 60, 8, 60)
    components, clock, terminals = _skewed_world(region)
    source, sink = (30, 2, 30), (32, 2, 30)
    data = NS(net_id="d", source=_end("ff0", "Q", terminals, source), sinks=[_end("ff1", "D", terminals, sink)])
    rec = Recorder()
    solid, owner = JournaledSet(rec, name="solid"), JournaledDict(rec)
    router = _GreedyRouter(region, rec, solid, owner, JournaledSet(rec, name="inverted"), frozenset(), bounds[4], frozenset())
    job = _RouteJob(key="d", net_id="d", src=source, sinks=[sink], bounds=bounds, protected=frozenset(), terminals={source, sink})
    assert router.run([job], "[test]") == []
    comp = NS(components=components, nets=[clock, data])

    # Padding alone can't: two cells of wire.
    shorts: list = []
    undo: list = []
    hold.fix_hold(region, comp, terminals, set(), shorts=shorts, edits=undo)
    assert shorts and shorts[0].slack < 0
    for pos, block in reversed(undo):
        region[pos] = block

    period_requirement: list = []
    _hold_with_detours(region, comp, terminals, set(), (), (), None, period_requirement, shorts, (router, {"d": job}))
    assert shorts == []
    timing = hold._HoldTiming(hold._World(region, frozenset(), frozenset(terminals.values())), comp, terminals, set())
    _, margins, _ = timing.margins()
    assert margins["ff1", "D"] >= hold._HOLD_MARGIN
    assert job.detour[sink] != sink and job.key in router.paths
