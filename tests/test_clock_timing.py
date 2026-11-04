"""The generated clock: blocks' measured timing, static timing per clock edge, and the smallest build that shows why."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from litemapy import Region, Schematic

from minecraft_v.cell_library import SCHEMATIC_MAP
from minecraft_v.placement_engine.hold import ClockRequirement, _HoldTiming, _SETUP_TICKS
from minecraft_v.placement_engine.template import _non_air_bounds
from minecraft_v.placement_engine.ir import (CURRENT_SCHEMA_VERSION, Component, ComponentList, ComponentType, Direction,
                                             NetConnection, NetEndpoint, PinRef)
from minecraft_v.placement_engine.macro import Macro
from minecraft_v.placement_engine.ir import Footprint
from minecraft_v.placement_engine.ram import (StackSpec, build_delay, build_group, build_ram, build_stack, build_window,
                                              settled_block)
from minecraft_v.placement_engine.ram_timing import _Bench, checker_words, measure_array, sample_columns

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from settle_schematic import headless_binary  # noqa: E402

NORMAL = {"east": (1, 0), "north": (0, -1), "south": (0, 1)}
needs_headless = pytest.mark.skipif(not Path(headless_binary()).exists(), reason="needs mchprs_headless")
needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")


def headless_changes(block: Macro, tmp_path: Path, steps: list[tuple[str, bool]], window: int) -> list[dict[str, list[tuple[int, bool]]]]:
    """For each pin toggled, (tick, level) of every output pin's changes, from mchprs_headless. Tick 0 is the toggle."""
    fp = block.footprint
    region = Region(0, 0, 0, fp.width, fp.height, fp.depth)
    for pos, b in settled_block(block).blocks or block.blocks:
        region[pos] = b
    Schematic(name="b", author="t", regions={"b": region}).save(str(tmp_path / "b.litematic"))
    with open(tmp_path / "b.pins", "w") as f:
        for pin in block.pins:
            nx, nz = NORMAL[pin.side.value]
            f.write(f"{pin.name} {pin.offset[0]} {pin.offset[1]} {fp.depth - 1 - pin.offset[2]} {nx} {nz}\n")
    inputs = [p.name for p in block.pins if p.direction == Direction.IN]
    lines = [f"settle {window}"]
    for k, (pin, on) in enumerate(steps):
        lines += [f"print s{k}t0", f"set {pin} {int(on)}"]
        for t in range(1, window):
            lines += ["settle 1", f"print s{k}t{t}"]
    (tmp_path / "b.script").write_text("\n".join(lines) + "\n")
    levers = [a for p in inputs for a in ("--add-lever", p)]
    run = subprocess.run([headless_binary(), "test", tmp_path / "b.litematic", tmp_path / "b.pins", tmp_path / "b.script",
                          "--modes", "redstone", *levers], capture_output=True, text=True, check=True)
    found: list[dict[str, list[tuple[int, bool]]]] = [{} for _ in steps]
    previous: dict[str, str] = {}
    for line in run.stdout.splitlines():
        if "] s" not in line:
            continue
        label, *values = line.split("]", 1)[1].split()
        k, t = label[1:].split("t")
        for value in values:
            name, level = value.split("=")
            if name in previous and previous[name] != level and name not in inputs:
                found[int(k)].setdefault(name, []).append((int(t), level == "1"))
            previous[name] = level
    return found


def sim_changes(block: Macro, steps: list[tuple[str, bool]]) -> list[dict[str, tuple[bool, int]]]:
    """For each pin toggled, each output's last rise and fall in the simulator."""
    bench = _Bench(settled_block(block))
    outs = [p.name for p in block.pins if p.direction == Direction.OUT]
    found = []
    for pin, on in steps:
        into: dict = {}
        bench.sim.toggle(bench.cells[pin], on, {o: bench.cells[o] for o in outs}, into)
        found.append({o: {up: t for (_, up), t in edges.items()} for o, edges in into.items()})
    return found


@needs_headless
@pytest.mark.parametrize("make, steps", [
    (build_window, [("clk", True), ("clk", False), ("clk", True), ("clk", False)]),
    (lambda: build_ram("r", 4, 2, 2, [0, 3, 3, 0], measure=False),
     [("aa[0]", True), ("aa[1]", True), ("aa[0]", False), ("aa[1]", False), ("ab[1]", True), ("ab[1]", False),
      ("wd[0]", True), ("we", True), ("we", False), ("wa[0]", True), ("wd[1]", True), ("we", True), ("aa[0]", True),
      ("wd[1]", False), ("wa[0]", False), ("we", False)]),
    (lambda: build_ram("r", 8, 3, 4, checker_words(8, 3), measure=False),
     [("ad[2]", True), ("ad[1]", True), ("ad[2]", False), ("wd[1]", True), ("we", True), ("we", False), ("ad[1]", False)]),
])
def test_simulator_agrees_with_mchprs(tmp_path, make, steps):
    block = make()
    theirs = headless_changes(block, tmp_path, steps, 80)
    ours = sim_changes(block, steps)
    for k, step in enumerate(steps):
        # Their lever and our driven repeater differ by the pin repeater's own tick.
        want = {o: {up: max(t for t, level in changes if level == up) - 1 for up in {level for _, level in changes}}
                for o, changes in theirs[k].items()}
        assert ours[k] == want, (k, step)


@pytest.mark.parametrize("depth, width, reads, high", [(4, 2, 2, 0), (8, 3, 4, 1), (16, 4, 2, 0), (16, 2, 4, 2)])
def test_sampled_rows_find_what_all_rows_do(depth, width, reads, high):
    block = settled_block(build_ram("t", depth, width, reads, checker_words(depth, width), high, 0, measure=False))
    abits = depth.bit_length() - 1 + high
    assert (measure_array(block, depth, width, abits, "abcd"[:reads])
            == measure_array(block, depth, width, abits, "abcd"[:reads], list(range(depth))))


def test_unsampled_columns_are_never_shorter(monkeypatch):
    depth, width = 8, 9
    def measure():
        block = settled_block(build_ram("t", depth, width, 2, checker_words(depth, width), measure=False))
        return measure_array(block, depth, width, 3, "ab")
    sampled, _ = measure()
    from minecraft_v.placement_engine import ram_timing
    monkeypatch.setattr(ram_timing, "sample_columns", lambda w: list(range(w)))
    full, _ = measure()
    assert len(sample_columns(width)) < width
    for pin, outs in full.items():
        for out, edges in outs.items():
            for key, ticks in edges.items():
                assert sampled[pin][out][key] >= ticks, (pin, out, key)


def test_read_arcs_see_both_directions_and_write_through():
    block = build_ram("r", 8, 3, 2)
    assert set(block.edges["aa[0]"]["ra[0]"]) == {(True, True), (True, False), (False, True), (False, False)}
    assert set(block.edges["wd[1]"]["ra[1]"]) == {(True, True), (False, False)}
    assert block.edges["we"]["ra[0]"].keys() == {(True, True), (True, False)}  # closing a write changes nothing read
    assert block.closing["we"] > 0 and set(block.guarded["we"]) == {f"wa[{b}]" for b in range(3)} | {f"wd[{b}]" for b in range(3)}


def test_window_arcs_are_a_pulse_on_the_falling_edge():
    edges = build_window().edges["clk"]["window"]
    assert edges.keys() == {(False, True), (False, False)}
    assert edges[False, True] < edges[False, False]
    # Routed wire ends outside the pin repeater, so an arc counts it; leaving it out made a design fail at its generated clock by these ticks.
    assert edges[False, True] == 7 and edges[False, False] == 25


def test_delay_tap_arcs_are_the_chain():
    edges = build_delay().edges["clk"]
    assert edges["t0"][True, True] < edges["t1"][True, True] < edges["t2"][True, True] < edges["t3"][True, True]
    assert all(edges[t][True, True] == edges[t][False, False] for t in edges)


def whole_block_arcs(block: Macro, pin: str, outs: list[str]) -> dict:
    bench = _Bench(settled_block(block))
    into: dict = {}
    for on in (True, False):
        bench.sim.toggle(bench.cells[pin], on, {o: bench.cells[o] for o in outs}, into)
    return into


def test_stack_arcs_from_wire_and_array_match_the_whole_block():
    words = checker_words(4, 2) + [0] * 4
    block = build_stack("s", 2, 2, 0, 3, 2, words)
    for pin, out in (("aa[0]", "ra[0]"), ("ab[1]", "rb[1]"), ("ad[0]", "rd[0]")):
        whole = whole_block_arcs(block, pin, [out])[out]
        assert all(block.edges[pin][out][key] >= ticks for key, ticks in whole.items()), (pin, out)
        assert max(block.edges[pin][out].values()) - max(whole.values()) <= 4, (pin, out)  # worse rows than address 0's


def test_group_arcs_join_lanes_and_arrays():
    nets = {f"aa[{b}]": f"a{b}" for b in range(2)}
    specs = [StackSpec(1, 2, 0, 2, 2, checker_words(2, 2) * 2, nets), StackSpec(1, 2, 0, 2, 2, checker_words(2, 2) * 2, nets)]
    block, own, lanes = build_group("g", specs)
    lane = lanes["a0"]
    assert {"s0_ra[0]", "s1_ra[0]"} <= block.edges[lane].keys()
    whole = whole_block_arcs(block, lane, ["s0_ra[0]", "s1_ra[0]"])
    for out, edges in whole.items():
        assert all(block.edges[lane][out][key] >= ticks for key, ticks in edges.items()), out


class _Wired(_HoldTiming):
    """Static timing alone, on wire given as numbers."""

    def __init__(self, comp: ComponentList, ticks: dict[str, int]):
        self.by_id = {c.id: c for c in comp.components}
        self.nets = {n.net_id: n for n in comp.nets}
        self.ticks = ticks

    def wire_ticks(self, n, s):
        return self.ticks.get(n.net_id, 1)


def netlist(cells: dict[str, ComponentType], nets: list[tuple[str, str, str, list[tuple[str, str]]]], metadata=None) -> ComponentList:
    pins: dict[str, dict[str, Direction]] = {cid: {} for cid in cells}
    for _, c, p, sinks in nets:
        pins[c][p] = Direction.OUT
        for a, b in sinks:
            pins[a][b] = Direction.IN
    comps = [Component(id=cid, type=kind, pins=[PinRef(name=n, direction=d) for n, d in pins[cid].items()],
                       metadata=(metadata or {}).get(cid, {})) for cid, kind in cells.items()]
    return ComponentList(schema_version=CURRENT_SCHEMA_VERSION, components=comps, nets=[
        NetConnection(net_id=name, source=NetEndpoint(component_id=c, pin_name=p),
                      sinks=[NetEndpoint(component_id=a, pin_name=b) for a, b in sinks]) for name, c, p, sinks in nets])


def _two_flops(inverted: bool, data: int = 20, clock_a: int = 3, clock_b: int = 3) -> ClockRequirement:
    cells = {"clk": ComponentType.INPUT_PIN, "a": ComponentType.DFF, "b": ComponentType.DFF, "n": ComponentType.NOT}
    clock_nets = [("ca", "clk", "clk", [("a", "C")])]
    if inverted:
        clock_nets += [("cn", "clk", "clk", [("n", "A")]), ("cb", "n", "Y", [("b", "C")])]
    else:
        clock_nets[0] = ("ca", "clk", "clk", [("a", "C")])
        clock_nets += [("cb", "clk", "clk", [("b", "C")])]
    nets = clock_nets + [("d", "a", "Q", [("b", "D")])]
    ticks = {"ca": clock_a, "cb": clock_b, "cn": 0, "d": data}
    return _Wired(netlist(cells, nets), ticks).clock_requirement()


def test_flop_to_flop_gets_a_period_and_its_own_skew():
    near = _two_flops(False, 20, 3, 3)
    assert near.full == 4 + 20 + _SETUP_TICKS and near.half_cycle == 0
    assert near.half_period == 13
    # A later capture clock is more time, a later launch less: skew counts per path, signed.
    assert _two_flops(False, 20, 3, 13).full == near.full - 10
    assert _two_flops(False, 20, 13, 3).full == near.full + 10


def test_inverted_clock_flop_gets_half_a_period():
    found = _two_flops(True)
    assert found.half_cycle > 0 and found.half_period == found.half_cycle


def _write_design(close: int = 30, data_wire: int = 2) -> ClockRequirement:
    """clk -> pulse block -> RAM write enable -> RAM output pin, and a flip-flop feeding the RAM's write data."""
    window = Macro(name="w", footprint=Footprint(width=1, height=1, depth=1), pins=[], blocks=[], arcs={}, q_out={}, setup={},
                   edges={"clk": {"window": {(False, True): 6, (False, False): 24}}})
    ram = Macro(name="r", footprint=Footprint(width=1, height=1, depth=1), pins=[], blocks=[], arcs={}, q_out={}, setup={},
                edges={"we": {"ra": {(True, True): 12}}, "wd": {"ra": {(True, True): 7}}}, closing={"we": close}, guarded={"we": ["wd"]})
    cells = {"clk": ComponentType.INPUT_PIN, "f": ComponentType.DFF, "win": ComponentType.CUSTOM, "ram": ComponentType.CUSTOM,
             "out": ComponentType.OUTPUT_PIN, "d": ComponentType.INPUT_PIN}
    nets = [("c", "clk", "clk", [("win", "clk"), ("f", "C")]), ("we", "win", "window", [("ram", "we")]),
            ("data", "f", "Q", [("ram", "wd")]), ("o", "ram", "ra", [("out", "out")]), ("dd", "d", "d", [("f", "D")])]
    comp = netlist(cells, nets, {"win": {"macro": "w"}, "ram": {"macro": "r"}})
    return _Wired(comp, {"c": 2, "we": 2, "data": data_wire, "o": 2, "dd": 1}).clock_requirement({"w": window, "r": ram})


def test_falling_clock_through_a_write_window_gets_half_a_period():
    found = _write_design()
    # Clock 1 + 2 of wire, the window opening 6, wire 2, write enable to output 12, wire 2, the pin's exit repeater.
    assert found.half_cycle == 1 + 2 + 6 + 2 + 12 + 2 + 1
    # Its full period path is the flip-flop's, not the window's closing edge.
    assert found.full == 1 + 2 + 4 + 2 + 7 + 2 + 1


def test_write_has_to_close_before_its_data_changes():
    found = _write_design(close=30, data_wire=2)
    closes = 1 + 2 + 24 + 2 + 30
    changes = 1 + 2 + 1 + 2  # the rising edge at the flip-flop, its fastest output, wire
    assert found.write_floor == closes - changes
    assert _write_design(close=30, data_wire=12).write_floor == closes - changes - 10
    assert found.half_period == found.write_floor


@needs_headless
@pytest.mark.parametrize("kind, file", [(ComponentType.DFF, "dff"), (ComponentType.DFFE, "dffe")])
def test_flip_flop_templates_keep_the_library_clock_to_q_and_setup(tmp_path, kind, file):
    """What static timing assumes of a flip-flop: Q changes within its library delay, and D settled _SETUP_TICKS before C is taken."""
    info = SCHEMATIC_MAP[kind]
    template = next(iter(Schematic.load(str(ROOT / "schematics" / f"{file}.litematic")).regions.values()))
    x0, y0, z0, x1, y1, z1 = _non_air_bounds(template)
    fp, margin = info.footprint, 3
    region = Region(0, 0, 0, fp.width + 2 * margin, fp.height + 3, fp.depth + 2 * margin)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                region[x - x0 + margin, y - y0 + 1, z - z0 + margin] = template[x, y, z]
    Schematic(name="f", author="t", regions={"f": region}).save(str(tmp_path / "f.litematic"))
    normal = {**NORMAL, "west": (-1, 0)}
    (tmp_path / "f.pins").write_text("".join(
        f"{p.name} {p.offset[0] + margin} {p.offset[1] + 1} {fp.depth - 1 - p.offset[2] + margin} "
        f"{normal[p.side.value][0]} {normal[p.side.value][1]}\n" for p in info.pins))
    levers = [a for p in info.pins if p.name != "Q" for a in ("--add-lever", p.name)]

    def q_values(lines: list[str]) -> list[tuple[str, str]]:
        (tmp_path / "s").write_text("\n".join(lines) + "\n")
        out = subprocess.run([headless_binary(), "test", tmp_path / "f.litematic", tmp_path / "f.pins", tmp_path / "s",
                              "--modes", "redstone", *levers], capture_output=True, text=True, check=True).stdout
        return [(line.split("]")[1].split()[0], line.split()[-1]) for line in out.splitlines() if "Q=" in line]

    enable = ["set E 1", "settle 20"] if kind == ComponentType.DFFE else []
    rise = enable + ["set D 1", "settle 30", "set C 1"]
    for t in range(1, 15):
        rise += ["settle 1", f"print t{t}"]
    first = next(label for label, value in q_values(rise) if value == "Q=1")
    assert int(first[1:]) <= info.propagation_delay_ticks
    taken = enable + ["set D 1", f"settle {_SETUP_TICKS}", "set C 1", "settle 20", "print end"]
    assert q_values(taken) == [("end", "Q=1")]


@needs_headless
@needs_yosys
def test_smallest_ram_runs_at_its_generated_clock(tmp_path):
    """examples/ramtile generated half 16 when half = path / 2, and writes get lost below half 30."""
    import clock_sweep

    assert clock_sweep.build("examples/ramtile", tmp_path, "0", "--tiled-ram", "--no-macros") == 0
    inputs, _ = clock_sweep.top_ports(tmp_path / "netlist.json")
    pins = {line.split()[0] for line in open(tmp_path / "design.pins")}
    inputs = [p for p in inputs if p in pins]
    import json
    generated = json.loads((tmp_path / "design.server.json").read_text())["clock"]["half_period_ticks"]
    reference = clock_sweep.trace(tmp_path, generated * 8, inputs, "clk", 60, 1, generated * 4, "redstone")
    assert clock_sweep.trace(tmp_path, generated, inputs, "clk", 60, 1, generated * 4, "redstone") == reference
    assert clock_sweep.trace(tmp_path, 16, inputs, "clk", 60, 1, generated * 4, "redstone") != reference
    assert generated <= 45  # not wildly above the 30 it needs
