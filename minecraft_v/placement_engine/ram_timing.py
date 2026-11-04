"""Timing arcs of the RAM blocks, measured by simulating them (tick_sim), for the clock's static timing (hold.py).

An array is measured with words alternating all ones and all zeros by the parity of their address, so moving any address
bit changes every output. Each address bit moves up and down at sample_rows, and writes land there: paths are lines with
repeaters at a regular pitch, so the ends and the middle find the longest; tests/test_ram_timing.py checks that against
all rows. Only sample_columns hold data and are read, and other columns' arcs take the longest of theirs. Bits above the row bits select the array: the array measured answers to 0, so one direction is seen for each,
and their arcs take the longer of the two.

A stack or group is wire around its arrays. That wire is simulated with the arrays left out, and its arcs joined with each
array's: simulating it all would go through every row of every bank, and arrays of a shape all take the same time.
"""
from __future__ import annotations

from .block_utils import BulkRegion
from .cosmetics import apply_dust_shapes
from .macro import Macro
from .propagate import comparator_entity, settle
from .tick_sim import Edges, TickSim

Pos = tuple[int, int, int]
PinEdges = dict[str, dict[str, Edges]]


def pin_cell(block: Macro, name: str) -> Pos:
    pin = next(p for p in block.pins if p.name == name)
    return pin.offset[0], pin.offset[1], block.footprint.depth - 1 - pin.offset[2]


def settled_region(block: Macro) -> BulkRegion:
    arrays, fp = block.settled, block.footprint
    region = BulkRegion(0, 0, 0, fp.width, fp.height, fp.depth)
    region.paste(arrays.xs, arrays.ys, arrays.zs, arrays.ids, arrays.palette)
    region.tile_entities.extend(comparator_entity(p, level) for p, level in arrays.comparators)
    return region


def sample_rows(depth: int) -> list[int]:
    return sorted({0, 1, depth // 2 - 1, depth // 2, depth - 2, depth - 1} & set(range(depth)))


def sample_columns(width: int) -> list[int]:
    return sorted({0, 1, width // 2, width - 2, width - 1} & set(range(width)))


def column_mask(width: int) -> int:
    return sum(1 << c for c in sample_columns(width))


def checker_words(depth: int, width: int) -> list[int]:
    return [(bin(k).count("1") & 1) * column_mask(width) for k in range(depth)]


class _Bench:
    def __init__(self, block: Macro):
        self.sim = TickSim(settled_region(block))
        self.cells = {p.name: self.sim.cell(pin_cell(block, p.name)) for p in block.pins}
        self.level = dict.fromkeys(self.cells, False)
        self.edges: PinEdges = {}

    def toggle(self, pin: str, on: bool, outs: list[str] = ()) -> int:
        """Ticks until still; nothing if the pin is already there."""
        if self.level[pin] == on:
            return 0
        self.level[pin] = on
        return self.sim.toggle(self.cells[pin], on, {out: self.cells[out] for out in outs}, self.edges.setdefault(pin, {}))

    def bus(self, name: str, bits: int, value: int, outs: list[str] = ()) -> None:
        for bit in range(bits):
            self.toggle(f"{name}[{bit}]", bool(value >> bit & 1), outs)


def measure_array(block: Macro, depth: int, width: int, abits: int, ports: str, rows: list[int] | None = None
                  ) -> tuple[PinEdges, dict[str, int]]:
    """Input pin -> output pin -> Edges, and input pin -> ticks until the array is still after it falls, for one holding
    checker_words and answering when its upper address bits are 0."""
    bench = _Bench(block)
    rows = sample_rows(depth) if rows is None else rows
    row_bits = depth.bit_length() - 1
    columns = sample_columns(width)
    every = [f"r{p}[{i}]" for p in ports for i in columns]
    for port in ports:
        name, outs = f"a{port}", [f"r{port}[{i}]" for i in columns]
        for k in rows:
            bench.bus(name, row_bits, k, outs)
            for bit in range(abits):
                if bit >= row_bits or not k >> bit & 1:
                    bench.toggle(f"{name}[{bit}]", True, outs)
                    bench.toggle(f"{name}[{bit}]", False, outs)
        bench.bus(name, row_bits, 0, outs)
    if "we" not in bench.cells:  # read-only
        _fill_columns(bench.edges, width, columns, ports)
        return bench.edges, {}

    def read_all(k: int) -> None:
        for port in ports:
            bench.bus(f"a{port}", row_bits, k)

    def write(k: int, word: int) -> None:
        bench.toggle("we", False)
        bench.bus("wa", abits, k)
        bench.bus("wd", width, word)
        bench.toggle("we", True)
        bench.toggle("we", False)

    closing = {"we": 0}
    mask = column_mask(width)
    for k in rows:
        read_all(k)
        for word in (0, mask):  # write enable: the word lands and the read follows, for both values
            write(k, mask ^ word)
            bench.bus("wd", width, word)
            bench.toggle("we", True, every)
            closing["we"] = max(closing["we"], bench.toggle("we", False, every))
        write(k, 0)
        bench.toggle("we", True)
        for i in columns:  # write through, the latch held open
            outs = [f"r{p}[{i}]" for p in ports]
            bench.toggle(f"wd[{i}]", True, outs)
            bench.toggle(f"wd[{i}]", False, outs)
        bench.toggle("we", False)
        for bit in range(abits):  # the write moves onto this row
            write(k, 0)
            bench.bus("wa", abits, k ^ 1 << bit)
            bench.bus("wd", width, mask)
            bench.toggle("we", True)
            bench.toggle(f"wa[{bit}]", bool(k >> bit & 1), every)
            bench.toggle(f"wa[{bit}]", not k >> bit & 1, every)
            bench.toggle("we", False)
    for bit in range(row_bits, abits):
        for name in [f"a{p}" for p in ports] + ["wa"]:
            for edges in bench.edges.get(f"{name}[{bit}]", {}).values():
                ticks = max(edges.values(), default=0)
                edges.update({key: ticks for key in ((a, b) for a in (False, True) for b in (False, True))})
    _fill_columns(bench.edges, width, columns, ports)
    return bench.edges, closing


def _fill_columns(edges: PinEdges, width: int, columns: list[int], ports: str) -> None:
    """Columns not simulated: the longest of those that were, per port. Their write data pin has no other column's arcs."""
    def worst(found: list[Edges]) -> Edges:
        total: Edges = {}
        for one in found:
            for key, ticks in one.items():
                total[key] = max(total.get(key, 0), ticks)
        return total

    through = worst([e for i in columns for e in edges.get(f"wd[{i}]", {}).values()])
    for i in set(range(width)) - set(columns):
        edges[f"wd[{i}]"] = {f"r{p}[{i}]": dict(through) for p in ports}
    for pin, outs in edges.items():
        if pin.startswith("wd["):
            continue
        for port in ports:
            sampled = [outs[f"r{port}[{i}]"] for i in columns if f"r{port}[{i}]" in outs]
            if sampled:
                total = worst(sampled)
                for i in set(range(width)) - set(columns):
                    outs[f"r{port}[{i}]"] = dict(total)


def measure_blocks(block: Macro) -> PinEdges:
    """A block with one input, as the window and delay blocks: its clock pin, up and down."""
    bench = _Bench(block)
    outs = [p.name for p in block.pins if p.name != "clk"]
    for on in (True, False):
        bench.toggle("clk", on, outs)
    return bench.edges


def measure_wire(blocks: dict[Pos, object], size: tuple[int, int, int], block: Macro, entries: dict[str, list[Pos]],
                 exits: list[tuple[Pos, object, list[str]]]) -> tuple[dict[tuple[str, Pos], Edges], dict[tuple[Pos, str], Edges]]:
    """The wire of a stack or group without its arrays. entries: input pin -> the cell before each array pin it reaches.
    exits: (an array's output pin cell, its block, the block's output pins it reaches). Returns ticks to those cells from
    the pins, and from those array pins to the block's."""
    region = BulkRegion(0, 0, 0, *size)
    for p, b in blocks.items():
        region[p] = b
    for p, b, _ in exits:
        region[p] = b
    apply_dust_shapes(region, [])
    settle(region, [])
    sim = TickSim(region)
    into: dict[tuple[str, Pos], Edges] = {}
    for pin, cells in entries.items():
        watch = {c: sim.cell(c) for c in cells}
        found: dict = {}
        drive = sim.cell(pin_cell(block, pin))
        for on in (True, False):
            sim.toggle(drive, on, watch, found)
        for c, edges in found.items():
            into[pin, c] = edges
    out: dict[tuple[Pos, str], Edges] = {}
    for p, _, pins in exits:
        watch = {pin: sim.cell(pin_cell(block, pin)) for pin in pins}
        found = {}
        drive = sim.cell(p)
        for on in (True, False):
            sim.toggle(drive, on, watch, found)
        for pin, edges in found.items():
            out[p, pin] = edges
    return into, out


def joined(*arcs: Edges) -> Edges:
    """Arcs one after the other."""
    total = arcs[0]
    for arc in arcs[1:]:
        next_total: Edges = {}
        for (a, b), first in total.items():
            for (c, d), second in arc.items():
                if b == c:
                    next_total[a, d] = max(next_total.get((a, d), 0), first + second)
        total = next_total
    return total
