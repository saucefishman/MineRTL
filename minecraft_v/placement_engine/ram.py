"""Tiled RAM: a generated array of latches with two read ports and one write port, kept as a macro.

No routing inside. Lines along X and lines along Z sit on different planes, so they never cross:

    y 9  read address lines (Z), decoder only, on glass and the blocks of its taps
    y 7  read wordlines (X), on the row blocks of plane 6
    y 6  S, a comparator per read port, its side repeater, the block that drives the bitline
    y 5  read bitlines (Z), under those blocks; second torch
    y 4  glass floor of the read bitlines; block between the torches
    y 3  lock lines (X), on the row blocks of plane 2; first torch
    y 2  storage: D repeater, E repeater that locks it, F
    y 1  write bitlines (Z)
    y 0  glass floor

A read port's comparator subtracts its active-low wordline from S, so a selected row drives the bitline with its bit
and any other row with nothing. Bitlines are wired ORs. Two torches carry the stored bit from F up to S.

East of the bits is the decoder: a NOR per row. Each address bit has a true and a complement line, and any line that
mismatches a row holds it high through a torch. Write address and write enable lines run on plane 5, under the row blocks.
Pins are addresses, write enable and bitlines. Writes take the self-closing pulse of build_window, see techmap/ram_2r1w.v.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from litemapy import BlockState

from minecraft_v.placement_engine.ir import CardinalDirection, Direction, Footprint, PinRef
from minecraft_v.progress import log, phase

from .block_utils import BulkRegion
from .cosmetics import apply_dust_shapes
from .macro import BlockArrays, Macro
from .propagate import comparator_entity, settle
from .ram_timing import PinEdges, checker_words, joined, measure_array, measure_blocks, measure_wire, pin_cell
from .tick_sim import Edges
from .verify import verify_support

Pos = tuple[int, int, int]

_COLUMN_PITCH = 6
_ROW_PITCH = 6
_REPEAT_EVERY = 2  # rows or columns between line repeaters; dust from a repeater reaches 15 cells

_Y_WRITE, _Y_STORE, _Y_LOCK, _Y_MIDDLE, _Y_READ, _Y_CELL, _Y_WORD, _Y_READ_ADDR = 1, 2, 3, 4, 5, 6, 7, 9
_BANK_PITCH = 16     # Y between a stack's banks; a whole number of tower inversions pairs
_STACK_NORTH = 5     # Z past a bank's output pins, for towers and the way out
_STACK_SOUTH_LOWER = 1  # Z past a bank's input pins of the tower of a pin on a lower plane
_STACK_SOUTH_UPPER = 4  # and of a pin on the upper read address plane
_STACK_SOUTH = 7
_TOWER_TICKS = 8     # torches per bank
_LANE_PITCH = 2      # Y between a group's lanes
_LANE_REPEAT = 12    # X between a lane's repeaters
_LANE_PIN_PITCH = 6  # least X between a group's input pins
_READ_PLANES = 4     # Y from one pair of read ports to the next
_LINE_PITCH = 2      # X between decoder lines; each line's torches sit between it and the next
_ADDR_PITCH = 4  # an address bit's true and complement lines; also the pins' X pitch
_DECODER_REPEAT = 12 // _ADDR_PITCH * _ADDR_PITCH  # X between repeaters on a row line in the decoder: under a true line, never a torch

# Write lines on plane 5 under the first read decoder, not beside it: narrower decoder, same bits. See build_ram.
_FOLD_FIXED: bool | None = None  # True always, False never; None, only groups too wide for a plot
_FOLD = bool(_FOLD_FIXED)
_FOLD_GROUP_WIDTH = 1000
_STACK_SOUTH_FOLDED = 10
_TOWER_Z_PITCH = 3  # Z between a column's towers when folded

_GLASS = BlockState("minecraft:glass")
_STONE = BlockState("minecraft:smooth_quartz")
_DUST = BlockState("minecraft:redstone_wire")
# TODO: other port counts: more read ports need another S per pair, more write ports a per-bit OR.
RAM_CELL = "ram_2r1w"
RAM4_CELL = "ram_stack"
DELAY_CELL = "ram_delay"
# Chain cells before each tap of build_delay, 4 ticks a cell. After clk falls: the first write's window; a gap, since an
# idle port's flip-flops hold 0 and switching to them as the window closes could reach its latches; settling for the second
# write's address through the deepest array; the second write's window.
_DELAY_TAPS = {12: "t0", 24: "t1", 84: "t2", 96: "t3"}
_DELAY_CELLS = 97
WINDOW_CELL = "ram_window"
WINDOW_MACRO = "ram_window"
BLOCK_CELLS = (RAM_CELL, RAM4_CELL, WINDOW_CELL, DELAY_CELL)


def _repeater(facing: str) -> BlockState:
    return BlockState("minecraft:repeater", facing=facing, delay="1", powered="false", locked="false")


def _torch(lit: bool) -> BlockState:
    return BlockState("minecraft:redstone_torch", lit="true" if lit else "false")


def _comparator(facing: str) -> BlockState:
    return BlockState("minecraft:comparator", facing=facing, mode="subtract", powered="false")


def _through_pin(edges: PinEdges) -> PinEdges:
    """Measured arcs start at an input pin repeater's output, but the routed wire ends a cell outside that repeater."""
    return {pin: {out: {key: ticks + 1 for key, ticks in found.items()} for out, found in outs.items()} for pin, outs in edges.items()}


def _opaque(name: str, size: tuple[int, int, int], blocks: dict[Pos, BlockState], pins: list[PinRef],
            edges: PinEdges, cells: int, closing: dict[str, int] | None = None) -> Macro:
    """Verification stops at the pins: the block is one opaque gate, checked by its own harness test."""
    size_x, size_y, size_z = size
    terminals: dict[str, Pos] = {}
    nets: dict[str, tuple[str, list[str]]] = {}
    port_net: dict[str, str] = {}
    for pin in pins:
        inner = f"tile:{pin.name}"
        terminals[inner] = (pin.offset[0], pin.offset[1], size_z - 1 - pin.offset[2])
        port_net[pin.name] = inner
        outside = f"{pin.name}:{pin.name}"
        nets[inner] = (outside, [inner]) if pin.direction == Direction.IN else (inner, [outside])
    return Macro(name=name, footprint=Footprint(width=size_x, height=size_y, depth=size_z), pins=pins,
                 blocks=list(blocks.items()), arcs=_longest(_through_pin(edges)),
                 q_out={}, setup={}, terminals=terminals, gate_boxes=[((0, 0, 0), size)], nets=nets, port_net=port_net,
                 cells=cells, edges=_through_pin(edges), closing={pin: ticks + 1 for pin, ticks in (closing or {}).items()})


def _longest(edges: PinEdges) -> dict[str, dict[str, int]]:
    return {pin: {out: max(e.values()) for out, e in outs.items() if e} for pin, outs in edges.items()}


_clock_block_edges: dict[str, PinEdges] = {}


def _with_clock_edges(block: Macro) -> Macro:
    """A block fed only by the clock, timed by simulating it."""
    edges = _clock_block_edges.get(block.name)
    if edges is None:
        edges = _clock_block_edges[block.name] = measure_blocks(settled_block(block))
    block.edges = _through_pin(edges)
    block.arcs = _longest(block.edges)
    return block


def build_window() -> Macro:
    """window = !clk & late clk: a comparator subtracting the inverted late clock from the inverted clock."""
    size = (7, 2, 7)
    y = 1
    blocks: dict[Pos, BlockState] = {(x, 0, z): _GLASS for x in range(size[0]) for z in range(size[2])}

    def slow(facing: str) -> BlockState:
        return BlockState("minecraft:repeater", facing=facing, delay="4", powered="false", locked="false")

    blocks[(2, y, 6)] = _repeater("south")  # clk pin
    blocks[(2, y, 5)] = _STONE
    blocks[(1, y, 5)] = BlockState("minecraft:redstone_wall_torch", facing="west", lit="true")
    blocks[(3, y, 5)] = BlockState("minecraft:redstone_wall_torch", facing="east", lit="true")
    blocks[(1, y, 4)] = _comparator("south")
    for z in range(4):
        blocks[(1, y, z)] = _repeater("south")  # window pin at z = 0
    # Late clock: out along z = 5 and back along z = 4 into the comparator's side.
    blocks[(4, y, 5)] = slow("west")
    blocks[(5, y, 5)] = slow("west")
    blocks[(6, y, 5)] = _DUST
    blocks[(6, y, 4)] = _DUST
    blocks[(5, y, 4)] = slow("east")
    blocks[(4, y, 4)] = slow("east")
    blocks[(3, y, 4)] = _repeater("east")
    blocks[(2, y, 4)] = _repeater("east")
    pins = [PinRef(name="clk", direction=Direction.IN, side=CardinalDirection.SOUTH, offset=(2, y, 0)),
            PinRef(name="window", direction=Direction.OUT, side=CardinalDirection.NORTH, offset=(1, y, size[2] - 1))]
    return _with_clock_edges(_opaque(WINDOW_MACRO, size, blocks, pins, {}, 1))


def build_ram(name: str, depth: int, width: int, reads: int = 2, init: list[int] | None = None,
              high_bits: int = 0, high: int = 0, measure: bool = True) -> Macro:
    """``reads`` is 2 or 4. ``init`` holds a preloaded word per row. With ``high_bits``, addresses have that many more
    bits, and this array answers only when they equal ``high``: several can share address lines and OR their bitlines.
    ``measure``: time it, by shape, or leave that to a caller who only wants the blocks."""
    row_bits = depth.bit_length() - 1
    if row_bits < 1 or depth != 1 << row_bits:
        raise ValueError(f"{name}: depth {depth} is not a power of two above 1")
    abits = row_bits + high_bits
    decoder_x = _COLUMN_PITCH * width + 2
    read_a_x = decoder_x
    read_b_x = read_a_x + _ADDR_PITCH * abits
    enable_x = read_b_x + _ADDR_PITCH * abits  # write lines have their own X, so each pin can have its own tower
    write_x = enable_x + _LINE_PITCH
    write_end = write_x + _ADDR_PITCH * abits
    size_x = write_end
    if _FOLD and reads == 4:
        # Plane 5 is free east of the bits: read rows' blocks are above it and lock lines below. A column then has three
        # pins, whose towers build_stack spreads along Z. Lock lines flow west, so they only reach the enable's torch.
        write_x, enable_x = read_a_x, read_b_x
        write_end = enable_x + _LINE_PITCH
        size_x = read_b_x + _ADDR_PITCH * abits
    # Past the last row: the complement torch, the stub to it, the pin.
    last_row_gap = _ROW_PITCH * depth
    size_z = last_row_gap + 4
    stub_z = size_z - 2
    size_y = (_Y_READ_ADDR + _READ_PLANES if reads == 4 else _Y_READ_ADDR) + 1
    blocks: dict[Pos, BlockState] = {}
    pins: list[PinRef] = []

    def column_x(i: int) -> int:
        return _COLUMN_PITCH * i + 3

    def row_z(k: int) -> int:
        return _ROW_PITCH * k + 3

    def add_pin(pin: str, direction: Direction, side: CardinalDirection, pos: Pos) -> None:
        pins.append(PinRef(name=pin, direction=direction, side=side,
                           offset=(pos[0], pos[1], size_z - 1 - pos[2])))

    def row_line(y: int, z: int, end_x: int, high: bool = False) -> None:
        """Active low line along X on row blocks, driven by the decoder's torches at its east end. Pasted ``high``, a
        lock line holds its latches from the first tick, so a preload survives until the torches take over."""
        for x in range(end_x):
            blocks[(x, y - 1, z)] = _STONE
            if x < decoder_x:
                gap = x % _COLUMN_PITCH == 0
                repeat = gap and (x == decoder_x - 2 or x // _COLUMN_PITCH % _REPEAT_EVERY == 0)
            else:
                repeat = (x - decoder_x) % _DECODER_REPEAT == 0
            if repeat:
                blocks[(x, y, z)] = BlockState("minecraft:repeater", facing="east", delay="1", locked="false",
                                               powered="true" if high else "false")
            else:
                blocks[(x, y, z)] = BlockState("minecraft:redstone_wire", power="15") if high else _DUST

    def bitline(x: int, y: int, end_z: int) -> None:
        """Line along Z flowing north on glass. Repeaters sit in the gaps between rows, and its south end is one."""
        for z in range(end_z + 1):
            blocks[(x, y - 1, z)] = _GLASS
            repeat = z in (0, end_z) or (z <= last_row_gap and z % _ROW_PITCH == 0 and z // _ROW_PITCH % _REPEAT_EVERY == 0)
            blocks[(x, y, z)] = _repeater("south") if repeat else _DUST

    def tap(x: int, y: int, z: int) -> None:
        """Row line at y - 2 high while line x at y is low: the line powers a block, whose torch sits over the row."""
        blocks[(x, y - 1, z)] = _STONE
        blocks[(x + 1, y - 1, z)] = BlockState("minecraft:redstone_wall_torch", facing="east", lit="true")

    def address_lines(x: int, y: int, pin: str, rows: list[tuple[int, int]], bit: int) -> None:
        """A bit's true line at x and complement at x + 2; each (row number, row line z) mismatching taps one of them."""
        bitline(x, y, size_z - 1)
        add_pin(pin, Direction.IN, CardinalDirection.SOUTH, (x, y, size_z - 1))
        bitline(x + 2, y, stub_z - 2)
        blocks[(x + 1, y - 1, stub_z)] = _GLASS
        blocks[(x + 1, y, stub_z)] = _DUST
        blocks[(x + 2, y - 1, stub_z)] = _GLASS
        blocks[(x + 2, y, stub_z)] = _STONE
        blocks[(x + 2, y, stub_z - 1)] = BlockState("minecraft:redstone_wall_torch", facing="north", lit="true")
        for k, z in rows:
            tap(x if (k | high << row_bits) >> bit & 1 else x + 2, y, z)

    lock_rows = [(k, row_z(k) - 2) for k in range(depth)]
    ports = [("a", 0, -2, read_a_x), ("b", 0, 2, read_b_x)]
    if reads == 4:
        ports += [("c", _READ_PLANES, -2, read_a_x), ("d", _READ_PLANES, 2, read_b_x)]
    for port, up, side, line_x in ports:
        rows = [(k, row_z(k) + side) for k in range(depth)]
        for _, z in rows:
            row_line(_Y_WORD + up, z, line_x + _ADDR_PITCH * abits)
        for bit in range(abits):
            address_lines(line_x + _ADDR_PITCH * bit, _Y_READ_ADDR + up, f"a{port}[{bit}]", rows, bit)
    for _, z in lock_rows:
        row_line(_Y_LOCK, z, write_end, high=True)
    for bit in range(abits):
        address_lines(write_x + _ADDR_PITCH * bit, _Y_READ, f"wa[{bit}]", lock_rows, bit)
    bitline(enable_x, _Y_READ, size_z - 1)
    add_pin("we", Direction.IN, CardinalDirection.SOUTH, (enable_x, _Y_READ, size_z - 1))
    for _, z in lock_rows:
        tap(enable_x, _Y_READ, z)

    for i in range(width):
        sx = column_x(i)
        bitline(sx - 3, _Y_WRITE, size_z - 1)
        add_pin(f"wd[{i}]", Direction.IN, CardinalDirection.SOUTH, (sx - 3, _Y_WRITE, size_z - 1))
        for port, up, x in (("a", 0, sx + 2), ("b", 0, sx - 2)) + ((("c", _READ_PLANES, sx + 2), ("d", _READ_PLANES, sx - 2)) if reads == 4 else ()):
            bitline(x, _Y_READ + up, last_row_gap)
            add_pin(f"r{port}[{i}]", Direction.OUT, CardinalDirection.NORTH, (x, _Y_READ + up, 0))
        for k in range(depth):
            z = row_z(k)
            # Up from the write bitline to the latch.
            blocks[(sx - 2, _Y_WRITE, z)] = _GLASS
            blocks[(sx - 2, _Y_STORE, z)] = _DUST
            # Latch: locked by a powered row line. The row blocks are on z - 2.
            stored = bool(init and init[k] >> i & 1)
            blocks[(sx - 1, _Y_WRITE, z)] = _GLASS
            blocks[(sx - 1, _Y_WRITE, z - 1)] = _GLASS
            blocks[(sx - 1, _Y_STORE, z)] = BlockState("minecraft:repeater", facing="west", delay="1", locked="true",
                                                        powered="true" if stored else "false")
            blocks[(sx - 1, _Y_STORE, z - 1)] = BlockState("minecraft:repeater", facing="north", delay="1", locked="false", powered="true")
            blocks[(sx, _Y_STORE, z)] = _STONE
            blocks[(sx, _Y_WRITE, z)] = _GLASS
            # Twice inverted up to each S.
            blocks[(sx, _Y_LOCK, z)] = _torch(not stored)
            blocks[(sx, _Y_MIDDLE, z)] = _STONE
            for up in (0, _READ_PLANES) if reads == 4 else (0,):
                blocks[(sx, _Y_READ + up, z)] = _torch(stored)
                blocks[(sx, _Y_CELL + up, z)] = _STONE
                # Read taps, on either side of S.
                for x, facing, side_z, side_facing, out_x in ((sx + 1, "west", z - 1, "north", sx + 2),
                                                              (sx - 1, "east", z + 1, "south", sx - 2)):
                    blocks[(x, _Y_READ + up, z)] = _GLASS
                    blocks[(x, _Y_READ + up, side_z)] = _GLASS
                    blocks[(x, _Y_CELL + up, z)] = _comparator(facing)
                    blocks[(x, _Y_CELL + up, side_z)] = _repeater(side_facing)
                    blocks[(out_x, _Y_CELL + up, z)] = _STONE
            if reads == 4:  # on up the ladder to the second S
                blocks[(sx, _Y_WORD, z)] = _torch(not stored)
                blocks[(sx, _Y_WORD + 1, z)] = _STONE

    edges, closing = _array_timing(depth, width, reads, high_bits).edges_closing() if measure else ({}, {})
    block = _opaque(name, (size_x, size_y, size_z), blocks, pins, edges, depth * width, closing)
    if closing:
        block.guarded = {"we": [p.name for p in pins if p.name.startswith(("wa[", "wd["))]}
    return block


def build_delay(name: str = "ram_delay") -> Macro:
    """Clock delayed three ways, for the write turns: outputs are repeater chain taps."""
    size = (_DELAY_CELLS + 2, 2, 4)
    y = 1
    blocks: dict[Pos, BlockState] = {(x, 0, z): _GLASS for x in range(size[0]) for z in range(size[2])}
    blocks[(0, y, 3)] = _repeater("south")
    blocks[(0, y, 2)] = _DUST
    pins = [PinRef(name="clk", direction=Direction.IN, side=CardinalDirection.SOUTH, offset=(0, y, 0))]
    taps = _DELAY_TAPS
    for x in range(1, size[0]):
        cell = x - 1
        if cell in taps:
            blocks[(x, y, 2)] = _DUST
            blocks[(x, y, 1)] = _repeater("south")
            blocks[(x, y, 0)] = _repeater("south")
            pins.append(PinRef(name=taps[cell], direction=Direction.OUT, side=CardinalDirection.NORTH, offset=(x, y, size[2] - 1)))
        else:
            blocks[(x, y, 2)] = BlockState("minecraft:repeater", facing="west", delay="4", powered="false", locked="false")
    return _with_clock_edges(_opaque(name, size, blocks, pins, {}, 1))


def build_stack(name: str, row_bits: int, banks: int, base: int, abits: int, width: int, init: list[int],
                towers: dict[str, tuple[int, int, int]] | None = None, on_lanes: frozenset[str] = frozenset(),
                banks_out: list | None = None) -> Macro:
    """``banks`` 4-read arrays one above the other, as one block, holding bank numbers ``base`` up. Every input pin feeds a
    torch tower beside its line; each bank's line taps the tower. Each output has a tower too, which every bank's bitline
    powers at its own level, so the top of the tower is the OR of the banks. Every pin ends up on the south face. Lines near towers are repeaters, which ignore
    them. A tower inverts twice per 4 blocks, so its tap levels agree. Input pins in ``on_lanes`` get no way out to the
    south face: ``towers`` collects each one's tower base (x, y, z) instead, for build_group to feed.
    With ``banks_out``, the banks' own blocks are left out, and it gets one (origin, array arguments, size) per bank: the
    caller settles them separately, see settled_block.
    """
    depth = 1 << row_bits
    high_bits = abits - row_bits
    bank_args = [(depth, width, tuple(init[k * depth:(k + 1) * depth]), high_bits, base + k) for k in range(banks)]
    arrays = [build_ram(name, depth, width, 4, list(a[2]), high_bits, a[4], measure=False) for a in bank_args[:1 if banks_out is not None else banks]]
    bank = arrays[0].footprint
    north = _STACK_NORTH
    south = _STACK_SOUTH_FOLDED if _FOLD else _STACK_SOUTH
    # Outputs run over the top to the south face, with the inputs: a net can't cross a block, and logic reads and writes.
    # Lower and upper read lines take a Y each, as their columns touch; 1 mod 4 keeps tower polarity.
    exit_y = (bank.height + _BANK_PITCH * (banks - 1) + 3) // 4 * 4 + 1
    size = (bank.width + 1, exit_y + _READ_PLANES + 1, north + bank.depth + south)
    blocks: dict[Pos, BlockState] = {}
    for k, array in enumerate(arrays if banks_out is None else ()):
        for (x, y, z), block in array.blocks:
            blocks[(x, y + _BANK_PITCH * k, z + north)] = block
    if banks_out is not None:
        banks_out.extend(((0, _BANK_PITCH * k, north), a, (bank.width, bank.height, bank.depth)) for k, a in enumerate(bank_args))

    def line(x: int, y: int, z: int, facing: str = "south") -> None:
        blocks[(x, y, z)] = _repeater(facing)
        blocks[(x, y - 1, z)] = _GLASS

    def tower(x: int, z: int, bottom: int, top: int | None = None) -> None:
        """Up from ``bottom`` through the last bank's level or to ``top``, in the state of an idle low signal."""
        top = bottom + _BANK_PITCH * (banks - 1) if top is None else top
        for y in range(bottom, top + 1, 2):
            blocks[(x, y, z)] = _STONE
            if y < top:
                blocks[(x, y + 1, z)] = _torch((y - bottom) // 2 % 2 == 0)

    pins: list[PinRef] = []
    # Folded: towers on lanes nearest, so wire from pins on the south face crosses none; lowest first of the others, as a
    # line out runs under the further towers.
    columns: dict[int, list] = {}
    for pin in arrays[0].pins:
        if pin.direction == Direction.IN:
            columns.setdefault(pin.offset[0], []).append(pin)
    tower_step = {pin.name: step for column in columns.values()
                  for step, pin in enumerate(sorted(column, key=lambda p: (p.name not in on_lanes, p.offset[1])))}
    for pin in arrays[0].pins:
        x, y = pin.offset[0], pin.offset[1]
        if pin.direction == Direction.IN:
            # Upper read pins' towers sit further out: a lower line reaches its pin under them. So does write enable's,
            # whose tower would otherwise touch the first write address line.
            z = north + bank.depth + (_STACK_SOUTH_UPPER if y > _Y_READ_ADDR or pin.name == "we" else _STACK_SOUTH_LOWER)
            if _FOLD:
                z = north + bank.depth + _STACK_SOUTH_LOWER + _TOWER_Z_PITCH * tower_step[pin.name]
            tower(x + 1, z, y)
            for k in range(banks):
                level = y + _BANK_PITCH * k
                blocks[(x, level, z)] = _DUST
                blocks[(x, level - 1, z)] = _GLASS
                for approach in range(north + bank.depth, z):
                    line(x, level, approach)
            if pin.name in on_lanes:
                towers[pin.name] = (x + 1, y, z)
                continue
            for out in range(z + 1, size[2]):
                line(x + 1, y, out)
            pins.append(PinRef(name=pin.name, direction=Direction.IN, side=CardinalDirection.SOUTH, offset=(x + 1, y, 0)))
        else:
            upper = y > _Y_READ
            # An upper line steps aside, as the lower line's tower passes its level. Mirrored, so that neighbouring
            # bits' steps don't touch.
            tower_x = x + (1 if pin.name[1] == "d" else -1) if upper else x
            for k in range(banks):
                level = y + _BANK_PITCH * k
                if upper:
                    blocks[(x, level, north - 1)] = _DUST
                    blocks[(x, level - 1, north - 1)] = _GLASS
                    blocks[(tower_x, level, north - 1)] = _DUST
                    blocks[(tower_x, level - 1, north - 1)] = _GLASS
                    line(tower_x, level, north - 2)
                else:
                    line(x, level, north - 1)
                    line(x, level, north - 2)
            top = exit_y + _READ_PLANES if upper else exit_y
            tower(tower_x, north - 3, y, top)
            for z in range(north - 2, size[2]):
                if z == north - 2 or z == size[2] - 1 or (z - north) % _LANE_REPEAT == 0:
                    blocks[(tower_x, top, z)] = _repeater("north")
                else:
                    blocks[(tower_x, top, z)] = _DUST
                blocks[(tower_x, top - 1, z)] = _GLASS
            pins.append(PinRef(name=pin.name, direction=Direction.OUT, side=CardinalDirection.SOUTH, offset=(tower_x, top, 0)))
    edges: PinEdges = {}
    closing: dict[str, int] = {}
    if banks_out is None and not on_lanes:
        local: list = []
        shell = build_stack(name, row_bits, banks, base, abits, width, init, banks_out=local)
        names = [p.name for p in shell.pins]
        feeds = {pin: [(k, pin) for k in range(banks)] for pin in names if pin not in _array_timing(depth, width, 4, high_bits).outputs()}
        leaves = {(k, pin): pin for k in range(banks) for pin in names if pin not in feeds}
        edges, closing = _joined_timing(shell, local, feeds, leaves)
    block = _opaque(name, size, blocks, pins, edges, len(init) * width, closing)
    if closing:
        block.guarded = _write_guards(feeds)
    return block


@dataclass
class StackSpec:
    row_bits: int
    banks: int
    base: int
    abits: int
    width: int
    init: list[int]
    nets: dict[str, object]  # input pin -> the net on it


def _lone_lower_pin(specs: list[StackSpec]) -> tuple[int, str, str] | None:
    """A (stack, lower pin, upper pin) whose lower read-address pin has a net to itself under a shared upper one. The
    unfolded layout can't build it: the upper pin's tower reaches down its lane through the lower pin's line out."""
    users: dict[object, int] = {}
    for spec in specs:
        for net in spec.nets.values():
            users[net] = users.get(net, 0) + 1
    for k, spec in enumerate(specs):
        for bit in range(spec.abits):
            for lower, upper in (("aa", "ac"), ("ab", "ad")):
                alone, above = spec.nets.get(f"{lower}[{bit}]"), spec.nets.get(f"{upper}[{bit}]")
                if alone is not None and above is not None and users[alone] == 1 and users[above] > 1:
                    return k, f"{lower}[{bit}]", f"{upper}[{bit}]"
    return None


def build_group(name: str, specs: list[StackSpec], banks_out: list | None = None
                ) -> tuple[Macro, dict[str, tuple[int, str]], dict[object, str]]:
    """Stacks side by side, every net two or more of them share on a prebuilt lane below them. Other inputs stay on their
    stacks' own pins, carried out to the south face. Returns the block, then what its pins connect to: (stack, stack pin)
    per own pin, and lane pin per shared net.

    Each lane has its own Y, at one Z, so no lane meets another. Stack towers reach down to their lanes. A tower inverts
    every 2 blocks, so a lane an odd number of steps down carries the inverted signal, from a torch at its pin. All input
    pins are 1 mod 4 in Y, so one polarity suits every tower on a lane.
    """
    users: dict[object, list[tuple[int, str]]] = {}
    for k, spec in enumerate(specs):
        for pin, net in spec.nets.items():
            users.setdefault(net, []).append((k, pin))
    shared = [net for net, ends in users.items() if len(ends) > 1]
    if not _FOLD:
        lone = _lone_lower_pin(specs)
        if lone:
            raise ValueError(f"{name}: stack {lone[0]} has {lone[1]} alone but {lone[2]} shared")
    lane_pins = {net: f"lane{n}" for n, net in enumerate(shared)}
    towers: list[dict[str, tuple[int, int, int]]] = [{} for _ in specs]
    local: list[list] = [[] for _ in specs]
    built = [build_stack(f"{name}_{k}", spec.row_bits, spec.banks, spec.base, spec.abits, spec.width, spec.init, towers[k],
                         frozenset(pin for pin, net in spec.nets.items() if net in lane_pins),
                         local[k] if banks_out is not None else None)
             for k, spec in enumerate(specs)]
    lift = _LANE_PITCH * len(shared) + 2
    lane_z = max(m.footprint.depth for m in built) + 1
    xs = [sum(m.footprint.width + 1 for m in built[:k]) for k in range(len(built))]
    size = (xs[-1] + built[-1].footprint.width + 1, lift + max(m.footprint.height for m in built), lane_z + 4)
    blocks: dict[Pos, BlockState] = {}
    pins: list[PinRef] = []
    own_pins: dict[str, tuple[int, str]] = {}
    if banks_out is not None:
        for k, stack_banks in enumerate(local):
            banks_out.extend(((xs[k] + o[0], lift + o[1], o[2]), args, size) for o, args, size in stack_banks)

    def floor(x: int, y: int, z: int) -> None:
        blocks.setdefault((x, y - 1, z), _GLASS)

    def dust(x: int, y: int, z: int) -> None:
        blocks[(x, y, z)] = _DUST
        floor(x, y, z)

    for k, sub in enumerate(built):
        depth = sub.footprint.depth
        for (x, y, z), block in sub.blocks:
            blocks[(x + xs[k], y + lift, z)] = block
        for pin in sub.pins:
            group_pin = f"s{k}_{pin.name}"
            own_pins[group_pin] = (k, pin.name)
            x, y = pin.offset[0] + xs[k], pin.offset[1] + lift
            if pin.direction == Direction.OUT:
                for z in range(depth, size[2]):
                    blocks[(x, y, z)] = _repeater("north")
                    floor(x, y, z)
                pins.append(PinRef(name=group_pin, direction=Direction.OUT, side=CardinalDirection.SOUTH, offset=(x, y, 0)))
            else:
                for z in range(depth, size[2]):
                    blocks[(x, y, z)] = _repeater("south")
                    floor(x, y, z)
                pins.append(PinRef(name=group_pin, direction=Direction.IN, side=CardinalDirection.SOUTH, offset=(x, y, 0)))

    pin_step = max(_LANE_PIN_PITCH, size[0] // max(len(shared), 1))
    for n, net in enumerate(shared):
        y = 1 + _LANE_PITCH * n
        inverted = False
        taps: list[int] = []
        for k, pin_name in users[net]:
            x, pin_y, z = towers[k][pin_name]
            x += xs[k]
            top = pin_y + lift
            inverted = (top - y) // 2 % 2 == 1
            # Up the tower: a block every 2, idle state set from the top, where an idle pin is low.
            for level in range(y, top + 1, 2):
                blocks[(x, level, z)] = _STONE
                if level < top:
                    blocks[(x, level + 1, z)] = _torch((top - level) // 2 % 2 == 0)
            # A read pin sharing its column with the upper read pin's further tower goes around it, on x + 2.
            around = pin_name[:2] in ("aa", "ab")
            if _FOLD:
                around = any(t[0] == towers[k][pin_name][0] and t[2] > z for t in towers[k].values())
            tap_x = x + 2 if around else x
            if around:
                blocks[(x + 1, y, z)] = _repeater("east")
                floor(x + 1, y, z)
                dust(tap_x, y, z)
            for cell_z in range(z + 1, lane_z - 1):
                if (lane_z - cell_z) % _LANE_REPEAT == 0:  # a shorter stack is further from its lane
                    blocks[(tap_x, y, cell_z)] = _repeater("south")
                    floor(tap_x, y, cell_z)
                else:
                    dust(tap_x, y, cell_z)
            blocks[(tap_x, y, lane_z - 1)] = _repeater("south")  # the lane may have little signal left here
            floor(tap_x, y, lane_z - 1)
            taps.append(tap_x)
        entry = min(n * pin_step + 1, size[0] - 1)

        def lay(x: int, repeat: bool, facing: str = "east") -> None:
            if repeat:
                blocks[(x, y, lane_z)] = BlockState("minecraft:repeater", facing=facing, delay="1", locked="false",
                                                    powered="true" if inverted else "false")
            else:
                blocks[(x, y, lane_z)] = BlockState("minecraft:redstone_wire", power="14") if inverted else _DUST
            floor(x, y, lane_z)

        lay(entry, False)
        for step, facing in ((-1, "east"), (1, "west")):
            since = 0
            x = entry + step
            while 0 <= x < size[0]:
                repeat = since >= _LANE_REPEAT and x not in taps
                lay(x, repeat, facing)
                since = 0 if repeat else since + 1
                x += step
        pin_z = lane_z + 3
        if inverted:
            blocks[(entry, y, lane_z + 1)] = BlockState("minecraft:redstone_wall_torch", facing="north", lit="true")
            blocks[(entry, y, lane_z + 2)] = _STONE
        else:
            for z in (lane_z + 1, lane_z + 2):
                blocks[(entry, y, z)] = _repeater("south")
                floor(entry, y, z)
        blocks[(entry, y, pin_z)] = _repeater("south")
        floor(entry, y, pin_z)
        pins.append(PinRef(name=lane_pins[net], direction=Direction.IN, side=CardinalDirection.SOUTH, offset=(entry, y, 0)))
    cells = sum(len(spec.init) * spec.width for spec in specs)
    block = _opaque(name, size, blocks, pins, {}, cells)
    if banks_out is None:  # with the arrays in the blocks: time the same thing without them
        timed = build_group(name, specs, [])[0]
        block.edges, block.closing, block.guarded = timed.edges, timed.closing, timed.guarded
        block.arcs = _longest(block.edges)
        return block, own_pins, lane_pins
    pin_of = {(k, pin): lane_pins[net] for net, ends in users.items() if net in lane_pins for k, pin in ends}
    feeds: dict[str, list[tuple[int, str]]] = {}
    leaves: dict[tuple[int, str], str] = {}
    placed: list = []
    for k, (spec, stack_banks) in enumerate(zip(specs, local)):
        depth = 1 << spec.row_bits
        outputs = _array_timing(depth, spec.width, 4, spec.abits - spec.row_bits).outputs()
        for j, item in enumerate(stack_banks):
            n = len(placed)
            placed.append(((xs[k] + item[0][0], lift + item[0][1], item[0][2]), item[1], item[2]))
            for pin in _array_timing(depth, spec.width, 4, spec.abits - spec.row_bits).pins:
                if pin in outputs:
                    leaves[n, pin] = f"s{k}_{pin}"
                else:
                    target = pin_of.get((k, pin), f"s{k}_{pin}")
                    if target in {p.name for p in pins}:
                        feeds.setdefault(target, []).append((n, pin))
    edges, closing = _joined_timing(block, placed, feeds, leaves)
    block.edges, block.closing = _through_pin(edges), {pin: ticks + 1 for pin, ticks in closing.items()}
    block.guarded = _write_guards(feeds)
    block.arcs = _longest(block.edges)
    return block, own_pins, lane_pins


_built: dict[str, Macro] = {}
_cache_dir: Path | None = None
_CODE = ("ram.py", "ram_timing.py", "tick_sim.py", "macro.py", "propagate.py", "cosmetics.py", "verify.py", "block_utils.py")


def set_block_cache(directory: Path | None) -> None:
    global _cache_dir
    _cache_dir = directory


def settled_block(block: Macro, banks: list[tuple[Pos, Macro]] = ()) -> Macro:
    """The block alone, its dust shaped and everything in the state it rests in with its pins low, as arrays. A build
    pastes this and only looks inside if the wire outside touches it or drives a pin high.

    banks: arrays already settled alone, with their origins in the block. They are pasted and treated as opaque boxes, so
    only the wire around them is shaped and settled here; the result is the same as settling it all at once."""
    size = block.footprint.width, block.footprint.height, block.footprint.depth
    region = BulkRegion(0, 0, 0, *size)
    arrays = block.arrays_oriented(False, False)
    region.paste(arrays.xs, arrays.ys, arrays.zs, arrays.ids, arrays.palette)
    boxes = []
    for (ox, oy, oz), bank in banks:
        w, h, d = bank.footprint.width, bank.footprint.height, bank.footprint.depth
        box = (ox, oy, oz), (ox + w, oy + h, oz + d)
        if region._Region__blocks[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]].any():
            raise ValueError(f"{block.name}: wire inside the box of a bank at {(ox, oy, oz)}")
        settled = bank.settled
        region.paste(settled.xs + ox, settled.ys + oy, settled.zs + oz, settled.ids, settled.palette)
        region.tile_entities.extend(comparator_entity((x + ox, y + oy, z + oz), level) for (x, y, z), level in settled.comparators)
        boxes.append(box)
    apply_dust_shapes(region, boxes)
    settle(region, boxes)
    region.tile_entities.sort(key=lambda e: e.position)
    unsupported = verify_support(region, {}, skip=boxes)
    if unsupported:
        log(f"[ram] {block.name}: {len(unsupported)} unsupported block(s), first {unsupported[0].pos}")
    ids = region._Region__blocks
    cells = np.argwhere(ids != 0).astype(np.int32)
    return replace(block, blocks=[], settled=BlockArrays(cells[:, 0], cells[:, 1], cells[:, 2], ids[ids != 0].astype(np.int32),
                                                         list(region._Region__palette),
                                                         [(e.position, int(e.to_nbt()["OutputSignal"])) for e in region.tile_entities]))


def _save(path: Path, block: Macro, extra: object) -> None:
    arrays = block.settled
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}")
    with open(partial, "wb") as f:
        palette = [(b.id, dict(b.properties())) for b in arrays.palette]
        pickle.dump((replace(block, settled=None), extra, palette, arrays.comparators), f)
        np.savez(f, xs=arrays.xs.astype(np.uint16), ys=arrays.ys.astype(np.uint16), zs=arrays.zs.astype(np.uint16),
                 ids=arrays.ids.astype(np.uint32))
    os.replace(partial, path)


def _load(path: Path) -> tuple[Macro, object] | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        block, extra, palette, comparators = pickle.load(f)
        palette = [BlockState(name, **props) for name, props in palette]
        data = np.load(f)
        block.settled = BlockArrays(data["xs"].astype(np.int32), data["ys"].astype(np.int32), data["zs"].astype(np.int32),
                                    data["ids"].astype(np.int32), palette, comparators)
    return block, extra


def _cached(key: object, build: Callable[[], tuple[Macro, object]],
            finish: Callable[[Macro], Macro] = settled_block, subdirectory: str = "") -> tuple[Macro, object]:
    """A settled block and what else its build returned, from cache_dir if this code has built this block before."""
    digest = hashlib.sha256(repr((key, _ADDR_PITCH, *(["fold"] if _FOLD else []))).encode())
    for name in _CODE:
        digest.update((Path(__file__).parent / name).read_bytes())
    path = None if _cache_dir is None else _cache_dir / subdirectory / f"{digest.hexdigest()[:24]}.block"
    if path is not None and (found := _load(path)) is not None:
        log(f"[ram] {found[0].name}: {len(found[0].settled.ids)} blocks from {path}")
        return found
    with phase("[ram] building and settling a block"):
        block, extra = build()
        block = finish(block)
    if path is not None:
        _save(path, block, extra)
    return block, extra


@dataclass
class _ArrayTiming:
    """What simulating an array of a shape found. pins: name -> cell in the array."""
    edges: PinEdges
    closing: dict[str, int]
    pins: dict[str, Pos]
    output_names: frozenset[str]

    def edges_closing(self) -> tuple[PinEdges, dict[str, int]]:
        return self.edges, self.closing

    def outputs(self) -> frozenset[str]:
        return self.output_names


_array_times: dict[tuple, _ArrayTiming] = {}


def _array_timing(depth: int, width: int, reads: int, high_bits: int) -> _ArrayTiming:
    """Contents and which upper address an array answers to don't change its timing, so each shape is simulated once."""
    shape = depth, width, reads, high_bits, _FOLD
    found = _array_times.get(shape)
    if found is not None:
        return found
    digest = hashlib.sha256(repr((shape, _ADDR_PITCH)).encode())
    for code in _CODE:
        digest.update((Path(__file__).parent / code).read_bytes())
    path = None if _cache_dir is None else _cache_dir / "timing" / f"{digest.hexdigest()[:24]}.pickle"
    if path is not None and path.exists():
        with open(path, "rb") as f:
            found = pickle.load(f)
    else:
        with phase(f"[ram] timing an array of {depth}x{width}, {reads} reads"):
            block = settled_block(build_ram("timed", depth, width, reads, checker_words(depth, width), high_bits, 0, measure=False))
            edges, closing = measure_array(block, depth, width, depth.bit_length() - 1 + high_bits, "abcd"[:reads])
            found = _ArrayTiming(edges, closing, {p.name: pin_cell(block, p.name) for p in block.pins},
                                 frozenset(p.name for p in block.pins if p.direction == Direction.OUT))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f"{path.name}.{os.getpid()}")
            with open(partial, "wb") as f:
                pickle.dump(found, f)
            os.replace(partial, path)
    _array_times[shape] = found
    return found


def _joined_timing(shell: Macro, banks: list, feeds: dict[str, list[tuple[int, str]]],
                   leaves: dict[tuple[int, str], str]) -> tuple[PinEdges, dict[str, int]]:
    """A block's arcs: its wire measured without its arrays, joined with theirs. banks: (origin, array arguments, size).
    feeds: input pin -> (bank, array pin) it reaches. leaves: (bank, array output pin) -> the block's output pin."""
    timing = [_array_timing(args[0], args[1], 4, args[3]) for _, args, _ in banks]

    def cell(n: int, pin: str) -> Pos:
        origin, _, _ = banks[n]
        return tuple(a + b for a, b in zip(origin, timing[n].pins[pin]))

    def entry(n: int, pin: str) -> Pos:
        x, y, z = cell(n, pin)
        return x, y, z + 1  # the wire's last cell, south of the array's pin

    entries = {pin: [entry(n, array_pin) for n, array_pin in ends] for pin, ends in feeds.items()}
    exits = {}
    for (n, pin), out in leaves.items():
        exits.setdefault(cell(n, pin), (_repeater("south"), []))[1].append(out)
    into, out = measure_wire(dict(shell.blocks), (shell.footprint.width, shell.footprint.height, shell.footprint.depth), shell,
                             entries, [(p, b, pins) for p, (b, pins) in exits.items()])
    edges: PinEdges = {}
    closing: dict[str, int] = {}
    pin_repeater: Edges = {(False, False): 1, (True, True): 1}
    for pin, ends in feeds.items():
        for n, array_pin in ends:
            wire = into.get((pin, entry(n, array_pin)))
            if not wire:
                raise ValueError(f"{shell.name}: {pin} doesn't reach {array_pin} of bank {n}")
            wire = joined(wire, pin_repeater)
            if array_pin in timing[n].closing:
                closing[pin] = max(closing.get(pin, 0), max(wire.values()) + timing[n].closing[array_pin])
            for array_out, arc in timing[n].edges.get(array_pin, {}).items():
                leaving = leaves.get((n, array_out))
                if leaving is None:
                    continue
                total = joined(wire, arc, out[cell(n, array_out), leaving])
                have = edges.setdefault(pin, {}).setdefault(leaving, {})
                for key, ticks in total.items():
                    have[key] = max(have.get(key, 0), ticks)
    return edges, closing


def _write_guards(feeds: dict[str, list[tuple[int, str]]]) -> dict[str, list[str]]:
    """Each block pin carrying write enable -> those carrying write address and data to the same arrays."""
    enabled = {pin: {n for n, array_pin in ends if array_pin == "we"} for pin, ends in feeds.items()}
    return {pin: [other for other, ends in feeds.items()
                  if any(n in banks and array_pin.startswith(("wa[", "wd[")) for n, array_pin in ends)]
            for pin, banks in enabled.items() if banks}


def _bank_block(args: tuple) -> Macro:
    """One bank of a stack, settled alone: the same bank in another stack or group is read from the cache."""
    depth, width, init, high_bits, high = args
    return _cached(("bank", *args), lambda: (build_ram("bank", depth, width, 4, list(init), high_bits, high, measure=False), None),
                   subdirectory="banks")[0]


def _bank_job(args: tuple) -> tuple[tuple, tuple]:
    block = _bank_block(args)
    arrays = block.settled
    palette = [(b.id, dict(b.properties())) for b in arrays.palette]
    return args, (replace(block, settled=None), palette, arrays.comparators, arrays.xs.astype(np.uint16),
                  arrays.ys.astype(np.uint16), arrays.zs.astype(np.uint16), arrays.ids.astype(np.uint32))


def _settled_banks(jobs: list[tuple]) -> dict[tuple, Macro]:
    """Each distinct bank settled, in as many processes as RAM_WORKERS says (the number of cores by default)."""
    unique = sorted(dict.fromkeys(jobs), key=lambda a: -a[0] * a[1])
    workers = min(int(os.environ.get("RAM_WORKERS", os.cpu_count() or 1)), len(unique))
    if workers <= 1:
        return {args: _bank_block(args) for args in unique}
    found: dict[tuple, Macro] = {}
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for args, (block, palette, comparators, xs, ys, zs, ids) in pool.imap_unordered(_bank_job, unique, chunksize=1):
            palette = [BlockState(name, **props) for name, props in palette]
            block.settled = BlockArrays(xs.astype(np.int32), ys.astype(np.int32), zs.astype(np.int32), ids.astype(np.int32),
                                        palette, comparators)
            found[args] = block
    return found


def built_block(name: str) -> Macro:
    return _built[name]


def block_for_cell(cell_type: str, parameters: dict[str, str]) -> Macro:
    """The block for a synthesized cell. Arrays differing only in contents are separate blocks."""
    if cell_type == WINDOW_CELL:
        block = build_window()
    elif cell_type == DELAY_CELL:
        block = build_delay()
    elif cell_type == RAM_CELL:
        depth, width = int(parameters["DEPTH"], 2), int(parameters["WIDTH"], 2)
        block = build_ram(f"ram_{depth}x{width}", depth, width)
    else:
        spec = stack_spec(parameters, {})
        words = len(spec.init)
        digest = hashlib.sha256(repr(spec.init).encode()).hexdigest()[:8]
        name = f"stack_{words}x{spec.width}_{spec.base}_{spec.abits}_{digest}"
        if name in _built:
            return _built[name]
        block, _ = _cached(("stack", spec.row_bits, spec.banks, spec.base, spec.abits, spec.width, spec.init),
                           lambda: (build_stack(name, spec.row_bits, spec.banks, spec.base, spec.abits, spec.width, spec.init), None))
    return _built.setdefault(block.name, block)


def stack_spec(parameters: dict[str, str], nets: dict[str, object]) -> StackSpec:
    row_bits, banks, base, abits, width = (int(parameters[k], 2) for k in ("ROW_BITS", "BANKS", "BASE", "ABITS", "WIDTH"))
    words = banks << row_bits
    # Written lsb last by yosys. An undefined bit powers up 0.
    bits = parameters["INIT"].replace("x", "0").zfill(words * width)[::-1]
    init = [int(bits[k * width:(k + 1) * width][::-1], 2) for k in range(words)]
    return StackSpec(row_bits, banks, base, abits, width, init, nets)


def _group_width(specs: list[StackSpec]) -> int:
    """X of a group of stacks side by side, unfolded: bits, then three decoders."""
    return sum(_COLUMN_PITCH * s.width + 3 * _ADDR_PITCH * s.abits + 6 for s in specs)


def group_block(specs: list[StackSpec]) -> tuple[Macro, dict[str, tuple[int, str]], dict[object, str]]:
    global _FOLD
    if _FOLD_FIXED is None:
        _FOLD = _group_width(specs) > _FOLD_GROUP_WIDTH or _lone_lower_pin(specs) is not None
    numbers: dict[object, int] = {}  # a block depends on which pins share a net, not on what nets are called
    shape = [(s.row_bits, s.banks, s.base, s.abits, s.width, s.init,
              [(pin, numbers.setdefault(net, len(numbers))) for pin, net in s.nets.items()]) for s in specs]
    name = f"group_{hashlib.sha256(repr(shape).encode()).hexdigest()[:8]}"

    def build() -> tuple[Macro, object]:
        banks: list = []
        block, own, lanes = build_group(name, specs, banks)
        settled = _settled_banks([args for _, args, _ in banks])
        block = settled_block(block, [(origin, settled[args]) for origin, args, _ in banks])
        return block, (own, {numbers[net]: pin for net, pin in lanes.items()})

    block, (own, lanes) = _cached(("group", shape), build, lambda block: block)
    _built[block.name] = block
    return block, own, {net: lanes[number] for net, number in numbers.items() if number in lanes}
