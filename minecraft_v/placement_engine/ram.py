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

from litemapy import BlockState

from minecraft_v.placement_engine.ir import CardinalDirection, Direction, Footprint, PinRef

from .macro import Macro

Pos = tuple[int, int, int]

_COLUMN_PITCH = 6
_ROW_PITCH = 6
_REPEAT_EVERY = 2  # rows or columns between line repeaters; dust from a repeater reaches 15 cells

_Y_WRITE, _Y_STORE, _Y_LOCK, _Y_MIDDLE, _Y_READ, _Y_CELL, _Y_WORD, _Y_READ_ADDR = 1, 2, 3, 4, 5, 6, 7, 9
_READ_PLANES = 4     # Y from one pair of read ports to the next
_LINE_PITCH = 2      # X between decoder lines; each line's torches sit between it and the next
_ADDR_PITCH = 4      # an address bit's true and complement lines
_DECODER_REPEAT = 12 # X between repeaters on a row line in the decoder; on even X, under a line, never under a torch

_GLASS = BlockState("minecraft:glass")
_STONE = BlockState("minecraft:smooth_quartz")
_DUST = BlockState("minecraft:redstone_wire")
# TODO: other port counts: more read ports need another S per pair, more write ports a per-bit OR.
RAM_CELL = "ram_2r1w"
RAM4_CELL = "ram_4r1w"
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


def _opaque(name: str, size: tuple[int, int, int], blocks: dict[Pos, BlockState], pins: list[PinRef],
            arcs: dict[str, dict[str, int]], cells: int) -> Macro:
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
                 blocks=list(blocks.items()), arcs=arcs, q_out={}, setup={}, terminals=terminals,
                 gate_boxes=[((0, 0, 0), size)], nets=nets, port_net=port_net, cells=cells)


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
    return _opaque(WINDOW_MACRO, size, blocks, pins, {"clk": {"window": 8}}, 1)


def build_ram(name: str, depth: int, width: int, reads: int = 2, init: list[int] | None = None, banked: bool = False) -> Macro:
    """``reads`` is 2 or 4. ``init`` holds a preloaded word per row. ``banked`` adds an enable per read port: a disabled
    port selects no row, so several arrays can OR their bitlines."""
    abits = depth.bit_length() - 1
    if abits < 1 or depth != 1 << abits:
        raise ValueError(f"{name}: depth {depth} is not a power of two above 1")
    decoder_x = _COLUMN_PITCH * width + 2
    enable = _LINE_PITCH if banked else 0
    read_a_x = decoder_x + enable
    read_b_x = read_a_x + _ADDR_PITCH * abits + enable
    write_x = decoder_x + _LINE_PITCH  # after the write enable line
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
            tap(x if k >> bit & 1 else x + 2, y, z)

    lock_rows = [(k, row_z(k) - 2) for k in range(depth)]
    write_end = write_x + _ADDR_PITCH * abits
    ports = [("a", 0, -2, read_a_x), ("b", 0, 2, read_b_x)]
    if reads == 4:
        ports += [("c", _READ_PLANES, -2, read_a_x), ("d", _READ_PLANES, 2, read_b_x)]
    for port, up, side, line_x in ports:
        rows = [(k, row_z(k) + side) for k in range(depth)]
        for _, z in rows:
            row_line(_Y_WORD + up, z, line_x + _ADDR_PITCH * abits)
        for bit in range(abits):
            address_lines(line_x + _ADDR_PITCH * bit, _Y_READ_ADDR + up, f"a{port}[{bit}]", rows, bit)
        if banked:
            enable_x = line_x - enable
            bitline(enable_x, _Y_READ_ADDR + up, size_z - 1)
            add_pin(f"e{port}", Direction.IN, CardinalDirection.SOUTH, (enable_x, _Y_READ_ADDR + up, size_z - 1))
            for _, z in rows:
                tap(enable_x, _Y_READ_ADDR + up, z)
    for _, z in lock_rows:
        row_line(_Y_LOCK, z, write_end, high=True)
    for bit in range(abits):
        address_lines(write_x + _ADDR_PITCH * bit, _Y_READ, f"wa[{bit}]", lock_rows, bit)
    bitline(decoder_x, _Y_READ, size_z - 1)
    add_pin("we", Direction.IN, CardinalDirection.SOUTH, (decoder_x, _Y_READ, size_z - 1))
    for _, z in lock_rows:
        tap(decoder_x, _Y_READ, z)

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

    # Worst paths, in repeater ticks. A written bit goes through its latch and both torches before the read tap.
    along_decoder = 1 + (size_x - decoder_x) // _DECODER_REPEAT
    along_row = along_decoder + 1 + width // _REPEAT_EVERY
    along_column = 1 + depth // _REPEAT_EVERY + 1
    decode = 1 + depth // _REPEAT_EVERY + 2  # along the address line, complement torch, tap torch
    tap_ticks = 2  # side repeater or torch, then comparator
    read = decode + along_row + tap_ticks + along_column
    written = 4 + (2 if reads == 4 else 0) + along_column
    outs = [port for port, _, _, _ in ports]
    arcs: dict[str, dict[str, int]] = {}
    for bit in range(abits):
        for port in outs:
            arcs[f"a{port}[{bit}]"] = {f"r{port}[{i}]": read for i in range(width)}
        arcs[f"wa[{bit}]"] = {f"r{p}[{i}]": decode + along_row + written for p in outs for i in range(width)}
    if banked:
        for port in outs:
            arcs[f"e{port}"] = {f"r{port}[{i}]": read for i in range(width)}
    arcs["we"] = {f"r{p}[{i}]": decode + along_row + written for p in outs for i in range(width)}
    for i in range(width):
        arcs[f"wd[{i}]"] = {f"r{p}[{i}]": along_column + written for p in outs}
    return _opaque(name, (size_x, size_y, size_z), blocks, pins, arcs, depth * width)


def build_delay(name: str = "ram_delay") -> Macro:
    """Clock delayed three ways, for the write turns: outputs are repeater chain taps."""
    size = (_DELAY_CELLS + 2, 2, 4)
    y = 1
    blocks: dict[Pos, BlockState] = {(x, 0, z): _GLASS for x in range(size[0]) for z in range(size[2])}
    blocks[(0, y, 3)] = _repeater("south")
    blocks[(0, y, 2)] = _DUST
    pins = [PinRef(name="clk", direction=Direction.IN, side=CardinalDirection.SOUTH, offset=(0, y, 0))]
    arcs: dict[str, int] = {}
    ticks = 1
    taps = _DELAY_TAPS
    for x in range(1, size[0]):
        cell = x - 1
        if cell in taps:
            blocks[(x, y, 2)] = _DUST
            blocks[(x, y, 1)] = _repeater("south")
            blocks[(x, y, 0)] = _repeater("south")
            pins.append(PinRef(name=taps[cell], direction=Direction.OUT, side=CardinalDirection.NORTH, offset=(x, y, size[2] - 1)))
            arcs[taps[cell]] = ticks + 2
        else:
            blocks[(x, y, 2)] = BlockState("minecraft:repeater", facing="west", delay="4", powered="false", locked="false")
            ticks += 4
    return _opaque(name, size, blocks, pins, {"clk": arcs}, 1)


_built: dict[str, Macro] = {}


def built_block(name: str) -> Macro:
    return _built[name]


def block_for_cell(cell_type: str, parameters: dict[str, str]) -> Macro:
    """The block for a synthesized cell. Arrays differing only in contents are separate blocks."""
    if cell_type == WINDOW_CELL:
        block = build_window()
    elif cell_type == DELAY_CELL:
        block = build_delay()
    else:
        depth, width = int(parameters["DEPTH"], 2), int(parameters["WIDTH"], 2)
        if cell_type == RAM_CELL:
            block = build_ram(f"ram_{depth}x{width}", depth, width)
        else:
            # Written lsb last by yosys. An undefined bit powers up 0.
            bits = parameters["INIT"].replace("x", "0").zfill(depth * width)[::-1]
            init = [int(bits[k * width:(k + 1) * width][::-1], 2) for k in range(depth)]
            digest = hashlib.sha256(bits.encode()).hexdigest()[:8]
            block = build_ram(f"ram4_{depth}x{width}_{digest}", depth, width, 4, init, banked=True)
    return _built.setdefault(block.name, block)
