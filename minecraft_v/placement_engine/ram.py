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
import os

from litemapy import BlockState

from minecraft_v.placement_engine.ir import CardinalDirection, Direction, Footprint, PinRef

from .macro import Macro

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
_ADDR_PITCH = int(os.environ.get("RAM_ADDR_PITCH", 4))  # an address bit's true and complement lines; also the pins' X pitch
_DECODER_REPEAT = 12 // _ADDR_PITCH * _ADDR_PITCH  # X between repeaters on a row line in the decoder: under a true line, never a torch

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
GROUP_CELL = "ram_group"
BLOCK_CELLS = (RAM_CELL, RAM4_CELL, WINDOW_CELL, DELAY_CELL, GROUP_CELL)


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


def build_ram(name: str, depth: int, width: int, reads: int = 2, init: list[int] | None = None,
              high_bits: int = 0, high: int = 0) -> Macro:
    """``reads`` is 2 or 4. ``init`` holds a preloaded word per row. With ``high_bits``, addresses have that many more
    bits, and this array answers only when they equal ``high``: several can share address lines and OR their bitlines."""
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


def build_stack(name: str, row_bits: int, banks: int, base: int, abits: int, width: int, init: list[int],
                towers: dict[str, tuple[int, int, int]] | None = None) -> Macro:
    """``banks`` 4-read arrays one above the other, as one block, holding bank numbers ``base`` up. Every input pin feeds a
    torch tower beside its line; each bank's line taps the tower. Each output has a tower too, which every bank's bitline
    powers at its own level, so the top of the tower is the OR of the banks. Lines near towers are repeaters, which ignore
    them. A tower inverts twice per 4 blocks, so its tap levels agree. With ``towers``, input pins get no way out to the
    south face: it collects each pin's tower base (x, y, z) instead, for build_group to feed.
    """
    depth = 1 << row_bits
    high_bits = abits - row_bits
    arrays = [build_ram(name, depth, width, 4, init[k * depth:(k + 1) * depth], high_bits, base + k) for k in range(banks)]
    bank = arrays[0].footprint
    north = _STACK_NORTH
    south = _STACK_SOUTH
    size = (bank.width + 1, bank.height + _BANK_PITCH * (banks - 1), north + bank.depth + south)
    blocks: dict[Pos, BlockState] = {}
    for k, array in enumerate(arrays):
        for (x, y, z), block in array.blocks:
            blocks[(x, y + _BANK_PITCH * k, z + north)] = block

    def line(x: int, y: int, z: int, facing: str = "south") -> None:
        blocks[(x, y, z)] = _repeater(facing)
        blocks[(x, y - 1, z)] = _GLASS

    def tower(x: int, z: int, bottom: int) -> None:
        """Up from ``bottom`` through the last bank's level, in the state of an idle low signal."""
        top = bottom + _BANK_PITCH * (banks - 1)
        for y in range(bottom, top + 1, 2):
            blocks[(x, y, z)] = _STONE
            if y < top:
                blocks[(x, y + 1, z)] = _torch((y - bottom) // 2 % 2 == 0)

    pins: list[PinRef] = []
    arcs_in = _TOWER_TICKS * banks
    for pin in arrays[0].pins:
        x, y = pin.offset[0], pin.offset[1]
        if pin.direction == Direction.IN:
            # Upper read pins' towers sit further out: a lower line reaches its pin under them. So does write enable's,
            # whose tower would otherwise touch the first write address line.
            z = north + bank.depth + (_STACK_SOUTH_UPPER if y > _Y_READ_ADDR or pin.name == "we" else _STACK_SOUTH_LOWER)
            tower(x + 1, z, y)
            for k in range(banks):
                level = y + _BANK_PITCH * k
                blocks[(x, level, z)] = _DUST
                blocks[(x, level - 1, z)] = _GLASS
                for approach in range(north + bank.depth, z):
                    line(x, level, approach)
            if towers is not None:
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
            tower(tower_x, north - 3, y)
            top = y + _BANK_PITCH * (banks - 1)
            line(tower_x, top, north - 4)
            line(tower_x, top, north - 5)
            pins.append(PinRef(name=pin.name, direction=Direction.OUT, side=CardinalDirection.NORTH,
                               offset=(tower_x, top, size[2] - 1)))
    arcs = {pin: {out: ticks + arcs_in for out, ticks in outs.items()} for pin, outs in arrays[0].arcs.items()}
    return _opaque(name, size, blocks, pins, arcs, len(init) * width)


def build_group(name: str, row_bits: int, banks: int, stacks: int, abits: int, width: int) -> Macro:
    """``stacks`` stacks side by side sharing every input, on prebuilt lanes below them. Bank numbers run on up the stacks.

    Each input's lane has its own Y, at one Z, so no lane meets another. Stack towers reach down to their lanes. A tower
    inverts every 2 blocks, so a lane an odd number of steps down carries the inverted signal, from a torch at its pin.
    TODO: contents, and stacks of different memories.
    """
    words = banks << row_bits
    towers: list[dict[str, tuple[int, int, int]]] = [{} for _ in range(stacks)]
    built = [build_stack(name, row_bits, banks, k * banks, abits, width, [0] * words, towers[k]) for k in range(stacks)]
    stack = built[0].footprint
    inputs = sorted(towers[0], key=lambda n: (towers[0][n][1], towers[0][n][0]))
    lift = _LANE_PITCH * len(inputs) + 2
    pitch = stack.width + 1
    lane_z = stack.depth + 1
    size = (pitch * stacks, lift + stack.height, lane_z + 4)
    blocks: dict[Pos, BlockState] = {}
    pins: list[PinRef] = []
    for k, sub in enumerate(built):
        for (x, y, z), block in sub.blocks:
            blocks[(x + k * pitch, y + lift, z)] = block
        for pin in sub.pins:
            if pin.direction == Direction.OUT:
                name_part, index = pin.name[:-1].split("[")
                pins.append(PinRef(name=f"{name_part}[{k * width + int(index)}]", direction=Direction.OUT,
                                   side=CardinalDirection.NORTH,
                                   offset=(pin.offset[0] + k * pitch, pin.offset[1] + lift, size[2] - 1)))

    def floor(x: int, y: int, z: int) -> None:
        blocks.setdefault((x, y - 1, z), _GLASS)

    def dust(x: int, y: int, z: int) -> None:
        blocks[(x, y, z)] = _DUST
        floor(x, y, z)

    pin_step = max(_LANE_PIN_PITCH, size[0] // max(len(inputs), 1))
    for n, pin_name in enumerate(inputs):
        y = 1 + _LANE_PITCH * n
        inverted = False
        taps: list[int] = []
        for k in range(stacks):
            x, pin_y, z = towers[k][pin_name]
            x += k * pitch
            steps = (pin_y + lift - y) // 2
            inverted = steps % 2 == 1
            # Up the tower: a block every 2, idle state set from the top, where an idle pin is low.
            for level in range(y, pin_y + lift + 1, 2):
                powered = (pin_y + lift - level) // 2 % 2 == 1
                blocks[(x, level, z)] = _STONE
                if level < pin_y + lift:
                    blocks[(x, level + 1, z)] = _torch(not powered)
            # A read pin sharing its column with the upper read pin's further tower goes around it, on x + 2.
            around = pin_name[:2] in ("aa", "ab")
            tap_x = x + 2 if around else x
            if around:
                blocks[(x + 1, y, z)] = _repeater("east")
                floor(x + 1, y, z)
                dust(tap_x, y, z)
            for cell_z in range(z + 1, lane_z - 1):
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
        pins.append(PinRef(name=pin_name, direction=Direction.IN, side=CardinalDirection.SOUTH, offset=(entry, y, 0)))
    along = size[0] // _LANE_REPEAT + lift // 2
    arcs = {pin: {f"{o[:2]}[{k * width + int(o[3:-1])}]": ticks + along for k in range(stacks) for o, ticks in outs.items()}
            for pin, outs in built[0].arcs.items()}
    return _opaque(name, size, blocks, pins, arcs, stacks * words * width)


_built: dict[str, Macro] = {}


def built_block(name: str) -> Macro:
    return _built[name]


def block_for_cell(cell_type: str, parameters: dict[str, str]) -> Macro:
    """The block for a synthesized cell. Arrays differing only in contents are separate blocks."""
    if cell_type == WINDOW_CELL:
        block = build_window()
    elif cell_type == DELAY_CELL:
        block = build_delay()
    elif cell_type == GROUP_CELL:
        row_bits, banks, stacks, abits, width = (int(parameters[k], 2) for k in ("ROW_BITS", "BANKS", "STACKS", "ABITS", "WIDTH"))
        block = build_group(f"group_{stacks}x{banks}x{row_bits}x{width}_{abits}", row_bits, banks, stacks, abits, width)
    elif cell_type == RAM_CELL:
        depth, width = int(parameters["DEPTH"], 2), int(parameters["WIDTH"], 2)
        block = build_ram(f"ram_{depth}x{width}", depth, width)
    else:
        row_bits, banks, base, abits, width = (int(parameters[k], 2) for k in ("ROW_BITS", "BANKS", "BASE", "ABITS", "WIDTH"))
        words = banks << row_bits
        # Written lsb last by yosys. An undefined bit powers up 0.
        bits = parameters["INIT"].replace("x", "0").zfill(words * width)[::-1]
        init = [int(bits[k * width:(k + 1) * width][::-1], 2) for k in range(words)]
        digest = hashlib.sha256(bits.encode()).hexdigest()[:8]
        block = build_stack(f"stack_{words}x{width}_{base}_{abits}_{digest}", row_bits, banks, base, abits, width, init)
    return _built.setdefault(block.name, block)
