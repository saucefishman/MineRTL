"""Tiled RAM: a generated array of latches with two read ports and one write port, kept as a macro.

No routing inside. Lines along X and lines along Z sit on different planes, so they never cross:

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

Pins are active low row lines and plain bitlines. Decoding stays in gates, see techmap/ram_2r1w.v, and writes take the
self-closing pulse of build_window.
"""
from __future__ import annotations

from litemapy import BlockState

from minecraft_v.placement_engine.ir import CardinalDirection, Direction, Footprint, PinRef

from .macro import Macro

Pos = tuple[int, int, int]

_COLUMN_PITCH = 6
_ROW_PITCH = 6
_REPEAT_EVERY = 2  # rows or columns between line repeaters; dust from a repeater reaches 15 cells

_Y_WRITE, _Y_STORE, _Y_LOCK, _Y_MIDDLE, _Y_READ, _Y_CELL, _Y_WORD = 1, 2, 3, 4, 5, 6, 7

_GLASS = BlockState("minecraft:glass")
_STONE = BlockState("minecraft:smooth_quartz")
_DUST = BlockState("minecraft:redstone_wire")
# TODO: other port counts: more read ports need another S per pair, more write ports a per-bit OR.
RAM_CELL = "ram_2r1w"
WINDOW_CELL = "ram_window"
WINDOW_MACRO = "ram_window"


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


def build_ram(name: str, depth: int, width: int) -> Macro:
    size_x = _COLUMN_PITCH * width + 1
    size_z = _ROW_PITCH * depth + 1
    size_y = _Y_WORD + 1
    blocks: dict[Pos, BlockState] = {}
    pins: list[PinRef] = []

    def column_x(i: int) -> int:
        return _COLUMN_PITCH * i + 3

    def row_z(k: int) -> int:
        return _ROW_PITCH * k + 3

    def add_pin(pin: str, direction: Direction, side: CardinalDirection, pos: Pos) -> None:
        pins.append(PinRef(name=pin, direction=direction, side=side,
                           offset=(pos[0], pos[1], size_z - 1 - pos[2])))

    def row_line(pin: str, y: int, z: int) -> None:
        """Active low line along X on row blocks, from its pin at the east end."""
        for x in range(size_x):
            blocks[(x, y - 1, z)] = _STONE
            gap = x % _COLUMN_PITCH == 0
            blocks[(x, y, z)] = _repeater("east") if gap and (x == size_x - 1 or x // _COLUMN_PITCH % _REPEAT_EVERY == 0) \
                else _DUST
        add_pin(pin, Direction.IN, CardinalDirection.EAST, (size_x - 1, y, z))

    def bitline(x: int, y: int) -> None:
        """Line along Z flowing north on glass. Repeaters sit in the gaps between rows."""
        for z in range(size_z):
            blocks[(x, y - 1, z)] = _GLASS
            gap = z % _ROW_PITCH == 0
            blocks[(x, y, z)] = _repeater("south") if gap and (z in (0, size_z - 1) or z // _ROW_PITCH % _REPEAT_EVERY == 0) \
                else _DUST

    for k in range(depth):
        z = row_z(k)
        row_line(f"nra[{k}]", _Y_WORD, z - 2)
        row_line(f"nrb[{k}]", _Y_WORD, z + 2)
        row_line(f"lock[{k}]", _Y_LOCK, z - 2)

    for i in range(width):
        sx = column_x(i)
        bitline(sx - 3, _Y_WRITE)
        add_pin(f"wd[{i}]", Direction.IN, CardinalDirection.SOUTH, (sx - 3, _Y_WRITE, size_z - 1))
        bitline(sx + 2, _Y_READ)
        add_pin(f"ra[{i}]", Direction.OUT, CardinalDirection.NORTH, (sx + 2, _Y_READ, 0))
        bitline(sx - 2, _Y_READ)
        add_pin(f"rb[{i}]", Direction.OUT, CardinalDirection.NORTH, (sx - 2, _Y_READ, 0))
        for k in range(depth):
            z = row_z(k)
            # Up from the write bitline to the latch.
            blocks[(sx - 2, _Y_WRITE, z)] = _GLASS
            blocks[(sx - 2, _Y_STORE, z)] = _DUST
            # Latch: locked by a powered row line. The row blocks are on z - 2.
            blocks[(sx - 1, _Y_WRITE, z)] = _GLASS
            blocks[(sx - 1, _Y_WRITE, z - 1)] = _GLASS
            blocks[(sx - 1, _Y_STORE, z)] = _repeater("west")
            blocks[(sx - 1, _Y_STORE, z - 1)] = _repeater("north")
            blocks[(sx, _Y_STORE, z)] = _STONE
            blocks[(sx, _Y_WRITE, z)] = _GLASS
            # Twice inverted up to S; the pasted states are a stored 0.
            blocks[(sx, _Y_LOCK, z)] = _torch(True)
            blocks[(sx, _Y_MIDDLE, z)] = _STONE
            blocks[(sx, _Y_READ, z)] = _torch(False)
            blocks[(sx, _Y_CELL, z)] = _STONE
            # Read taps, on either side of S.
            for x, facing, side_z, side_facing, out_x in ((sx + 1, "west", z - 1, "north", sx + 2),
                                                          (sx - 1, "east", z + 1, "south", sx - 2)):
                blocks[(x, _Y_READ, z)] = _GLASS
                blocks[(x, _Y_READ, side_z)] = _GLASS
                blocks[(x, _Y_CELL, z)] = _comparator(facing)
                blocks[(x, _Y_CELL, side_z)] = _repeater(side_facing)
                blocks[(out_x, _Y_CELL, z)] = _STONE

    # Worst paths, in repeater ticks. A written bit goes through its latch and both torches before the read tap.
    along_row = 1 + width // _REPEAT_EVERY + 1
    along_column = 1 + depth // _REPEAT_EVERY + 1
    tap = 2  # side repeater or torch, then comparator
    read = along_row + tap + along_column
    written = 4 + along_column
    arcs: dict[str, dict[str, int]] = {}
    for k in range(depth):
        arcs[f"nra[{k}]"] = {f"ra[{i}]": read for i in range(width)}
        arcs[f"nrb[{k}]"] = {f"rb[{i}]": read for i in range(width)}
        arcs[f"lock[{k}]"] = {f"{p}[{i}]": along_row + written for p in ("ra", "rb") for i in range(width)}
    for i in range(width):
        arcs[f"wd[{i}]"] = {f"{p}[{i}]": along_column + written for p in ("ra", "rb")}
    return _opaque(name, (size_x, size_y, size_z), blocks, pins, arcs, depth * width)
