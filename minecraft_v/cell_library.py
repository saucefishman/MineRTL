from dataclasses import dataclass

from minecraft_v.placement_ir import (
    CardinalDirection,
    ComponentType,
    Direction,
    Footprint,
    PinRef,
)

CELL_TYPE_MAP: dict[str, ComponentType] = {
    "$_AND_": ComponentType.AND,
    "$_OR_": ComponentType.OR,
    "$_NOT_": ComponentType.NOT,
    "$_XOR_": ComponentType.XOR,
    "$_DFF_P_": ComponentType.DFF,
    "$_DFFE_PP_": ComponentType.DFFE,
    "fulladder": ComponentType.FULL_ADDER,
    "$_MUX_": ComponentType.MUX,
}

@dataclass
class SchematicInfo:
    file_prefix: str
    footprint: Footprint
    pins: list[PinRef]

SCHEMATIC_MAP: dict[ComponentType, SchematicInfo] = {
    ComponentType.AND: SchematicInfo(
        file_prefix="and",
        footprint=Footprint(width=3, height=2, depth=3),
        pins=[
            PinRef(name="A", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(0, 0, 0)),
            PinRef(name="B", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(2, 0, 0)),
            PinRef(name="Y", direction=Direction.OUT, side=CardinalDirection.NORTH, offset=(1, 0, 2)),
        ],
    ),
    ComponentType.OR: SchematicInfo(
        file_prefix="or",
        footprint=Footprint(width=3, height=1, depth=3),
        pins=[
            PinRef(name="A", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(0, 0, 0)),
            PinRef(name="B", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(2, 0, 0)),
            PinRef(name="Y", direction=Direction.OUT, side=CardinalDirection.NORTH, offset=(1, 0, 2)),
        ],
    ),
    ComponentType.NOT: SchematicInfo(
        file_prefix="not",
        footprint=Footprint(width=1, height=2, depth=3),
        pins=[
            PinRef(name="A", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(0, 0, 0)),
            PinRef(name="Y", direction=Direction.OUT, side=CardinalDirection.NORTH, offset=(0, 0, 2)),
        ],
    ),
    ComponentType.XOR: SchematicInfo(
        file_prefix="xor",
        footprint=Footprint(width=3, height=2, depth=7),
        pins=[
            PinRef(name="A", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(0, 0, 0)),
            PinRef(name="B", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(2, 0, 0)),
            PinRef(name="Y", direction=Direction.OUT, side=CardinalDirection.NORTH, offset=(1, 0, 6)),
        ],
    ),
    ComponentType.DFF: SchematicInfo(
        file_prefix="dff",
        footprint=Footprint(width=4, height=1, depth=2),
        pins=[
            PinRef(name="C", direction=Direction.IN,  side=CardinalDirection.EAST, offset=(3, 0, 0)),
            PinRef(name="D", direction=Direction.IN,  side=CardinalDirection.SOUTH,  offset=(0, 0, 0)),
            PinRef(name="Q", direction=Direction.OUT, side=CardinalDirection.NORTH,  offset=(0, 0, 1)),
        ],
    ),
    ComponentType.DFFE: SchematicInfo(
        file_prefix="dffe",
        footprint=Footprint(width=7, height=1, depth=3),
        pins=[
            PinRef(name="C", direction=Direction.IN,  side=CardinalDirection.EAST, offset=(0, 0, 0)),
            PinRef(name="D", direction=Direction.IN,  side=CardinalDirection.SOUTH,  offset=(0, 0, 0)),
            PinRef(name="E", direction=Direction.IN,  side=CardinalDirection.EAST,  offset=(6, 0, 2)),
            PinRef(name="Q", direction=Direction.OUT, side=CardinalDirection.NORTH,  offset=(0, 0, 2)),
        ],
    ),
    ComponentType.FULL_ADDER: SchematicInfo(
        file_prefix="fulladder",
        footprint=Footprint(width=8, height=4, depth=15),
        pins=[
            PinRef(name="A", direction=Direction.IN,  side=CardinalDirection.SOUTH,  offset=(3, 0, 0)),
            PinRef(name="B", direction=Direction.IN,  side=CardinalDirection.SOUTH,  offset=(5, 0, 2)),
            PinRef(name="Cin", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(7, 0, 1)),
            PinRef(name="S", direction=Direction.OUT, side=CardinalDirection.NORTH,  offset=(1, 0, 14)),
            PinRef(name="Cout", direction=Direction.OUT, side=CardinalDirection.NORTH,  offset=(5, 0, 14)),
        ],
    ),
    ComponentType.MUX: SchematicInfo(
        file_prefix="mux",
        footprint=Footprint(width=6, height=3, depth=6),
        pins=[
            PinRef(name="A", direction=Direction.IN,  side=CardinalDirection.SOUTH,  offset=(2, 0, 0)),
            PinRef(name="B", direction=Direction.IN,  side=CardinalDirection.SOUTH,  offset=(4, 0, 0)),
            PinRef(name="S", direction=Direction.IN,  side=CardinalDirection.WEST, offset=(0, 0, 3)),
            PinRef(name="Y", direction=Direction.OUT, side=CardinalDirection.NORTH,  offset=(3, 0, 5)),
        ],
    ),
}

def apply_schematic_pin(pin_name: str, direction: Direction, schematic: SchematicInfo) -> PinRef:
    template = next((p for p in schematic.pins if p.name == pin_name and p.direction == direction), None)
    if template is None:
        raise ValueError(f"pin '{pin_name}' not in schematic for '{schematic.file_prefix}'")
    return PinRef(name=pin_name, direction=direction, side=template.side, offset=template.offset)
