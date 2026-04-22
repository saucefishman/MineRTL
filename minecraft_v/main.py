import argparse
from collections import defaultdict
from dataclasses import dataclass

from minecraft_v.models import Module, Netlist
from minecraft_v.placement_ir import (
    CURRENT_SCHEMA_VERSION,
    CardinalDirection,
    Component,
    ComponentList,
    ComponentType,
    Direction,
    Footprint,
    NetConnection,
    NetEndpoint,
    PinRef,
)

_CELL_TYPE_MAP = {
    "$_AND_": ComponentType.AND,
    "$_OR_": ComponentType.OR,
    "$_NOT_": ComponentType.NOT,
    "$_XOR_": ComponentType.XOR,
    "$_DFF_P_": ComponentType.DFF,
    "$_DFFE_PP_": ComponentType.DFF,
    "fulladder": ComponentType.FULL_ADDER,
    "$_MUX_": ComponentType.MUX,
}

@dataclass
class SchematicInfo:
    file_prefix: str  # e.g. "dff" for dff.litematic
    footprint: Footprint
    pins: list[PinRef]  


_SCHEMATIC_MAP: dict[ComponentType, SchematicInfo] = {
    ComponentType.AND: SchematicInfo(
        file_prefix="and",
        footprint=Footprint(width=3, height=2, depth=3),
        pins=[
            PinRef(name="A", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(2, 0, 0)),
            PinRef(name="B", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(0, 0, 0)),
            PinRef(name="Y", direction=Direction.OUT, side=CardinalDirection.NORTH, offset=(1, 0, 2)),
        ],
    ),
    ComponentType.OR: SchematicInfo(
        file_prefix="or",
        footprint=Footprint(width=3, height=1, depth=3),
        pins=[
            PinRef(name="A", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(0, 0, 0)),
            PinRef(name="B", direction=Direction.IN,  side=CardinalDirection.SOUTH, offset=(2, 0, 2)),
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


def _apply_schematic(pin_name: str, direction: Direction, schematic: SchematicInfo) -> PinRef:
    # Copies side/offset from the schematic template onto a netlist-derived pin.
    template = next((p for p in schematic.pins if p.name == pin_name and p.direction == direction), None)
    if template is None:
        raise ValueError(f"pin '{pin_name}' not in schematic for '{schematic.file_prefix}'")
    return PinRef(name=pin_name, direction=direction, side=template.side, offset=template.offset)


def module_to_component_list(module: Module) -> ComponentList:
    components: list[Component] = []
    # bit_id -> [(component_id, pin_name, Direction)]
    bit_endpoints: dict[int, list[tuple[str, str, Direction]]] = defaultdict(list)

    for port_name, port in module.ports.items():
        pin_dir = Direction.OUT if port.direction == "input" else Direction.IN
        comp_type = ComponentType.INPUT_PIN if port.direction == "input" else ComponentType.OUTPUT_PIN
        pins = []
        for i, bit in enumerate(port.bits):
            if isinstance(bit, int):
                pin_name = port_name if len(port.bits) == 1 else f"{port_name}[{i}]"
                pins.append(PinRef(name=pin_name, direction=pin_dir))
                bit_endpoints[bit].append((port_name, pin_name, pin_dir))
        components.append(Component(id=port_name, type=comp_type, pins=pins))

    for cell_name, cell in module.cells.items():
        if cell.type not in _CELL_TYPE_MAP:
            known = ", ".join(sorted(_CELL_TYPE_MAP))
            raise ValueError(
                f"Unknown cell type '{cell.type}' in cell '{cell_name}'. "
                f"Known types: {known}"
            )

        comp_type = _CELL_TYPE_MAP[cell.type]
        schematic = _SCHEMATIC_MAP[comp_type]
        pins = []
        for pin_name, dir_str in cell.port_directions.items():
            pin_dir = Direction.IN if dir_str == "input" else Direction.OUT
            pins.append(_apply_schematic(pin_name, pin_dir, schematic))
            for bit in cell.connections.get(pin_name, []):
                if isinstance(bit, int):
                    bit_endpoints[bit].append((cell_name, pin_name, pin_dir))

        components.append(Component(
            id=cell_name,
            type=comp_type,
            pins=pins,
            params=dict(cell.parameters),
            footprint=schematic.footprint,
        ))

    nets: list[NetConnection] = []
    for bit_id, endpoints in bit_endpoints.items():
        sources = [(cid, pn) for cid, pn, d in endpoints if d == Direction.OUT]
        sinks = [NetEndpoint(component_id=cid, pin_name=pn) for cid, pn, d in endpoints if d == Direction.IN]
        if not sources or not sinks:
            continue
        nets.append(NetConnection(
            net_id=f"net_{bit_id}",
            source=NetEndpoint(component_id=sources[0][0], pin_name=sources[0][1]),
            sinks=sinks,
        ))

    return ComponentList(
        schema_version=CURRENT_SCHEMA_VERSION,
        components=components,
        nets=nets,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist", required=True)
    args = parser.parse_args()

    with open(args.netlist) as f:
        netlist = Netlist.model_validate_json(f.read())


    for module_name, module in netlist.modules.items():
        print(f"Module: {module_name}")
        component_list = module_to_component_list(module)
        print(component_list.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
