import argparse
from collections import defaultdict
from pathlib import Path

from minecraft_v.models import Module, Netlist
from minecraft_v.placement_ir import (
    CURRENT_SCHEMA_VERSION,
    Component,
    ComponentList,
    ComponentType,
    Direction,
    NetConnection,
    NetEndpoint,
    PinRef,
)
from minecraft_v.placement_engine import build_litematic_from_component_list
from minecraft_v.cell_library import CELL_TYPE_MAP, SCHEMATIC_MAP, apply_schematic_pin


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
        if cell.type not in CELL_TYPE_MAP:
            known = ", ".join(sorted(CELL_TYPE_MAP))
            raise ValueError(
                f"Unknown cell type '{cell.type}' in cell '{cell_name}'. "
                f"Known types: {known}"
            )

        comp_type = CELL_TYPE_MAP[cell.type]
        schematic = SCHEMATIC_MAP[comp_type]
        pins = []
        for pin_name, dir_str in cell.port_directions.items():
            pin_dir = Direction.IN if dir_str == "input" else Direction.OUT
            pins.append(apply_schematic_pin(pin_name, pin_dir, schematic))
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
    parser.add_argument("--out-litematic", type=str, default="build/result.litematic")
    parser.add_argument("--module", type=str, default="main")
    parser.add_argument("--schematics-dir", type=str, default="schematics")
    parser.add_argument("--schematic-name", type=str, default=None)
    args = parser.parse_args()

    out_litematic = Path(args.out_litematic)
    if out_litematic.suffix != ".litematic":
        raise SystemExit(f"--out-litematic must end in .litematic, got '{args.out_litematic}'")
    if out_litematic.is_dir():
        raise SystemExit(f"--out-litematic path is a directory: '{args.out_litematic}'")
    if not out_litematic.parent.exists():
        raise SystemExit(f"Output directory does not exist: '{out_litematic.parent}'")

    with open(args.netlist) as f:
        netlist = Netlist.model_validate_json(f.read())

    schematic_name = args.schematic_name or args.module

    module = netlist.modules.get(args.module)
    if module is None:
        available = ", ".join(sorted(netlist.modules))
        raise SystemExit(f"Module '{args.module}' not found. Available: {available}")
    component_list = module_to_component_list(module)
    build_litematic_from_component_list(component_list, schematics_dir=Path(args.schematics_dir), out_path=out_litematic, schematic_name=schematic_name)

    print(f"Wrote litematic: {out_litematic.resolve()}")


if __name__ == "__main__":
    main()
