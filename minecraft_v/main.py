import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from minecraft_v.build_utils import save_build_artifacts
from minecraft_v.cell_library import CELL_TYPE_MAP, SCHEMATIC_MAP, apply_schematic_pin
from minecraft_v.models import Module, Netlist
from minecraft_v.placement_engine import build_litematic_from_component_list
from minecraft_v.placement_engine.ir import (
    CURRENT_SCHEMA_VERSION,
    Component,
    ComponentList,
    ComponentType,
    Direction,
    NetConnection,
    NetEndpoint,
    PinRef,
)


def _load_output_pin_targets(path: str) -> dict[str, tuple[int, int]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise SystemExit("--output-pin-targets-json must contain a JSON object")

    normalized: dict[str, tuple[int, int]] = {}
    for pin_name, coord in payload.items():
        if not isinstance(pin_name, str) or not pin_name:
            raise SystemExit("output pin target keys must be non-empty strings")
        if isinstance(coord, dict):
            x = coord.get("x")
            y = coord.get("y")
        elif isinstance(coord, (list, tuple)) and len(coord) == 3:
            x, y, _z_ignored = coord
        else:
            raise SystemExit(
                f"Invalid coord for '{pin_name}'. Use [x, y, z] or {{\"x\":...,\"y\":...,\"z\":...}}"
            )
        if not isinstance(x, int) or not isinstance(y, int):
            raise SystemExit(f"Invalid coord for '{pin_name}': x and y must be integers")
        normalized[pin_name] = (x, y)
    return normalized


def _bit_bus_index(module: Module) -> dict[int, tuple[str, int]]:
    """Net bit -> (vector name, index) for bits of multi-bit netnames; user names win over Yosys ones."""
    out: dict[int, tuple[str, int]] = {}
    for name, nn in sorted(module.netnames.items(), key=lambda kv: (-kv[1].hide_name, kv[0])):
        if len(nn.bits) < 2:
            continue
        for i, bit in enumerate(nn.bits):
            if isinstance(bit, int):
                out[bit] = (name, i)
    return out


def _scope_index(hierarchy: dict, top: str) -> dict[tuple[str, ...], list[str]]:
    """Instantiation-site chain -> instance path (['ad', 'f0']), from the pre-flatten JSON (``hierarchy.json``):
    ``flatten -noscopeinfo`` adds the site of every instance a cell sits in to its ``src``."""
    modules = hierarchy.get("modules", {})
    out: dict[tuple[str, ...], list[str]] = {}

    def walk(mod: str, sites: tuple[str, ...], path: list[str], depth: int) -> None:
        if depth > 32:
            return
        for name, cell in modules.get(mod, {}).get("cells", {}).items():
            if cell.get("type") in modules:
                site = cell.get("attributes", {}).get("src", "")
                if site:
                    sub_sites, sub_path = (*sites, site), [*path, name.lstrip("\\")]
                    out[sub_sites] = sub_path
                    walk(cell["type"], sub_sites, sub_path, depth + 1)

    walk(top, (), [], 0)
    return out


def _instance_path(src: str, scopes: dict[tuple[str, ...], list[str]]) -> list[str]:
    """Instance path of a flattened cell: the deepest instance whose whole site chain is in its ``src``.
    ``src`` is a Yosys string pool ('|'-joined, not in hierarchy order: outer site, own line, inner site).
    [] = top module."""
    entries = set(src.split("|"))
    best: list[str] = []
    for sites, path in scopes.items():
        if len(path) > len(best) and entries.issuperset(sites):
            best = path
    return best


def module_to_component_list(module: Module, scopes: dict[tuple[str, ...], list[str]] | None = None) -> ComponentList:
    """scopes: see _scope_index; tags cells with metadata['instance']."""
    components: list[Component] = []
    bit_bus = _bit_bus_index(module)
    scopes = scopes or {}
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
        bus: tuple[str, int] | None = None
        for schematic_pin in schematic.pins:
            pin_name = schematic_pin.name
            if pin_name in cell.port_directions:
                pin_dir = Direction.IN if cell.port_directions[pin_name] == "input" else Direction.OUT
            else:
                pin_dir = schematic_pin.direction
            pin_ref = apply_schematic_pin(pin_name, pin_dir, schematic)
            const_val: str | None = None
            for bit in cell.connections.get(pin_name, []):
                if isinstance(bit, int):
                    bit_endpoints[bit].append((cell_name, pin_name, pin_dir))
                    if pin_dir == Direction.OUT and bus is None:
                        bus = bit_bus.get(bit)
                elif isinstance(bit, str) and bit in ('0', '1', 'x', 'z'):
                    const_val = bit
            if const_val is not None:
                pin_ref = pin_ref.model_copy(update={"const_value": const_val})
            pins.append(pin_ref)

        components.append(Component(
            id=cell_name,
            type=comp_type,
            pins=pins,
            params=dict(cell.parameters),
            footprint=schematic.footprint,
            metadata={k: v for k, v in (
                ("bus", list(bus) if bus else None),
                ("instance", _instance_path(cell.attributes.get("src", ""), scopes)),
            ) if v},
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
    parser.add_argument("--allow-routing-failures", type=bool, default=False)
    parser.add_argument("--output-pin-targets-json", type=str, default=None)
    parser.add_argument("--generate-clock", type=str, default=None)
    parser.add_argument("--route-passes", type=int, default=None,
                        help="Routing passes incl. final hard pass; 1 = single greedy pass (default: _ROUTE_MAX_PASSES)")
    parser.add_argument("--wire-timeout", type=float, default=None,
                        help="Per-search A* time limit in seconds for non-final passes; 0 disables "
                             "(default: _ROUTE_SEARCH_TIMEOUT)")
    parser.add_argument("--route-workers", type=int, default=1,
                        help="Processes for negotiation passes (needs --route-passes > 1); 1 = sequential")
    parser.add_argument("--route-deadline", type=float, default=None,
                        help="Seconds of routing before giving up on remaining nets (writes a partial result; "
                             "use with --allow-routing-failures True)")
    parser.add_argument("--snapshots", action="store_true",
                        help="Save debug litematic snapshots during routing to <artifacts>/snapshots (slow)")
    parser.add_argument("--placer", choices=("legacy", "bary", "sa"), default="sa",
                        help="Gate placement: legacy list order, barycenter row ordering, or barycenter + "
                             "simulated annealing")
    parser.add_argument("--place-y-moves", action=argparse.BooleanOptionalAction, default=True,
                        help="Let --placer sa move gates between Y levels (default: on)")
    parser.add_argument("--place-z-moves", action=argparse.BooleanOptionalAction, default=True,
                        help="Let --placer sa move gates between depth layers within their slack (default: on)")
    parser.add_argument("--place-mirror", action=argparse.BooleanOptionalAction, default=True,
                        help="Let --placer bary|sa mirror gates in X (default: on)")
    parser.add_argument("--flat", action="store_true",
                        help="Put every component on the same Y level (for small projects)")
    parser.add_argument("--place-seed", type=int, default=0, help="RNG seed for --placer sa (default 0)")
    parser.add_argument("--place-workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="forked chains per annealing temperature for --placer sa (default "
                             "min(4, cpus); the result depends on the count, 1 = serial)")
    parser.add_argument("--place-slice-weight", type=float, default=None,
                        help="--placer sa: pull gates driving consecutive bits of a vector into the same X/Z "
                             "(bit k sits on Y level k, so slices stack straight up); 0 = off (default: "
                             "_PLACE_W_SLICE)")
    parser.add_argument("--place-hier-weight", type=float, default=None,
                        help="--placer bary|sa: keep each module instance's gates together (pull to the "
                             "instance centroid, rows grouped by instance); 0 = off (default: _PLACE_W_HIER)")
    parser.add_argument("--bend-cost", type=float, default=None,
                        help="A* extra cost for a flat wire turning (default 1; a flat step costs 1)")
    parser.add_argument("--gutter", type=int, default=None,
                        help="X gap between neighbouring gates in a row (default: auto from routing demand, 6-8)")
    parser.add_argument("--io-margin", type=int, default=None,
                        help="margin around the layout / IO rows (default 7)")
    parser.add_argument("--routing-gutter", type=int, default=None,
                        help="Z gap between layer rows, the routing channel depth (default: auto, 6-8)")
    parser.add_argument("--y-spacing", type=int, default=None,
                        help="extra Y between levels beyond the tallest component (default 3)")
    parser.add_argument("--no-pin-extensions", action="store_true",
                        help="ignore --output-pin-targets-json (no output pin extensions)")
    args = parser.parse_args()

    out_litematic = Path(args.out_litematic)
    if out_litematic.suffix != ".litematic":
        raise SystemExit(f"--out-litematic must end in .litematic, got '{args.out_litematic}'")
    if out_litematic.is_dir():
        raise SystemExit(f"--out-litematic path is a directory: '{args.out_litematic}'")
    out_litematic.parent.mkdir(parents=True, exist_ok=True)

    with open(args.netlist) as f:
        netlist = Netlist.model_validate_json(f.read())

    schematic_name = args.schematic_name or args.module

    module = netlist.modules.get(args.module)
    if module is None:
        available = ", ".join(sorted(netlist.modules))
        raise SystemExit(f"Module '{args.module}' not found. Available: {available}")
    hierarchy_path = Path(args.netlist).with_name("hierarchy.json")
    scopes = (_scope_index(json.loads(hierarchy_path.read_text()), args.module)
              if hierarchy_path.exists() else {})
    component_list = module_to_component_list(module, scopes)

    save_build_artifacts(component_list)

    allow_routing_failures = args.allow_routing_failures
    output_pin_targets = (
        _load_output_pin_targets(args.output_pin_targets_json)
        if args.output_pin_targets_json and not args.no_pin_extensions
        else None
    )
    generate_clock_pin = args.generate_clock
    if generate_clock_pin is not None and generate_clock_pin not in (c.id for c in component_list.components if c.type == ComponentType.INPUT_PIN):
        raise SystemExit(f"Input pin '{generate_clock_pin}' for clock generation does not exist. Available pins: "
                         f"{list(c.id for c in component_list.components if c.type == ComponentType.INPUT_PIN)}")

    build_litematic_from_component_list(component_list, schematics_dir=Path(args.schematics_dir),
                                        out_path=out_litematic, schematic_name=schematic_name,
                                        allow_routing_failures=allow_routing_failures,
                                        output_pin_targets=output_pin_targets,
                                        clock_pin=generate_clock_pin,
                                        **({"route_passes": args.route_passes} if args.route_passes is not None else {}),
                                        **({"wire_timeout": args.wire_timeout or None} if args.wire_timeout is not None else {}),
                                        route_workers=args.route_workers,
                                        route_deadline=args.route_deadline,
                                        snapshots=args.snapshots,
                                        placer=args.placer,
                                        place_y_moves=args.place_y_moves,
                                        place_z_moves=args.place_z_moves,
                                        place_mirror=args.place_mirror,
                                        place_seed=args.place_seed,
                                        place_workers=args.place_workers,
                                        place_slice_weight=args.place_slice_weight,
                                        place_hier_weight=args.place_hier_weight,
                                        bend_cost=args.bend_cost,
                                        flat=args.flat,
                                        **{k: v for k, v in (
                                            ("gutter", args.gutter),
                                            ("component_width_spacing", args.io_margin),
                                            ("component_depth_spacing", args.routing_gutter),
                                            ("additional_component_height_spacing", args.y_spacing),
                                        ) if v is not None})

    print(f"Wrote litematic: {out_litematic.resolve()}")


if __name__ == "__main__":
    main()
