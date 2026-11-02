import argparse
import atexit
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from minecraft_v.build_utils import save_build_artifacts
from minecraft_v.cell_library import CELL_TYPE_MAP, SCHEMATIC_MAP, apply_schematic_pin
from minecraft_v.models import Module, Netlist
from minecraft_v.placement_engine import build_litematic_from_component_list, pathfinding
from minecraft_v.placement_engine.builder import build_macro
from minecraft_v.placement_engine.macro import extract_macros
from minecraft_v.placement_engine.ram import BLOCK_CELLS, RAM4_CELL, block_for_cell, built_block, group_block, stack_spec
from minecraft_v.placement_engine.probes import DEBUG_CELLS, extract_probes
from minecraft_v.placement_engine.server_file import SERVER_FILE_SUFFIX
from minecraft_v.placement_engine.resume import ResumeStore
from minecraft_v.progress import log, phase, set_debug
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


def _init_high_bits(module: Module) -> set[int]:
    """Net bits whose register starts at 1, from the netlist's "init" attributes."""
    high: set[int] = set()
    for net in module.netnames.values():
        init = net.attributes.get("init")
        if init is None:
            continue
        for bit, value in zip(reversed(net.bits), str(init).zfill(len(net.bits))):
            if value == "1" and isinstance(bit, int):
                high.add(bit)
    return high


def _add_ram_group(stacks, components: list[Component], bit_endpoints) -> None:
    """All of a design's RAM stacks as one block, whose lanes carry the nets they share."""
    specs = []
    outputs: dict[tuple[int, str], int] = {}
    for k, (cell_name, cell) in enumerate(stacks):
        nets: dict[str, int] = {}
        for port, bits in cell.connections.items():
            for i, bit in enumerate(bits):
                pin = f"{port}[{i}]" if len(bits) > 1 or port != "we" else port
                if port in ("ra", "rb", "rc", "rd"):
                    if isinstance(bit, int):
                        outputs[(k, pin)] = bit
                elif isinstance(bit, int):
                    nets[pin] = bit
                else:
                    raise ValueError(f"{cell_name}: {pin} is tied to {bit}")
        specs.append(stack_spec(cell.parameters, nets))
    block, own_pins, lane_pins = group_block(specs)
    pins = []
    by_name = {p.name: p for p in block.pins}
    for bit, name in lane_pins.items():
        pins.append(by_name[name])
        bit_endpoints[bit].append(("ram_group", name, Direction.IN))
    for name, (k, stack_pin) in own_pins.items():
        pin = by_name[name]
        bit = outputs.get((k, stack_pin)) if pin.direction == Direction.OUT else specs[k].nets[stack_pin]
        if bit is not None:
            pins.append(pin)
            bit_endpoints[bit].append(("ram_group", name, pin.direction))
    components.append(Component(id="ram_group", type=ComponentType.CUSTOM, pins=pins, footprint=block.footprint,
                                metadata={"macro": block.name, "bank": True, "south_pins": True}))


def module_to_component_list(module: Module) -> ComponentList:
    components: list[Component] = []
    init_high = _init_high_bits(module)
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

    stacks = [(name, cell) for name, cell in module.cells.items() if cell.type == RAM4_CELL]
    grouped = {name for name, _ in stacks} if len(stacks) > 1 else set()
    if grouped:
        _add_ram_group(stacks, components, bit_endpoints)

    for cell_name, cell in module.cells.items():
        if cell.type in DEBUG_CELLS or cell_name in grouped:
            continue
        if cell.type in BLOCK_CELLS:
            block = block_for_cell(cell.type, cell.parameters)
            pins = []
            for pin in block.pins:
                port, _, index = pin.name.rstrip("]").partition("[")
                bit = cell.connections[port][int(index or 0)]
                # TODO: tie off constant pins, so memories with a fixed write word or an unused address bit can map.
                if not isinstance(bit, int) and pin.direction == Direction.IN:
                    raise ValueError(f"{cell_name}: {pin.name} is tied to {bit}")
                if isinstance(bit, int):
                    pins.append(pin)
                    bit_endpoints[bit].append((cell_name, pin.name, pin.direction))
            components.append(Component(id=cell_name, type=ComponentType.CUSTOM, pins=pins, footprint=block.footprint,
                                        metadata={"macro": block.name, **({"bank": True, "south_pins": True} if cell.type == RAM4_CELL else {"thin": True})}))
            continue
        if cell.type not in CELL_TYPE_MAP:
            known = ", ".join(sorted(CELL_TYPE_MAP))
            raise ValueError(
                f"Unknown cell type '{cell.type}' in cell '{cell_name}'. "
                f"Known types: {known}"
            )

        comp_type = CELL_TYPE_MAP[cell.type]
        schematic = SCHEMATIC_MAP[comp_type]
        pins = []
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
                elif isinstance(bit, str) and bit in ('0', '1', 'x', 'z'):
                    const_val = bit
            if const_val is not None:
                pin_ref = pin_ref.model_copy(update={"const_value": const_val})
            pins.append(pin_ref)

        starts_high = comp_type in (ComponentType.DFF, ComponentType.DFFE) and \
            any(bit in init_high for bit in cell.connections.get("Q", []))
        components.append(Component(
            id=cell_name,
            type=comp_type,
            pins=pins,
            params=dict(cell.parameters),
            footprint=schematic.footprint,
            metadata={"init": 1} if starts_high else {},
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


_MACRO_FANOUT = 2  # nets above this fanout are shared between instances, not grouped into them
_MACRO_MIN_CELLS = 8  # smallest repeated subcircuit worth a macro
_MACRO_ROW_TILES = 0  # instances per array, sharing lanes; 0 = all instances of a pattern in one array
_MACRO_BANK_COLUMNS = 4  # tiles per row in fixed two-row banks facing one lane plane
_MACRO_CHANNEL = int(os.environ.get("RAM_CHANNEL", 24))  # Z kept clear past a long macro's input face
_MACRO_EAST_ROOM = 20  # X past a bank stack's pin end where gates may go; the rest go on top
_MACRO_HEADROOM = 32  # layers kept above the tallest macro
_BANK_GUTTER = 12
_BANK_ROUTING_GUTTER = 12
_BANK_Y_SPACING = 5


def _true_or_false(value: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError(f"expected True or False, got '{value}'")
    return value.lower() == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist", required=True)
    parser.add_argument("--out-litematic", type=str, default="build/result.litematic")
    parser.add_argument("--module", type=str, default="main")
    parser.add_argument("--schematics-dir", type=str, default="schematics")
    parser.add_argument("--schematic-name", type=str, default=None)
    parser.add_argument("--allow-routing-failures", type=bool, default=False)
    parser.add_argument("--output-pin-targets-json", type=str, default=None)
    parser.add_argument("--generate-clock", type=str, default="auto",
                        help="Input pin to drive with a generated clock; auto: _clk or clk if present; none: no clock")
    parser.add_argument("--server-clock", type=int, nargs="?", const=0, default=None, metavar="HALF_PERIOD",
                        help="Save the space of the generated clock: give the clock pin a lever, and MCHPRS toggles "
                             "it every HALF_PERIOD redstone ticks (default: from the critical path, the generated "
                             "clock's period). synth.sh turns this on when its mchprs_headless has --run-clock")
    parser.add_argument("--no-server-clock", dest="server_clock", action="store_const", const=None,
                        help="Generate the clock even if synth.sh finds mchprs_headless")
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
    parser.add_argument("--placer", choices=("legacy", "bary", "sa", "poisson"), default="poisson",
                        help="Gate placement: legacy list order, barycenter row ordering, barycenter + "
                             "simulated annealing, or electrostatic 3D placement without rows")
    parser.add_argument("--place-y-moves", action=argparse.BooleanOptionalAction, default=True,
                        help="Let --placer sa move gates between Y levels (default: on)")
    parser.add_argument("--place-z-moves", action=argparse.BooleanOptionalAction, default=True,
                        help="Let --placer sa move gates between depth layers within their slack (default: on)")
    parser.add_argument("--place-flow-rule", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep same-level sinks in later layers than their driver in --placer sa (default: on)")
    parser.add_argument("--place-mirror", action=argparse.BooleanOptionalAction, default=True,
                        help="Let --placer bary|sa mirror gates in X (default: on)")
    parser.add_argument("--flat", action="store_true",
                        help="Put every component on the same Y level (for small projects)")
    parser.add_argument("--settle", action=argparse.BooleanOptionalAction, default=True,
                        help="Save blocks in the states they settle to, so that power-up pulses can't overwrite "
                             "flip-flops (default: on)")
    parser.add_argument("--place-seed", type=int, default=0, help="RNG seed for --placer sa (default 0)")
    parser.add_argument("--place-workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="forked chains per annealing temperature for --placer sa (default "
                             "min(4, cpus); the result depends on the count, 1 = serial)")
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
    parser.add_argument("--debug", type=_true_or_false, nargs="?", const=True, default=False,
                        help="Log progress and slow searches, save litematic snapshots during routing to "
                             "<artifacts>/snapshots (slow), print A* counters")
    parser.add_argument("--debug-prints", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep $write/$display: mark the blocks they read and write <out>.server.json for MCHPRS "
                             "/watch and mchprs_headless --probes (default: on)")
    parser.add_argument("--macros", action=argparse.BooleanOptionalAction, default=True,
                        help="Route repeated subcircuits once and stamp them; the rest routes around them (default: on; "
                             "needs --placer poisson, ignored with --flat)")
    parser.add_argument("--macro-min-instances", type=int, default=None,
                        help="--macros: use macros only if some pattern repeats this often (default: enough for a "
                             "bank; designs with only small repeats route better as gates)")
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="Testing: save the placement and routed paths here and reuse them on the next build. "
                             "Use a new directory after changing the placer or netlist")
    parser.add_argument("--resume-reroute", type=str, default="",
                        help="--resume-dir: comma-separated nets to search again instead of reusing their paths")
    parser.add_argument("--macro-cache", type=str, default="build/macro_cache",
                        help="--macros: directory keeping routed macros between builds")
    args = parser.parse_args()

    set_debug(args.debug)
    if args.debug and pathfinding._astar is not None:
        atexit.register(lambda: print(f"[astar-stats] {pathfinding._astar.stats()}", file=sys.stderr))

    out_litematic = Path(args.out_litematic)
    if out_litematic.suffix != ".litematic":
        raise SystemExit(f"--out-litematic must end in .litematic, got '{args.out_litematic}'")
    if out_litematic.is_dir():
        raise SystemExit(f"--out-litematic path is a directory: '{args.out_litematic}'")
    out_litematic.parent.mkdir(parents=True, exist_ok=True)
    # MCHPRS loads a sibling sidecar: one from an older build would read wrong blocks
    out_litematic.with_suffix(SERVER_FILE_SUFFIX).unlink(missing_ok=True)
    out_litematic.with_suffix(".probes.json").unlink(missing_ok=True)  # the old name

    with phase("[load] reading netlist"), open(args.netlist) as f:
        netlist = Netlist.model_validate_json(f.read())

    schematic_name = args.schematic_name or args.module

    module = netlist.modules.get(args.module)
    if module is None:
        available = ", ".join(sorted(netlist.modules))
        raise SystemExit(f"Module '{args.module}' not found. Available: {available}")
    component_list = module_to_component_list(module)
    probes = extract_probes(module) if args.debug_prints else None

    save_build_artifacts(component_list)

    allow_routing_failures = args.allow_routing_failures
    output_pin_targets = (
        _load_output_pin_targets(args.output_pin_targets_json)
        if args.output_pin_targets_json and not args.no_pin_extensions
        else None
    )
    generate_clock_pin = None if args.generate_clock == "none" else args.generate_clock
    if generate_clock_pin == "auto":
        pins = {c.id for c in component_list.components if c.type == ComponentType.INPUT_PIN}
        generate_clock_pin = next((pin for pin in ("_clk", "clk") if pin in pins), None)
    if generate_clock_pin is not None and generate_clock_pin not in (c.id for c in component_list.components if c.type == ComponentType.INPUT_PIN):
        raise SystemExit(f"Input pin '{generate_clock_pin}' for clock generation does not exist. Available pins: "
                         f"{list(c.id for c in component_list.components if c.type == ComponentType.INPUT_PIN)}")

    if args.server_clock is not None and args.generate_clock == "none":
        raise SystemExit("--server-clock needs a clock pin, but --generate-clock is none")
    if args.server_clock is not None and args.server_clock < 0:
        raise SystemExit("--server-clock half period can't be negative")

    macros = None
    flat_comp = component_list
    rams = {c.metadata["macro"]: built_block(c.metadata["macro"])
            for c in component_list.components if c.type == ComponentType.CUSTOM}
    if rams:
        macros = rams
    if args.macros and args.placer == "poisson" and not args.flat:
        def build(name, sub, ports, cells):
            return build_macro(name, sub, ports, cells, Path(args.schematics_dir),
                               cache_dir=Path(args.macro_cache) / f"row{_MACRO_ROW_TILES}",
                               io_lane_pins=_MACRO_ROW_TILES != 1)

        def report(cls, macro):
            log(f"[macro] {len(cls.instances)} x {macro.name} ({macro.cells} cells)")

        component_list, macros, instances = extract_macros(component_list, build, _MACRO_FANOUT, _MACRO_MIN_CELLS,
                                                           _MACRO_ROW_TILES,
                                                           _MACRO_BANK_COLUMNS,
                                                           (args.macro_min_instances if args.macro_min_instances is not None
                                                            else max(2, 2 * _MACRO_BANK_COLUMNS)),
                                                           on_class=report)
        macros = {**macros, **rams}
        log(f"[macro] {len(instances)} instances; top level {len(component_list.components)} components, "
              f"{len(component_list.nets)} nets")
        if any(c.metadata.get("bank") for c in component_list.components):
            # Gates around a bank stack: what mem and regsbig route clean with. Without banks, the usual auto spacing.
            args.gutter = _BANK_GUTTER if args.gutter is None else args.gutter
            args.routing_gutter = _BANK_ROUTING_GUTTER if args.routing_gutter is None else args.routing_gutter
            args.y_spacing = _BANK_Y_SPACING if args.y_spacing is None else args.y_spacing

    build_litematic_from_component_list(component_list, schematics_dir=Path(args.schematics_dir),
                                        out_path=out_litematic, schematic_name=schematic_name,
                                        allow_routing_failures=allow_routing_failures,
                                        output_pin_targets=output_pin_targets,
                                        clock_pin=generate_clock_pin,
                                        **({"route_passes": args.route_passes} if args.route_passes is not None else {}),
                                        **({"wire_timeout": args.wire_timeout or None} if args.wire_timeout is not None else {}),
                                        route_workers=args.route_workers,
                                        route_deadline=args.route_deadline,
                                        snapshots=args.debug,
                                        placer=args.placer,
                                        place_y_moves=args.place_y_moves,
                                        place_z_moves=args.place_z_moves,
                                        place_flow_rule=args.place_flow_rule,
                                        place_mirror=args.place_mirror,
                                        place_seed=args.place_seed,
                                        place_workers=args.place_workers,
                                        flat=args.flat,
                                        settled=args.settle,
                                        macros=macros,
                                        flat_comp=flat_comp,
                                        probes=probes,
                                        server_clock=args.server_clock,
                                        resume=ResumeStore(args.resume_dir, set(filter(None, args.resume_reroute.split(","))))
                                        if args.resume_dir else None,
                                        macro_channel=_MACRO_CHANNEL,
                                        macro_east_room=_MACRO_EAST_ROOM,
                                        macro_headroom=_MACRO_HEADROOM,
                                        **{k: v for k, v in (
                                            ("gutter", args.gutter),
                                            ("component_width_spacing", args.io_margin),
                                            ("component_depth_spacing", args.routing_gutter),
                                            ("additional_component_height_spacing", args.y_spacing),
                                        ) if v is not None})

    log(f"Wrote litematic: {out_litematic.resolve()}")


if __name__ == "__main__":
    main()
