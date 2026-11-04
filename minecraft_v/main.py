import argparse
import atexit
import json
import sys
from collections import defaultdict
from pathlib import Path

from minecraft_v.build_utils import save_build_artifacts
from minecraft_v.cell_library import CELL_TYPE_MAP, SCHEMATIC_MAP, SCHEMATICS_DIR, apply_schematic_pin
from minecraft_v.models import Module, Netlist
from minecraft_v.placement_engine import build_litematic_from_component_list, pathfinding
from minecraft_v.placement_engine.builder import build_macro, set_first_fanout, set_long_edge_split
from minecraft_v.placement_engine.macro import extract_macros
from minecraft_v.placement_engine.ram import BLOCK_CELLS, RAM4_CELL, block_for_cell, built_block, group_block, set_block_cache, stack_spec
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


def _load_output_pin_targets(path: str) -> dict[str, tuple[int, ...]]:
    """Pin -> (x, y), or (x, y, 3) where {"tile": 3} asks for a 3x3 tile of lamps (lit while the pin is low)."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise SystemExit("--output-pin-targets-json must contain a JSON object")

    normalized: dict[str, tuple[int, ...]] = {}
    for pin_name, coord in payload.items():
        if not isinstance(pin_name, str) or not pin_name:
            raise SystemExit("output pin target keys must be non-empty strings")
        tile = 1
        if isinstance(coord, dict):
            x = coord.get("x")
            y = coord.get("y")
            tile = coord.get("tile", 1)
        elif isinstance(coord, (list, tuple)) and len(coord) == 3:
            x, y, _z_ignored = coord
        else:
            raise SystemExit(
                f"Invalid coord for '{pin_name}'. Use [x, y, z] or {{\"x\":...,\"y\":...,\"z\":...}}"
            )
        if not isinstance(x, int) or not isinstance(y, int):
            raise SystemExit(f"Invalid coord for '{pin_name}': x and y must be integers")
        if tile not in (1, 3):
            raise SystemExit(f"Invalid tile for '{pin_name}': 1 or 3, not {tile!r}")
        normalized[pin_name] = (x, y) if tile == 1 else (x, y, tile)
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
                elif bit == "1":  # TODO: drive a pin tied high. Low is what an unconnected pin reads anyway.
                    raise SystemExit(f"{cell_name}: RAM input {pin} is tied to {bit}, which RAM blocks can't take yet; "
                                     "build with --no-tiled-ram or drive it from logic")
        specs.append(stack_spec(cell.parameters, nets))
    block, own_pins, lane_pins = group_block(specs)
    pins = []
    by_name = {p.name: p for p in block.pins}
    for bit, name in lane_pins.items():
        pins.append(by_name[name])
        bit_endpoints[bit].append(("ram_group", name, Direction.IN))
    for name, (k, stack_pin) in own_pins.items():
        pin = by_name[name]
        bit = outputs.get((k, stack_pin)) if pin.direction == Direction.OUT else specs[k].nets.get(stack_pin)
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
                # TODO: drive a pin tied high. Low is what an unconnected pin reads anyway.
                if bit == "1" and pin.direction == Direction.IN:
                    raise SystemExit(f"{cell_name}: block input {pin.name} is tied to {bit}, which blocks can't take yet; "
                                     "build with --no-tiled-ram or drive it from logic")
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
_MACRO_CHANNEL = 24  # Z kept clear past a long macro's input face
_MACRO_EAST_ROOM = 20  # X past a bank stack's pin end where gates may go; the rest go on top
_MACRO_HEADROOM = 32  # layers kept above the tallest macro
_BANK_GUTTER = 12
_BANK_ROUTING_GUTTER = 12
_BANK_Y_SPACING = 5


def main():
    parser = argparse.ArgumentParser(description="Place and route a yosys netlist into a Minecraft .litematic. "
                                     "synth.sh runs yosys and then this; see README.md.")
    parser.add_argument("--netlist", required=True, help="netlist.json written by synth.sh")
    parser.add_argument("--out-litematic", type=str, default="build/result.litematic",
                        help="Output schematic (default: build/result.litematic)")
    parser.add_argument("--module", type=str, default="main",
                        help="Top module; synth.sh flattens everything under it (default: main)")
    parser.add_argument("--schematic-name", type=str, default=None,
                        help="Name stored in the schematic (default: the module name)")
    parser.add_argument("--allow-routing-failures", action=argparse.BooleanOptionalAction, default=False,
                        help="Write the build even if some nets fail to route (default: off)")
    parser.add_argument("--output-pin-targets-json", type=str, default=None,
                        help="JSON of output pin target positions; synth.sh passes the project's pin_targets.json")
    parser.add_argument("--pin-extensions", action=argparse.BooleanOptionalAction, default=True,
                        help="Extend output pins to --output-pin-targets-json (default: on)")
    parser.add_argument("--generate-clock", type=str, default="auto",
                        help="Input pin to drive with a generated clock; auto: _clk or clk if present; none: no clock")
    parser.add_argument("--server-clock", type=int, nargs="?", const=0, default=None, metavar="HALF_PERIOD",
                        help="Save the space of the generated clock: give the clock pin a lever, and MCHPRS toggles "
                             "it every HALF_PERIOD redstone ticks (default: from the critical path, the generated "
                             "clock's period). synth.sh turns this on when its mchprs_headless has --run-clock")
    parser.add_argument("--no-server-clock", dest="server_clock", action="store_const", const=None,
                        help="Generate the clock even if synth.sh finds mchprs_headless")
    parser.add_argument("--route-deadline", type=float, default=None,
                        help="Seconds of routing before giving up on remaining nets and writing a partial result "
                             "(implies --allow-routing-failures)")
    parser.add_argument("--place-seed", type=int, default=0,
                        help="Placement seed (default 0). Routing is deterministic per placement: try another seed "
                             "if a build fails nets")
    parser.add_argument("--flat", action=argparse.BooleanOptionalAction, default=False,
                        help="Put every component on the same Y level (for small projects; output pin targets often "
                             "fail flat, see --no-pin-extensions)")
    parser.add_argument("--settle", action=argparse.BooleanOptionalAction, default=True,
                        help="Save blocks in the states they settle to, so that power-up pulses can't overwrite "
                             "flip-flops (default: on)")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False,
                        help="Log progress and slow searches, save litematic snapshots during routing to "
                             "<artifacts>/snapshots (slow), print A* counters")
    parser.add_argument("--debug-prints", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep $write/$display: mark the blocks they read and write <out>.server.json for MCHPRS "
                             "/watch and mchprs_headless --probes (default: on)")
    parser.add_argument("--macros", action=argparse.BooleanOptionalAction, default=True,
                        help="Route repeated subcircuits once and stamp them; the rest routes around them (default: on; "
                             "ignored with --flat)")
    # Tests: allow macros for patterns that repeat this often (default: enough for a bank)
    parser.add_argument("--macro-min-instances", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="Testing: save the placement and routed paths here and reuse them on the next build. "
                             "Use a new directory after changing the placer or netlist")
    parser.add_argument("--resume-reroute", type=str, default="",
                        help="--resume-dir: comma-separated nets to search again instead of reusing their paths")
    parser.add_argument("--macro-cache", type=str, default="build/macro_cache",
                        help="Directory keeping routed macros and settled RAM blocks between builds")
    args = parser.parse_args()

    set_debug(args.debug)
    if args.debug:
        atexit.register(lambda: print(f"[astar-stats] {pathfinding._astar.stats()}", file=sys.stderr))

    out_litematic = Path(args.out_litematic)
    if out_litematic.suffix != ".litematic":
        raise SystemExit(f"--out-litematic must end in .litematic, got '{args.out_litematic}'")
    if out_litematic.is_dir():
        raise SystemExit(f"--out-litematic path is a directory: '{args.out_litematic}'")
    out_litematic.parent.mkdir(parents=True, exist_ok=True)
    # MCHPRS loads a sibling sidecar: one from an older build would read wrong blocks
    out_litematic.with_suffix(SERVER_FILE_SUFFIX).unlink(missing_ok=True)

    with phase("[load] reading netlist"), open(args.netlist) as f:
        netlist = Netlist.model_validate_json(f.read())

    schematic_name = args.schematic_name or args.module

    module = netlist.modules.get(args.module)
    if module is None:
        available = ", ".join(sorted(netlist.modules))
        raise SystemExit(f"Module '{args.module}' not found. Available: {available}")
    set_block_cache(Path(args.macro_cache) / "ram")
    component_list = module_to_component_list(module)
    probes = extract_probes(module) if args.debug_prints else None

    save_build_artifacts(component_list)

    allow_routing_failures = args.allow_routing_failures or args.route_deadline is not None
    output_pin_targets = (
        _load_output_pin_targets(args.output_pin_targets_json)
        if args.output_pin_targets_json and args.pin_extensions
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
    pathfinding.set_foot_repeaters(not rams)
    set_first_fanout(64 if rams else 0)
    set_long_edge_split(sum(c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN) for c in component_list.components))
    spacing = {}
    if args.macros and not args.flat:
        def build(name, sub, ports, cells):
            return build_macro(name, sub, ports, cells, SCHEMATICS_DIR,
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
            # Gates around a bank stack: what routes clean around banks. Without banks, the usual auto spacing.
            spacing = {"gutter": _BANK_GUTTER, "component_depth_spacing": _BANK_ROUTING_GUTTER,
                       "additional_component_height_spacing": _BANK_Y_SPACING}

    reroute = set(filter(None, args.resume_reroute.split(",")))
    if reroute and not args.resume_dir:
        log("[resume] --resume-reroute does nothing without --resume-dir")
    known = {n.net_id for n in component_list.nets} | {n.net_id for n in flat_comp.nets}
    if unknown := sorted(reroute - known):
        log(f"[resume] --resume-reroute: no such net {', '.join(unknown)}")

    build_litematic_from_component_list(component_list, schematics_dir=SCHEMATICS_DIR,
                                        out_path=out_litematic, schematic_name=schematic_name,
                                        allow_routing_failures=allow_routing_failures,
                                        output_pin_targets=output_pin_targets,
                                        clock_pin=generate_clock_pin,
                                        route_deadline=args.route_deadline,
                                        snapshots=args.debug,
                                        place_seed=args.place_seed,
                                        flat=args.flat,
                                        settled=args.settle,
                                        macros=macros,
                                        flat_comp=flat_comp,
                                        probes=probes,
                                        server_clock=args.server_clock,
                                        resume=ResumeStore(args.resume_dir, reroute) if args.resume_dir else None,
                                        macro_channel=_MACRO_CHANNEL,
                                        macro_east_room=_MACRO_EAST_ROOM,
                                        macro_headroom=_MACRO_HEADROOM,
                                        **spacing)

    log(f"Wrote litematic: {out_litematic.resolve()}")


if __name__ == "__main__":
    main()
