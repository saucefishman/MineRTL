from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from litemapy import BlockState, Region, Schematic
from minecraft_v.cell_library import SCHEMATIC_MAP
from minecraft_v.placement_ir import (
    CardinalDirection,
    Component,
    ComponentList,
    ComponentType,
    Footprint,
    NetConnection,
    NetEndpoint,
    PinRef,
)

AIR = BlockState("minecraft:air")
STONE = BlockState("minecraft:stone")
REDSTONE = BlockState("minecraft:redstone_wire")

def _block_str(block: BlockState) -> str:
    return str(block)

def _is_air(block: BlockState) -> bool:
    text = _block_str(block)
    return text == "minecraft:air" or text.startswith("minecraft:air[")

def _is_redstone_wire(block: BlockState) -> bool:
    return "redstone_wire" in _block_str(block)

def _non_air_bounds(template: Region) -> tuple[int, int, int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    zs: list[int] = []
    for pos in template.block_positions():
        if not _is_air(template[pos]):
            x, y, z = pos
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        return (
            template.min_x(),
            template.min_y(),
            template.min_z(),
            template.max_x(),
            template.max_y(),
            template.max_z(),
        )
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

def _needs_support(block: BlockState) -> bool:
    name = _block_str(block)
    return (
        "minecraft:repeater" in name
        or "minecraft:comparator" in name
        or "minecraft:redstone_wire" in name
        or "minecraft:redstone_torch[" in name
        or name == "minecraft:redstone_torch"
    )

def _ensure_support(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    cell: tuple[int, int, int],
) -> None:
    x, y, z = cell
    if y <= 0:
        return
    below = (x, y - 1, z)
    if below in solid:
        return
    if _is_air(workspace[x, y - 1, z]):
        workspace[x, y - 1, z] = STONE
        solid.add(below)

def _paste_template(
    workspace: Region,
    template: Region,
    dest_origin: tuple[int, int, int],
    solid: set[tuple[int, int, int]],
) -> None:
    tmin_x, tmin_y, tmin_z, _, _, _ = _non_air_bounds(template)
    dx, dy, dz = dest_origin
    placed: list[tuple[tuple[int, int, int], BlockState]] = []
    for pos in template.block_positions():
        block = template[pos]
        if _is_air(block):
            continue
        x, y, z = pos
        wx = dx + (x - tmin_x)
        wy = dy + (y - tmin_y)
        wz = dz + (z - tmin_z)
        workspace[wx, wy, wz] = block
        solid.add((wx, wy, wz))
        placed.append(((wx, wy, wz), block))
    for cell, block in placed:
        if _needs_support(block):
            _ensure_support(workspace, solid, cell)

def _pin_world(
    origin: tuple[int, int, int],
    pin: PinRef,
    footprint: Footprint,
) -> tuple[int, int, int]:
    ox, oy, oz = origin
    px, py, pz = pin.offset
    return (ox + px, oy + py, oz + (footprint.depth - 1 - pz))

_SIDE_NORMAL: dict[str, tuple[int, int, int]] = {
    CardinalDirection.SOUTH.value: (0, 0, 1),
    CardinalDirection.NORTH.value: (0, 0, -1),
    CardinalDirection.EAST.value: (1, 0, 0),
    CardinalDirection.WEST.value: (-1, 0, 0),
}

_OPPOSITE_SIDE: dict[str, str] = {
    CardinalDirection.SOUTH.value: CardinalDirection.NORTH.value,
    CardinalDirection.NORTH.value: CardinalDirection.SOUTH.value,
    CardinalDirection.EAST.value: CardinalDirection.WEST.value,
    CardinalDirection.WEST.value: CardinalDirection.EAST.value,
}

def _io_repeater_facing(component_type: ComponentType, side: CardinalDirection) -> str:
    if component_type == ComponentType.OUTPUT_PIN:
        return _OPPOSITE_SIDE[side.value]
    return side.value

def _inside_footprint(
    pos: tuple[int, int, int],
    origin: tuple[int, int, int],
    footprint: Footprint,
) -> bool:
    ox, oy, oz = origin
    x, y, z = pos
    return (
        ox <= x < ox + footprint.width
        and oy <= y < oy + footprint.height
        and oz <= z < oz + footprint.depth
    )

def _io_repeater_cell(
    pin_world: tuple[int, int, int],
    origin: tuple[int, int, int],
    footprint: Footprint,
    side: CardinalDirection,
    bounds: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[int, int, int]:
    nx, ny, nz = _SIDE_NORMAL[side.value]
    x, y, z = pin_world
    cx, cy, cz = x + nx, y + ny, z + nz
    while _inside_footprint((cx, cy, cz), origin, footprint):
        cx += nx
        cy += ny
        cz += nz
    if bounds is not None:
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        if not (
            min_x <= cx <= max_x
            and min_y <= cy <= max_y
            and min_z <= cz <= max_z
        ):
            raise ValueError(
                f"IO repeater cell {(cx, cy, cz)} is outside workspace bounds {bounds}; "
                f"pin_world={pin_world} origin={origin} side={side.value}"
            )
    return (cx, cy, cz)

def _default_io_side(component: Component, pin: PinRef) -> CardinalDirection:
    if pin.side is not None:
        return pin.side
    if component.type == ComponentType.INPUT_PIN:
        return CardinalDirection.SOUTH
    if component.type == ComponentType.OUTPUT_PIN:
        return CardinalDirection.NORTH
    raise ValueError(f"Missing pin.side for {component.id} pin {pin.name}")

_DIRS_6: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

def _bfs_path_3d(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    net_id: str,
    bounds: tuple[int, int, int, int, int, int],
) -> list[tuple[int, int, int]]:
    min_x, min_y, min_z, max_x, max_y, max_z = bounds

    def walkable(pos: tuple[int, int, int]) -> bool:
        x, y, z = pos
        if not (min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z):
            return False
        if pos in solid:
            return False
        block = workspace[x, y, z]
        if _is_air(block):
            return True
        if _is_redstone_wire(block):
            owner = dust_owner.get(pos)
            return owner is None or owner == net_id
        return False

    seeds: list[tuple[int, int, int]] = []
    if walkable(start):
        seeds.append(start)
    else:
        for dx, dy, dz in _DIRS_6:
            cand = (start[0] + dx, start[1] + dy, start[2] + dz)
            if walkable(cand):
                seeds.append(cand)
    if not seeds:
        raise ValueError(f"No walkable cell near start {start} for net {net_id}")

    if walkable(goal):
        goal_set: set[tuple[int, int, int]] = {goal}
    else:
        goal_set = set()
        for dx, dy, dz in _DIRS_6:
            cand = (goal[0] + dx, goal[1] + dy, goal[2] + dz)
            if walkable(cand):
                goal_set.add(cand)
        if not goal_set:
            raise ValueError(f"No walkable cell near goal {goal} for net {net_id}")

    queue: deque[tuple[int, int, int]] = deque(seeds)
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {
        s: None for s in seeds
    }
    reached: tuple[int, int, int] | None = None
    while queue:
        current = queue.popleft()
        if current in goal_set:
            reached = current
            break
        cx, cy, cz = current
        for dx, dy, dz in _DIRS_6:
            npos = (cx + dx, cy + dy, cz + dz)
            if npos in came_from:
                continue
            if not walkable(npos):
                continue
            came_from[npos] = current
            queue.append(npos)
    if reached is None:
        raise ValueError(f"No 3D route for net {net_id} from {start} to {goal}")

    path: list[tuple[int, int, int]] = []
    cur: tuple[int, int, int] | None = reached
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path

def _lay_redstone_path(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    path: Iterable[tuple[int, int, int]],
    net_id: str,
) -> None:
    for x, y, z in path:
        existing = workspace[x, y, z]
        if _is_redstone_wire(existing):
            owner = dust_owner.get((x, y, z))
            if owner is not None and owner != net_id:
                raise ValueError(f"Dust collision at {(x, y, z)} for nets {owner} vs {net_id}")
        workspace[x, y, z] = REDSTONE
        dust_owner[(x, y, z)] = net_id
        below = (x, y - 1, z)
        if y > 0 and _is_air(workspace[below[0], below[1], below[2]]) and below not in solid:
            workspace[below[0], below[1], below[2]] = STONE
            solid.add(below)

@dataclass
class _Placed:
    component: Component
    origin: tuple[int, int, int]

def _expand_multibit_io(comp: ComponentList) -> ComponentList:
    renames: dict[tuple[str, str], tuple[str, str]] = {}
    new_components: list[Component] = []
    for c in comp.components:
        is_io = c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
        if not is_io or len(c.pins) <= 1:
            new_components.append(c)
            continue
        for pin in c.pins:
            new_id = pin.name if pin.name != c.id else f"{c.id}_{pin.name}"
            new_pin = PinRef(
                name=pin.name,
                direction=pin.direction,
                side=pin.side,
                offset=(0, 0, 0),
            )
            new_components.append(
                Component(
                    id=new_id,
                    type=c.type,
                    pins=[new_pin],
                    params=dict(c.params),
                    footprint=Footprint(width=1, height=1, depth=1),
                )
            )
            renames[(c.id, pin.name)] = (new_id, pin.name)

    def rebind(ep: NetEndpoint) -> NetEndpoint:
        key = (ep.component_id, ep.pin_name)
        if key in renames:
            new_id, new_name = renames[key]
            return NetEndpoint(component_id=new_id, pin_name=new_name)
        return ep

    new_nets = [
        NetConnection(
            net_id=net.net_id,
            source=rebind(net.source),
            sinks=[rebind(sink) for sink in net.sinks],
        )
        for net in comp.nets
    ]
    return ComponentList(
        schema_version=comp.schema_version,
        components=new_components,
        nets=new_nets,
    )

def _layout_components(
    components: list[Component],
    *,
    gutter: int,
    base_y: int,
    io_margin: int = 2,
    routing_gutter: int = 3,
) -> list[_Placed]:
    inputs = [c for c in components if c.type == ComponentType.INPUT_PIN]
    outputs = [c for c in components if c.type == ComponentType.OUTPUT_PIN]
    gates = [
        c
        for c in components
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
    ]

    output_row_z = io_margin
    gate_row_z = output_row_z + routing_gutter + 1
    max_gate_depth = max((c.footprint.depth for c in gates), default=0)
    input_row_z = gate_row_z + max_gate_depth + routing_gutter

    placed: list[_Placed] = []
    for row, row_z in ((inputs, input_row_z), (gates, gate_row_z), (outputs, output_row_z)):
        cursor_x = 0
        for component in row:
            origin = (cursor_x, base_y, row_z)
            placed.append(_Placed(component=component, origin=origin))
            cursor_x += component.footprint.width + gutter
    return placed

def _compute_workspace_dims(
    placed: list[_Placed],
    *,
    base_y: int,
    io_margin: int,
    routing_headroom: int,
) -> tuple[int, int, int]:
    max_x = 1
    max_y = base_y + 1
    max_z = 1
    for item in placed:
        ox, oy, oz = item.origin
        fp = item.component.footprint
        max_x = max(max_x, ox + fp.width)
        max_y = max(max_y, oy + fp.height)
        max_z = max(max_z, oz + fp.depth)
    width = max_x + io_margin
    depth = max_z + io_margin
    height = max_y + routing_headroom + io_margin
    return (width, height, depth)

def _load_template_region(schematics_dir: Path, prefix: str) -> Region:
    path = schematics_dir / f"{prefix}.litematic"
    if not path.is_file():
        raise FileNotFoundError(f"Missing schematic template: {path}")
    schematic = Schematic.load(str(path))
    if not schematic.regions:
        raise ValueError(f"Schematic has no regions: {path}")
    return next(iter(schematic.regions.values()))

def build_litematic_from_component_list(
    comp: ComponentList,
    schematics_dir: Path,
    out_path: Path,
    *,
    gutter: int = 8,
    workspace_size: tuple[int, int, int] | None = None,
    base_y: int = 40,
    schematic_name: str = "build",
    io_margin: int = 2,
    routing_headroom: int = 4,
) -> Path:
    comp = _expand_multibit_io(comp)
    placed = _layout_components(
        comp.components, gutter=gutter, base_y=base_y, io_margin=io_margin
    )
    if workspace_size is None:
        width, height, depth = _compute_workspace_dims(
            placed,
            base_y=base_y,
            io_margin=io_margin,
            routing_headroom=routing_headroom,
        )
    else:
        width, height, depth = workspace_size
    workspace = Region(0, 0, 0, width, height, depth)
    solid: set[tuple[int, int, int]] = set()
    dust_owner: dict[tuple[int, int, int], str] = {}
    ws_bounds = (0, 0, 0, width - 1, height - 1, depth - 1)

    ref_mc_version = 2975
    ref_lm_version = 6
    ref_lm_sub = 1

    for item in placed:
        c = item.component
        if c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN, ComponentType.CUSTOM):
            continue
        if c.type not in SCHEMATIC_MAP:
            raise ValueError(f"No schematic template registered for component type {c.type}")
        info = SCHEMATIC_MAP[c.type]
        template = _load_template_region(schematics_dir, info.file_prefix)
        ref = Schematic.load(str(schematics_dir / f"{info.file_prefix}.litematic"))
        ref_mc_version = int(ref.mc_version)
        ref_lm_version = int(ref.lm_version)
        ref_lm_sub = int(ref.lm_subversion)
        _paste_template(workspace, template, item.origin, solid)

    pin_world: dict[tuple[str, str], tuple[int, int, int]] = {}
    pin_terminal: dict[tuple[str, str], tuple[int, int, int]] = {}
    for item in placed:
        c = item.component
        origin = item.origin
        is_io = c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
        for pin in c.pins:
            pw = _pin_world(origin, pin, c.footprint)
            pin_world[(c.id, pin.name)] = pw
            if is_io:
                pin_terminal[(c.id, pin.name)] = pw
            else:
                side = pin.side
                if side is None:
                    pin_terminal[(c.id, pin.name)] = pw
                else:
                    nx, ny, nz = _SIDE_NORMAL[side.value]
                    pin_terminal[(c.id, pin.name)] = (pw[0] + nx, pw[1] + ny, pw[2] + nz)

    for item in placed:
        c = item.component
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
            continue
        origin = item.origin
        for pin in c.pins:
            side = _default_io_side(c, pin)
            pw = pin_world[(c.id, pin.name)]
            cell = _io_repeater_cell(pw, origin, c.footprint, side, bounds=ws_bounds)
            facing = _io_repeater_facing(c.type, side)
            workspace[cell[0], cell[1], cell[2]] = BlockState(
                "minecraft:repeater",
                facing=facing,
                delay="1",
            )
            solid.add(cell)
            _ensure_support(workspace, solid, cell)

    bounds = ws_bounds

    for net in comp.nets:
        src_pin = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
        for sink in net.sinks:
            dst_pin = _pin_for_endpoint(
                pin_terminal, sink.component_id, sink.pin_name
            )
            path = _bfs_path_3d(
                workspace, solid, dust_owner, src_pin, dst_pin, net.net_id, bounds
            )
            _lay_redstone_path(workspace, solid, dust_owner, path, net.net_id)

    schematic = workspace.as_schematic(name=schematic_name, author="minecraft-v", description="merged placement")
    schematic.mc_version = ref_mc_version
    schematic.lm_version = ref_lm_version
    schematic.lm_subversion = ref_lm_sub

    out_path.parent.mkdir(parents=True, exist_ok=True)
    schematic.save(str(out_path))
    return out_path

def _pin_for_endpoint(
    pin_world: dict[tuple[str, str], tuple[int, int, int]],
    component_id: str,
    pin_name: str,
) -> tuple[int, int, int]:
    key = (component_id, pin_name)
    if key in pin_world:
        return pin_world[key]
    raise ValueError(f"Unknown pin endpoint {component_id}.{pin_name}")
