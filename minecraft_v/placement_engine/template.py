from __future__ import annotations
from pathlib import Path
from litemapy import BlockState, Region, Schematic
from minecraft_v.placement_ir import CardinalDirection, Component, ComponentType, Footprint, PinRef
from .constants import _SIDE_NORMAL, _OPPOSITE_SIDE
from .block_utils import _is_air, _needs_support, _ensure_support


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
            template.min_x(), template.min_y(), template.min_z(),
            template.max_x(), template.max_y(), template.max_z(),
        )
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


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
        wx, wy, wz = dx + (x - tmin_x), dy + (y - tmin_y), dz + (z - tmin_z)
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
        if not (min_x <= cx <= max_x and min_y <= cy <= max_y and min_z <= cz <= max_z):
            raise ValueError(
                f"IO repeater cell {(cx, cy, cz)} outside bounds {bounds}; "
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


def _load_template_region(schematics_dir: Path, prefix: str) -> Region:
    path = schematics_dir / f"{prefix}.litematic"
    if not path.is_file():
        raise FileNotFoundError(f"Missing schematic template: {path}")
    schematic = Schematic.load(str(path))
    if not schematic.regions:
        raise ValueError(f"Schematic has no regions: {path}")
    return next(iter(schematic.regions.values()))
