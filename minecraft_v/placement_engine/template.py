from __future__ import annotations

from pathlib import Path

from litemapy import BlockState, Region, Schematic

from minecraft_v.placement_engine.ir import CardinalDirection, Component, ComponentType, Footprint, PinRef
from .block_utils import _is_air, _needs_support, _ensure_support, _is_redstone_wire, _is_torch
from .constants import _SIDE_NORMAL, _OPPOSITE_SIDE


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


_MIRROR_X_DIR = {"east": "west", "west": "east"}


def _mirror_block_x(block: BlockState) -> BlockState:
    """``block`` reflected across the YZ plane: east and west swap in ``facing`` and wire sides."""
    props = dict(block.properties())
    changed = {}
    if props.get("facing") in _MIRROR_X_DIR:
        changed["facing"] = _MIRROR_X_DIR[props["facing"]]
    if "east" in props and "west" in props:
        changed["east"], changed["west"] = props["west"], props["east"]
    return block.with_properties(**changed) if changed else block


def _mirror_component_x(component: Component) -> Component:
    """Copy of ``component`` mirrored in X within its footprint; mirroring twice restores it.

    Pins move to ``width - 1 - x`` and east/west sides swap; ``metadata['mirror_x']`` tells
    ``_paste_template`` (via the builder) to reflect the schematic the same way.
    """
    w = component.footprint.width
    pins = [
        pin.model_copy(update={
            "offset": (w - 1 - pin.offset[0], pin.offset[1], pin.offset[2]),
            "side": CardinalDirection(_MIRROR_X_DIR[pin.side.value])
            if pin.side is not None and pin.side.value in _MIRROR_X_DIR else pin.side,
        })
        for pin in component.pins
    ]
    metadata = {**component.metadata, "mirror_x": not component.metadata.get("mirror_x", False)}
    return component.model_copy(update={"pins": pins, "metadata": metadata})


def _paste_template(
        workspace: Region,
        template: Region,
        dest_origin: tuple[int, int, int],
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str] | None = None,
        schematic_net_id: str | None = None,
        pin_positions: frozenset[tuple[int, int, int]] = frozenset(),
        mirror_width: int | None = None,
) -> None:
    """Paste ``template``'s non-air blocks at ``dest_origin``; ``mirror_width`` reflects it in X
    within that width (see ``_mirror_component_x``)."""
    tmin_x, tmin_y, tmin_z, _, _, _ = _non_air_bounds(template)
    dx, dy, dz = dest_origin
    placed: list[tuple[tuple[int, int, int], BlockState]] = []
    for pos in template.block_positions():
        block = template[pos]
        if _is_air(block):
            continue
        x, y, z = pos
        rx = x - tmin_x
        if mirror_width is not None:
            rx = mirror_width - 1 - rx
            block = _mirror_block_x(block)
        wx, wy, wz = dx + rx, dy + (y - tmin_y), dz + (z - tmin_z)
        workspace[wx, wy, wz] = block
        # `solid` doubles as "can support dust": skip dust, repeaters, comparators, torches.
        if not (_needs_support(block) or _is_torch(block)):
            solid.add((wx, wy, wz))
        placed.append(((wx, wy, wz), block))
        if (dust_owner is not None and schematic_net_id is not None
                and (wx, wy, wz) not in pin_positions
                and (_is_redstone_wire(block) or _is_torch(block))):
            dust_owner[(wx, wy, wz)] = schematic_net_id
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
