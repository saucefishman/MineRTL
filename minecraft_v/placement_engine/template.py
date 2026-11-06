from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path

from litemapy import BlockState, Region, Schematic

from minecraft_v.placement_engine.ir import CardinalDirection, Component, ComponentType, Footprint, PinRef
from .block_utils import _is_air, _needs_support, _is_redstone_wire, _is_torch
from .constants import _SIDE_NORMAL, _OPPOSITE_SIDE, STONE


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


_OPPOSITE_DIR = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _rotate_block_180(block: BlockState) -> BlockState:
    """``block`` turned half way about Y: every horizontal ``facing`` and wire side swaps with its opposite."""
    props = dict(block.properties())
    changed = {}
    if props.get("facing") in _OPPOSITE_DIR:
        changed["facing"] = _OPPOSITE_DIR[props["facing"]]
    for a, b in (("east", "west"), ("north", "south")):
        if a in props and b in props:
            changed[a], changed[b] = props[b], props[a]
    return block.with_properties(**changed) if changed else block


def _rotate_component_180(component: Component) -> Component:
    """Copy of ``component`` turned half way about Y within its footprint; ``metadata['rotate_180']`` tells the stamp."""
    w, d = component.footprint.width, component.footprint.depth
    pins = [
        pin.model_copy(update={
            "offset": (w - 1 - pin.offset[0], pin.offset[1], d - 1 - pin.offset[2]),
            "side": CardinalDirection(_OPPOSITE_DIR[pin.side.value]) if pin.side is not None else None,
        })
        for pin in component.pins
    ]
    metadata = {**component.metadata, "rotate_180": not component.metadata.get("rotate_180", False)}
    return component.model_copy(update={"pins": pins, "metadata": metadata})


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


_Plan = list[tuple[tuple[int, int, int], BlockState, bool, bool, bool]]  # relative pos, block, needs support, torch, wire
_plans: dict[tuple[int, int | None, int], tuple[Region, object, _Plan]] = {}  # keeps the template so its id stays its own


def _template_plan(template: Region, mirror_width: int | None,
                   block_states: dict[tuple[int, int, int], dict[str, str]] | None) -> _Plan:
    """``template``'s non-air blocks relative to its first non-air corner, mirrored and overridden, as pasted."""
    key = (id(template), mirror_width, id(block_states))
    hit = _plans.get(key)
    if hit is not None and hit[0] is template and hit[1] is block_states:
        return hit[2]
    tmin_x, tmin_y, tmin_z, _, _, _ = _non_air_bounds(template)
    plan: _Plan = []
    for pos in template.block_positions():
        block = template[pos]
        if _is_air(block):
            continue
        x, y, z = pos
        rx = x - tmin_x
        relative = (rx, y - tmin_y, z - tmin_z)
        if block_states is not None and relative in block_states:
            block = block.with_properties(**block_states[relative])
        if mirror_width is not None:
            rx = mirror_width - 1 - rx
            block = _mirror_block_x(block)
        plan.append(((rx, y - tmin_y, z - tmin_z), block, _needs_support(block), _is_torch(block),
                     _is_redstone_wire(block)))
    _plans[key] = (template, block_states, plan)
    return plan


def _paste_template(
        workspace: Region,
        template: Region,
        dest_origin: tuple[int, int, int],
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str] | None = None,
        schematic_net_id: str | None = None,
        pin_positions: frozenset[tuple[int, int, int]] = frozenset(),
        mirror_width: int | None = None,
        block_states: dict[tuple[int, int, int], dict[str, str]] | None = None,
        write: Callable[[tuple[int, int, int], BlockState], None] | None = None,
) -> None:
    """Paste ``template``'s non-air blocks at ``dest_origin``; ``mirror_width`` reflects it in X
    within that width (see ``_mirror_component_x``). ``block_states`` overrides properties of
    blocks by position relative to the template's first non-air corner, e.g. to paste a flip-flop already holding a 1.
    write: palette_writer(workspace), for many pastes into one big workspace."""
    if write is None:
        write = workspace.__setitem__
    dx, dy, dz = dest_origin
    owned = dust_owner is not None and schematic_net_id is not None
    supported = []
    for (rx, ry, rz), block, needs_support, torch, wire in _template_plan(template, mirror_width, block_states):
        cell = (dx + rx, dy + ry, dz + rz)
        write(cell, block)
        # `solid` doubles as "can support dust": skip dust, repeaters, comparators, torches.
        if needs_support:
            supported.append(cell)
        elif not torch:
            solid.add(cell)
        if owned and (wire or torch) and cell not in pin_positions:
            dust_owner[cell] = schematic_net_id
    for x, y, z in supported:
        if y <= 0:
            continue
        below = (x, y - 1, z)
        if below in solid:
            continue
        if _is_air(workspace[below]):
            write(below, STONE)
            solid.add(below)


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


@cache
def _load_template_region(schematics_dir: Path, prefix: str) -> Region:
    """Shared between callers: don't change it."""
    path = schematics_dir / f"{prefix}.litematic"
    if not path.is_file():
        raise FileNotFoundError(f"Missing schematic template: {path}")
    schematic = Schematic.load(str(path))
    if not schematic.regions:
        raise ValueError(f"Schematic has no regions: {path}")
    return next(iter(schematic.regions.values()))
