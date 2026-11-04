from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from minecraft_v.placement_engine.ir import (
    Component,
    ComponentList,
    ComponentType,
    Footprint,
    NetConnection,
    NetEndpoint,
    PinRef,
)

_BIT_INDEX_RE = re.compile(r'\[(\d+)\]$')
_BIT_NAME_RE = re.compile(r'^(.*)\[(\d+)\]$')


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
            rid, rname = renames[key]
            return NetEndpoint(component_id=rid, pin_name=rname)
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


def _io_bit_index(c: Component) -> int | None:
    """Return bit index from an IO component's ID (e.g. 'a[2]' → 2), or None."""
    if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
        return None
    m = _BIT_INDEX_RE.search(c.id)
    return int(m.group(1)) if m else None


def _io_base_name(comp_id: str) -> str:
    """Strip bit-index suffix: 'a[2]' → 'a', 'reset' → 'reset'."""
    m = _BIT_NAME_RE.match(comp_id)
    return m.group(1) if m else comp_id


def _assign_io_x_positions(
        io_components: list[Component],
        gates_placed: list[_Placed],
        nets: list[NetConnection],
        io_margin: int,
        gutter: int,
        ws_width: int,
        flat: bool = False,
        x_order: dict[str, float] | None = None,
) -> dict[str, int]:
    """
    Assign X positions to IO components to minimise total net distance.

    Multi-bit groups (same base name, e.g. 'a[0]'/'a[1]') share one X column
    so their bits are vertically aligned.  Groups are ordered by the median
    world-X of the gate pins they connect to, or by the median x_order of their
    members (component id -> sort key, e.g. output pin target X) where given.
    With ``flat`` every IO is on one Y level, so each gets its own column (bits
    of a group stay adjacent, in order).
    """
    if not io_components:
        return {}

    io_ids = {c.id for c in io_components}

    # (component_id, pin_name) → world-X for every placed gate pin
    gate_pin_x: dict[tuple[str, str], float] = {}
    for item in gates_placed:
        ox = item.origin[0]
        for pin in item.component.pins:
            gate_pin_x[(item.component.id, pin.name)] = float(ox + pin.offset[0])

    # Collect gate-pin X coords reachable from each IO component via nets
    gate_xs_for: dict[str, list[float]] = {c.id: [] for c in io_components}
    for net in nets:
        endpoints = [(net.source.component_id, net.source.pin_name)] + [
            (s.component_id, s.pin_name) for s in net.sinks
        ]
        gate_xs: list[float] = []
        io_on_net: list[str] = []
        for cid, pname in endpoints:
            px = gate_pin_x.get((cid, pname))
            if px is not None:
                gate_xs.append(px)
            elif cid in io_ids:
                io_on_net.append(cid)
        for io_id in io_on_net:
            gate_xs_for[io_id].extend(gate_xs)

    def _median(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[len(s) // 2]

    individual_x: dict[str, float] = {c.id: _median(gate_xs_for[c.id]) for c in io_components}

    # Group bits by base name; group desired X = median over member desired Xs
    groups: dict[str, list[str]] = defaultdict(list)
    for c in io_components:
        groups[_io_base_name(c.id)].append(c.id)
    if flat:
        for ids in groups.values():
            ids.sort(key=lambda cid: (_io_bit_index(next(c for c in io_components if c.id == cid)) or 0, cid))

    x_order = x_order or {}

    def _group_key(ids: list[str]) -> float:
        fixed = [x_order[cid] for cid in ids if cid in x_order]
        return _median(fixed) if fixed else _median([individual_x[cid] for cid in ids])

    sorted_groups = sorted((_group_key(ids), base) for base, ids in groups.items())

    io_layer_width = (1 + gutter) * (len(io_components) if flat else len(sorted_groups))

    result: dict[str, int] = {}
    x = io_margin + (ws_width - io_layer_width) // 2
    for _, base in sorted_groups:
        for cid in groups[base]:
            result[cid] = x
            if flat:
                x += 1 + gutter
        if not flat:
            x += 1 + gutter  # expanded IO footprint is always 1 wide

    return result


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
    depth = max_z + 2  # I/O repeater at max_z, with the input lever's stone behind it
    height = max_y + routing_headroom + io_margin
    return (width, height, depth)
