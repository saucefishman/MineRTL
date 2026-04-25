from __future__ import annotations

import re
from collections import defaultdict, deque
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


def _majority_y(ys: list[int]) -> int:
    """Plurality-vote Y level; tiebreak = max Y."""
    count: dict[int, int] = defaultdict(int)
    for y in ys:
        count[y] += 1
    max_count = max(count.values())
    # TODO: tiebreak to lowest Y value
    return min(y for y, cnt in count.items() if cnt == max_count)


def _build_dependency_layers(
        gates: list[Component],
        nets: list[NetConnection],
) -> dict[str, int]:
    """Return depth per gate: 0 = driven directly from inputs, N = N hops into graph.

    Cycles are broken by removing back edges detected during DFS.
    """
    gate_ids = {c.id for c in gates}

    successors: dict[str, set[str]] = {c.id: set() for c in gates}
    predecessors: dict[str, set[str]] = {c.id: set() for c in gates}
    for net in nets:
        src = net.source.component_id
        for sink in net.sinks:
            dst = sink.component_id
            if src in gate_ids and dst in gate_ids and src != dst:
                successors[src].add(dst)
                predecessors[dst].add(src)

    # Iterative DFS — detect back edges (cycle edges) and remove them.
    # Sort gate_ids and successor lists for deterministic traversal regardless of hash seed.
    visited: set[str] = set()
    in_stack: set[str] = set()
    for start in sorted(gate_ids):
        if start in visited:
            continue
        dfs_stack: list[tuple[str, list[str], int]] = [(start, sorted(successors[start]), 0)]
        in_stack.add(start)
        visited.add(start)
        while dfs_stack:
            node, nbrs, idx = dfs_stack[-1]
            if idx < len(nbrs):
                dfs_stack[-1] = (node, nbrs, idx + 1)
                nb = nbrs[idx]
                if nb not in visited:
                    visited.add(nb)
                    in_stack.add(nb)
                    dfs_stack.append((nb, sorted(successors[nb]), 0))
                elif nb in in_stack:
                    # Back edge — remove to break cycle
                    successors[node].discard(nb)
                    predecessors[nb].discard(node)
            else:
                in_stack.discard(node)
                dfs_stack.pop()

    in_deg = {gid: len(predecessors[gid]) for gid in gate_ids}
    depth: dict[str, int] = {gid: 0 for gid in gate_ids}
    queue: deque[str] = deque(sorted(gid for gid in gate_ids if in_deg[gid] == 0))
    while queue:
        node = queue.popleft()
        for nb in successors[node]:
            depth[nb] = max(depth[nb], depth[node] + 1)
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    return depth


def _assign_component_y_levels(
        components: list[Component],
        nets: list[NetConnection],
        *,
        y_stride: int,
) -> dict[str, int]:
    """Return Y offset per component ID.

    Indexed IO (e.g. ``a[2]``) → bit_index * y_stride (fixed).
    Gates → least-loaded Y among candidate levels drawn from both input
    sources and output sinks already assigned. Using both directions lets
    gates near the output boundary (e.g. DFFEs driving indexed outputs)
    land on the correct bit layer even when their inputs are unconstrained.
    Non-indexed IO (scalar shared) → majority Y of connected gates.
    """
    y_level: dict[str, int] = {}

    for c in components:
        idx = _io_bit_index(c)
        if idx is not None:
            y_level[c.id] = idx * y_stride

    gates = [c for c in components if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)]
    gate_ids = {c.id for c in gates}

    net_source_cid: dict[str, str] = {}
    gate_input_nets: dict[str, list[str]] = defaultdict(list)
    gate_output_nets: dict[str, list[str]] = defaultdict(list)
    net_sink_cids: dict[str, list[str]] = defaultdict(list)
    for net in nets:
        src = net.source.component_id
        net_source_cid[net.net_id] = src
        if src in gate_ids:
            gate_output_nets[src].append(net.net_id)
        for sink in net.sinks:
            net_sink_cids[net.net_id].append(sink.component_id)
            if sink.component_id in gate_ids:
                gate_input_nets[sink.component_id].append(net.net_id)

    io_ids = {c.id for c in components if c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)}
    io_nets = [n for n in nets if n.source.component_id in io_ids or any(s.component_id in io_ids for s in n.sinks)]
    depths = _build_dependency_layers(gates, io_nets)
    max_depth = max(depths.values(), default=0)
    gates_by_depth: dict[int, list[Component]] = defaultdict(list)
    for gate in gates:
        gates_by_depth[depths[gate.id]].append(gate)

    def _run_pass(depth_order: list[int]) -> None:
        gate_count_per_y: dict[int, int] = defaultdict(int)
        for existing_y in y_level.values():
            gate_count_per_y[existing_y] += 0
        for gate in gates:
            if gate.id in y_level:
                gate_count_per_y[y_level[gate.id]] += 1
        for d in depth_order:
            for gate in gates_by_depth[d]:
                if gate.id in y_level:
                    gate_count_per_y[y_level[gate.id]] -= 1
                input_ys = [
                    y_level[net_source_cid[nid]]
                    for nid in gate_input_nets[gate.id]
                    if net_source_cid[nid] in y_level
                ]
                output_ys = [
                    y_level[sink_cid]
                    for nid in gate_output_nets[gate.id]
                    for sink_cid in net_sink_cids[nid]
                    if sink_cid in y_level
                ]
                candidates = set(input_ys + output_ys)
                assigned = min(candidates, key=lambda y: gate_count_per_y[y]) if candidates else 0
                y_level[gate.id] = assigned
                gate_count_per_y[assigned] += 1

    _run_pass(list(range(max_depth + 1)))  # forward
    _run_pass(list(range(max_depth, -1, -1)))  # backward

    for c in components:
        if c.id in y_level:
            continue
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
            continue
        connected_ys: list[int] = []
        for net in nets:
            if net.source.component_id == c.id:
                for sink in net.sinks:
                    if sink.component_id in y_level:
                        connected_ys.append(y_level[sink.component_id])
            else:
                for sink in net.sinks:
                    if sink.component_id == c.id:
                        src = net.source.component_id
                        if src in y_level:
                            connected_ys.append(y_level[src])
        y_level[c.id] = _majority_y(connected_ys) if connected_ys else 0

    return y_level


def _layout_components(
        components: list[Component],
        nets: list[NetConnection],
        *,
        gutter: int,
        base_y: int,
        io_margin: int = 4,
        routing_gutter: int = 10,
        y_level: dict[str, int] | None = None,
) -> list[_Placed]:
    inputs = [c for c in components if c.type == ComponentType.INPUT_PIN]
    outputs = [c for c in components if c.type == ComponentType.OUTPUT_PIN]
    gates = [
        c for c in components
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
    ]

    def comp_y(c: Component) -> int:
        if y_level is not None and c.id in y_level:
            return base_y + y_level[c.id]
        return base_y

    gates_by_y: dict[int, list[Component]] = defaultdict(list)
    for c in gates:
        gates_by_y[comp_y(c)].append(c)

    _REGISTER_TYPES = {ComponentType.DFF, ComponentType.DFFE, ComponentType.DLATCH}
    _non_register_ids = {c.id for c in gates if c.type not in _REGISTER_TYPES}
    _feedforward_nets = [n for n in nets if n.source.component_id in _non_register_ids]

    local_depth: dict[str, int] = {}
    for y_gates in gates_by_y.values():
        ld = _build_dependency_layers(y_gates, _feedforward_nets)
        local_depth.update(ld)

    max_depth = max(local_depth.values(), default=0)

    layer_gates: dict[int, list[Component]] = defaultdict(list)
    for c in gates:
        layer_gates[local_depth[c.id]].append(c)

    output_row_z = 1
    z_cursor = output_row_z + routing_gutter
    layer_z: dict[int, int] = {}
    for layer_idx in range(max_depth, -1, -1):
        layer_z[layer_idx] = z_cursor
        layer_depth = max((c.footprint.depth for c in layer_gates.get(layer_idx, [])), default=0)
        z_cursor += layer_depth + routing_gutter
    input_row_z = z_cursor

    placed: list[_Placed] = []
    x_cursors: dict[tuple[int, int], int] = {}

    for c in outputs:
        y = comp_y(c)
        key = (max_depth + 1, y)
        cursor_x = x_cursors.get(key, io_margin)
        placed.append(_Placed(component=c, origin=(cursor_x, y, output_row_z)))
        x_cursors[key] = cursor_x + c.footprint.width + gutter

    for layer_idx in range(max_depth, -1, -1):
        z = layer_z[layer_idx]
        for c in layer_gates.get(layer_idx, []):
            y = comp_y(c)
            key = (layer_idx, y)
            cursor_x = x_cursors.get(key, io_margin)
            placed.append(_Placed(component=c, origin=(cursor_x, y, z)))
            x_cursors[key] = cursor_x + c.footprint.width + gutter

    for c in inputs:
        y = comp_y(c)
        key = (-1, y)
        cursor_x = x_cursors.get(key, io_margin)
        placed.append(_Placed(component=c, origin=(cursor_x, y, input_row_z)))
        x_cursors[key] = cursor_x + c.footprint.width + gutter

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
    depth = max_z + 1  # +1 so the I/O repeater fits at depth-1; shadow falls out of bounds
    height = max_y + routing_headroom + io_margin
    return (width, height, depth)
