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


def _io_y_index(c: Component, clock_pin: str | None = None) -> int | None:
    """Bit index that fixes an IO's Y level, or None if it follows its gates.

    Scalar inputs ('i0') count as bit 0, so bit k of every module input sits on
    the same Y level.  The generated clock pin (``clock_pin``) is exempt and
    follows its gates like a scalar output.
    """
    idx = _io_bit_index(c)
    if idx is None and c.type == ComponentType.INPUT_PIN and c.id != clock_pin:
        return 0
    return idx


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
        ws_width: int
) -> dict[str, int]:
    """
    Assign X positions to IO components to minimise total net distance.

    Multi-bit groups (same base name, e.g. 'a[0]'/'a[1]') share one X column
    so their bits are vertically aligned.  Groups are ordered by the median
    world-X of the gate pins they connect to.
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

    sorted_groups = sorted(
        (_median([individual_x[cid] for cid in ids]), base)
        for base, ids in groups.items()
    )

    io_layer_width = (1 + gutter) * len(sorted_groups)

    result: dict[str, int] = {}
    x = io_margin + (ws_width - io_layer_width) // 2
    for _, base in sorted_groups:
        for cid in groups[base]:
            result[cid] = x
        x += 1 + gutter  # expanded IO footprint is always 1 wide

    return result


def _compute_output_cone_y(
        gates: list[Component],
        all_components: list[Component],
        nets: list[NetConnection],
        y_stride: int,
) -> dict[str, int | None]:
    """
    For each gate, determine which indexed output-IO cones it belongs to via
    backward BFS.  Returns the unique Y level if the gate is exclusively in
    one cone, None if it is shared across multiple cones or unreachable.

    Only OUTPUT_PIN components with a bit index drive the cone analysis; input
    IOs are ignored here because address/control bits fan forward into shared
    logic and would pollute the cone labelling.
    """
    gate_ids = {c.id for c in gates}

    # comp → nets where it appears as a SINK (receives signal)
    comp_as_sink: dict[str, list[str]] = defaultdict(list)
    net_source_comp: dict[str, str] = {}
    for net in nets:
        net_source_comp[net.net_id] = net.source.component_id
        for sink in net.sinks:
            comp_as_sink[sink.component_id].append(net.net_id)

    indexed_outputs: list[tuple[str, int]] = [
        (c.id, _io_bit_index(c))
        for c in all_components
        if c.type == ComponentType.OUTPUT_PIN and _io_bit_index(c) is not None
    ]

    if not indexed_outputs:
        return {c.id: None for c in gates}

    gate_cone_ys: dict[str, set[int]] = {c.id: set() for c in gates}

    for io_id, bit_idx in indexed_outputs:
        y = bit_idx * y_stride  # type: ignore[operator]
        visited: set[str] = {io_id}
        queue: deque[str] = deque([io_id])
        while queue:
            comp_id = queue.popleft()
            if comp_id in gate_ids:
                gate_cone_ys[comp_id].add(y)
            for net_id in comp_as_sink.get(comp_id, []):
                src = net_source_comp.get(net_id)
                if src is not None and src not in visited:
                    visited.add(src)
                    queue.append(src)

    return {
        c.id: (next(iter(gate_cone_ys[c.id])) if len(gate_cone_ys[c.id]) == 1 else None)
        for c in gates
    }


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
        clock_pin: str | None = None,
) -> dict[str, int]:
    """Return Y offset per component ID.

    Indexed IO (e.g. ``a[2]``) → bit_index * y_stride (fixed); scalar inputs
    other than ``clock_pin`` count as bit 0.
    Gates exclusively in one indexed-output's backward fanin cone → forced to
    that cone's Y level (avoids the heuristic misassigning independent data
    paths that happen to share control/address logic).
    Remaining gates → least-loaded Y among candidates from connected
    already-assigned components; two passes (forward + backward).
    Scalar outputs and ``clock_pin`` → majority Y of connected gates.
    """
    y_level: dict[str, int] = {}

    for c in components:
        idx = _io_y_index(c, clock_pin)
        if idx is not None:
            y_level[c.id] = idx * y_stride

    gates = [c for c in components if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)]
    gate_ids = {c.id for c in gates}

    # Pre-seed gates that live exclusively in one indexed output's fanin cone.
    # Gates reachable from multiple cones remain unset and are handled by the
    # heuristic passes below.
    cone_y = _compute_output_cone_y(gates, components, nets, y_stride)
    fixed_y_gates: set[str] = set()
    for g in gates:
        cy = cone_y.get(g.id)
        if cy is not None:
            y_level[g.id] = cy
            fixed_y_gates.add(g.id)

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
    for g in gates:
        gates_by_depth[depths[g.id]].append(g)

    def _run_pass(depth_order: list[int]) -> None:
        gate_count_per_y: dict[int, int] = defaultdict(int)
        for existing_y in y_level.values():
            gate_count_per_y[existing_y] += 0
        for g in gates:
            if g.id in y_level:
                gate_count_per_y[y_level[g.id]] += 1
        for d in depth_order:
            for g in gates_by_depth[d]:
                if g.id in fixed_y_gates:
                    continue  # cone-assigned — never reassign
                if g.id in y_level:
                    gate_count_per_y[y_level[g.id]] -= 1
                input_ys = [
                    y_level[net_source_cid[nid]]
                    for nid in gate_input_nets[g.id]
                    if net_source_cid[nid] in y_level
                ]
                output_ys = [
                    y_level[sink_cid]
                    for nid in gate_output_nets[g.id]
                    for sink_cid in net_sink_cids[nid]
                    if sink_cid in y_level
                ]
                candidates = set(input_ys + output_ys)
                assigned = min(candidates, key=lambda y: gate_count_per_y[y]) if candidates else 0
                y_level[g.id] = assigned
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


_REGISTER_TYPES = {ComponentType.DFF, ComponentType.DFFE, ComponentType.DLATCH}


def _feedforward_nets(gates: list[Component], nets: list[NetConnection]) -> list[NetConnection]:
    """Nets driven by combinational gates — register outputs are feedback and impose no layer order."""
    non_register_ids = {c.id for c in gates if c.type not in _REGISTER_TYPES}
    return [n for n in nets if n.source.component_id in non_register_ids]


def _assign_gate_layers(
        gates: list[Component],
        nets: list[NetConnection],
        gate_y: dict[str, int],
) -> dict[str, int]:
    """Depth layer per gate, computed separately within each Y level (0 = nearest the inputs)."""
    gates_by_y: dict[int, list[Component]] = defaultdict(list)
    for c in gates:
        gates_by_y[gate_y[c.id]].append(c)

    ff_nets = _feedforward_nets(gates, nets)
    local_depth: dict[str, int] = {}
    for y_gates in gates_by_y.values():
        local_depth.update(_build_dependency_layers(y_gates, ff_nets))
    return local_depth


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
    gates = [
        c for c in components
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
    ]

    def comp_y(c: Component) -> int:
        if y_level is not None and c.id in y_level:
            return base_y + y_level[c.id]
        return base_y

    local_depth = _assign_gate_layers(gates, nets, {c.id: comp_y(c) for c in gates})

    layer_gates: dict[int, list[Component]] = defaultdict(list)
    for c in gates:
        layer_gates[local_depth[c.id]].append(c)

    return _pack_layout(
        components, nets, layer_gates, comp_y,
        gutter=gutter, io_margin=io_margin, routing_gutter=routing_gutter,
    )


def _pack_layout(
        components: list[Component],
        nets: list[NetConnection],
        layer_gates: dict[int, list[Component]],
        comp_y,
        *,
        gutter: int,
        io_margin: int,
        routing_gutter: int,
) -> list[_Placed]:
    """Pack gates into rows and place IO around them.

    ``layer_gates`` maps layer index → gates in X order (rows are split by Y).
    Higher layers sit nearer the output row; empty layers take no space.
    """
    inputs = [c for c in components if c.type == ComponentType.INPUT_PIN]
    outputs = [c for c in components if c.type == ComponentType.OUTPUT_PIN]
    layer_order = sorted((idx for idx, gs in layer_gates.items() if gs), reverse=True)

    output_row_z = 1
    z_cursor = output_row_z + routing_gutter
    layer_z: dict[int, int] = {}
    layer_widths = {}
    for layer_idx in layer_order:
        layer_z[layer_idx] = z_cursor
        layer_depth = max((c.footprint.depth for c in layer_gates.get(layer_idx, [])), default=0)
        z_cursor += layer_depth + routing_gutter

        for c in layer_gates.get(layer_idx, []):
            key = (layer_idx, comp_y(c))
            if key not in layer_widths:
                layer_widths[key] = 0
            layer_widths[key] += c.footprint.width + gutter
    input_row_z = z_cursor

    placed: list[_Placed] = []
    x_cursors: dict[tuple[int, int], int] = {}
    max_layer_width = max(layer_widths.values(), default=0)

    # Place gates first so their X positions are known for IO optimisation.
    for layer_idx in layer_order:
        z = layer_z[layer_idx]

        for c in layer_gates.get(layer_idx, []):
            y = comp_y(c)
            key = (layer_idx, y)
            cursor_x = x_cursors.get(key, io_margin + (max_layer_width - layer_widths[key]) // 2)
            placed.append(_Placed(component=c, origin=(cursor_x, y, z)))
            x_cursors[key] = cursor_x + c.footprint.width + gutter

    # Assign IO X positions: minimise net distance, same X per multi-bit group.
    input_xs = _assign_io_x_positions(inputs, placed, nets, io_margin, gutter, max_layer_width)
    output_xs = _assign_io_x_positions(outputs, placed, nets, io_margin, gutter, max_layer_width)

    for c in outputs:
        y = comp_y(c)
        placed.append(_Placed(component=c, origin=(output_xs[c.id], y, output_row_z)))

    for c in inputs:
        y = comp_y(c)
        placed.append(_Placed(component=c, origin=(input_xs[c.id], y, input_row_z)))

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
    depth = max_z + 2  # I/O repeater at max_z, with the input lever's stone behind it
    height = max_y + routing_headroom + io_margin
    return (width, height, depth)
