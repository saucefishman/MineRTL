#!/usr/bin/env python3
"""
Simulate net power states for N clock cycles from component_list.json.

Usage:
    uv run simulate.py [options]

Options:
    --component-list PATH   component_list.json path (default: build/artifacts/component_list.json)
    --cycles N              number of clock cycles to simulate (default: 8)
    --clock COMP_ID         which INPUT_PIN component is the clock (auto-detected if omitted)
    --inputs KEY=VAL        set INPUT_PIN value: COMP_ID.PIN_NAME=0|1 (repeatable)
    --show-all              print all nets each cycle (default: only output pins)

Example:
    uv run simulate.py --cycles 4 --show-all
    uv run simulate.py --cycles 8 --inputs reset.reset=1
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, deque

from minecraft_v.placement_engine.ir import load_component_list, ComponentType, Direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEQ_TYPES = {ComponentType.DFF, ComponentType.DFFE, ComponentType.DLATCH}
_COMB_TYPES = {ComponentType.AND, ComponentType.OR, ComponentType.NOT,
               ComponentType.XOR, ComponentType.FULL_ADDER, ComponentType.MUX}


def _const_val(cv: str | None) -> int | None:
    if cv is None:
        return None
    return 1 if cv == "1" else 0


def _eval_component(ctype: ComponentType, inputs: dict[str, int]) -> dict[str, int]:
    """Combinational evaluation. Returns output pin values."""
    A = inputs.get("A", 0)
    B = inputs.get("B", 0)
    if ctype == ComponentType.AND:
        return {"Y": A & B}
    if ctype == ComponentType.OR:
        return {"Y": A | B}
    if ctype == ComponentType.NOT:
        return {"Y": 1 - A}
    if ctype == ComponentType.XOR:
        return {"Y": A ^ B}
    if ctype == ComponentType.FULL_ADDER:
        Cin = inputs.get("Cin", 0)
        s = A + B + Cin
        return {"S": s & 1, "Cout": (s >> 1) & 1}
    if ctype == ComponentType.MUX:
        S = inputs.get("S", 0)
        return {"Y": B if S else A}
    return {}


# ---------------------------------------------------------------------------
# Build lookup tables from ComponentList
# ---------------------------------------------------------------------------

def build_sim(cl):
    comp_by_id = {c.id: c for c in cl.components}

    # pin_const[(comp_id, pin_name)] = 0|1 for const-valued pins
    pin_const: dict[tuple[str, str], int] = {}
    for c in cl.components:
        for p in c.pins:
            v = _const_val(p.const_value)
            if v is not None:
                pin_const[(c.id, p.name)] = v

    # pin_net[(comp_id, pin_name)] = net_id  (only non-const pins)
    pin_net: dict[tuple[str, str], str] = {}
    # net_source[net_id] = (comp_id, pin_name)
    net_source: dict[str, tuple[str, str]] = {}
    # net_sinks[net_id] = [(comp_id, pin_name), ...]
    net_sinks: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for net in cl.nets:
        src_key = (net.source.component_id, net.source.pin_name)
        pin_net[src_key] = net.net_id
        net_source[net.net_id] = src_key
        for sink in net.sinks:
            sk = (sink.component_id, sink.pin_name)
            pin_net[sk] = net.net_id
            net_sinks[net.net_id].append(sk)

    return comp_by_id, pin_const, pin_net, net_source, net_sinks


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def _interactive_loop(cl, snapshots, comp_by_id, pin_net, net_source, max_cycle: int) -> None:
    try:
        import readline  # noqa: F401 — enables arrow keys / history on most platforms
    except ImportError:
        pass

    def _show_component(comp_id: str, snap: dict[str, int]) -> None:
        comp = comp_by_id.get(comp_id)
        if comp is None:
            print(f"  (not found: {comp_id!r})")
            return
        print(f"\n  [{comp.type.value}] {comp_id}")
        out_pins = [p for p in comp.pins if p.direction == Direction.OUT]
        in_pins  = [p for p in comp.pins if p.direction == Direction.IN]
        if out_pins:
            print("  outputs:")
            for p in out_pins:
                nid = pin_net.get((comp_id, p.name))
                v = snap.get(nid, 0) if nid else "?"
                src_comp, src_pin = net_source.get(nid, ("?", "?")) if nid else ("?", "?")
                print(f"    {p.name} = {v}  net={nid}  (src: {src_comp}.{src_pin})")
        if in_pins:
            print("  inputs:")
            for p in in_pins:
                cv = p.const_value
                if cv is not None:
                    print(f"    {p.name} = {cv}  (const)")
                    continue
                nid = pin_net.get((comp_id, p.name))
                v = snap.get(nid, 0) if nid else "?"
                src_comp, src_pin = net_source.get(nid, ("?", "?")) if nid else ("?", "?")
                print(f"    {p.name} = {v}  net={nid}  (src: {src_comp}.{src_pin})")

    current_cycle = max_cycle
    print(f"\nInteractive mode. Cycles simulated: 1–{max_cycle}.")
    print("Commands: <number> = select cycle | <filter> = search components | q = quit\n")

    while True:
        try:
            raw = input(f"[cycle {current_cycle}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw.lower() == "q":
            break

        if raw.isdigit():
            n = int(raw)
            if n < 1 or n > max_cycle:
                print(f"  cycle must be 1–{max_cycle}")
            else:
                current_cycle = n
            continue

        # Component filter: substring match on component id
        snap = snapshots[current_cycle]
        matches = [c for c in cl.components if raw.lower() in c.id.lower()]
        if not matches:
            print(f"  no components matching {raw!r}")
        for comp in matches:
            _show_component(comp.id, snap)


def simulate(cl, n_cycles: int, clock_id: str | None, input_vals: dict[str, int],
             show_all: bool, interactive: bool = False) -> None:
    comp_by_id, pin_const, pin_net, _net_source, _ = build_sim(cl)

    # Detect clock component
    input_comps = [c for c in cl.components if c.type == ComponentType.INPUT_PIN]
    if clock_id is None:
        clk_comps = [c for c in input_comps if "clk" in c.id.lower()]
        if len(clk_comps) == 1:
            clock_id = clk_comps[0].id
        elif len(clk_comps) > 1:
            clock_id = clk_comps[0].id
            print(f"[sim] multiple clock candidates, using {clock_id!r}", file=sys.stderr)
        else:
            print("[sim] no clock detected — FFs will not toggle", file=sys.stderr)

    # Build topo order for combinational components
    # Node = comp_id; edges: if output net of comp A feeds input pin of comb comp B → A→B
    comb_ids = {c.id for c in cl.components if c.type in _COMB_TYPES}
    comb_in_degree: dict[str, int] = {cid: 0 for cid in comb_ids}
    comb_dependents: dict[str, list[str]] = defaultdict(list)  # net_id → downstream comb ids

    for cid in comb_ids:
        comp = comp_by_id[cid]
        for p in comp.pins:
            if p.direction != Direction.IN:
                continue
            if p.const_value is not None:
                continue
            net_id = pin_net.get((cid, p.name))
            if net_id is None:
                continue
            src_comp_id, _ = _net_source.get(net_id, (None, None))
            if src_comp_id in comb_ids:
                comb_in_degree[cid] += 1
                comb_dependents[net_id].append(cid)

    # Kahn's algorithm
    queue = deque(cid for cid, deg in comb_in_degree.items() if deg == 0)
    topo: list[str] = []
    remaining = dict(comb_in_degree)
    while queue:
        cid = queue.popleft()
        topo.append(cid)
        comp = comp_by_id[cid]
        for p in comp.pins:
            if p.direction != Direction.OUT:
                continue
            net_id = pin_net.get((cid, p.name))
            if net_id is None:
                continue
            for dep in comb_dependents.get(net_id, []):
                remaining[dep] -= 1
                if remaining[dep] == 0:
                    queue.append(dep)
    if len(topo) != len(comb_ids):
        print("[sim] warning: combinational loop detected — partial evaluation order used", file=sys.stderr)
        topo.extend(cid for cid in comb_ids if cid not in set(topo))

    # State: net_value[net_id] = 0|1
    net_value: dict[str, int] = {net.net_id: 0 for net in cl.nets}

    # Sequential state: ff_q[comp_id] = Q output value
    ff_q: dict[str, int] = {c.id: 0 for c in cl.components if c.type in _SEQ_TYPES}

    # Net id for each INPUT_PIN's output pin
    input_pin_net: dict[str, str] = {}  # comp_id → net_id
    for c in cl.components:
        if c.type == ComponentType.INPUT_PIN:
            for p in c.pins:
                if p.direction == Direction.OUT:
                    nid = pin_net.get((c.id, p.name))
                    if nid:
                        input_pin_net[c.id] = nid

    # Net id for each sequential component's Q output
    ff_q_net: dict[str, str] = {}  # comp_id → net_id
    for c in cl.components:
        if c.type in _SEQ_TYPES:
            for p in c.pins:
                if p.direction == Direction.OUT:
                    nid = pin_net.get((c.id, p.name))
                    if nid:
                        ff_q_net[c.id] = nid

    def apply_inputs(clk_val: int) -> None:
        for c in input_comps:
            nid = input_pin_net.get(c.id)
            if nid is None:
                continue
            if c.id == clock_id:
                net_value[nid] = clk_val
            else:
                net_value[nid] = input_vals.get(c.id, 0)

    def apply_ff_outputs() -> None:
        for cid, q in ff_q.items():
            nid = ff_q_net.get(cid)
            if nid:
                net_value[nid] = q

    def get_pin_val(comp_id: str, pin_name: str) -> int:
        cv = pin_const.get((comp_id, pin_name))
        if cv is not None:
            return cv
        nid = pin_net.get((comp_id, pin_name))
        if nid is None:
            return 0
        return net_value.get(nid, 0)

    def resolve_comb() -> None:
        for cid in topo:
            comp = comp_by_id[cid]
            ins = {p.name: get_pin_val(cid, p.name)
                   for p in comp.pins if p.direction == Direction.IN}
            outs = _eval_component(comp.type, ins)
            for pin_name, val in outs.items():
                nid = pin_net.get((cid, pin_name))
                if nid:
                    net_value[nid] = val

    def sample_ffs(clk_was_low: bool) -> None:
        """Sample D inputs and update Q (rising-edge for DFF/DFFE, level for DLATCH)."""
        for c in cl.components:
            if c.type == ComponentType.DFF:
                if clk_was_low:  # rising edge
                    ff_q[c.id] = get_pin_val(c.id, "D")
            elif c.type == ComponentType.DFFE:
                if clk_was_low:
                    E = get_pin_val(c.id, "E")
                    if E:
                        ff_q[c.id] = get_pin_val(c.id, "D")
            elif c.type == ComponentType.DLATCH:
                E = get_pin_val(c.id, "E")
                if E:
                    ff_q[c.id] = get_pin_val(c.id, "D")

    def print_cycle(cycle: int) -> None:
        print(f"\n=== Cycle {cycle} ===")
        if show_all:
            for nid in sorted(net_value):
                src_comp, src_pin = _net_source.get(nid, ("?", "?"))
                print(f"  {nid}: {net_value[nid]}  (src: {src_comp}.{src_pin})")
            print("  --- outputs ---")
        for c in cl.components:
            if c.type == ComponentType.OUTPUT_PIN:
                print(f"  {c.id}:")
                for p in c.pins:
                    nid = pin_net.get((c.id, p.name))
                    v = net_value.get(nid, 0) if nid else 0
                    src_comp, src_pin = _net_source.get(nid, ("?", "?")) if nid else ("?", "?")
                    print(f"    {p.name}={v} (src: {src_comp}.{src_pin})")

    # Initial state: all 0s
    apply_ff_outputs()
    apply_inputs(clk_val=0)
    resolve_comb()

    snapshots: dict[int, dict[str, int]] = {}

    for cycle in range(1, n_cycles + 1):
        # Rising edge
        apply_inputs(clk_val=1)
        apply_ff_outputs()
        resolve_comb()
        sample_ffs(clk_was_low=True)
        apply_ff_outputs()

        # Falling edge — resolve steady state with clk=0
        apply_inputs(clk_val=0)
        resolve_comb()

        snapshots[cycle] = dict(net_value)
        if not interactive:
            print_cycle(cycle)

    if interactive:
        _interactive_loop(cl, snapshots, comp_by_id, pin_net, _net_source, n_cycles)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--component-list",
                        default="build/artifacts/component_list.json",
                        help="path to component_list.json")
    parser.add_argument("--cycles", type=int, default=8,
                        help="number of clock cycles (default: 8)")
    parser.add_argument("--clock", default=None,
                        help="INPUT_PIN component id to use as clock")
    parser.add_argument("--inputs", action="append", default=[],
                        metavar="COMP_ID=VAL",
                        help="set INPUT_PIN value: comp_id=0|1 (repeatable)")
    parser.add_argument("--show-all", action="store_true",
                        help="print all nets (default: output pins only)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="interactive component inspector after simulation")
    args = parser.parse_args()

    path = Path(args.component_list)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(1)

    cl = load_component_list(path)

    input_vals: dict[str, int] = {}
    for item in args.inputs:
        if "=" not in item:
            print(f"error: --inputs must be COMP_ID=VAL, got {item!r}", file=sys.stderr)
            sys.exit(1)
        k, v = item.split("=", 1)
        input_vals[k.strip()] = int(v.strip())

    simulate(cl, args.cycles, args.clock, input_vals, args.show_all, args.interactive)


if __name__ == "__main__":
    main()
