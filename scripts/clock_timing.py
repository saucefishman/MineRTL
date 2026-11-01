"""Build a design and report clock skew and hold margins from the routed wire.

Delays are in redstone ticks (repeaters and torches on the wire, the library's gate delays). Margin
is the earliest new data at a flip-flop's D or E, launched from any flip-flop, minus that flip-flop's
own clock arrival. Below about 1 tick, new data can get in on the same edge.

Usage: clock_timing.py <minecraft_v args...>
"""
import sys
from collections import Counter

from minecraft_v.cell_library import SCHEMATIC_MAP
from minecraft_v.main import main
from minecraft_v.placement_engine import builder
from minecraft_v.placement_engine.hold import wire_arrivals
from minecraft_v.placement_engine.ir import ComponentType
from minecraft_v.placement_engine.verify import _World

SEQUENTIAL = (ComponentType.DFF, ComponentType.DFFE, ComponentType.DLATCH)
CLOCK_NAMES = ("clk", "_clk")


def sink_delays(world, source, stop):
    return wire_arrivals(world, source, stop)[0]


def report(comp, workspace, pin_terminal):
    world = _World(workspace, frozenset(), frozenset(pin_terminal.values()))
    terminals = frozenset(pin_terminal.values())
    by_id = {c.id: c for c in comp.components}
    gate = lambda c: SCHEMATIC_MAP[c.type].propagation_delay_ticks if c.type in SCHEMATIC_MAP else 0
    nets_from: dict[str, list] = {}
    for n in comp.nets:
        nets_from.setdefault(n.source.component_id, []).append(n)
    delays = {}

    def net_arrivals(n):
        if n.net_id not in delays:
            src = pin_terminal[n.source.component_id, n.source.pin_name]
            sinks = {pin_terminal[s.component_id, s.pin_name] for s in n.sinks}
            delays[n.net_id] = sink_delays(world, src, terminals - sinks - {src})
        return delays[n.net_id]

    def propagate(starts, into):
        """Earliest time at each sink pin; passes through combinational components."""
        best = {}
        heap = [(t, i, node) for i, (node, t) in enumerate(starts.items())]
        heapq.heapify(heap)
        counter = len(heap)
        while heap:
            t, _, cid = heapq.heappop(heap)
            if t > best.get(cid, 1 << 30):
                continue
            for n in nets_from.get(cid, []):
                arr = net_arrivals(n)
                for s in n.sinks:
                    at = t + arr.get(pin_terminal[s.component_id, s.pin_name], 1 << 20)
                    if by_id[s.component_id].type in SEQUENTIAL:
                        key = (s.component_id, s.pin_name)
                        into[key] = min(into.get(key, 1 << 30), at)
                    elif by_id[s.component_id].type != ComponentType.OUTPUT_PIN:
                        out = at + gate(by_id[s.component_id])
                        if out < best.get(s.component_id, 1 << 30):
                            best[s.component_id] = out
                            counter += 1
                            heapq.heappush(heap, (out, counter, s.component_id))

    roots = {c.id: 1 for c in comp.components if c.type == ComponentType.INPUT_PIN and c.id in CLOCK_NAMES}
    pins = {}
    propagate(roots, pins)
    clock = {cid: t for (cid, pin), t in pins.items() if pin == "C" and t < 1 << 19}
    if not clock:
        print("[timing] no clocked flip-flops")
        return
    skew = max(clock.values()) - min(clock.values())
    print(f"[skew] {len(clock)} flip-flops, clock arrives {min(clock.values())}..{max(clock.values())} ticks, skew {skew}")
    launches = {cid: t + gate(by_id[cid]) for cid, t in clock.items()}
    data = {}
    propagate(launches, data)
    margins = {(c, p): t - clock[c] for (c, p), t in data.items() if c in clock and p != "C" and t < 1 << 19}
    if not margins:
        return
    hist = Counter(min(m // 5 * 5, 50) for m in margins.values())
    print("[skew] hold margin (ticks, bucket of 5, count): " + " ".join(f"{k}:{hist[k]}" for k in sorted(hist)))
    worst = sorted(margins.items(), key=lambda kv: kv[1])[:5]
    print(f"[skew] worst {worst}")
    print(f"[skew] pins with margin < 1: {sum(m < 1 for m in margins.values())} of {len(margins)}"
          f" (D {sum(m < 1 and p == 'D' for (_, p), m in margins.items())},"
          f" E {sum(m < 1 and p == 'E' for (_, p), m in margins.items())})")


original = builder._compute_critical_path


def capture(comp, dust_owner, workspace, pin_terminal, macros=None):
    result = original(comp, dust_owner, workspace, pin_terminal, macros)
    print(f"[skew] critical path {result} ticks")
    report(comp, workspace, pin_terminal)
    return result


builder._compute_critical_path = capture
main()
