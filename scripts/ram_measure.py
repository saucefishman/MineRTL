"""Measure a design's RAM group without settling it. Usage: ram_measure.py <netlist.json>...

Prints each stack's size and what its X goes to, the group's size and what its Y goes to, and each lane's span.
Environment that changes the layout (RAM_FOLD_DECODER and so on) applies.
"""
import json
import sys
from collections import Counter

from minecraft_v.placement_engine.ram import RAM4_CELL, _COLUMN_PITCH, build_group, stack_spec


def specs_of(path: str) -> list:
    module = json.load(open(path))["modules"]["main"]
    specs = []
    for cell in module["cells"].values():
        if cell["type"] != RAM4_CELL:
            continue
        nets = {}
        for port, bits in cell["connections"].items():
            if port in ("ra", "rb", "rc", "rd"):
                continue
            for i, bit in enumerate(bits):
                pin = f"{port}[{i}]" if len(bits) > 1 or port != "we" else port
                if isinstance(bit, int):
                    nets[pin] = bit
        specs.append(stack_spec(cell["parameters"], nets))
    return specs


def main() -> None:
    for path in sys.argv[1:]:
        specs = specs_of(path)
        print(path)
        users = Counter(net for spec in specs for net in spec.nets.values())
        block, own, lanes = build_group("g", specs, [])
        fp = block.footprint
        print(f" stacks {len(specs)}, input nets {len(users)}, lanes {len(lanes)}, own pins {len(own)}, "
              f"lane pins {len(block.pins) - len(own)}, placed bits {sum(len(s.init) * s.width for s in specs)}")
        shared_ways = Counter(users[n] for n in lanes)
        print(f" lanes by stacks sharing them {sorted(shared_ways.items())}")
        for k, spec in enumerate(specs):
            cells = _COLUMN_PITCH * spec.width + 2
            words = len(spec.init)
            own_k = sum(1 for _, (s, _) in own.items() if s == k)
            on_lanes = sum(1 for n in spec.nets.values() if n in lanes)
            print(f" stack {k}: {words}x{spec.width} banks {spec.banks} abits {spec.abits} base {spec.base}; "
                  f"cells X {cells}, input pins on lanes {on_lanes}, group pins {own_k}")
        print(f" group XYZ {fp.width} x {fp.height} x {fp.depth}, pins {len(block.pins)}, "
              f"blocks without banks {len(block.blocks)}")
        ys = Counter(p.offset[1] for p in block.pins)
        print(f" pin Y range {min(ys)}..{max(ys)}, distinct {len(ys)}")
        arcs = [t for outs in block.arcs.values() for t in outs.values()]
        print(f" max arc {max(arcs)}")


if __name__ == "__main__":
    main()
