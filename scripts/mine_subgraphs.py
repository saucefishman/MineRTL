"""Find repeated connected subgraphs in a post-yosys netlist, respecting cell type and pin role.

    uv run --with networkx python scripts/mine_subgraphs.py build/artifacts/netlist.json

Two views:
- components: drop nets with fanout > F, split into connected components, group them by
  isomorphism (WL hash, confirmed with networkx). These are the maximal macro candidates.
- tiles: inside the largest repeated component, the repeated subtrees hanging off its
  single-fanout outputs, k levels up from the leaves.
For each pattern: instances, share of cells and nets, nets it hides (all endpoints in one
instance) and pins it still exposes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher


class Netlist:
    def __init__(self, path: str, module: str = "main"):
        mod = json.load(open(path))["modules"][module]
        self.cells = {name: c for name, c in mod["cells"].items()}
        self.driver: dict[int, tuple[str, str]] = {}
        self.sinks: dict[int, list[tuple[str, str]]] = defaultdict(list)
        self.pins: dict[str, dict[str, int]] = defaultdict(dict)
        outputs = {"Y", "Q", "S_out", "Cout"}
        for name, c in self.cells.items():
            directions = c.get("port_directions") or {}
            for port, bits in c["connections"].items():
                for bit in bits:
                    if not isinstance(bit, int):
                        continue
                    self.pins[name][port] = bit
                    is_out = directions.get(port) == "output" if directions else port in outputs
                    if c["type"] == "fulladder" and not directions:
                        is_out = port in ("S", "Cout")
                    if is_out:
                        self.driver[bit] = (name, port)
                    else:
                        self.sinks[bit].append((name, port))
        for port in mod["ports"].values():
            for bit in port["bits"]:
                if isinstance(bit, int) and port["direction"] == "output":
                    self.sinks[bit].append(("<io>", "out"))
        self.nets = set(self.driver) | set(self.sinks)

    def fanout(self, bit: int) -> int:
        return len(self.sinks[bit])

    def graph(self, cells, max_fanout: float = float("inf")) -> nx.DiGraph:
        """Labelled bipartite cell/net graph: edge label = pin role, so isomorphism respects pins."""
        g = nx.DiGraph()
        cells = set(cells)
        for name in cells:
            g.add_node(name, label=self.cells[name]["type"])
            for port, bit in self.pins[name].items():
                if self.fanout(bit) > max_fanout:
                    continue
                net = f"n{bit}"
                g.add_node(net, label="net")
                if self.driver.get(bit) == (name, port):
                    g.add_edge(name, net, label=port)
                else:
                    g.add_edge(net, name, label=port)
        return g

    def stats(self, instances: list[set[str]]) -> dict:
        hidden = exposed = 0
        for inst in instances:
            touching = {b for n in inst for b in self.pins[n].values()}
            for bit in touching:
                ends = {self.driver[bit][0]} if bit in self.driver else {"<in>"}
                ends |= {n for n, _ in self.sinks[bit]}
                if ends <= inst:
                    hidden += 1
                else:
                    exposed += 1
        n = len(instances)
        cells = sum(len(i) for i in instances)
        return {"instances": n, "cells": len(instances[0]), "cell_share": cells / len(self.cells),
                "hidden_nets_each": hidden // n, "hidden_share": hidden / len(self.nets),
                "exposed_nets_each": exposed // n}


def components(nl: Netlist, max_fanout: int) -> list[list[set[str]]]:
    """Isomorphism classes of connected components once nets above max_fanout are removed."""
    g = nl.graph(nl.cells, max_fanout)
    comps = [{n for n in c if not n.startswith("n") or n in nl.cells}
             for c in nx.weakly_connected_components(g)]
    by_hash: dict[str, list[tuple[set[str], nx.DiGraph]]] = defaultdict(list)
    for comp in comps:
        sub = nl.graph(comp, max_fanout)
        by_hash[nx.weisfeiler_lehman_graph_hash(sub, node_attr="label", edge_attr="label", iterations=4)].append((comp, sub))
    classes = []
    for group in by_hash.values():
        reps: list[tuple[nx.DiGraph, list[set[str]]]] = []
        for comp, sub in group:
            for rep, members in reps:
                # An exact match on the small ones; huge WL-equal components are checked against the first only.
                gm = DiGraphMatcher(rep, sub, node_match=lambda a, b: a["label"] == b["label"],
                                  edge_match=lambda a, b: a["label"] == b["label"])
                if gm.is_isomorphic():
                    members.append(comp)
                    break
            else:
                reps.append((sub, [comp]))
        classes.extend(members for _, members in reps)
    return sorted(classes, key=lambda c: -len(c) * len(c[0]))


def subtree_tiles(nl: Netlist, slice_cells: set[str], levels: int) -> list[set[str]]:
    """Cells within `levels` mux levels of the storage leaves, grouped under their level-k root."""
    level: dict[str, int] = {}
    def depth(name: str) -> int:
        if name in level:
            return level[name]
        level[name] = 0
        if nl.cells[name]["type"].startswith("$_DFF"):
            return 0
        best = 0
        for port, bit in nl.pins[name].items():
            drv = nl.driver.get(bit)
            if drv and drv[0] in slice_cells and drv[0] != name and port != "S":
                best = max(best, depth(drv[0]))
        level[name] = best + 1
        return level[name]
    for name in slice_cells:
        depth(name)
    tiles: dict[frozenset, set[str]] = {}
    # A level-k tile is a pair of port trees over the same 2^k storage cells.
    roots = [n for n in slice_cells if level[n] == levels]
    def leaves(name: str) -> set[str]:
        if level[name] == 0:
            return {name}
        out = set()
        for port, bit in nl.pins[name].items():
            drv = nl.driver.get(bit)
            if drv and drv[0] in slice_cells and port != "S":
                out |= leaves(drv[0])
        return out
    def cone(name: str) -> set[str]:
        out = {name}
        for port, bit in nl.pins[name].items():
            drv = nl.driver.get(bit)
            if drv and drv[0] in slice_cells and port != "S" and level[name] > 0:
                out |= cone(drv[0])
        return out
    for root in roots:
        key = frozenset(leaves(root))
        tiles.setdefault(key, set()).update(cone(root))
    return list(tiles.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("netlist")
    parser.add_argument("--module", default="main")
    parser.add_argument("--fanouts", type=int, nargs="*", default=[1, 2, 3, 4, 8])
    args = parser.parse_args()
    nl = Netlist(args.netlist, args.module)
    print(f"{len(nl.cells)} cells, {len(nl.nets)} nets, types {dict(Counter(c['type'] for c in nl.cells.values()))}")
    best = None
    for fanout in args.fanouts:
        classes = [c for c in components(nl, fanout) if len(c) > 1 and len(c[0]) > 1]
        print(f"\n-- components, nets with fanout > {fanout} cut")
        for members in classes[:6]:
            s = nl.stats(members)
            types = dict(Counter(nl.cells[n]["type"] for n in members[0]))
            print(f"  {s['instances']:4} x {s['cells']:4} cells ({s['cell_share']:5.1%} of cells), "
                  f"hides {s['hidden_nets_each']:4} nets each ({s['hidden_share']:5.1%} of nets), "
                  f"exposes {s['exposed_nets_each']:3}: {types}")
        if classes and (best is None or len(classes[0]) * len(classes[0][0]) > len(best) * len(best[0])):
            best = classes[0]
    if best is None:
        return
    print("\n-- tiles inside the largest repeated component")
    for k in range(0, 8):
        tiles = [t for inst in best for t in subtree_tiles(nl, inst, k)]
        if not tiles:
            break
        sizes = Counter(len(t) for t in tiles)
        size = sizes.most_common(1)[0][0]
        same = [t for t in tiles if len(t) == size]
        s = nl.stats(same)
        print(f"  level {k}: {s['instances']:5} x {s['cells']:4} cells ({s['cell_share']:5.1%}), "
              f"hides {s['hidden_nets_each']:4} each ({s['hidden_share']:5.1%} of nets), exposes {s['exposed_nets_each']:3}")


if __name__ == "__main__":
    main()
