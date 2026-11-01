"""The repeated subcircuit finder: exact isomorphism and disjoint instances, on hand-made and example netlists."""
import shutil
import subprocess
from pathlib import Path

import pytest

from minecraft_v.main import Netlist, module_to_component_list
from minecraft_v.placement_engine.ir import (
    CURRENT_SCHEMA_VERSION, Component, ComponentList, ComponentType, Direction, NetConnection, NetEndpoint, PinRef,
)
from minecraft_v.placement_engine.subgraphs import RepeatedClass, find_repeated

ROOT = Path(__file__).resolve().parent.parent
_INPUTS = {ComponentType.AND: ("A", "B"), ComponentType.XOR: ("A", "B"), ComponentType.NOT: ("A",)}


def _netlist(gates: list[tuple[str, ComponentType, tuple[str, ...], str]], inputs: list[str]) -> ComponentList:
    """Gates as (id, type, input nets, output net). Input nets are driven by pins named after them."""
    comps = [Component(id=n, type=ComponentType.INPUT_PIN, pins=[PinRef(name=n, direction=Direction.OUT)]) for n in inputs]
    source = {n: (n, n) for n in inputs}
    sinks: dict[str, list[tuple[str, str]]] = {n: [] for n in inputs}
    for gid, kind, ins, out in gates:
        names = _INPUTS[kind]
        comps.append(Component(id=gid, type=kind, pins=[PinRef(name=p, direction=Direction.IN) for p in names]
                               + [PinRef(name="Y", direction=Direction.OUT)]))
        source[out] = (gid, "Y")
        sinks.setdefault(out, [])
        for net, pin in zip(ins, names):
            sinks.setdefault(net, []).append((gid, pin))
    nets = [NetConnection(net_id=n, source=NetEndpoint(component_id=source[n][0], pin_name=source[n][1]),
                          sinks=[NetEndpoint(component_id=c, pin_name=p) for c, p in sinks[n]]) for n in source if sinks[n]]
    return ComponentList(schema_version=CURRENT_SCHEMA_VERSION, components=comps, nets=nets)


def _assert_exact(comp: ComponentList, classes: list[RepeatedClass]) -> None:
    """Checks every class against the netlist alone, nothing from the finder's own bookkeeping."""
    by_id = {c.id: c for c in comp.components}
    pin_net, ends = {}, {}
    for net in comp.nets:
        ends[net.net_id] = [net.source.component_id, *(s.component_id for s in net.sinks)]
        for e in [net.source, *net.sinks]:
            pin_net[(e.component_id, e.pin_name)] = net.net_id
    used: set[str] = set()
    for cls in classes:
        assert len(cls.instances) >= 2
        for m in cls.instances:
            inside = {m[t] for t in cls.cells}
            assert len(inside) == len(cls.cells)
            assert not inside & used
            used |= inside
            assert len({m[t] for t in cls.nets}) == len(cls.nets)
            for t in cls.cells:
                a, b = by_id[t], by_id[m[t]]
                assert a.type == b.type and a.metadata.get("init", 0) == b.metadata.get("init", 0)
                assert {(p.name, p.const_value) for p in a.pins} == {(p.name, p.const_value) for p in b.pins}
                for p in a.pins:
                    net = pin_net.get((t, p.name))
                    assert pin_net.get((m[t], p.name)) == (None if net is None else m[net])
            for t, label in cls.nets.items():
                outside = [e for e in ends[m[t]] if e not in inside]
                assert (label == "internal") == (not outside)
                if outside:
                    assert (label == "output") == (ends[m[t]][0] in inside)


def _tile(k: int, kind: ComponentType, depth: int, shared: str) -> list[tuple[str, ComponentType, tuple[str, ...], str]]:
    """A chain of gates, each also on a net shared between tiles."""
    gates = [(f"g{k}_0", kind, (f"in{k}", shared), f"n{k}_0")]
    gates += [(f"g{k}_{i}", kind, (f"n{k}_{i - 1}", shared), f"n{k}_{i}") for i in range(1, depth)]
    return gates


def test_finds_several_patterns_inside_one_connected_circuit():
    """Each tile feeds the next, so no tile is a whole component once high-fanout nets are cut."""
    gates = []
    for k in range(7):
        kind, depth = (ComponentType.AND, 8) if k < 4 else (ComponentType.XOR, 10)
        tile = _tile(k, kind, depth, "s")
        if k:
            tile[0] = (tile[0][0], kind, (f"j{k - 1}", "s"), tile[0][3])
        gates += tile + [(f"join{k}", ComponentType.NOT, (f"n{k}_{depth - 1}",), f"j{k}")]
    comp = _netlist(gates, ["s", "in0"])
    classes = find_repeated(comp)
    _assert_exact(comp, classes)
    kinds = {next(c for c in comp.components if c.id == t).type for cls in classes for t in cls.cells}
    assert {ComponentType.AND, ComponentType.XOR} <= kinds
    assert len(classes) >= 2


def test_a_chain_does_not_overlap_itself():
    gates = [(f"g{i}", ComponentType.AND, (f"n{i - 1}" if i else "a", "s"), f"n{i}") for i in range(64)]
    comp = _netlist(gates, ["a", "s"])
    classes = find_repeated(comp)
    _assert_exact(comp, classes)
    assert classes
    assert all(len(c.cells) >= 8 for c in classes)


def test_a_symmetric_pair_agrees_on_shared_nets():
    """Two trees over one set of cells look alike; only their select nets tell them apart."""
    gates = []
    for k in range(4):
        gates += [(f"n{k}_{i}", ComponentType.NOT, (f"in{k}",) if i == 0 else (f"x{k}_{i - 1}",), f"x{k}_{i}") for i in range(8)]
        gates += [(f"a{k}", ComponentType.AND, (f"x{k}_7", "s0"), f"p{k}"), (f"b{k}", ComponentType.AND, (f"x{k}_7", "s1"), f"q{k}")]
    comp = _netlist(gates, ["s0", "s1", *(f"in{k}" for k in range(4))])
    classes = find_repeated(comp)
    _assert_exact(comp, classes)
    assert len(classes) == 1 and len(classes[0].instances) == 4
    cls = classes[0]
    assert sum(len({m[n] for m in cls.instances}) == 1 for n in cls.nets) == 2  # s0 and s1 each on one lane


def test_nothing_repeated_finds_nothing():
    comp = _netlist(_tile(0, ComponentType.AND, 12, "s"), ["s", "in0"])
    assert find_repeated(comp) == []


def test_a_spent_time_limit_still_returns_valid_classes():
    gates = [g for k in range(4) for g in _tile(k, ComponentType.AND, 8, "s")]
    comp = _netlist(gates, ["s", *(f"in{k}" for k in range(4))])
    _assert_exact(comp, find_repeated(comp, time_limit=0.0))


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not installed")
def test_cpu_finds_its_memory_and_registers_within_the_limit(tmp_path):
    subprocess.run(["./synth.sh", "examples/cpu", "--synth-only", "--artifacts-dir", str(tmp_path)], cwd=ROOT, check=True,
                   capture_output=True)
    netlist = Netlist.model_validate_json((tmp_path / "netlist.json").read_text())
    comp = module_to_component_list(netlist.modules["main"])
    classes = find_repeated(comp, time_limit=30.0)
    _assert_exact(comp, classes)
    sizes = [(len(c.instances), len(c.cells)) for c in classes]
    assert sizes[0][0] == 16 and sizes[0][1] >= 190
    assert (16, 48) in sizes
