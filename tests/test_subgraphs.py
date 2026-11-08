"""The repeated subcircuit finder: exact isomorphism and disjoint instances, on hand-made and example netlists."""

from minecraft_v.placement_engine.ir import (
    CURRENT_SCHEMA_VERSION, CardinalDirection, Component, ComponentList, ComponentType, Direction, NetConnection, NetEndpoint, PinRef,
)
from minecraft_v.placement_engine.subgraphs import RepeatedClass, find_repeated

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



def _cores_with_blocks(blocks: list[str]) -> ComponentList:
    """A chain into a CUSTOM block (a ROM, say) and another chain out of it, per block, on a shared select net."""
    comps = [Component(id=n, type=ComponentType.INPUT_PIN, pins=[PinRef(name=n, direction=Direction.OUT)])
             for n in ["s", *(f"i{k}" for k in range(len(blocks)))]]
    nets: dict[str, list] = {}

    def net(name: str, source: tuple[str, str], sink: tuple[str, str]) -> None:
        nets.setdefault(name, [source, []])[1].append(sink)

    for k, block in enumerate(blocks):
        comps.append(Component(id=f"b{k}", type=ComponentType.CUSTOM, metadata={"macro": block},
                               pins=[PinRef(name="D", direction=Direction.IN), PinRef(name="Q", direction=Direction.OUT)]))
        for side in "gh":
            for i in range(9):
                comps.append(Component(id=f"{side}{k}_{i}", type=ComponentType.AND if side == "g" else ComponentType.XOR,
                                       pins=[PinRef(name="A", direction=Direction.IN), PinRef(name="B", direction=Direction.IN),
                                             PinRef(name="Y", direction=Direction.OUT)]))
                net("s", ("s", "s"), (f"{side}{k}_{i}", "B"))
        net(f"i{k}", (f"i{k}", f"i{k}"), (f"g{k}_0", "A"))
        net(f"q{k}", (f"b{k}", "Q"), (f"h{k}_0", "A"))
        for side in "gh":
            for i in range(1, 9):
                net(f"{side}{k}_{i - 1}", (f"{side}{k}_{i - 1}", "Y"), (f"{side}{k}_{i}", "A"))
        net(f"d{k}", ("g%d_8" % k, "Y"), (f"b{k}", "D"))
    return ComponentList(schema_version=CURRENT_SCHEMA_VERSION, components=comps, nets=[
        NetConnection(net_id=n, source=NetEndpoint(component_id=src[0], pin_name=src[1]),
                      sinks=[NetEndpoint(component_id=c, pin_name=p) for c, p in sinks]) for n, (src, sinks) in nets.items()])


def test_blocks_with_different_contents_stay_out_of_instances():
    """Two cores read ROMs that differ only in contents: a macro stamped from one would run its program in both."""
    comp = _cores_with_blocks(["rom_a", "rom_b"])
    classes = find_repeated(comp)
    assert classes and all(len(cls.instances) == 2 for cls in classes)
    custom = {c.id for c in comp.components if c.type == ComponentType.CUSTOM}
    assert not custom & {m[t] for cls in classes for m in cls.instances for t in cls.cells}
    for cls in classes:  # the nets to a block are ports, and each instance keeps its own block's
        ports = {m[n] for m in cls.instances for n, label in cls.nets.items() if label != "internal"}
        assert {"d0", "d1"} <= ports or {"q0", "q1"} <= ports


def test_blocks_with_same_contents_stay_out_of_instances_too():
    comp = _cores_with_blocks(["rom_a", "rom_a", "rom_a"])
    classes = find_repeated(comp)
    custom = {c.id for c in comp.components if c.type == ComponentType.CUSTOM}
    assert classes and not custom & {m[t] for cls in classes for m in cls.instances for t in cls.cells}


def test_macros_leave_each_cores_block_in_the_top_netlist():
    """extract_macros on cores with different ROMs: every ROM stays a top-level component, wired to its own instance."""
    from minecraft_v.placement_engine.ir import Footprint
    from minecraft_v.placement_engine.macro import Macro, extract_macros

    def build(name, sub, ports, cells):
        io = [c for c in sub.components if c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)]
        pins = [PinRef(name=c.id, direction=Direction.IN if c.type == ComponentType.INPUT_PIN else Direction.OUT,
                       side=CardinalDirection.SOUTH if c.type == ComponentType.INPUT_PIN else CardinalDirection.NORTH,
                       offset=(k, k, 0)) for k, c in enumerate(io)]
        assert not any(c.type == ComponentType.CUSTOM for c in sub.components)  # a tile holds no block
        return Macro(name=name, footprint=Footprint(width=4, height=2, depth=4), pins=pins, blocks=[], arcs={}, q_out={},
                     setup={}, cells=cells,
                     nets={n.net_id: (f"{n.source.component_id}:{n.source.pin_name}",
                                      [f"{s.component_id}:{s.pin_name}" for s in n.sinks]) for n in sub.nets},
                     port_net={port: net for net, port in ports.items()})

    comp = _cores_with_blocks(["rom_a", "rom_b"])
    top, macros, instances = extract_macros(comp, build, 2, 8, row_tiles=1, bank_columns=0, min_instances=2)
    assert macros and instances
    blocks = {c.id: c.metadata["macro"] for c in top.components if c.type == ComponentType.CUSTOM and c.id.startswith("b")}
    assert blocks == {"b0": "rom_a", "b1": "rom_b"}
    ends = {net.net_id: {net.source.component_id, *(s.component_id for s in net.sinks)} for net in top.nets}
    macro_ids = {i.component_id for i in instances}
    assert all(ends[net] & macro_ids for net in ("d0", "q0", "d1", "q1"))  # blocks connect to the macro instances
