"""The compiled per-net checks say what verify's say, on random scenes full of latches and contacts."""
import random

import pytest
from litemapy import BlockState, Region

from minecraft_v.placement_engine.builder import _GreedyRouter
from minecraft_v.placement_engine.ripup import JournaledDict, JournaledSet, Recorder
from minecraft_v.placement_engine.verify import Either, Excluding, find_gate_contacts, find_inverted_contacts, find_latch

SIZE = 9
FACINGS = ("north", "south", "east", "west")
FAULT_LATCH, FAULT_INVERTED, FAULT_GATE = 1, 2, 4


def random_block(rng: random.Random) -> BlockState:
    roll = rng.random()
    if roll < 0.40:
        return BlockState("minecraft:redstone_wire")
    if roll < 0.55:
        return BlockState("minecraft:repeater", facing=rng.choice(FACINGS))
    if roll < 0.65:
        return BlockState("minecraft:redstone_wall_torch", facing=rng.choice(FACINGS))
    if roll < 0.72:
        return BlockState("minecraft:redstone_torch")
    if roll < 0.90:
        return BlockState("minecraft:stone")
    if roll < 0.95:
        return BlockState("minecraft:glass")
    return BlockState("minecraft:redstone_block")


def scene(seed: int):
    rng = random.Random(seed)
    region = Region(0, 0, 0, SIZE, SIZE, SIZE)
    cells = [(x, y, z) for x in range(SIZE) for y in range(SIZE) for z in range(SIZE)]
    filled = rng.sample(cells, int(len(cells) * rng.uniform(0.25, 0.7)))
    for pos in filled:
        region[pos] = random_block(rng)
    terminals = frozenset(rng.sample(cells, 12))
    footprint = frozenset(rng.sample(cells, rng.randrange(0, 150)))
    gates = frozenset(rng.sample(cells, rng.randrange(0, 200)))
    rec = Recorder()
    router = _GreedyRouter(region, rec, JournaledSet(rec, name="solid"), JournaledDict(rec),
                           JournaledSet(rec, name="inverted"), footprint, SIZE, terminals, gate_cells=gates)
    assert router.net_checks is not None
    source = rng.choice(filled)
    sinks = rng.sample(sorted(terminals), 3)
    own = rng.sample(filled, min(len(filled), 60))
    journal = [("block", None, pos, None, None) for pos in own] + [("set", None, own[0], False, None)]
    return region, router, terminals, footprint, gates, source, sinks, frozenset(own), journal


@pytest.mark.parametrize("seed", range(300))
def test_same_answers_as_verify(seed):
    region, router, terminals, footprint, gates, source, sinks, own, journal = scene(seed)
    through = {source, *sinks}
    faults = router.net_checks.faults(source, sinks, through, journal)
    stop = Excluding(terminals, through)
    assert bool(faults & FAULT_LATCH) == bool(find_latch(region, source, stop))
    assert bool(faults & FAULT_INVERTED) == bool(
        find_inverted_contacts(region, source, sinks, stop, footprint, Either(terminals, own)))
    assert bool(faults & FAULT_GATE) == bool(find_gate_contacts(region, own, gates))


def test_scenes_have_every_fault():
    """Otherwise the comparison above says little."""
    seen = 0
    for seed in range(300):
        _, router, _, _, _, source, sinks, _, journal = scene(seed)
        seen |= router.net_checks.faults(source, sinks, {source, *sinks}, journal)
    assert seen == FAULT_LATCH | FAULT_INVERTED | FAULT_GATE


@pytest.mark.parametrize("seed", range(100))
def test_signal_returns_same_as_python(seed):
    """NetChecks.signal_returns is wire._signal_returns compiled, over the grid."""
    from minecraft_v.placement_engine.wire import _signal_returns
    region, router, *_ = scene(seed)
    rng = random.Random(seed + 1000)
    cells = [(x, y, z) for x in range(SIZE) for y in range(SIZE) for z in range(SIZE)]
    for pos in cells:
        if rng.random() < 0.7:
            router.dust_owner[pos] = rng.choice("aab")
    for pos in rng.sample(cells, 40):
        router.dust_owner.pop(pos, None)
    for _ in range(40):
        start = rng.choice(cells)
        target, cut = ((min(SIZE - 1, max(0, c + rng.randint(-2, 2))) for c in start) for _ in range(2))
        target, cut = tuple(target), tuple(cut)
        net = rng.choice("ab")
        assert (router.net_checks.signal_returns(net, start, target, cut)
                == _signal_returns(router.indexed, router.dust_owner, net, start, target, cut))
