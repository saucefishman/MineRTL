"""The _signals kernel against the Python it stands in for: same results on random redstone, ties and order included."""
from __future__ import annotations

import gzip
import random
from types import SimpleNamespace

import numpy as np
import pytest
from litemapy import BlockState, Region, Schematic

from minecraft_v.placement_engine import block_utils, hold, propagate, verify
from minecraft_v.placement_engine.block_utils import BulkRegion, _pack_states, save_schematic
from minecraft_v.placement_engine.cosmetics import apply_dust_shapes
from minecraft_v.placement_engine.ir import ComponentType
from minecraft_v.placement_engine.propagate import propagate_states, settle
from minecraft_v.placement_engine.verify import HiddenBoxes, verify_signals

pytest.importorskip("minecraft_v.placement_engine._signals")

SIZE = 14
FACINGS = ("north", "south", "west", "east")


def _block(rng: random.Random) -> BlockState:
    roll = rng.random()
    facing = rng.choice(FACINGS)
    if roll < .45:
        return BlockState("minecraft:redstone_wire", power=str(rng.choice((0, 0, 7, 15))))
    if roll < .6:
        return BlockState("minecraft:repeater", facing=facing, delay=str(rng.randint(1, 4)),
                          powered=rng.choice(("true", "false")), locked="false")
    if roll < .66:
        return BlockState("minecraft:comparator", facing=facing, mode=rng.choice(("compare", "subtract")), powered="false")
    if roll < .72:
        return BlockState("minecraft:redstone_torch", lit=rng.choice(("true", "false")))
    if roll < .78:
        return BlockState("minecraft:redstone_wall_torch", facing=facing, lit="true")
    if roll < .82:
        return BlockState("minecraft:redstone_lamp", lit="false")
    if roll < .85:
        return BlockState("minecraft:lever", face=rng.choice(("floor", "wall", "ceiling")), facing=facing,
                          powered=rng.choice(("true", "false")))
    if roll < .87:
        return BlockState("minecraft:redstone_block")
    if roll < .9:
        return BlockState("minecraft:glass")
    if roll < .93:
        return BlockState("minecraft:smooth_stone_slab", type=rng.choice(("top", "bottom", "double")))
    return BlockState("minecraft:stone")


def _world(seed: int, fill: float = .35) -> Region:
    rng = random.Random(seed)
    region = BulkRegion(0, 0, 0, SIZE, SIZE, SIZE)
    for x in range(SIZE):
        for y in range(SIZE):
            for z in range(SIZE):
                if rng.random() < fill:
                    region[x, y, z] = _block(rng)
    return region


def _same_world_twice(seed: int, **kwargs) -> tuple[Region, Region]:
    return _world(seed, **kwargs), _world(seed, **kwargs)


def _named(region: Region) -> list[str]:
    palette, blocks = region._Region__palette, region._Region__blocks
    return [str(palette[i]) for i in blocks.ravel().tolist()]


@pytest.fixture
def python_only(monkeypatch):
    def off():
        for module in (verify, propagate, block_utils):
            monkeypatch.setattr(module, "_signals", None)
    return off


def _redstone(region: Region, rng: random.Random, count: int) -> list[tuple[int, int, int]]:
    cells = [(x, y, z) for x in range(SIZE) for y in range(SIZE) for z in range(SIZE) if region[x, y, z].id != "minecraft:air"]
    rng.shuffle(cells)
    return cells[:count]


@pytest.mark.parametrize("seed", range(30))
def test_verify_signals_same_violations_in_the_same_order(seed, monkeypatch):
    region = _world(seed, fill=.5)
    rng = random.Random(seed)
    cells = _redstone(region, rng, 60)
    nets = {f"n{i}": (cells[i * 4], cells[i * 4 + 1:i * 4 + 4]) for i in range(len(cells) // 4)}
    owner = {c: f"n{rng.randrange(len(nets))}" for c in _redstone(region, rng, 100)}
    blocked = HiddenBoxes(frozenset(_redstone(region, rng, 40)), [((0, 0, 0), (3, 3, 3))])
    fast = verify_signals(region, nets, owner, blocked)
    with monkeypatch.context() as m:
        m.setattr(verify, "_signals", None)
        slow = verify_signals(region, nets, owner, blocked)
    assert fast == slow


@pytest.mark.parametrize("seed", range(30))
def test_dust_shapes_same(seed, python_only):
    fast, slow = _same_world_twice(seed)
    boxes = [((4, 4, 4), (8, 8, 8))] if seed % 2 else []
    apply_dust_shapes(fast, boxes)
    python_only()
    apply_dust_shapes(slow, boxes)
    assert _named(fast) == _named(slow)
    assert fast._Region__palette == slow._Region__palette


@pytest.mark.parametrize("seed", range(60))
def test_settle_same(seed, python_only, monkeypatch):
    monkeypatch.setattr(propagate, "_MAX_STEPS_PER_ITEM", 20)  # random redstone oscillates: give up as soon, both ways
    fast, slow = _same_world_twice(seed, fill=.2 + .5 * (seed % 3) / 2)
    boxes = [((3, 3, 3), (9, 9, 9))] if seed % 2 else []
    lines = propagate_states(fast, boxes)
    settle(fast, boxes)
    python_only()
    assert propagate_states(slow, boxes) == lines
    settle(slow, boxes)
    assert _named(fast) == _named(slow)
    assert fast._Region__palette == slow._Region__palette
    assert sorted((e.position, int(e.to_nbt()["OutputSignal"])) for e in fast.tile_entities) == \
           sorted((e.position, int(e.to_nbt()["OutputSignal"])) for e in slow.tile_entities)


@pytest.mark.parametrize("nbits", [2, 3, 7, 13, 32])
def test_pack_states_same_bits(nbits, python_only):
    rng = np.random.default_rng(nbits)
    blocks = rng.integers(0, 1 << nbits, (17, 9, 5), dtype=np.uint64).astype(np.uint32)
    fast = np.array(_pack_states(blocks, nbits))
    python_only()
    assert np.array_equal(fast, np.array(_pack_states(blocks, nbits)))


def test_saved_file_is_one_gzip_member_of_the_same_nbt(tmp_path, monkeypatch):
    region = _world(1)
    monkeypatch.setattr(block_utils, "_GZIP_CHUNK", 1 << 10)
    schematic = Schematic(name="t", author="t", regions={"main": region})
    schematic.save(str(tmp_path / "plain.litematic"))
    save_schematic(schematic, str(tmp_path / "threaded.litematic"))
    loaded = Schematic.load(str(tmp_path / "threaded.litematic"))
    assert _named(loaded.regions["main"]) == _named(region)
    plain = gzip.open(tmp_path / "plain.litematic").read()
    threaded = gzip.open(tmp_path / "threaded.litematic").read()
    assert len(plain) == len(threaded)
    # Times in the metadata can tick over between the saves, and only those.
    differ = np.flatnonzero(np.frombuffer(plain, np.uint8) != np.frombuffer(threaded, np.uint8))
    assert plain.find(b"TimeCreated") < differ.min() if len(differ) else True
    assert differ.max() < plain.find(b"TotalBlocks") if len(differ) else True


def _timing_world(seed: int):
    """Flip-flops joined by random wire, with a clock net: data often beats its clock. Returns what fix_hold takes."""
    rng = random.Random(seed)
    region = BulkRegion(0, 0, 0, 40, 4, 40)
    for x in range(40):
        for z in range(40):
            region[x, 0, z] = BlockState("minecraft:stone")
    ff = SimpleNamespace(id="", type=ComponentType.DFF)
    components, nets, terminals = [SimpleNamespace(id="clk", type=ComponentType.INPUT_PIN)], [], {}
    wire = BlockState("minecraft:redstone_wire", power="0")
    n = 5
    for i in range(n):
        components.append(SimpleNamespace(id=f"ff{i}", type=ff.type))

    def endpoint(cid: str, pin: str, pos):
        terminals[cid, pin] = pos
        return SimpleNamespace(component_id=cid, pin_name=pin)

    # the clock: one long wire along z = 0, stubs running south to each flip-flop's C pin
    length = 6 + 7 * n
    for x in range(length):
        region[x % 40, 1, 0] = wire
    clock_sinks = []
    for i in range(n):
        cx = 3 + 7 * i
        for z in range(1, 4):
            region[cx, 1, z] = wire
        clock_sinks.append(endpoint(f"ff{i}", "C", (cx, 1, 3)))
    nets.append(SimpleNamespace(net_id="clock", source=endpoint("clk", "Y", (0, 1, 0)), sinks=clock_sinks))
    # data: ff[i].Q to ff[(i + 1) % n].D, straight runs along z = row
    for i in range(n):
        row = 8 + 3 * i
        start = rng.randrange(0, 6)
        stop = start + rng.randrange(6, 30)
        for x in range(start, stop + 1):
            region[x, 1, row] = wire
        nets.append(SimpleNamespace(net_id=f"d{i}", source=endpoint(f"ff{i}", "Q", (start, 1, row)),
                                    sinks=[endpoint(f"ff{(i + 1) % n}", "D", (stop, 1, row))]))
    return region, SimpleNamespace(components=components, nets=nets), terminals


@pytest.mark.parametrize("seed", range(10))
def test_fix_hold_same(seed, monkeypatch, capsys):
    region, comp, terminals = _timing_world(seed)
    fast = hold.fix_hold(region, comp, terminals, set())
    printed = capsys.readouterr().out
    assert "[hold]" in printed
    other, comp2, terminals2 = _timing_world(seed)
    with monkeypatch.context() as m:
        m.setattr(hold, "signal_grid", lambda *a: None)
        slow = hold.fix_hold(other, comp2, terminals2, set())
    assert fast == slow
    assert capsys.readouterr().out == printed
    assert _named(region) == _named(other)
