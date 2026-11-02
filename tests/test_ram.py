import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dataclasses import replace

import numpy as np
from litemapy import BlockState, Region

from minecraft_v.placement_engine.block_utils import BulkRegion
from minecraft_v.placement_engine.cosmetics import apply_dust_shapes
from minecraft_v.placement_engine import ram
from minecraft_v.placement_engine.ir import Direction
from minecraft_v.placement_engine.propagate import comparator_entity, settle
from minecraft_v.placement_engine.ram import (StackSpec, build_group, build_ram, build_stack, build_window, group_block,
                                              set_block_cache, settled_block)

ROOT = Path(__file__).parent.parent


def _map(tmp_path: Path, source: str) -> dict:
    (tmp_path / "main.v").write_text(source)
    out = tmp_path / "out.json"
    subprocess.run(["yosys", "-q", "-p", f"read_verilog -sv {tmp_path}/main.v; read_verilog -lib {ROOT}/techmap/ram_cells.v; "
                    f"hierarchy -top main; proc; opt; memory_collect; techmap -map {ROOT}/techmap/ram_2r1w.v t:$mem_v2; "
                    f"write_json {out}"], check=True, capture_output=True)
    return json.loads(out.read_text())["modules"]["main"]["cells"]


_MEMORY = """module main(input clk, input wen, input [1:0] wa, input [1:0] d, input [1:0] a, input [1:0] b,
                    output [1:0] x, output [1:0] y);
    reg [1:0] m[0:3];
    {init}
    assign x = m[a];
    {second_read}
    always @(posedge clk) if (wen) m[wa] <= d;
endmodule"""


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
@pytest.mark.parametrize("init, second_read, tiled", [
    ("", "assign y = m[b];", True),
    ("", "assign y = 0;", False),  # one read port
    ("initial m[0] = 1;", "assign y = m[b];", False),  # preloaded
])
def test_only_the_supported_layout_is_tiled(tmp_path, init, second_read, tiled):
    types = {c["type"] for c in _map(tmp_path, _MEMORY.format(init=init, second_read=second_read)).values()}
    assert ("ram_2r1w" in types) == tiled
    assert ("$mem_v2" in types) != tiled


@pytest.mark.parametrize("macro", [build_ram("ram", 4, 2), build_window()])
def test_pins_are_repeaters_on_their_faces(macro):
    blocks = dict(macro.blocks)
    fp = macro.footprint
    for pin in macro.pins:
        pos = pin.offset[0], pin.offset[1], fp.depth - 1 - pin.offset[2]
        assert blocks[pos].id == "minecraft:repeater"
        assert macro.terminals[f"tile:{pin.name}"] == pos
        assert (pos[2] == fp.depth - 1 if pin.side.value == "south" else
                pos[2] == 0 if pin.side.value == "north" else pos[0] == fp.width - 1)


def test_array_pins():
    pins = {p.name: p.direction for p in build_ram("ram", 4, 2).pins}
    assert len(pins) == 3 * 2 + 1 + 3 * 2
    assert pins["aa[1]"] == pins["ab[0]"] == pins["wa[1]"] == pins["we"] == pins["wd[1]"] == Direction.IN
    assert pins["ra[1]"] == pins["rb[0]"] == Direction.OUT


def test_depth_must_be_a_power_of_two():
    with pytest.raises(ValueError):
        build_ram("ram", 3, 2)


def test_blocks_fit_the_footprint():
    macro = build_ram("ram", 2, 2)
    size = macro.footprint.width, macro.footprint.height, macro.footprint.depth
    assert all(0 <= p[a] < size[a] for p, _ in macro.blocks for a in range(3))


def _spec(words: int = 8) -> StackSpec:
    init = [(i * 5 + 3) % 8 for i in range(words)]
    nets = {f"{port}[{i}]": f"{port}{i}" for port in ("aa", "ab", "ac", "ad", "wa") for i in range(3)}
    nets.update({f"wd[{i}]": f"wd{i}" for i in range(3)})
    nets["we"] = "we"
    return StackSpec(3, 1, 0, 3, 3, init, nets)


def _world(block, cells_of, high: list[str], margin: int = 6) -> BulkRegion:
    """The block in a larger region, with a lever-driven block on the given pins. cells_of: its blocks, as a BlockArrays."""
    fp = block.footprint
    region = BulkRegion(0, 0, 0, fp.width + 2 * margin, fp.height + 2 * margin, fp.depth + 2 * margin)
    region.paste(cells_of.xs + margin, cells_of.ys + margin, cells_of.zs + margin, cells_of.ids, cells_of.palette)
    region.tile_entities.extend(comparator_entity((x + margin, y + margin, z + margin), level) for (x, y, z), level in cells_of.comparators)
    pins = {pin.name: pin for pin in block.pins}
    high = [f"s0_{name}" for name in high]
    for name in high:
        x, y, z = block.terminals[f"tile:{name}"]
        x, y, z = x + margin, y + margin, z + margin
        region[x, y, z + 1] = BlockState("minecraft:stone")
        region[x, y + 1, z + 1] = BlockState("minecraft:lever", face="floor", facing="north", powered="true")
    assert all(pins[name].direction == Direction.IN for name in high)
    return region


def _cells(region: BulkRegion) -> dict:
    blocks, palette = region._Region__blocks, region._Region__palette
    props = {}
    for x, y, z in np.argwhere(blocks != 0).tolist():
        props[x, y, z] = palette[blocks[x, y, z]]
    return props


@pytest.mark.parametrize("high", [
    [],
    ["aa[0]", "ab[2]"],  # another row read
    ["aa[0]", "aa[1]", "aa[2]", "ab[0]", "ac[1]", "ad[2]"],
    ["we"],  # a write held open: row 0 takes wd
    ["we", "wd[0]", "wd[2]", "wa[1]"],
])
def test_settled_block_with_wire_on_it_settles_like_the_whole_thing(high):
    block, _, _ = build_group("g", [_spec(), replace(_spec(), base=0, nets={})])
    whole = _world(block, block.arrays_oriented(False, False), high)
    apply_dust_shapes(whole)
    settle(whole)
    alone = settled_block(block)
    fp = block.footprint
    box = ((6, 6, 6), (6 + fp.width, 6 + fp.height, 6 + fp.depth))
    region = _world(alone, alone.settled, high)
    apply_dust_shapes(region, [box])
    settle(region, [box])
    assert _cells(region) == _cells(whole)
    saved = lambda r: sorted((e.position, int(e.to_nbt()["OutputSignal"])) for e in r.tile_entities)
    assert saved(region) == saved(whole)


def test_block_cache_returns_the_same_block(tmp_path):
    set_block_cache(tmp_path)
    try:
        specs = [_spec(), replace(_spec(), init=[1] * 8)]
        made, own, lanes = group_block(specs)
        again, own_again, lanes_again = group_block([replace(s, nets={p: f"other_{n}" for p, n in s.nets.items()}) for s in specs])
    finally:
        set_block_cache(None)
    assert len(list(tmp_path.glob("*.block"))) == 1  # the group; its banks are in tmp_path/banks
    assert own == own_again and {f"other_{n}": pin for n, pin in lanes.items()} == lanes_again
    assert made.footprint == again.footprint and made.pins == again.pins
    assert made.settled.palette == again.settled.palette and made.settled.comparators == again.settled.comparators
    assert np.array_equal(made.settled.ids, again.settled.ids) and np.array_equal(made.settled.xs, again.settled.xs)


def test_fold_narrows_the_stack_and_keeps_pins_and_bits(monkeypatch):
    spec = _spec()
    plain = build_stack("s", spec.row_bits, spec.banks, spec.base, spec.abits, spec.width, spec.init)
    monkeypatch.setattr(ram, "_FOLD", True)
    folded = build_stack("s", spec.row_bits, spec.banks, spec.base, spec.abits, spec.width, spec.init)
    assert folded.footprint.width == plain.footprint.width - 3 * ram._ADDR_PITCH - ram._LINE_PITCH
    assert {(p.name, p.direction) for p in folded.pins} == {(p.name, p.direction) for p in plain.pins}
    count = lambda m, kind: sum(block.id == kind for _, block in m.blocks)
    assert count(folded, "minecraft:comparator") == count(plain, "minecraft:comparator")


def test_fold_group_with_a_lone_lower_read_pin_under_a_shared_upper_pin(monkeypatch):
    """The layout the guard refuses unfolded: lane towers come nearest, so the lone pin's way out crosses none."""
    shared = {"ac[0]": "n", "wa[0]": "m"}
    first = replace(_spec(), nets={**_spec().nets, **shared, "aa[0]": "alone"})
    second = replace(_spec(), base=0, nets=dict(shared))
    with pytest.raises(ValueError):
        build_group("g", [first, second])
    monkeypatch.setattr(ram, "_FOLD", True)
    block, own, lanes = build_group("g", [first, second])
    assert set(lanes) == {"n", "m"} and "s0_aa[0]" in own
    assert all(0 <= p[a] < s for p, _ in block.blocks for a, s in enumerate(
        (block.footprint.width, block.footprint.height, block.footprint.depth)))


def test_preload_survives_settling_and_comparators_keep_their_signal():
    spec = _spec()
    block = settled_block(build_stack("s", spec.row_bits, spec.banks, spec.base, spec.abits, spec.width, spec.init))
    # Row 0 is read at rest, holding 3 = 011: two bits on each of four read ports.
    assert sorted(level for _, level in block.settled.comparators) == [15] * 8


def test_bulk_region_saves_what_region_saves():
    rng = np.random.default_rng(1)
    for size, kinds in (((5, 3, 4), 3), ((17, 9, 11), 30), ((64, 2, 3), 2)):
        plain, bulk = Region(0, 0, 0, *size), BulkRegion(0, 0, 0, *size)
        palette = [BlockState("minecraft:air")] + [BlockState(f"minecraft:b{i}") for i in range(kinds)]
        for pos, pick in zip(np.ndindex(size), rng.integers(0, kinds + 1, np.prod(size)).tolist()):
            plain[pos] = palette[pick]
            bulk[pos] = palette[pick]
        ours, theirs = bulk.to_nbt(), plain.to_nbt()
        assert np.array_equal(np.asarray(ours["BlockStates"]), np.asarray(theirs["BlockStates"]))
        assert ours.keys() == theirs.keys() and ours["BlockStatePalette"] == theirs["BlockStatePalette"]


@pytest.mark.parametrize("workers", ["1", "2"])
def test_group_settled_bank_by_bank_equals_settled_whole(monkeypatch, workers):
    monkeypatch.setenv("RAM_WORKERS", workers)
    shared = {"aa[0]": "n0", "ab[1]": "n1", "ac[0]": "n2", "we": "n3", "wd[1]": "n4"}
    first = replace(_spec(), nets={**_spec().nets, **shared})
    second = StackSpec(3, 2, 1, 4, 3, [(i * 3 + 1) % 8 for i in range(16)], dict(shared))
    specs = [first, second]
    whole = settled_block(build_group("g", specs)[0])
    made = group_block(specs)[0]
    cells = lambda b: sorted((x, y, z, repr((b.settled.palette[i].id, sorted(dict(b.settled.palette[i].properties()).items()))))
                             for x, y, z, i in zip(b.settled.xs.tolist(), b.settled.ys.tolist(), b.settled.zs.tolist(),
                                                   b.settled.ids.tolist()))
    assert cells(made) == cells(whole)
    assert made.settled.comparators == whole.settled.comparators
