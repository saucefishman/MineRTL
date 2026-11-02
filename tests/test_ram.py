import json
import shutil
import subprocess
from pathlib import Path

import pytest

from minecraft_v.placement_engine.ir import Direction
from minecraft_v.placement_engine.ram import build_ram, build_window

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
    assert len(pins) == 3 * 4 + 3 * 2
    assert pins["lock[3]"] == pins["nrb[0]"] == pins["wd[1]"] == Direction.IN
    assert pins["ra[1]"] == pins["rb[0]"] == Direction.OUT


def test_blocks_fit_the_footprint():
    macro = build_ram("ram", 2, 2)
    size = macro.footprint.width, macro.footprint.height, macro.footprint.depth
    assert all(0 <= p[a] < size[a] for p, _ in macro.blocks for a in range(3))
