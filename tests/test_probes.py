"""$write/$display through yosys and a full build: format parsing, stripping, the sidecar and its marker blocks."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from litemapy import Schematic

from minecraft_v.placement_engine.probes import PROBE_MARKER, parse_format

ROOT = Path(__file__).resolve().parent.parent
DESIGN = """module main(input clk, input en, input [3:0] a, output reg o);
    wire [3:0] sum = a + 4'd3;
    always @(posedge clk) begin
        o <= a[0];
        if (en) $write("%c", {4'b0100, sum});
        $display("a=%0d {x} %h", a, sum);
    end
endmodule
"""


def test_parse_format():
    assert parse_format("{8:> c}") == [
        {"bits": list(range(8)), "radix": "c", "signed": False, "width": 0, "pad": " ", "justify": ">"}]
    tokens = parse_format("r{4:=0d-u}{{}}{8:>02h-s}\n")
    assert [t.get("text") for t in tokens] == ["r", None, "{}", None, "\n"]
    assert tokens[3]["bits"] == list(range(4, 12)) and tokens[3]["width"] == 2 and tokens[3]["signed"]
    assert [t["bits"] for t in parse_format("{16:> c}")] == [list(range(8, 16)), list(range(8))]
    with pytest.raises(ValueError):
        parse_format("{0:> t}")


def build(tmp_path, *flags):
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src/main.v").write_text(DESIGN)
    out = tmp_path / "out.litematic"
    subprocess.run([ROOT / "synth.sh", tmp_path / "src", "--artifacts-dir", tmp_path / "art", "--no-settle",
                    "--out-litematic", out, *flags], cwd=ROOT, check=True)
    return out


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_prints_stripped_by_default(tmp_path):
    out = build(tmp_path)
    assert out.exists() and not out.with_suffix(".probes.json").exists()
    assert "$print" not in (tmp_path / "art/netlist.json").read_text()


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_probes_tap_marked_blocks(tmp_path):
    out = build(tmp_path, "--debug-prints")
    sidecar = json.loads(out.with_suffix(".probes.json").read_text())
    assert sidecar["version"] == 1
    write, display = sidecar["probes"]
    assert write["format"][0]["radix"] == "c" and "pos" in write["enable"]
    assert [t["text"] for t in display["format"] if "text" in t][0] == "a="
    assert len(write["data"]) == 8 and {"const": 1} in write["data"] and {"const": 0} in write["data"]

    region = Schematic.load(str(out)).regions["main"]
    assert write["trigger"]["edge"] == "rising"
    assert any(region[x, y, z] == PROBE_MARKER for x in region.range_x() for y in region.range_y()
               for z in region.range_z())
    for probe in (write, display):
        for entry in (probe["trigger"], probe["enable"], *probe["data"]):
            if "pos" in entry:
                assert "repeater" in region[tuple(entry["pos"])].id or "torch" in region[tuple(entry["pos"])].id
