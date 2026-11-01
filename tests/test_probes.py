"""$write/$display through yosys and a full build: format parsing, stripping, the sidecar and its marker blocks."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from litemapy import Schematic

from minecraft_v.placement_engine.probes import PROBE_MARKER, parse_format

ROOT = Path(__file__).resolve().parent.parent
QUIET = "module main(input clk, input a, output reg o);\n    always @(posedge clk) o <= a;\nendmodule\n"
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


def build(tmp_path, *flags, design=DESIGN, env=None):
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src/main.v").write_text(design)
    out = tmp_path / "out.litematic"
    subprocess.run([ROOT / "synth.sh", tmp_path / "src", "--artifacts-dir", tmp_path / "art", "--no-settle",
                    "--out-litematic", out, "--no-server-clock", *flags], cwd=ROOT, check=True, env=env)
    return out


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_prints_stripped_when_off(tmp_path):
    (tmp_path / "out.server.json").write_text("{}")
    out = build(tmp_path, "--no-debug-prints")
    assert out.exists() and not out.with_suffix(".server.json").exists()
    assert "$print" not in (tmp_path / "art/netlist.json").read_text()


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_probes_tap_marked_blocks(tmp_path):
    out = build(tmp_path)
    sidecar = json.loads(out.with_suffix(".server.json").read_text())
    assert sidecar["version"] == 2
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


HEADLESS = Path(os.environ.get("MCHPRS_HEADLESS", Path.home() / "Projects/MCHPRS/target/release/mchprs_headless"))
# Word 2 is a flip-flop in the macro; `read` comes through the read mux. Neither has a wire to anything but the print.
IN_MACRO = """module main(input clk, output reg [2:0] a);
    initial a = 0;
    reg [{d}:0] n = 16'h1234;
    reg [{d}:0] data[0:7];
    wire [{d}:0] read = data[a];
    always @(posedge clk) begin
        a <= a + 3'd1;
        n <= {{n[{d}-1:0], n[{d}] ^ n[3]}};
        data[a] <= n;
        $display("%0d %0d %0d", a, data[2], read);
    end
endmodule
"""


@pytest.mark.skipif(shutil.which("yosys") is None or not HEADLESS.is_file(), reason="needs yosys and mchprs_headless")
@pytest.mark.parametrize("bits, macro, flags", [(16, "bank_", ()), (4, "array_", ("--macro-min-instances", "2"))])
def test_prints_read_inside_macros(tmp_path, bits, macro, flags):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.v").write_text(IN_MACRO.format(d=bits - 1))
    out = tmp_path / "out.litematic"
    run = subprocess.run([ROOT / "synth.sh", tmp_path / "src", "--artifacts-dir", tmp_path / "art", "--out-litematic", out,
                          "--server-clock", "--macro-cache", tmp_path / "cache", *flags],
                         cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": "0", "MCHPRS_HEADLESS": str(HEADLESS)},
                         capture_output=True, text=True, check=True)
    assert "[verify] 0 violation(s)" in run.stdout, run.stdout[-2000:]

    sidecar = json.loads(out.with_suffix(".server.json").read_text())
    layout = json.loads((tmp_path / "art/component_layout.json").read_text())
    banks = [c for c in layout if c["id"].startswith(macro)]
    assert banks

    def in_bank(pos):  # the only region starts at the origin, so sidecar positions are world positions
        return any(all(c["origin"][i] <= pos[i] < c["origin"][i] + c["footprint"][k] for i, k in
                       enumerate(("width", "height", "depth"))) for c in banks)
    assert any(in_bank(d["pos"]) for d in sidecar["probes"][0]["data"] if "pos" in d)

    (tmp_path / "pins.txt").write_text("")
    (tmp_path / "run.txt").write_text(f"settle {sidecar['clock']['half_period_ticks'] * 2 * 20}\n")
    run = subprocess.run([HEADLESS, "test", out, tmp_path / "pins.txt", tmp_path / "run.txt", "--modes", "redpiler",
                          "--probes", out.with_suffix(".server.json"), "--run-clock"],
                         capture_output=True, text=True, check=True)
    # Each word holds its last write once written; reads and word 2 show the old value on the edge that writes.
    want = []
    words = [0] * 8
    mask = (1 << bits) - 1
    n = 0x1234 & mask
    for edge in range(16):
        a = edge % 8
        want.append(f"{a} {words[2]} {words[a]}")
        words[a] = n
        n = (n << 1) & mask | (n >> (bits - 1) ^ n >> 3) & 1
    assert [line for line in run.stdout.splitlines() if line[:1].isdigit()][:16] == want


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_no_prints_no_sidecar(tmp_path):
    (tmp_path / "out.server.json").write_text("{}")
    out = build(tmp_path, design=QUIET)
    assert out.exists() and not out.with_suffix(".server.json").exists()


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_server_clock_lever(tmp_path):
    out = build(tmp_path, "--server-clock", "50", design=QUIET)
    sidecar = json.loads(out.with_suffix(".server.json").read_text())
    assert sidecar["probes"] == [] and sidecar["clock"]["pin"] == "clk" and sidecar["clock"]["half_period_ticks"] == 50
    schematic = Schematic.load(str(out))
    assert "clock" not in schematic.regions
    lever = schematic.regions["main"][tuple(sidecar["clock"]["pos"])]
    assert lever.id == "minecraft:lever" and lever["powered"] == "false"


def fake_headless(tmp_path, help_text="--run-clock"):
    headless = tmp_path / "headless"
    headless.write_text(f"#!/bin/sh\necho '{help_text}'\n")
    headless.chmod(0o755)
    return headless


def synth_auto(tmp_path, headless, design=QUIET):
    """synth.sh with no clock flag, seeing `headless` as MCHPRS's binary."""
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src/main.v").write_text(design)
    out = tmp_path / "out.litematic"
    subprocess.run([ROOT / "synth.sh", tmp_path / "src", "--artifacts-dir", tmp_path / "art", "--no-settle",
                    "--out-litematic", out], cwd=ROOT, check=True, env={**os.environ, "MCHPRS_HEADLESS": str(headless)})
    return out


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_server_clock_automatic_with_mchprs(tmp_path):
    out = synth_auto(tmp_path, fake_headless(tmp_path))
    assert json.loads(out.with_suffix(".server.json").read_text())["clock"]["half_period_ticks"] > 0
    assert "clock" not in Schematic.load(str(out)).regions


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_generated_clock_without_mchprs(tmp_path):
    out = synth_auto(tmp_path, tmp_path / "missing")
    assert not out.with_suffix(".server.json").exists()
    assert "clock" in Schematic.load(str(out)).regions


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_server_clock_without_clock_pin(tmp_path):
    out = synth_auto(tmp_path, fake_headless(tmp_path), design="module main(input a, output o);\n    assign o = ~a;\nendmodule\n")
    assert out.exists() and not out.with_suffix(".server.json").exists()


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_generated_clock_with_old_mchprs(tmp_path):
    out = synth_auto(tmp_path, fake_headless(tmp_path, "--probes"))
    assert not out.with_suffix(".server.json").exists()
    assert "clock" in Schematic.load(str(out)).regions
