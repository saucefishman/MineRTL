"""Whole builds through the macro flow: yosys netlists of small memories, routed and verified."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from litemapy import Region

from minecraft_v.placement_engine.builder import _GreedyRouter, _wall_vias
from minecraft_v.placement_engine.ir import Component, ComponentType, Footprint, NetConnection, NetEndpoint
from minecraft_v.placement_engine.layout import _Placed
from minecraft_v.placement_engine.macro import _lane_plane, _row_gap, _row_shift

ROOT = Path(__file__).resolve().parent.parent
MEMORY = """module main(input clk,
    input [{a}:0]raddr0, output [{d}:0]rdata0,
    input [{a}:0]raddr1, output [{d}:0]rdata1,
    input wen, input [{a}:0]waddr, input [{d}:0]wdata);
    reg [{d}:0]data[0:{w}];
    assign rdata0 = data[raddr0];
    assign rdata1 = data[raddr1];
    always @(posedge clk) begin
        if (wen) begin
            data[waddr] <= wdata;
        end
    end
endmodule
"""


def _build(tmp: Path, project: Path, *flags: str) -> tuple[str, list[dict], Path]:
    out = tmp / "result.litematic"
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    run = subprocess.run(
        ["./synth.sh", str(project), "--artifacts-dir", str(tmp / "artifacts"), "--route-passes", "1",
         "--allow-routing-failures", "True", "--out-litematic", str(out), "--macros", "--macro-cache", str(tmp / "cache"), *flags],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    log = run.stdout + run.stderr
    assert run.returncode == 0, log[-2000:]
    return log, json.loads((tmp / "artifacts" / "component_layout.json").read_text()), out


def _memory(tmp: Path, words: int, bits: int) -> Path:
    project = tmp / "project"
    project.mkdir()
    a = len(bin(words - 1)) - 3
    (project / "main.v").write_text(MEMORY.format(a=a, d=bits - 1, w=words - 1))
    return project


def _assert_clean(log: str) -> None:
    assert re.search(r"\[wire\] done \(\d+ nets, 0 failed\)", log), log[-2000:]
    assert "skipped net" not in log
    assert "[verify] open" not in log and "[verify] short" not in log


def _overlap(a: dict, b: dict) -> bool:
    keys = ("width", "height", "depth")
    return all(a["origin"][i] < b["origin"][i] + b["footprint"][keys[i]]
               and b["origin"][i] < a["origin"][i] + a["footprint"][keys[i]] for i in range(3))


needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not installed")


@needs_yosys
def test_bank_build_is_clean_and_gates_stay_out_of_the_stack(tmp_path):
    log, layout, _ = _build(tmp_path, _memory(tmp_path, 8, 16))
    _assert_clean(log)
    assert re.search(r"\[verify\] 0 violation\(s\) in \d+ flattened nets", log), log[-2000:]
    banks = [c for c in layout if c["id"].startswith("bank_")]
    assert len(banks) >= 2
    assert len({(c["origin"][0], c["origin"][2]) for c in banks}) == 1  # stacked in Y
    gates = [c for c in layout if c["type"] not in ("INPUT_PIN", "OUTPUT_PIN") and c not in banks]
    assert gates
    assert not any(_overlap(g, b) for g in gates for b in banks)
    assert not any(_overlap(a, b) for i, a in enumerate(banks) for b in banks[i + 1:])


@needs_yosys
def test_array_build_with_too_few_instances_for_a_bank_is_clean(tmp_path):
    log, layout, _ = _build(tmp_path, _memory(tmp_path, 8, 4), "--macro-min-instances", "2")
    _assert_clean(log)
    assert "[verify] 0 violation(s)" in log
    assert not [c for c in layout if c["id"].startswith("bank_")]
    assert [c for c in layout if c["id"].startswith("array_")]


@needs_yosys
def test_too_few_repeats_for_a_bank_stay_gates_by_default(tmp_path):
    log, layout, _ = _build(tmp_path, _memory(tmp_path, 8, 4))
    _assert_clean(log)
    assert "[macro] 0 instances" in log
    assert not [c for c in layout if c["type"] == "CUSTOM"]


@needs_yosys
@pytest.mark.parametrize("example", ["counter", "regs"])
def test_design_without_repeats_builds_the_same_with_macros_on(tmp_path, example):
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    on, layout_on, _ = _build(tmp_path / "on", ROOT / "examples" / example)
    off, layout_off, _ = _build(tmp_path / "off", ROOT / "examples" / example, "--no-macros")
    assert "[macro] 0 instances" in on
    _assert_clean(on)
    _assert_clean(off)
    assert layout_on == layout_off


@needs_yosys
def test_no_macros_builds_without_mining(tmp_path):
    log, layout, _ = _build(tmp_path, _memory(tmp_path, 8, 4), "--no-macros")
    assert "[macro]" not in log
    assert not [c for c in layout if c["type"] == "CUSTOM"]


@pytest.mark.parametrize("per_y", [1, 2, 3, 5])
def test_lane_planes_leave_room_for_bridges(per_y):
    near = [_lane_plane(per_y, 0, k) for k in range(per_y)]
    far = [_lane_plane(per_y, 1, k) for k in range(per_y)]
    assert near[0] >= 3  # a landing and the spacer before the row
    assert all(b - a == 2 for a, b in zip(near, near[1:]))
    assert far[0] - near[-1] == 3  # a bridge lands between the sets, clear of the other's floors
    assert _row_gap(per_y) - far[-1] >= 4


def test_row_shift_keeps_the_rows_taps_apart():
    shift = _row_shift({3, 20, 40}, 51, 54, 2)
    upright = {k * 54 + x for k in range(2) for x in (3, 20, 40)}
    turned = {k * 54 + shift + 50 - x for k in range(2) for x in (3, 20, 40)}
    assert min(abs(a - b) for a in upright for b in turned) >= 3


@pytest.mark.parametrize("src_z, sink_z, near_far_edge", [(50, 250, lambda z: z >= 184), (250, 50, lambda z: z <= 115)])
def test_wall_via_sits_by_the_far_edge(src_z, sink_z, near_far_edge):
    wall = _Placed(Component(id="bank", type=ComponentType.CUSTOM, pins=[], footprint=Footprint(width=100, height=10, depth=100)),
                   (0, 0, 100))
    pins = {("g", "Y"): (50, 20, src_z), ("a", "A"): (50, 5, sink_z)}
    net = NetConnection(net_id="n", source=NetEndpoint(component_id="g", pin_name="Y"),
                        sinks=[NetEndpoint(component_id="a", pin_name="A")])
    (_, y, z), = _wall_vias(net, pins, [wall])
    assert y > 10
    assert near_far_edge(z)


def test_waypoint_stays_out_of_other_nets_keepouts():
    """A search can never enter such a cell, so a waypoint there would fail the whole net."""
    router = SimpleNamespace(region=Region(0, 0, 0, 40, 40, 40), footprint_blocked=set(), dust_owner={})
    router._via_open = lambda via, job: _GreedyRouter._via_open(router, via, job)
    job = SimpleNamespace(net_id="n", protected={(20, 20, 20)})
    assert _GreedyRouter._open_via(router, (20, 20, 20), job) not in (None, (20, 20, 20))


def test_macros_are_on_by_default(tmp_path):
    out = tmp_path / "result.litematic"
    run = subprocess.run(
        ["./synth.sh", str(_memory(tmp_path, 8, 16)), "--artifacts-dir", str(tmp_path / "artifacts"), "--route-passes", "1",
         "--allow-routing-failures", "True", "--out-litematic", str(out)],
        cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": "0"}, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert "[macro]" in run.stdout
