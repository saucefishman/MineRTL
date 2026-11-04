import shutil
import sys
from pathlib import Path

import pytest
from litemapy import Schematic

from minecraft_v.cell_library import SCHEMATICS_DIR
from minecraft_v.main import module_to_component_list
from minecraft_v.models import Cell, Module, Port
from minecraft_v.placement_engine import build_litematic_from_component_list
from minecraft_v.placement_engine.ir import CardinalDirection, ComponentType
from minecraft_v.placement_engine.template import _io_repeater_facing



def _make_module(ports=None, cells=None):
    return Module(ports=ports or {}, cells=cells or {})


def _make_cell(type_, port_directions, connections, parameters=None):
    return Cell(
        hide_name=1,
        type=type_,
        parameters=parameters or {},
        port_directions=port_directions,
        connections=connections,
    )


def _single_gate_module(type_: str, *, has_b: bool = True, has_s: bool = False) -> Module:
    ports = {
        "a": Port(direction="input", bits=[2]),
        "y": Port(direction="output", bits=[10]),
    }
    conns = {"A": [2], "Y": [10]}
    pds = {"A": "input", "Y": "output"}
    if has_b:
        ports["b"] = Port(direction="input", bits=[3])
        conns["B"] = [3]
        pds["B"] = "input"
    if has_s:
        ports["s"] = Port(direction="input", bits=[4])
        conns["S"] = [4]
        pds["S"] = "input"
    return _make_module(
        ports=ports,
        cells={"g": _make_cell(type_, pds, conns)},
    )


def _inverter_module() -> Module:
    return _make_module(
        ports={
            "a_in": Port(direction="input", bits=[2]),
            "y_out": Port(direction="output", bits=[3]),
        },
        cells={
            "inv": _make_cell(
                "$_NOT_",
                {"A": "input", "Y": "output"},
                {"A": [2], "Y": [3]},
            )
        },
    )


def _build(module: Module, out: Path, capsys, **kwargs):
    """Build with routing failures raising; return the loaded main region.

    Also fails on anything the post-route checks report: [verify] opens / shorts /
    unsupported blocks, and ignored pin-target failures.
    """
    build_litematic_from_component_list(
        module_to_component_list(module),
        schematics_dir=SCHEMATICS_DIR,
        out_path=out,
        schematic_name=out.stem,
        allow_routing_failures=False,
        **kwargs,
    )
    printed = capsys.readouterr().out
    problems = [line for line in printed.splitlines() if line.startswith(("[verify]", "IGNORING"))]
    assert not problems, "\n".join(problems)
    assert out.is_file()
    return Schematic.load(str(out)).regions["main"]


def test_io_repeater_facing_rules():
    # facing = input direction: an input pin's repeater reads from the pin (its side),
    # an output pin's reads from the wire arriving on the far side.
    assert _io_repeater_facing(ComponentType.INPUT_PIN, CardinalDirection.SOUTH) == "south"
    assert _io_repeater_facing(ComponentType.INPUT_PIN, CardinalDirection.NORTH) == "north"
    assert _io_repeater_facing(ComponentType.OUTPUT_PIN, CardinalDirection.NORTH) == "south"
    assert _io_repeater_facing(ComponentType.OUTPUT_PIN, CardinalDirection.SOUTH) == "north"


def test_build_litematic_from_netlist_smoke(tmp_path: Path, capsys):
    region = _build(_inverter_module(), tmp_path / "test_inv.litematic", capsys)
    blocks = [str(region[pos]) for pos in region.block_positions()]
    assert any(b != "minecraft:air" for b in blocks)
    assert any("repeater" in b for b in blocks)


def test_output_pin_target_layer_wiring(tmp_path: Path, capsys):
    target = (2, 5)
    region = _build(_inverter_module(), tmp_path / "test_inv_targets.litematic", capsys,
                    output_pin_targets={"y_out": target})
    # Targets are x-offset to the middle of the output layer (see _route_output_pin_extensions).
    tx, ty = target
    etx = tx + (region.width - 1) // 2
    assert "redstone_wire" in str(region[etx, ty, 0])


def test_output_pin_target_lamp_tile(tmp_path: Path, capsys):
    target = (2, 5, 3)
    region = _build(_inverter_module(), tmp_path / "test_inv_tile.litematic", capsys,
                    output_pin_targets={"y_out": target})
    etx = target[0] + (region.width - 1) // 2
    lamps = {(etx + i, target[1] + j, 0) for i in range(3) for j in range(3)}
    assert all(region[p].id == "minecraft:redstone_lamp" for p in lamps)
    assert region[etx + 1, target[1], 6].id == "minecraft:repeater"  # the tile's input
    assert "redstone_wire" in str(region[etx + 1, target[1], 7])  # wire up to it


def test_lamp_tiles_touch_without_lighting_each_other(tmp_path: Path):
    """Tiles side by side and stacked, each from its own lever: every lamp follows its own tile's input, inverted."""
    import itertools
    import subprocess
    from litemapy import Region as LmRegion
    from minecraft_v.placement_engine.builder import _lamp_tile
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from settle_schematic import headless_binary
    binary = headless_binary()
    if not Path(binary).is_file() and not shutil.which(binary):
        pytest.skip("needs mchprs_headless")
    region = LmRegion(0, 0, 0, 11, 12, 10)
    pins, tiles = [], [f"{c}{r}" for c in range(3) for r in range(3)]
    for c, r in itertools.product(range(3), range(3)):
        blocks, (ix, iy, iz) = _lamp_tile(1 + 3 * c, 1 + 3 * r, 0)
        for pos, block in blocks.items():
            assert region[pos].id == "minecraft:air"
            region[pos] = block
        pins.append(f"in{c}{r} {ix} {iy} {iz - 1} 0 1")
        pins += [f"l{c}{r}_{i}{j} {1 + 3 * c + i} {1 + 3 * r + j} 0 0 -1" for i in range(3) for j in range(3)]
    Schematic(name="tiles", regions={"t": region}).save(str(tmp_path / "tiles.litematic"))
    (tmp_path / "pins.txt").write_text("\n".join(pins) + "\n")
    lines = []
    for state in [{t: 0 for t in tiles}] + [{t: int(t == one) for t in tiles} for one in tiles] + \
                 [{t: int(t != one) for t in tiles} for one in tiles]:
        lines += [f"set in{t} {v}" for t, v in state.items()] + ["settle 20"]
        lines += [f"expect l{t}_{i}{j} {1 - v}" for t, v in state.items() for i in range(3) for j in range(3)]
    (tmp_path / "script.txt").write_text("\n".join(lines) + "\n")
    result = subprocess.run([binary, "test", tmp_path / "tiles.litematic", tmp_path / "pins.txt", tmp_path / "script.txt",
                             "--modes", "noupdate,redstone,redpiler", *(a for t in tiles for a in ("--add-lever", f"in{t}"))],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:]


@pytest.mark.parametrize(
    "gate_type, kwargs",
    [
        ("$_AND_", {"has_b": True}),
        ("$_OR_", {"has_b": True}),
        ("$_XOR_", {"has_b": True}),
        ("$_NOT_", {"has_b": False}),
        pytest.param("$_MUX_", {"has_b": True, "has_s": True}, marks=pytest.mark.xfail(
            raises=ValueError, strict=False,
            reason="known layout bug: IO repeater cell lands outside the workspace; "
                   "not strict because it depends on the placement")),
    ],
)
def test_single_gate_routes_clean(tmp_path: Path, capsys, gate_type, kwargs):
    _build(_single_gate_module(gate_type, **kwargs), tmp_path / "gate.litematic", capsys)


@pytest.mark.parametrize("gate, cells", [
    ("DFF", [(0, 0, 1), (0, 0, 0)]),
    ("DFFE", [(0, 0, 2), (0, 0, 0)]),
])
def test_init_high_flip_flop_pastes_powered(gate, cells):
    from litemapy import Region
    from minecraft_v.cell_library import INIT_HIGH_STATES, SCHEMATIC_MAP
    from minecraft_v.placement_engine.template import _load_template_region, _paste_template

    kind = ComponentType[gate]
    template = _load_template_region(SCHEMATICS_DIR, SCHEMATIC_MAP[kind].file_prefix)
    for block_states, want in ((None, "false"), (INIT_HIGH_STATES[kind], "true")):
        region = Region(0, 0, 0, 12, 3, 8)
        _paste_template(region, template, (1, 1, 1), set(), block_states=block_states)
        for x, y, z in cells:
            assert region[1 + x, 1 + y, 1 + z]["powered"] == want


def test_dffe_template_fits_its_footprint_and_keeps_the_clock_dust():
    from minecraft_v.cell_library import SCHEMATIC_MAP
    from minecraft_v.placement_engine.template import _load_template_region, _non_air_bounds

    info = SCHEMATIC_MAP[ComponentType.DFFE]
    template = _load_template_region(SCHEMATICS_DIR, info.file_prefix)
    x0, y0, z0, x1, y1, z1 = _non_air_bounds(template)
    assert (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1) == (info.footprint.width, info.footprint.height, info.footprint.depth)
    # Dust on the quartz is what carries the clock into the gate past the lock tap.
    assert template[x0 + 5, y0 + 1, z0 + 3].id == "minecraft:redstone_wire"


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_synth_works_without_mchprs_headless(tmp_path: Path):
    import os
    import subprocess

    out = tmp_path / "result.litematic"
    run = subprocess.run(
        ["./synth.sh", "examples/counter", "--artifacts-dir", str(tmp_path / "artifacts"),
         "--out-litematic", str(out)],
        cwd=Path(__file__).resolve().parent.parent,
        env={**os.environ, "MCHPRS_HEADLESS": str(tmp_path / "missing")}, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert out.exists()


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_synth_runs_on_a_project_path_with_spaces(tmp_path: Path):
    import subprocess

    root = Path(__file__).resolve().parent.parent
    project = tmp_path / "my project"
    project.mkdir()
    (project / "main.v").write_text((root / "examples/counter/counter.v").read_text())
    run = subprocess.run([root / "synth.sh", project, "--synth-only", "--artifacts-dir", tmp_path / "art dir"],
                         cwd=root, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert (tmp_path / "art dir" / "netlist.json").exists()
