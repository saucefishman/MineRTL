"""Check the tiled RAM in mchprs_headless on all modes.

  check_ram.py array <depth> <width> <out_dir>           the bare array: every word written twice, read on both ports
  check_ram.py build <netlist.json> <out_dir> <seed>     examples/ramtile built, random traffic against a model of the RTL
"""
import os
import random
import subprocess
import sys
from pathlib import Path

from litemapy import Region, Schematic

from minecraft_v.placement_engine.ir import Direction
from minecraft_v.placement_engine.ram import build_ram
from settle_schematic import headless_binary

ROOT = Path(__file__).parent.parent
MODES = "noupdate,redstone,redpiler"
ARRAY_SETTLE = 80
HALF_PERIOD = 40
DEPTH, WIDTH, ABITS = 4, 2, 2
NORMAL = {"east": (1, 0), "north": (0, -1), "south": (0, 1)}


def array_script(depth: int, width: int, seed: int) -> str:
    rng = random.Random(seed)
    mask = (1 << width) - 1
    words = [rng.randrange(mask + 1) for _ in range(depth)]
    lines = [f"set {line}[{k}] 1" for k in range(depth) for line in ("lock", "nra", "nrb")] + [f"settle {ARRAY_SETTLE}"]
    for k, word in [(k, ~w & mask) for k, w in enumerate(words)] + list(enumerate(words)):
        lines += [f"set wd[{i}] {word >> i & 1}" for i in range(width)]
        lines += [f"settle {ARRAY_SETTLE}", f"set lock[{k}] 0", f"settle {ARRAY_SETTLE}", f"set lock[{k}] 1", f"settle {ARRAY_SETTLE}"]
    lines += [f"set wd[{i}] 0" for i in range(width)] + [f"settle {ARRAY_SETTLE}"]
    for a in range(depth):
        b = depth - 1 - a
        lines += [f"set nra[{a}] 0", f"set nrb[{b}] 0", f"settle {ARRAY_SETTLE}"]
        lines += [f"expect ra[{i}] {words[a] >> i & 1}" for i in range(width)]
        lines += [f"expect rb[{i}] {words[b] >> i & 1}" for i in range(width)]
        lines += [f"set nra[{a}] 1", f"set nrb[{b}] 1", f"settle {ARRAY_SETTLE}"]
        lines += [f"expect {port}[{i}] 0" for port in ("ra", "rb") for i in range(width)]
    return "\n".join(lines) + "\n"


def check_array(depth: int, width: int, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    ram = build_ram("ram", depth, width)
    fp = ram.footprint
    region = Region(0, 0, 0, fp.width, fp.height, fp.depth)
    for pos, block in ram.blocks:
        region[pos] = block
    Schematic(name="ram", author="MineRTL", regions={"ram": region}).save(str(out / "ram.litematic"))
    with open(out / "ram.pins", "w") as f:
        for pin in ram.pins:
            nx, nz = NORMAL[pin.side.value]
            f.write(f"{pin.name} {pin.offset[0]} {pin.offset[1]} {fp.depth - 1 - pin.offset[2]} {nx} {nz}\n")
    (out / "ram.script").write_text(array_script(depth, width, depth * 100 + width))
    print(f"{fp.width}x{fp.height}x{fp.depth}, {fp.width * fp.height * fp.depth / depth / width:.0f} volume per bit, "
          f"{len(ram.blocks) / depth / width:.0f} blocks per bit")
    levers = [arg for pin in ram.pins if pin.direction == Direction.IN for arg in ("--add-lever", pin.name)]
    return subprocess.run([headless_binary(), "test", out / "ram.litematic", out / "ram.pins", out / "ram.script",
                           "--modes", MODES, *levers]).returncode


def build_script(seed: int, steps: int = 60) -> str:
    rng = random.Random(seed)
    mem = [0] * DEPTH
    lines = ["set clk 0", "set wen 0"]

    def drive(name: str, width: int, value: int) -> None:
        lines.extend(f"set {name}[{i}] {value >> i & 1}" for i in range(width))

    def expect_reads(a: int, b: int) -> None:
        lines.extend(f"expect rdata{port}[{i}] {mem[addr] >> i & 1}" for port, addr in ((0, a), (1, b)) for i in range(WIDTH))

    lines.append(f"settle {HALF_PERIOD}")
    for addr in range(DEPTH):  # no reset, so contents are defined by writing them
        mem[addr] = rng.randrange(1 << WIDTH)
        drive("waddr", ABITS, addr)
        drive("wdata", WIDTH, mem[addr])
        lines += ["set wen 1", f"settle {HALF_PERIOD}", f"pulse clk {HALF_PERIOD}", "set wen 0", f"settle {HALF_PERIOD}"]
    for _ in range(steps):
        a, b, addr, value = (rng.randrange(DEPTH), rng.randrange(DEPTH), rng.randrange(DEPTH), rng.randrange(1 << WIDTH))
        wen = rng.random() < 0.6
        drive("raddr0", ABITS, a)
        drive("raddr1", ABITS, b)
        drive("waddr", ABITS, addr)
        drive("wdata", WIDTH, value)
        lines += [f"set wen {int(wen)}", f"settle {HALF_PERIOD}"]
        expect_reads(a, b)
        lines += [f"pulse clk {HALF_PERIOD}", f"settle {HALF_PERIOD}"]
        if wen:
            mem[addr] = value
        expect_reads(a, b)
    return "\n".join(lines) + "\n"


def check_build(netlist: str, out: Path, seed: str, *args: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    litematic, pins = out / f"s{seed}.litematic", out / f"s{seed}.pins"
    (out / f"s{seed}_artifacts").mkdir(exist_ok=True)
    subprocess.run([sys.executable, ROOT / "scripts/build_with_pins.py", pins, "--netlist", netlist, "--route-passes", "1",
                    "--generate-clock", "none", "--place-seed", seed, "--out-litematic", litematic, *args], check=True,
                   env={**os.environ, "MINECRAFT_V_ARTIFACTS_DIR": str(out / f"s{seed}_artifacts")})
    script = out / f"s{seed}.script"
    script.write_text(build_script(int(seed)))
    return subprocess.run([headless_binary(), "test", litematic, pins, script, "--modes", MODES, "--add-lever", "clk"]).returncode


def main() -> int:
    mode, *rest = sys.argv[1:]
    if mode == "array":
        return check_array(int(rest[0]), int(rest[1]), Path(rest[2]))
    return check_build(rest[0], Path(rest[1]), rest[2], *rest[3:])


if __name__ == "__main__":
    sys.exit(main())
