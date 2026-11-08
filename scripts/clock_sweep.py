"""Find the fastest clock a built design really runs at, in mchprs_headless, against the same build on a very slow clock.

Random inputs change as the clock rises and outputs are read just before the next rise: what static timing assumes of
the outside world. A half period passes if every output on every clock matches the slow run. Random traffic can only
show a clock is too fast, so the fastest pass is a lower bound on the true minimum.

Usage: clock_sweep.py build <project_dir> <out_dir> [seed] [minecraft_v args...]    synthesize and build, once per out_dir
       clock_sweep.py sweep <out_dir> <half,half,...|auto> [clocks] [stimulus seed] [modes]
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from settle_schematic import headless_binary
from minecraft_v.build_utils import artifact_exists, load_artifact_json

ROOT = Path(__file__).parent.parent
SLOW_FACTOR = 8      # the reference run's half period, against the generated one
POWER_UP_FACTOR = 4  # generated half periods of quiet before the first clock
POOL = 4             # values per bus width


def top_ports(netlist: Path) -> tuple[list[str], list[str]]:
    """(input, output) names as the pins file spells them."""
    modules = load_artifact_json(netlist)["modules"]
    top = next(m for name, m in modules.items() if m.get("attributes", {}).get("top") or name == "main")
    names: tuple[list[str], list[str]] = ([], [])
    for name, port in top["ports"].items():
        width = len(port["bits"])
        names[port["direction"] != "input"].extend([name] if width == 1 else [f"{name}[{b}]" for b in range(width)])
    return names


def build(project: str, out: Path, seed: str, *args: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    env_args = ["env", f"MINECRAFT_V_ARTIFACTS_DIR={out}"]
    if not artifact_exists(out / "netlist.json"):
        flags = [a for a in args if a in ("--tiled-ram", "--no-tiled-ram")]
        subprocess.run([*env_args, "bash", "synth.sh", project, "--synth-only", *flags], cwd=ROOT, check=True)
    rest = [a for a in args if a not in ("--tiled-ram", "--no-tiled-ram")]
    return subprocess.run([*env_args, sys.executable, ROOT / "scripts/build_with_pins.py", out / "design.pins", "--netlist",
                           out / "netlist.json", "--server-clock", "--place-seed", seed,
                           "--out-litematic", out / "design.litematic", *rest], cwd=ROOT).returncode


def script(half: int, inputs: list[str], clock: str, clocks: int, seed: int, settle_first: int) -> str:
    rng = random.Random(seed)
    buses: dict[str, list[str]] = {}
    for p in inputs:
        if p != clock:
            buses.setdefault(p.split("[")[0], []).append(p)
    # Buses of a width share a few values, so that a read address often names a word just written. Reads through a memory
    # of hundreds of words would otherwise hardly ever see a write.
    pool = {width: [rng.getrandbits(width) for _ in range(POOL)] for width in {len(bits) for bits in buses.values()}}
    lines = [f"set {p} 0" for p in inputs] + [f"settle {settle_first}"]
    for k in range(clocks):
        lines.append(f"set {clock} 1")
        rate = rng.choice((0.1, 0.4, 0.8))  # few changes as well as many: a word holding still shows a slow bitline falling
        for bits in buses.values():
            if rng.random() < rate:
                value = rng.choice(pool[len(bits)])
                lines += [f"set {p} {value >> b & 1}" for b, p in enumerate(bits)]
        lines += [f"settle {half}", f"set {clock} 0", f"settle {half}", f"print c{k}"]
    return "\n".join(lines) + "\n"


def trace(out: Path, half: int, inputs: list[str], clock: str, clocks: int, seed: int, first: int, mode: str) -> list[str]:
    path = out / f"sweep_{half}.script"
    path.write_text(script(half, inputs, clock, clocks, seed, first))
    levers = [arg for p in inputs for arg in ("--add-lever", p)]
    run = subprocess.run([headless_binary(), "test", out / "design.litematic", out / "design.pins", path, "--modes", mode, *levers],
                         capture_output=True, text=True)
    path.unlink()
    lines = [re.sub(r"\s+", " ", line.split("]", 1)[1]).strip() for line in run.stdout.splitlines() if re.search(r"\] c\d+ ", line)]
    if run.returncode or len(lines) != clocks:
        raise RuntimeError(f"half {half}: {len(lines)} of {clocks} clocks, exit {run.returncode}\n{run.stdout[-500:]}\n{run.stderr[-500:]}")
    return lines


def first_difference(got: list[str], want: list[str]) -> str:
    for k, (a, b) in enumerate(zip(got, want)):
        if a != b:
            bad = sorted(set(a.split()) - set(b.split()))
            return f"clock {k}, {len(bad)} wrong: {' '.join(bad[:4])}"
    return ""


def sweep(out: Path, halves: str, clocks: int, seed: int, modes: str) -> int:
    inputs, _ = top_ports(out / "netlist.json")
    pins = {line.split()[0] for line in open(out / "design.pins")}
    inputs = [p for p in inputs if p in pins]
    side = json.loads((out / "design.server.json").read_text())["clock"]
    clock, generated = side["pin"], side["half_period_ticks"]
    slow = generated * SLOW_FACTOR
    first = generated * POWER_UP_FACTOR
    if halves == "auto":
        grid = sorted({max(1, generated * n // 16) for n in range(1, 25)})
    else:
        grid = [int(h) for h in halves.split(",")]
    worst_fail = 0
    for mode in modes.split(","):
        want = trace(out, slow, inputs, clock, clocks, seed, first, mode)
        again = trace(out, slow // 2, inputs, clock, clocks, seed, first, mode)
        if want != again:
            print(f"[{mode}] reference differs between half {slow} and {slow // 2}: {first_difference(again, want)}")
            return 2
        changing = sum(len(set(col)) > 1 for col in zip(*(line.split()[1:] for line in want)))
        print(f"[{mode}] generated half {generated}, reference half {slow}, {changing} outputs change")
        for half in grid:
            why = first_difference(trace(out, half, inputs, clock, clocks, seed, first, mode), want)
            print(f"[{mode}] half {half:5} ({half / generated:.2f}x) {'FAIL ' + why if why else 'ok'}", flush=True)
            if why:
                worst_fail = max(worst_fail, half)
    print(f"largest failing half {worst_fail}; generated {generated}; ratio {generated / max(worst_fail + 1, 1):.2f}")
    return int(worst_fail >= generated)


def main() -> int:
    mode, *rest = sys.argv[1:]
    if mode == "build":
        return build(rest[0], Path(rest[1]).resolve(), rest[2] if len(rest) > 2 else "0", *rest[3:])
    return sweep(Path(rest[0]).resolve(), rest[1], int(rest[2]) if len(rest) > 2 else 60, int(rest[3]) if len(rest) > 3 else 1,
                 rest[4] if len(rest) > 4 else "redstone")


if __name__ == "__main__":
    sys.exit(main())
