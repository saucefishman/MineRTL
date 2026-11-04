"""Build examples/memread on one placement seed and run it in mchprs_headless on all modes.

Counts through the preloaded words several times, then stores the count on some clocks and reads it back: count changes
on the same edge, so a word that holds badly stores count + 1. The clock is gated (_clk & switch), and hold fixing has to
see through that gate.

Usage: check_memread.py <netlist.json> <out_dir> <seed> [minecraft_v args...]
"""
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from settle_schematic import headless_binary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
WORDS = [int(line, 16) for line in (ROOT / "examples/memread/memread.hex").read_text().split()]
CLOCK_WIDTH, SETTLE = 200, 250


def expect(words: list[int], count: int) -> list[str]:
    q = words[count & 7]
    return [f"expect count[{b}] {count >> b & 1}" for b in range(16)] + [f"expect q[{b}] {q >> b & 1}" for b in range(16)]


def script() -> str:
    words = WORDS
    lines = ["set switch 1", "set wen 0", f"settle {SETTLE}"]
    count = 0
    lines += expect(words, count)
    for _ in range(40):
        lines += [f"pulse _clk {CLOCK_WIDTH}", f"settle {SETTLE}"]
        count += 1
        lines += expect(words, count)
    words = list(words)
    rng = random.Random(0)
    for _ in range(80):
        wen = rng.random() < 0.5
        lines += [f"set wen {int(wen)}", f"settle {SETTLE}", f"pulse _clk {CLOCK_WIDTH}", f"settle {SETTLE}"]
        if wen:
            words[count & 7] = count & 0xffff
        count += 1
        lines += expect(words, count)
    return "\n".join(lines) + "\n"


def main() -> int:
    netlist, out, seed, *args = sys.argv[1:]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    litematic, pins = out / f"s{seed}.litematic", out / f"s{seed}.pins"
    env = {**os.environ, "MINECRAFT_V_ARTIFACTS_DIR": str(out / f"s{seed}_artifacts")}
    (out / f"s{seed}_artifacts").mkdir(exist_ok=True)
    subprocess.run([sys.executable, ROOT / "scripts/build_with_pins.py", pins, "--netlist", netlist, "--route-passes", "1",
                    "--no-macros", "--generate-clock", "none", "--place-seed", seed, "--out-litematic", litematic, *args],
                   check=True, env=env, stdout=subprocess.DEVNULL if os.environ.get("QUIET") else None)
    script_path = out / f"s{seed}.script"
    script_path.write_text(script())
    return subprocess.run([headless_binary(), "test", litematic, pins, script_path, "--modes", "noupdate,redstone,redpiler"]).returncode


if __name__ == "__main__":
    sys.exit(main())
