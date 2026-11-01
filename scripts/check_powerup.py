"""Show why a build has to paste settled, and that settling keeps state.

examples/powerup fails unsettled and passes settled. It depends on layout, so a router or placer change can hide the
failure on --place-seed 7: then sweep seeds. examples/powerup_inverted has clocks that rest high, and has to pass settled,
then still capture on the clock's real edge.

Usage: check_powerup.py <out_dir> [seed]
"""
import os
import subprocess
import sys
from pathlib import Path

from settle_schematic import headless_binary

ROOT = Path(__file__).parent.parent
SCRIPT = "set switch 1\nsettle 250\n" + "expect q 1\npulse _clk 200\nsettle 250\n" * 4
INVERTED_SCRIPT = ("settle 250\nexpect q[0] 1\nexpect q[1] 0\nexpect q[2] 1\nexpect q[3] 0\n"
                   "pulse _clk 200\nsettle 250\nexpect q[0] 0\nexpect q[1] 1\nexpect q[2] 0\nexpect q[3] 1\n")


def build(example: str, out: Path, seed: str, script_text: str) -> tuple[Path, Path, Path, Path]:
    """Builds unsettled and settled copies. Returns them with the pins and script to test them with."""
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MINECRAFT_V_ARTIFACTS_DIR": str(out)}
    subprocess.run([ROOT / "synth.sh", ROOT / "examples" / example, "--artifacts-dir", out, "--synth-only"], check=True, stdout=subprocess.DEVNULL)
    pins, raw, settled = out / "pins.txt", out / "unsettled.litematic", out / "settled.litematic"
    for flag, litematic in (("--no-settle", raw), ("--settle", settled)):
        subprocess.run([sys.executable, ROOT / "scripts/build_with_pins.py", pins, "--netlist", out / "netlist.json", "--route-passes", "1",
                        "--no-macros", "--no-debug-prints", "--generate-clock", "none", "--place-seed", seed, flag, "--out-litematic", litematic],
                       check=True, env=env, stdout=subprocess.DEVNULL)
    script = out / "run.txt"
    script.write_text(script_text)
    return raw, settled, pins, script


def run(name: str, litematic: Path, pins: Path, script: Path) -> int:
    print(f"== {name}", flush=True)
    return subprocess.run([headless_binary(), "test", litematic, pins, script, "--modes", "redstone,redpiler"]).returncode


def main() -> int:
    out = Path(sys.argv[1])
    seed = sys.argv[2] if len(sys.argv) > 2 else "7"
    raw, settled, pins, script = build("powerup", out / "powerup", seed, SCRIPT)
    unsettled_fails = run("powerup unsettled", raw, pins, script) != 0
    settled_passes = run("powerup settled", settled, pins, script) == 0
    _, settled, pins, script = build("powerup_inverted", out / "inverted", seed, INVERTED_SCRIPT)
    inverted_passes = run("inverted settled", settled, pins, script) == 0
    return 0 if unsettled_fails and settled_passes and inverted_passes else 1


if __name__ == "__main__":
    sys.exit(main())
