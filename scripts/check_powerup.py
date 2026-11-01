"""Show why a build has to paste settled: examples/powerup fails unsettled and passes settled.

Depends on layout, so a router or placer change can hide the failure on --place-seed 7. Then sweep seeds.

Usage: check_powerup.py <out_dir> [seed]
"""
import os
import subprocess
import sys
from pathlib import Path

from settle_schematic import headless_binary

ROOT = Path(__file__).parent.parent
SCRIPT = "set switch 1\nsettle 250\n" + "expect q 1\npulse _clk 200\nsettle 250\n" * 4


def main() -> int:
    out = Path(sys.argv[1])
    seed = sys.argv[2] if len(sys.argv) > 2 else "7"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MINECRAFT_V_ARTIFACTS_DIR": str(out)}
    subprocess.run([ROOT / "synth.sh", ROOT / "examples/powerup", "--artifacts-dir", out, "--synth-only"], check=True, stdout=subprocess.DEVNULL)
    pins, raw, settled = out / "pins.txt", out / "unsettled.litematic", out / "settled.litematic"
    subprocess.run([sys.executable, ROOT / "scripts/build_with_pins.py", pins, "--netlist", out / "netlist.json", "--route-passes", "1",
                    "--no-macros", "--no-debug-prints", "--generate-clock", "none", "--place-seed", seed, "--out-litematic", raw],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/settle_schematic.py", raw, out / "component_layout.json", settled],
                   check=True, stdout=subprocess.DEVNULL)
    script = out / "run.txt"
    script.write_text(SCRIPT)
    codes = {}
    for name, litematic in (("unsettled", raw), ("settled", settled)):
        print(f"== {name}")
        codes[name] = subprocess.run([headless_binary(), "test", litematic, pins, script, "--modes", "redstone,redpiler"]).returncode
    return 0 if codes["unsettled"] != 0 and codes["settled"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
