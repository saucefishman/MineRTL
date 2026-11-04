"""Build examples/video on one placement seed and play it in mchprs_headless on all modes.

Steps through every frame and round to the first again, expecting each frame of examples/video/video.hex on the dark
outputs, on every tile's middle lamp, and on all nine lamps of a few tiles. A tile lights while its pin is low.

Usage: check_video.py <netlist.json> <out_dir> <seed> [minecraft_v args...]. Env: QUIET=1 hides the build output,
FRAMES sets the clock count (default: every frame, and two more).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from settle_schematic import headless_binary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO = ROOT / "examples/video"
WORDS = [int(line, 16) for line in (VIDEO / "video.hex").read_text().split()]
PIXELS = 192
WHOLE_TILES = [0, 11, 50, 100, 180, 191]  # corners, and a few between
CLOCK_WIDTH, SETTLE = 100, 900


def pin(k: int) -> str:
    """Bit k of a frame: column 15 - k // 12, k % 12 rows from the bottom."""
    return f"dark{15 - k // 12}[{k % 12}]"


def lamps(placed: dict[str, list[int]]) -> dict[str, tuple[int, int, int]]:
    """Name -> lamp cell: every tile's middle lamp, and all of WHOLE_TILES'."""
    found = {}
    for k in range(PIXELS):
        x, y, z, _ = placed[pin(k)]
        for i in range(3):
            for j in range(3):
                if (i, j) == (1, 1) or k in WHOLE_TILES:
                    found[f"lamp{k}_{i}{j}"] = (x + i, y + j, z)
    return found


def expect(word: int, names: list[str]) -> list[str]:
    lines = [f"expect {pin(k)} {word >> k & 1}" for k in range(PIXELS)]
    lines += [f"expect {name} {1 - (word >> int(name[4:name.index('_')]) & 1)}" for name in names]
    return lines


def script(frames: int, names: list[str]) -> str:
    lines = ["set clk 0", f"settle {SETTLE}", *expect(WORDS[0], names)]
    for frame in range(1, frames + 1):
        lines += [f"pulse clk {CLOCK_WIDTH}", f"settle {SETTLE}", *expect(WORDS[frame % len(WORDS)], names)]
    return "\n".join(lines) + "\n"


def main() -> int:
    netlist, out, seed, *args = sys.argv[1:]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    litematic, pins = out / f"s{seed}.litematic", out / f"s{seed}.pins"
    artifacts = out / f"s{seed}_artifacts"
    artifacts.mkdir(exist_ok=True)
    env = {**os.environ, "MINECRAFT_V_ARTIFACTS_DIR": str(artifacts)}
    subprocess.run([sys.executable, ROOT / "scripts/build_with_pins.py", pins, "--netlist", netlist,
                    "--output-pin-targets-json", VIDEO / "pin_targets.json", "--generate-clock", "none",
                    "--place-seed", seed, "--out-litematic", litematic, *args],
                   check=True, env=env, stdout=subprocess.DEVNULL if os.environ.get("QUIET") else None)
    settled = out / f"s{seed}.settled.litematic"
    subprocess.run([sys.executable, ROOT / "scripts/settle_schematic.py", litematic, artifacts / "component_layout.json",
                    settled], check=True, stdout=subprocess.DEVNULL if os.environ.get("QUIET") else None)
    found = lamps(json.loads((artifacts / "pin_targets_placed.json").read_text()))
    with open(pins, "a") as f:
        f.writelines(f"{name} {x} {y} {z} 0 -1\n" for name, (x, y, z) in found.items())
    script_path = out / f"s{seed}.script"
    script_path.write_text(script(int(os.environ.get("FRAMES", len(WORDS) + 2)), list(found)))
    return subprocess.run([headless_binary(), "test", settled, pins, script_path,
                           "--modes", "noupdate,redstone,redpiler"]).returncode


if __name__ == "__main__":
    sys.exit(main())
