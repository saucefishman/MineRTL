"""Run the minecraft_v build and write the top level's pin positions for `mchprs_headless test`.

Usage: build_with_pins.py <pins.txt> <minecraft_v args...>

Env: STACK_DUMP_EVERY=<seconds> dumps every thread's stack that often to <artifacts dir>/stacks.txt, a poor man's profiler
for long builds (the artifacts dir is $MINECRAFT_V_ARTIFACTS_DIR, default build/artifacts).
"""
import faulthandler
import os
import sys

from minecraft_v.main import main
from minecraft_v.placement_engine import builder

if os.environ.get("STACK_DUMP_EVERY"):  # a poor man's profiler for long builds
    artifacts = os.environ.get("MINECRAFT_V_ARTIFACTS_DIR", "build/artifacts")
    os.makedirs(artifacts, exist_ok=True)
    stacks = open(artifacts + "/stacks.txt", "w")
    faulthandler.dump_traceback_later(int(os.environ["STACK_DUMP_EVERY"]), repeat=True, file=stacks)
pins_path = sys.argv[1]
sys.argv = [sys.argv[0], *sys.argv[2:]]
place_io = builder._place_io_repeaters
signs: list = []


def capture(*args, **kwargs):
    result = place_io(*args, **kwargs)
    signs[:] = result[2]  # macros run this too; the top level is the last call
    return result


builder._place_io_repeaters = capture
main()
with open(pins_path, "w") as f:
    for (x, y, z), (nx, nz), name in signs:
        f.write(f"{name} {x} {y} {z} {nx} {nz}\n")
