"""A preloaded RAM read back from a settled build before any write, then random traffic, against its model.

Usage: check_preload.py <dir with out.litematic and pins> <seed> <half period> <modes>. For a build of tests/designs/preload.
Env: STEPS sets the random write steps (default 30).
"""
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from settle_schematic import headless_binary  # noqa: E402

d, seed, H, modes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
r = random.Random(seed)
M = {i: (i * 7 + 1) % 4 for i in range(16)}
N = {i: (i * 5 + 3) % 8 for i in range(16)}
L: list[str] = []


def bus(name: str, width: int, value: int) -> None:
    L.extend(f"set {name}[{i}] {value >> i & 1}" for i in range(width))


def check(addrs: list[int]) -> None:
    for p, a in enumerate(addrs):
        word_a = M[a]
        L.extend(f"expect a{p}[{i}] {word_a >> i & 1}" for i in range(2))
        L.extend(f"expect b{p}[{i}] {N[a] >> i & 1}" for i in range(3))


L += ["set clk 0", "set wen0 0", "set wen1 0"]
for p in range(4):
    bus(f"addr{p}", 4, 0)
L += [f"settle {H}"]
check([0, 0, 0, 0])  # before anything has moved
for a in range(16):  # every word, before any write
    addrs = [a, (a + 3) % 16, (a + 7) % 16, (a + 11) % 16]
    for p, x in enumerate(addrs):
        bus(f"addr{p}", 4, x)
    L.append(f"settle {H}")
    check(addrs)
for _ in range(int(os.environ.get("STEPS", 30))):
    addrs = [r.randrange(16) for _ in range(4)]
    w0, w1, v0, v1 = r.randrange(16), r.randrange(16), r.randrange(4), r.randrange(4)
    e0, e1 = r.random() < .6, r.random() < .6
    for p, x in enumerate(addrs):
        bus(f"addr{p}", 4, x)
    bus("waddr0", 4, w0)
    bus("waddr1", 4, w1)
    bus("wdata0", 2, v0)
    bus("wdata1", 2, v1)
    L += [f"set wen0 {int(e0)}", f"set wen1 {int(e1)}", f"settle {H}"]
    check(addrs)
    L += [f"pulse clk {H}", f"settle {H}"]
    if e0:
        M[w0], N[w0] = v0, v0 | (v0 & 1) << 2
    if e1:
        M[w1], N[w1] = v1, v1 | (v1 & 1) << 2
    check(addrs)
with open(f"{d}/p.script", "w") as f:
    f.write("\n".join(L) + "\n")
sys.exit(subprocess.run([headless_binary(), "test", f"{d}/out.litematic", f"{d}/pins",
                         f"{d}/p.script", "--modes", modes, "--add-lever", "clk"]).returncode)
