"""A built examples/bpmem against a model of its RTL: random reads and writes, next_pc checked before and after each clock.

Usage: check_bpmem.py <dir with out.litematic and pins> <seed> <half period> <modes>. STEPS sets the clocks, default 40.
The model agrees with the RTL: run the script it writes, <dir>/b.script, on examples/bpmem/main.v to see.
"""
import os
import random
import subprocess
import sys

from settle_schematic import headless_binary


def script(seed: int, half: int, steps: int) -> str:
    r = random.Random(seed)
    btb: dict[int, int] = {}
    hist: dict[int, int] = {}
    takes: dict[int, int] = {}
    lines = ["set clk 0", "set wen0 0", "set wen1 0", f"settle {half}"]

    def bus(name: str, width: int, value: int) -> None:
        lines.extend(f"set {name}[{i}] {value >> i & 1}" for i in range(width))

    def next_pc(pc: int) -> int:
        index = pc & 511
        taken = takes.get(index << 2 | hist.get(index, 0), 1) >> 1
        return btb.get(index, 0) if taken else (pc + 2) & 0xffff

    def check(a: int, b: int) -> None:
        for name, pc in (("next_pc0", a), ("next_pc1", b)):
            value = next_pc(pc)
            lines.extend(f"expect {name}[{i}] {value >> i & 1}" for i in range(16))

    pool = [r.randrange(1 << 16) & ~1 for _ in range(8)] + [0, 510, 512, 1022]
    dests = [r.randrange(1 << 16) & ~1 for _ in range(4)]
    for _ in range(steps):
        a, b, w0, w1 = r.choice(pool), r.choice(pool), r.choice(pool), r.choice(pool)
        if r.random() < .15:
            w1 = w0
        d0, d1, t0, t1, e0, e1 = (r.choice(dests), r.choice(dests), r.random() < .6, r.random() < .6, r.random() < .7,
                                  r.random() < .7)
        bus("pc0", 16, a)
        bus("pc1", 16, b)
        bus("wpc0", 16, w0)
        bus("wpc1", 16, w1)
        bus("wdest0", 16, d0)
        bus("wdest1", 16, d1)
        lines += [f"set wtakes0 {int(t0)}", f"set wtakes1 {int(t1)}", f"set wen0 {int(e0)}", f"set wen1 {int(e1)}", f"settle {half}"]
        check(a, b)
        lines += [f"pulse clk {half}", f"settle {half}"]
        # Both ports read old contents; port 1's writes land after port 0's.
        old_btb, old_hist, old_takes = dict(btb), dict(hist), dict(takes)
        for enable, pc, dest, taken in ((e0, w0, d0, t0), (e1, w1, d1, t1)):
            if not enable:
                continue
            index = pc & 511
            history = old_hist.get(index, 0)
            at = index << 2 | history
            count = old_takes.get(at, 1)
            if old_btb.get(index, 0) != dest:
                btb[index] = dest
                takes[at] = 2 if taken else 1
            else:
                takes[at] = min(count + 1, 3) if taken else max(count - 1, 0)
            hist[index] = (history & 1) << 1 | int(taken)
        check(a, b)
    return "\n".join(lines) + "\n"


def main() -> int:
    build, seed, half, modes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    path = f"{build}/b.script"
    with open(path, "w") as f:
        f.write(script(seed, half, int(os.environ.get("STEPS", 40))))
    return subprocess.run([headless_binary(), "test", f"{build}/out.litematic", f"{build}/pins", path, "--modes", modes,
                           "--add-lever", "clk"]).returncode


if __name__ == "__main__":
    sys.exit(main())
