"""Run a built examples/floweypu_hi in mchprs_headless: it must print 1, 2, 3, then h and i on one clock.

Usage: check_floweypu.py <litematic> <pins> <half_period> [modes] [clocks] [expected text]

With expected text (examples/floweypu_loop), the characters must spell it, and halt must rise.

Modes run one after another (each headless world is about 2 GB resident). `clocks` defaults to 10: the
characters come on clocks 5 to 8 after reset, so 10 still shows any extra output on the two after.
"""
import re
import subprocess
import sys
from pathlib import Path

from settle_schematic import headless_binary

CLOCKS = 10
EXPECTED = [("1", ""), ("2", ""), ("3", ""), ("h", "i")]


def script(half: int, clocks: int = CLOCKS) -> str:
    lines = ["set rst 1", f"settle {half}", f"pulse clk {half}", f"settle {half}", "set rst 0", f"settle {half}"]
    for _ in range(clocks):
        lines += [f"pulse clk {half}", f"settle {half}", "print e0"]
    return "\n".join(lines) + "\n"


def characters(output: str) -> list[tuple[str, str]]:
    """Per clock with something printed: the characters on ports 0 and 1."""
    seen = []
    for line in output.splitlines():
        values = dict(re.findall(r"(\w+(?:\[\d+\])?)=(\d)", line))
        if "e0" not in values:
            continue
        chars = []
        for port in "01":
            if values.get(f"e{port}") == "1":
                chars.append(chr(sum(int(values.get(f"d{port}[{b}]", 0)) << b for b in range(8))))
            else:
                chars.append("")
        if any(chars):
            seen.append(tuple(chars))
    return seen


def main() -> int:
    litematic, pins, half, *rest = sys.argv[1:]
    modes = rest[0].split(",") if rest else ["redstone", "redpiler"]
    clocks = int(rest[1]) if len(rest) > 1 else CLOCKS
    text = rest[2] if len(rest) > 2 else None
    script_path = Path(litematic).with_suffix(".hi.script")
    script_path.write_text(script(int(half), clocks))
    probes = [name for name in (line.split()[0] for line in open(pins)) if re.fullmatch(r"(e[01]|d[01]\[\d\]|halt)", name)]
    if "e1" not in probes:
        print("missing character pins", probes)
        return 2
    failed = 0
    for mode in modes:
        out = subprocess.run([headless_binary(), "test", litematic, pins, script_path, "--modes", mode, "--add-lever", "clk",
                              "--add-lever", "rst"], capture_output=True, text=True).stdout
        got = characters(out)
        halts = [k for k, line in enumerate(l for l in out.splitlines() if "e0=" in l) if "halt=1" in line]
        ok = got == EXPECTED if text is None else "".join(a + b for a, b in got) == text and bool(halts)
        failed += not ok
        print(f"[{mode}] {'PASS' if ok else 'FAIL'}: {got}" + (f" halt from clock {halts[0] if halts else None}" if text else ""))
        if not got:
            print(out[-600:])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
