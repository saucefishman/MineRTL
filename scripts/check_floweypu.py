"""Run a built examples/floweypu_hi in mchprs_headless: it must print 1, 2, 3, then h and i on one clock.

Usage: check_floweypu.py <litematic> <pins> <half_period> [modes]
"""
import re
import subprocess
import sys
from pathlib import Path

from settle_schematic import headless_binary

CLOCKS = 20
EXPECTED = [("1", ""), ("2", ""), ("3", ""), ("h", "i")]


def script(half: int) -> str:
    lines = ["set rst 1", f"settle {half}", f"pulse clk {half}", f"settle {half}", "set rst 0", f"settle {half}"]
    for _ in range(CLOCKS):
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
    modes = rest[0] if rest else "redstone,redpiler"
    script_path = Path(litematic).with_suffix(".hi.script")
    script_path.write_text(script(int(half)))
    probes = [name for name in (line.split()[0] for line in open(pins)) if re.fullmatch(r"(e[01]|d[01]\[\d\]|halt)", name)]
    if "e1" not in probes:
        print("missing character pins", probes)
        return 2
    failed = 0
    for mode in modes.split(","):
        out = subprocess.run([headless_binary(), "test", litematic, pins, script_path, "--modes", mode, "--add-lever", "clk",
                              "--add-lever", "rst"], capture_output=True, text=True).stdout
        got = characters(out)
        ok = got == EXPECTED
        failed += not ok
        print(f"[{mode}] {'PASS' if ok else 'FAIL'}: {got}")
        if not got:
            print(out[-600:])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
