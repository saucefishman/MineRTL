"""Build examples/cpu on one placement seed and run its program in mchprs_headless on all modes.

A model of the ISA (see examples/cpu/cpu.v) runs examples/cpu/cpu.hex alongside, and every clock checks pc_out and the
out lamps against it. The default run covers two passes of the Fibonacci program, including the restart on carry.

Usage: check_cpu.py <netlist.json> <out_dir> <seed> [minecraft_v args...]. Env: QUIET=1 hides the build output,
CYCLES sets the clock count (default 200).
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from settle_schematic import headless_binary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
ROM = [int(line.split("//")[0], 16) for line in (ROOT / "examples/cpu/cpu.hex").read_text().splitlines()
       if line.split("//")[0].strip()]
CLOCK_WIDTH, SETTLE = 200, 250


class Cpu:
    def __init__(self) -> None:
        self.pc, self.regs, self.out, self.carry = 0, [0] * 4, 0, 0

    def step(self) -> None:
        instr = ROM[self.pc]
        op, rd, rs, imm = instr >> 12, instr >> 10 & 3, instr >> 8 & 3, instr & 0xff
        a, b = self.regs[rd], self.regs[rs]
        next_pc = (self.pc + 1) & 15
        if op == 0x0:
            self.regs[rd] = imm
        elif op in (0x1, 0x7):
            total = a + (imm if op == 0x7 else b)
            self.regs[rd], self.carry = total & 0xff, total >> 8
        elif op == 0x2:
            self.regs[rd], self.carry = (a - b) & 0xff, int(a < b)
        elif op == 0x3:
            self.regs[rd] = a & b
        elif op == 0x4:
            self.regs[rd] = a | b
        elif op == 0x5:
            self.regs[rd] = a ^ b
        elif op == 0x6:
            self.regs[rd] = b
        elif op == 0x8:
            self.out = b
        elif op == 0x9 or (op == 0xA and self.carry) or (op == 0xB and b == 0):
            next_pc = imm & 15
        self.pc = next_pc

    def expect(self) -> list[str]:
        return [f"expect out[{i}] {self.out >> i & 1}" for i in range(8)] + \
            [f"expect pc_out[{i}] {self.pc >> i & 1}" for i in range(4)]


def script(cycles: int) -> str:
    cpu = Cpu()
    lines = ["set clk 0", f"settle {SETTLE}", *cpu.expect()]
    for _ in range(cycles):
        cpu.step()
        lines += [f"pulse clk {CLOCK_WIDTH}", f"settle {SETTLE}", *cpu.expect()]
    return "\n".join(lines) + "\n"


def main() -> int:
    netlist, out, seed, *args = sys.argv[1:]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    litematic, pins = out / f"s{seed}.litematic", out / f"s{seed}.pins"
    env = {**os.environ, "MINECRAFT_V_ARTIFACTS_DIR": str(out / f"s{seed}_artifacts")}
    (out / f"s{seed}_artifacts").mkdir(exist_ok=True)
    subprocess.run([sys.executable, ROOT / "scripts/build_with_pins.py", pins, "--netlist", netlist,
                    "--generate-clock", "none", "--place-seed", seed, "--out-litematic", litematic, *args],
                   check=True, env=env, stdout=subprocess.DEVNULL if os.environ.get("QUIET") else None)
    script_path = out / f"s{seed}.script"
    script_path.write_text(script(int(os.environ.get("CYCLES", 200))))
    return subprocess.run([headless_binary(), "test", litematic, pins, script_path, "--modes", "noupdate,redstone,redpiler"]).returncode


if __name__ == "__main__":
    sys.exit(main())
