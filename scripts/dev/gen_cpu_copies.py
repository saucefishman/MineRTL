"""Write a design of N copies of examples/cpu, for scaling tests: `gen_cpu_copies.py N OUT_DIR`, then `./synth.sh OUT_DIR`.

Each copy runs its own program (cpu_<i>.hex, the example's with scrambled immediates), so nothing merges, and the
copies' outputs are XOR-reduced into main's, so nothing is removed. About 285 gates and 170 KB of netlist per copy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

CPU = Path(__file__).resolve().parents[2] / "examples" / "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("copies", type=int)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    n = args.copies
    args.out.mkdir(parents=True, exist_ok=True)
    core = (CPU / "cpu.v").read_text().replace(
        "module main(input clk, output [7:0] out, output [3:0] pc_out);",
        'module cpu_core #(parameter HEX = "cpu_0.hex") (input clk, output [7:0] out, output [3:0] pc_out);',
    ).replace('$readmemh("cpu.hex", rom)', "$readmemh(HEX, rom)")
    assert "cpu_core" in core and "readmemh(HEX" in core
    words = [w for w in (line.split("//")[0].strip() for line in (CPU / "cpu.hex").read_text().splitlines()) if w]
    lines = [core, "module main(input clk, output [7:0] out, output [3:0] pc_out);",
             f"  wire [7:0] o [0:{n - 1}]; wire [3:0] p [0:{n - 1}];"]
    for i in range(n):
        program = ["%04x" % ((int(w, 16) & 0xff00) | (((int(w, 16) & 0xff) * (2 * i + 3) + 7 * i + k) & 0xff))
                   for k, w in enumerate(words)]
        (args.out / f"cpu_{i}.hex").write_text("\n".join(program) + "\n")
        lines.append(f'  cpu_core #(.HEX("cpu_{i}.hex")) c{i}(.clk(clk), .out(o[{i}]), .pc_out(p[{i}]));')
    lines += [f"  wire [7:0] xo [0:{n - 1}]; wire [3:0] xp [0:{n - 1}];", "  assign xo[0] = o[0]; assign xp[0] = p[0];"]
    lines += [f"  assign xo[{i}] = xo[{i - 1}] ^ o[{i}]; assign xp[{i}] = xp[{i - 1}] ^ p[{i}];" for i in range(1, n)]
    lines += [f"  assign out = xo[{n - 1}]; assign pc_out = xp[{n - 1}];", "endmodule"]
    (args.out / "main.v").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
