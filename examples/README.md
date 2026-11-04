# Examples

Build any of these with `./synth.sh examples/<name>`. Each directory holds the Verilog (top module `main`), and a
`pin_targets.json` where the outputs should land somewhere specific.

## Getting started

| Example | Shows |
| --- | --- |
| `helloworld` | One gate (`y = a ^ b`): the smallest build, combinational logic only. |
| `counter` | A 4-bit counter: flip-flops, an adder, and the generated clock on `clk`. |
| `multimodule` | A 4-bit ripple-carry adder from four `full_adder` instances: module hierarchy and buses. |
| `sevenseg` | A decimal counter on a seven-segment display: `case` decoding, and `pin_targets.json` placing the outputs as a display. |
| `printing` | `$write`/`$display` debug prints: prints "Hello!" and a lap count, shown by MCHPRS `/watch` or `mchprs_headless test --probes`. |

## Memories

| Example | Shows |
| --- | --- |
| `regs` | A 4-word register file: a Verilog memory built from flip-flops. |
| `ramtile` | The smallest memory that becomes a tiled RAM block (2 reads, 1 write; `--no-tiled-ram` keeps it as flip-flops). `scripts/dev/check_ram.py` tests it against a model. |
| `memread` | A memory loaded with `$readmemh` and read out in order. `scripts/dev/check_memread.py` runs it. |

## Showcase

| Example | Shows |
| --- | --- |
| `cpu` | A single-cycle 8-bit CPU: four registers in a tiled RAM block, a carry flag, and 16-bit instructions (load immediate, add/sub/and/or/xor/mov, add immediate, out, jump, jump on carry or zero) from a 16-word ROM loaded with `$readmemh`. Its program shows Fibonacci numbers on the `out` lamps. Builds in seconds; `scripts/dev/check_cpu.py` runs it against a model of the ISA. |
| `video` | A 16x12 frame buffer loaded 8 bits at a time, its outputs laid out as a screen by `pin_targets.json`. Over 1,100 nets: it takes minutes, so `scripts/bench_examples.py` skips it unless named. |
