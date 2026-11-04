# scripts

Run from the repo root with `uv run python scripts/<name>.py`. Scripts that run circuits need `mchprs_headless`: set `$MCHPRS_HEADLESS`, or `$MCHPRS_DIR` (the MCHPRS checkout), or put it on `PATH`.

- `bench_examples.py`: builds several examples in parallel, each in its own artifacts directory, and prints failures and `[verify]` violations.
- `build_with_pins.py`: runs a build and writes each top-level pin's position, for `mchprs_headless test`.
- `settle_schematic.py`: settles a build made with `--no-settle` in `mchprs_headless`; also the lookup for the binary.
- `add_clock.py`: adds a generated clock to a built schematic without rebuilding it.
- `compare_gate_rtl.py`: checks a built schematic net by net against the netlist and the RTL.
- `check_same_output.py`: hashes each example's placed blocks, to check a refactor leaves output unchanged.
- `clock_sweep.py`: finds the fastest clock a build really runs at, against a slow reference run.
- `clock_timing.py`: reports clock skew and hold margins from the routed wire.
- `simulate.py`: simulates net power states from `component_list.json` for N clock cycles.
- `dev/`: acceptance checks for single designs in `examples/` and `tests/designs/` (`check_cpu.py`, `check_memread.py`, `check_powerup.py`, `check_preload.py`, `check_ram.py`); each has its usage in its docstring.

Environment variables the scripts and build read, besides the `MCHPRS_*` ones: `MINECRAFT_V_ARTIFACTS_DIR` (where a build writes its netlist and artifacts), `RAM_WORKERS` (processes that settle RAM banks, default the core count), `STACK_DUMP_EVERY` (`build_with_pins.py`), `QUIET` (`dev/check_cpu.py`, `dev/check_memread.py`), `CYCLES` (`dev/check_cpu.py`), `STEPS` (`dev/check_preload.py`).

`compare_gate_rtl.py --rtl` needs `iverilog` and `vvp` on `PATH`, and `nbtlib` (in the dev dependencies).
