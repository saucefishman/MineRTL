![image](assets/logo.png)

# MineRTL: Verilog to Minecraft Redstone Synthesis

**MineRTL** is a hardware synthesis toolchain that takes standard Verilog HDL code and compiles it into functional 3D Minecraft redstone circuits.

## Features

- **Full Synthesis Pipeline:** Parses, elaborates, and optimizes Verilog using Yosys.
- **Custom Cell Library:** Standard logic gates (AND, OR, XOR, NOT) and D-Flip Flops seamlessly implemented in Minecraft.
- **Automated 3D Placement and Routing:** Electrostatic 3D placement and a compiled (Cython) A* router for 3D redstone wiring, with a verify pass that reports opens, shorts, weak signals, latches and hold faults.
- **Macros:** Repeated subcircuits are routed once and stamped.
- **Memories:** Verilog memories with a supported port layout become prebuilt, pre-settled RAM blocks instead of routed flip-flops.
- **Clocks and Debug Prints:** A generated clock sized from static timing, or a lever MCHPRS toggles; `$write`/`$display` show up in MCHPRS.
- **Litematica Integration:** Outputs `.litematic` files for easy importing into Minecraft worlds or fast simulation via MCHPRS.
- **Scale:** designs of over 12,000 nets, such as a pipelined CPU with a branch predictor, build in minutes.

## Prerequisites

1. Install [yosys](https://github.com/yosyshq/yosys) (Required for Verilog synthesis)
2. Install [uv](https://github.com/astral-sh/uv) (Fast Python package manager)
3. Optional: [our MCHPRS fork](https://github.com/saucefishman/MCHPRS), for settling builds, testing them without the game (`mchprs_headless`), the server clock, `/watch` of debug prints and `/lever` by input name. Upstream [MCHPRS](https://github.com/MCHPR/MCHPRS) doesn't have these.
4. Optional: [sv2v](https://github.com/zachjs/sv2v), to convert SystemVerilog designs to Verilog first

## Getting Started

1. **Install Python dependencies:**

```sh
uv sync
```

This also builds the Cython A* kernel; `uv run` rebuilds it when a `.pyx` file changes.

2. **Generate a schematic:**
   We provide several examples in the `examples/` directory, from the smallest (`helloworld`, `counter`, `multimodule`, `sevenseg`) to `video`; `examples/README.md` lists what each one shows. Synthesize the counter example with:

```sh
./synth.sh examples/counter
```

The resulting schematic will be saved to `build/result.litematic`. You can import this directly into a Minecraft world using the Litematica mod.

3. **Synthesis Flags & Configuration:**
   The `synth.sh` script automatically wraps our Python-based layout engine (`minecraft_v/main.py`) alongside Yosys. It supports several flags that you can pass directly:
   - `--out-litematic <path>`: Specifies custom output path for the `.litematic` file (Default: `build/result.litematic`).
   - `--out-litematic-dir <dir>`: Writes the `.litematic` to `<dir>/<project-dir name>.litematic` (e.g. `examples/counter` → `<dir>/counter.litematic`). Handled by `synth.sh`.
   - `--artifacts-dir <dir>`: Where the netlist and build artifacts go (Default: `build/artifacts`, or `$MINECRAFT_V_ARTIFACTS_DIR`). Use a separate directory per build to run builds in parallel.
   - `--module <name>`: Sets the top-level Verilog module to synthesize (Default: `main`).
   - `--schematics-dir <dir>`: Sets the directory containing our `.litematic` cell library objects (Default: `schematics`).
   - `--schematic-name <name>`: The name attached to the generated schematic in-game (Default: matches `--module`).
   - `--allow-routing-failures`: Keeps generating the structure even if some 3D routes fail to connect (useful for debugging physical placement).
   - `--synth-only`: Stops after writing the netlist (handled by `synth.sh`).
   - `--tiled-ram` / `--no-tiled-ram` (synth.sh): Memories with a supported port layout become tiled RAM blocks, settled once and cached in `--macro-cache` (Default: `build/macro_cache`). On by default; `--no-tiled-ram` maps every memory to flip-flops. Large memory designs build better with `--no-macros`.
   - `--no-macros`: Routes every gate on its own. By default repeated subcircuits are routed once and stamped, which on a small CPU builds about 2.4x faster and takes about 3x the volume. Designs without enough repeats stay as gates.
   - `--no-settle`: Saves blocks as built instead of in the states they settle to. Without settling, power-up pulses can overwrite flip-flops.
   - `--no-debug-prints`: Removes `$write`/`$display`. By default they stay when they're in a clocked block, inside macros too: blocks they read stand on or hang from lime concrete, and `<out>.server.json` is written next to the litematic for [MCHPRS fork](https://github.com/saucefishman/MCHPRS) `/watch` and `mchprs_headless test --probes`. Non-constant `$sformatf` isn't supported by Yosys: use `$display` directly.
   - `--output-pin-targets-json <file.json>`: Path to a JSON file explicitly dictating output coordinates. Note: `synth.sh` will _automatically_ use `pin_targets.json` if it finds it in your specific project directory; `--no-pin-extensions` ignores it.
   - `--generate-clock <pin|auto|none>`: Select the pin to be the clock pin and automatically generate a redstone clock for it. Its period comes from static timing of the built circuit. (Default: `auto`, which picks `_clk` or `clk`)
   - `--server-clock [half period]`: No generated clock, which gets long on large circuits: the clock pin gets a lever, and MCHPRS toggles it every that many redstone ticks, read from `<out>.server.json`. `synth.sh` turns this on when it finds an `mchprs_headless` with `--run-clock` (`$MCHPRS_HEADLESS`, or `target/release` under `$MCHPRS_DIR`, else `mchprs_headless` on `PATH`) and no clock flag is given; `--no-server-clock` or `--generate-clock <pin>` keeps the generated clock. (Default half period: from the critical path, the generated clock's period.) That check can't see the running server: if the clock stays still, rebuild and restart it, and stand on the plot, since plots only tick with a player.

   - `--place-seed <n>`: Placement seed (Default: 0). Routing is deterministic per placement, so try another seed if a build fails nets.
   - `--flat`: Puts every component on one Y level (for small designs).
   - `--resume-dir <dir>`, `--resume-reroute <nets>`: Save the placement and routed paths and reuse them on the next build; reroute only the listed nets.
   - `--route-deadline <s>`: Stops routing after that many seconds and writes a partial result (use with `--allow-routing-failures`).
   - `--debug`: Logs progress and slow searches, saves routing snapshots to `<artifacts>/snapshots` (slow), prints A* counters.

   Run `uv run python -m minecraft_v.main --help` for the full list.

   Example usage:

```sh
./synth.sh examples/counter --out-litematic counter.litematic --place-seed 1
```

## Running Builds in MCHPRS

We recommend running builds in [our MCHPRS fork](https://github.com/saucefishman/MCHPRS) instead of vanilla Minecraft. MCHPRS is a redstone-only server, and its redpiler compiler runs large circuits many times faster than vanilla. The fork also reads the `<out>.server.json` that each build writes next to its `.litematic`:

- **Server clock:** with `--server-clock` (on by default when `synth.sh` finds `mchprs_headless`), the server toggles the clock lever itself, so the build doesn't need a generated clock.
- **Debug prints:** `$display`/`$write` output appears in chat and the server log.
- **`/lever <input> [on|off|toggle]`:** sets inputs by their Verilog name.

To load a build, copy the `.litematic` and its `.server.json` into the server's `schems/` folder, stand on a plot and run `/load <name>.litematic -c`, or `/watch <name>.litematic -c` to re-paste on every rebuild. (WorldEdit's `//load` pastes the blocks only, without the server file.) `/rtps halfclock` runs large builds much faster. See the fork's README for setup and the full command list.

## Testing Builds Without the Game

`mchprs_headless` (in [our MCHPRS fork](https://github.com/saucefishman/MCHPRS), `crates/headless`) loads a `.litematic` and runs a script against it. Build it with `cargo build --release -p mchprs_headless`; scripts find it at `$MCHPRS_HEADLESS`, else `$MCHPRS_DIR/target/release/mchprs_headless` if `MCHPRS_DIR` is set, else `mchprs_headless` on `PATH`. When present, `synth.sh` uses it to settle the build and to turn on `--server-clock`.

```sh
uv run python scripts/build_with_pins.py pins.txt --netlist netlist.json --out-litematic out.litematic --generate-clock none
uv run python scripts/settle_schematic.py out.litematic component_layout.json settled.litematic
mchprs_headless test settled.litematic pins.txt script.txt --modes noupdate,redstone,redpiler
```

The pin file lists `name x y z nx nz` per pin. Script commands: `settle N`, `set NAME 0|1`, `pulse NAME N`, `expect NAME 0|1`, `print LABEL`. Add `--add-lever clk` for an input with no lever.

Other scripts in `scripts/`:

- `bench_examples.py`: builds several examples in parallel, each in its own artifacts directory, and prints failures and `[verify]` violations.
- `add_clock.py`: adds a generated clock to a build you've already tested, without rebuilding.
- `compare_gate_rtl.py`: checks a built schematic net by net against the netlist and RTL.
- `dev/`: acceptance checks for single examples (`check_preload.py`, `check_powerup.py`, ...). See `scripts/README.md`.

## Development & Testing

Run our test suite with:

```sh
uv run python -m pytest -q tests
```
