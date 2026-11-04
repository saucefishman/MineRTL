![MineRTL logo](assets/logo.png)

# MineRTL: Verilog to Minecraft Redstone Synthesis

**MineRTL** is a hardware synthesis toolchain that takes standard Verilog HDL code and compiles it into functional 3D Minecraft redstone circuits.

## Features

- **Full Synthesis Pipeline:** Parses, elaborates, and optimizes Verilog using Yosys.
- **Custom Cell Library:** AND, OR, XOR, NOT, MUX, full adder, D latch, and D flip-flops with and without enable, built in redstone.
- **Automated 3D Placement and Routing:** Electrostatic 3D placement and a compiled (Cython) A* router for 3D redstone wiring, with a verify pass that reports opens, shorts, weak signals, latches and hold faults.
- **Macros:** Repeated subcircuits are routed once and stamped.
- **Memories:** Verilog memories with a supported port layout become prebuilt, pre-settled RAM blocks instead of routed flip-flops.
- **Clocks and Debug Prints:** A generated clock sized from static timing, or a lever MCHPRS toggles; `$write`/`$display` show up in MCHPRS.
- **Litematica Integration:** Outputs `.litematic` files for easy importing into Minecraft worlds or fast simulation via MCHPRS.
- **Scale:** tested on designs of over 12,000 nets, which build in minutes.

## Prerequisites

macOS or Linux (on Windows, use WSL). Builds target Minecraft 1.20.4.

1. Install [yosys](https://github.com/yosyshq/yosys) (Required for Verilog synthesis; developed against 0.64)
2. Install [uv](https://github.com/astral-sh/uv) (Fast Python package manager; it fetches Python 3.13+ if needed)
3. A C++17 compiler (Xcode command line tools, GCC or Clang), to build the A* kernel
4. Optional: [our MCHPRS fork](https://github.com/saucefishman/MCHPRS), for testing builds without the game (`mchprs_headless`), the server clock, debug print output and `/lever` by input name. Upstream [MCHPRS](https://github.com/MCHPR/MCHPRS) doesn't have these.
5. Optional: [sv2v](https://github.com/zachjs/sv2v), to convert SystemVerilog designs to Verilog first

## Getting Started

1. **Install Python dependencies:**

```sh
uv sync
```

This also builds the Cython A* kernel; `uv run` rebuilds it when a `.pyx` file changes.

2. **Generate a schematic:**
   We provide several examples in the `examples/` directory, from the smallest (`helloworld`, `counter`, `multimodule`, `sevenseg`) to `video`; [`examples/README.md`](examples/README.md) lists what each one shows. Synthesize the counter example with:

```sh
./synth.sh examples/counter
```

The resulting schematic will be saved to `build/result.litematic`. Paste it into a world with the [Litematica](https://www.curseforge.com/minecraft/mc-mods/litematica) mod (Fabric, with MaLiLib, for 1.20.4): put the file in `.minecraft/schematics/`, load it there, and paste in creative mode. Or load it in MCHPRS (below), which is much faster for large builds. Each input gets a lever on a stone block and each output a redstone lamp, at the edges of the build. A generated clock starts on its own once pasted.

3. **Synthesis Flags & Configuration:**
   The `synth.sh` script automatically wraps our Python-based layout engine (`minecraft_v/main.py`) alongside Yosys. It supports several flags that you can pass directly:
   - `--out-litematic <path>`: Specifies custom output path for the `.litematic` file (Default: `build/result.litematic`).
   - `--out-litematic-dir <dir>`: Writes the `.litematic` to `<dir>/<project-dir name>.litematic` (e.g. `examples/counter` → `<dir>/counter.litematic`). Handled by `synth.sh`.
   - `--artifacts-dir <dir>`: Where the netlist and build artifacts go (Default: `build/artifacts`, or `$MINECRAFT_V_ARTIFACTS_DIR`). Use a separate directory per build to run builds in parallel.
   - `--module <name>`: The top-level Verilog module; everything under it is flattened into one build (Default: `main`).
   - `--schematic-name <name>`: The name attached to the generated schematic in-game (Default: matches `--module`).
   - `--allow-routing-failures`: Keeps generating the structure even if some 3D routes fail to connect (useful for debugging physical placement).
   - `--synth-only`: Stops after writing the netlist (handled by `synth.sh`).
   - `--tiled-ram` / `--no-tiled-ram` (synth.sh): Memories with a supported port layout become tiled RAM blocks, settled once and cached in `--macro-cache` (Default: `build/macro_cache`). On by default; `--no-tiled-ram` maps every memory to flip-flops and also skips the ABC optimization pass that comes with tiled RAM, so other logic can come out a little larger. Large memory designs build better with `--no-macros`.
   - `--no-macros`: Routes every gate on its own. By default repeated subcircuits are routed once and stamped, which builds designs with many repeated blocks faster but takes more volume. Designs without enough repeats stay as gates.
   - `--no-settle`: Saves blocks as built instead of in the states they settle to. Without settling, power-up pulses can overwrite flip-flops.
   - `--no-debug-prints`: Removes `$write`/`$display`. By default they stay when they're in a clocked block, inside macros too: blocks they read stand on or hang from lime concrete, and `<out>.server.json` is written next to the litematic for [MCHPRS fork](https://github.com/saucefishman/MCHPRS) `/watch` and `mchprs_headless test --probes`. Non-constant `$sformatf` isn't supported by Yosys: use `$display` directly.
   - `--output-pin-targets-json <file.json>`: Path to a JSON file explicitly dictating output coordinates. Note: `synth.sh` will _automatically_ use `pin_targets.json` if it finds it in your specific project directory; `--no-pin-extensions` ignores it.
   - `--generate-clock <pin|auto|none>`: Select the pin to be the clock pin and automatically generate a redstone clock for it. Its period comes from static timing of the built circuit. (Default: `auto`, which picks `_clk` or `clk`)
   - `--server-clock [half period]`: No generated clock, which gets long on large circuits: the clock pin gets a lever, and MCHPRS toggles it every that many redstone ticks, read from `<out>.server.json`. `synth.sh` turns this on when it finds an `mchprs_headless` with `--run-clock` (`$MCHPRS_HEADLESS`, or `target/release` under `$MCHPRS_DIR`, else `mchprs_headless` on `PATH`) and no clock flag is given; `--no-server-clock` or `--generate-clock <pin>` keeps the generated clock. (Default half period: from the critical path, the generated clock's period.) That check can't see the running server: if the clock stays still, rebuild and restart it, and stand on the plot, since plots only tick with a player.

   - `--place-seed <n>`: Placement seed (Default: 0). Routing is deterministic per placement, so try another seed if a build fails nets.
   - `--flat`: Puts every component on one Y level (for small designs). Output pin targets often fail to route flat: combine it with `--no-pin-extensions` for designs that have a `pin_targets.json`.
   - `--resume-dir <dir>`, `--resume-reroute <nets>`: Save the placement and routed paths and reuse them on the next build; reroute only the listed nets.
   - `--route-deadline <s>`: Stops routing after that many seconds and writes a partial result (implies `--allow-routing-failures`).
   - `--debug`: Logs progress and slow searches, saves routing snapshots to `<artifacts>/snapshots` (slow), prints A* counters.

   Run `uv run python -m minecraft_v --help` for the full list.

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

`mchprs_headless` (in [our MCHPRS fork](https://github.com/saucefishman/MCHPRS), `crates/headless`) loads a `.litematic` and runs a script against it. Build it with `cargo build --release -p mchprs_headless`; scripts find it at `$MCHPRS_HEADLESS`, else `$MCHPRS_DIR/target/release/mchprs_headless` if `MCHPRS_DIR` is set, else `mchprs_headless` on `PATH`. When it supports `--run-clock`, `synth.sh` turns on `--server-clock`. Builds are settled in Python, without it.

```sh
./synth.sh examples/counter --synth-only
uv run python scripts/build_with_pins.py pins.txt --netlist build/artifacts/netlist.json --out-litematic out.litematic --generate-clock none
mchprs_headless test out.litematic pins.txt script.txt --modes noupdate,redstone,redpiler
```

with a `script.txt` such as:

```text
settle 100
pulse clk 20
settle 100
expect count[0] 1
expect count[1] 0
```

The pin file lists `name x y z nx nz` per pin. Script commands: `settle N`, `set NAME 0|1`, `pulse NAME N`, `expect NAME 0|1`, `print LABEL`. `mchprs_headless test --add-lever NAME` adds a lever for an input that has none.

Other scripts in `scripts/`:

- `bench_examples.py`: builds several examples in parallel, each in its own artifacts directory, and prints failures and `[verify]` violations.
- `add_clock.py`: adds a generated clock to a build you've already tested, without rebuilding.
- `compare_gate_rtl.py`: checks a built schematic net by net against the netlist and RTL.
- `dev/`: acceptance checks for single designs (`check_preload.py`, `check_powerup.py`, ...), built from `examples/` and `tests/designs/`. See [`scripts/README.md`](scripts/README.md).

## Limitations

- RAM blocks support two read ports with one write port, and four read ports with two write ports, without partial (byte-enable) writes. 2-read/1-write blocks can't be preloaded (`initial`, `$readmemh`). Other memories are built from flip-flops. A RAM block input tied to constant 1 (say, a write enable that's always on) fails the build; use `--no-tiled-ram` or drive it from logic.
- `--flat` doesn't route output pin targets; use it with `--no-pin-extensions`.

## Development & Testing

Run our test suite with:

```sh
uv run python -m pytest -q tests
```

Tests that need `yosys` or `mchprs_headless` are skipped when those aren't found.
