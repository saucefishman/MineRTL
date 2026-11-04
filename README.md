![MineRTL logo](assets/logo.png)

# MineRTL: Verilog to Minecraft Redstone Synthesis

**MineRTL** turns circuits written in Verilog, the text language used to design real chips, into working Minecraft redstone. You describe what the circuit does, and MineRTL designs the redstone, lays it out in 3D and wires it.

The result is a `.litematic` schematic. Paste it into your world with the [Litematica](https://www.curseforge.com/minecraft/mc-mods/litematica) mod, or load it into [MCHPRS](https://github.com/saucefishman/MCHPRS), a Minecraft server built for redstone that runs it far faster than the normal game.

A 4-bit counter builds in about a second; a basic CPU in minutes.

## Features

- **Write Verilog, get redstone:** Describe a circuit in Verilog and get a redstone build that does the same thing, ready to paste with Litematica or load into MCHPRS. Inputs become levers and outputs become lamps.
- **Outputs where you want them:** Place output lamps as a display, such as a seven-segment digit or a screen (`pin_targets.json`).
- **Memory with data in it:** Memories can start out filled from a file with `$readmemh`, for lookup tables or a program.
- **Clock included:** The build comes with a redstone clock that's slow enough for the circuit, so you don't have to work out the timing.
- **Debug prints:** `$display` messages from your Verilog show up in chat when you run the build in MCHPRS.
- **Test without the game:** Script inputs and check outputs from the command line with `mchprs_headless`.
- **Scale:** Builds a complete basic CPU in minutes.

## Prerequisites

macOS or Linux (on Windows, use WSL). Builds target Minecraft 1.20.4.

1. [Yosys](https://github.com/yosyshq/yosys) (developed against 0.64)
2. [uv](https://github.com/astral-sh/uv), a Python package manager; it fetches Python 3.13+ if needed
3. A C++17 compiler (Xcode command line tools, GCC or Clang): part of the router is compiled on install
4. To play the build: Minecraft 1.20.4 with [Litematica](https://www.curseforge.com/minecraft/mc-mods/litematica) (a Fabric mod; it also needs MaLiLib), or [our MCHPRS fork](https://github.com/saucefishman/MCHPRS) (see [Running Builds in MCHPRS](#running-builds-in-mchprs))
5. Optional: [sv2v](https://github.com/zachjs/sv2v), to convert SystemVerilog designs to Verilog first

## Quick Start

1. **Install the Python dependencies** (this also compiles the router):

```sh
uv sync
```

2. **Build the counter example:**

```sh
./synth.sh examples/counter
```

This takes about a second and writes `build/result.litematic`, about 35×32×27 blocks.

3. **Paste it into a world:** copy `build/result.litematic` into `.minecraft/schematics/`. In a creative world with open flat ground, press `M` for the Litematica menu, choose *Load Schematics*, pick `result`, and load it. Move the outline where you want it, then paste it with Litematica's paste tool mode.

4. **Watch it run:** four lamps at the edge of the build (outputs `count[0]` to `count[3]`) count up in binary. The clock is built in and starts on its own. Inputs, when a design has any, are levers on stone blocks.

> If you've installed the MCHPRS fork's `mchprs_headless` (below), `synth.sh` leaves the clock out, so that MCHPRS can drive it, and the counter won't count in normal Minecraft. Add `--no-server-clock` to build a clock in.

[`examples/README.md`](examples/README.md) lists the other examples, from `helloworld` (one gate) to `video` (a frame buffer of over 1,100 wires that takes a few minutes).

## Your Own Design

Make a directory with your `.v` files, where the top module is called `main` (or pass `--module <name>`), and build it:

```sh
./synth.sh path/to/mydesign
```

The design has to be synthesizable Verilog: no `#` delays, and `initial` only for starting values of registers and memories. Its inputs and outputs become levers and lamps. A clock input named `clk` or `_clk` gets the built-in clock. If the directory has a `pin_targets.json`, outputs are routed to the positions it lists (see `examples/sevenseg`).

**If a build fails:** the build prints lines starting with `[verify]` for wires it couldn't connect or that interfere. Routing depends on where gates were placed, so the first thing to try is another placement: `--place-seed 1`, `--place-seed 2`, and so on.

## Build Options

Pass these to `synth.sh`:

- `--out-litematic <path>`: where to write the schematic (default: `build/result.litematic`). `--out-litematic-dir <dir>` writes `<dir>/<design dir name>.litematic` instead.
- `--module <name>`: the top-level module (default: `main`). Everything under it is flattened into one build.
- `--schematic-name <name>`: the name Litematica shows (default: the module name).
- `--place-seed <n>`: placement seed (default: 0). Try another one if a build fails to connect wires.
- `--generate-clock <pin|auto|none>`: which input gets the built-in clock (default: `auto`, which picks `clk` or `_clk`).
- `--server-clock` / `--no-server-clock`: leave the clock out and let MCHPRS toggle the clock lever instead. On by default when `synth.sh` finds the fork's `mchprs_headless`. Large designs need a long clock, which is big to build.
- `--no-tiled-ram`: build memories from flip-flops instead of compact RAM blocks.
- `--no-macros`: wire every gate on its own instead of copying repeated parts. Copies take more room, so a design can come out smaller without them; memory-heavy designs often route better with `--no-macros`.
- `--no-debug-prints`: drop `$display`/`$write`. By default they're kept when they're in a clocked block: the blocks they read stand on or hang from lime concrete, and a `<out>.server.json` next to the schematic tells MCHPRS what to print. Non-constant `$sformatf` isn't supported by Yosys; use `$display` directly.
- `--no-pin-extensions`: ignore `pin_targets.json`. `--output-pin-targets-json <file>` uses another file.
- `--flat`: put every gate on one level, for small designs.

Developer options (`--synth-only`, `--artifacts-dir`, `--resume-dir`, `--route-deadline`, `--debug`, `--no-settle`, `--allow-routing-failures`) are under [Development](#development). Run `./synth.sh --help` for every option.

## Running Builds in MCHPRS

We recommend running large builds in [our MCHPRS fork](https://github.com/saucefishman/MCHPRS) instead of normal Minecraft. MCHPRS is a redstone-only server, and its redpiler compiler runs large circuits many times faster. Upstream [MCHPRS](https://github.com/MCHPR/MCHPRS) works too, but only the fork reads the `<out>.server.json` each build writes next to its `.litematic`, which adds:

- **Server clock:** the server toggles the clock lever itself, so the build doesn't need a built-in clock.
- **Debug prints:** `$display`/`$write` output appears in chat and the server log.
- **`/lever <input> [on|off|toggle]`:** sets inputs by their Verilog name.

To load a build, copy the `.litematic` and its `.server.json` into the server's `schems/` folder, stand on a plot and run `/load <name>.litematic -c`, or `/watch <name>.litematic -c` to re-paste on every rebuild. (WorldEdit's `//load` pastes the blocks only, without the server file.) `/rtps halfclock` runs large builds much faster. If the server clock doesn't move, rebuild, restart the server, and stand on the plot: plots only run with a player on them. See the fork's README for setup and the full command list.

## Testing Builds Without the Game

`mchprs_headless` (in [our MCHPRS fork](https://github.com/saucefishman/MCHPRS), `crates/headless`) loads a `.litematic`, sets inputs and checks outputs from a script. Build it with `cargo build --release -p mchprs_headless`. Scripts look for it at `$MCHPRS_HEADLESS`, else `$MCHPRS_DIR/target/release/mchprs_headless`, else `mchprs_headless` on `PATH`. Builds work without it.

```sh
./synth.sh examples/counter --synth-only
uv run python scripts/build_with_pins.py pins.txt --netlist build/artifacts/netlist.json --out-litematic out.litematic --generate-clock none
mchprs_headless test out.litematic pins.txt script.txt --modes noupdate,redstone,redpiler
```

The first command stops after Yosys and writes the gates to `build/artifacts/netlist.json`. The second builds the schematic with a lever on `clk` instead of a clock, and writes `pins.txt` with the position of each input and output. You write `script.txt`, for example:

```text
settle 100
pulse clk 20
settle 100
expect count[0] 1
expect count[1] 0
```

Script commands: `settle N` (run N ticks), `set NAME 0|1`, `pulse NAME N`, `expect NAME 0|1`, `print LABEL`. `mchprs_headless test --add-lever NAME` adds a lever for an input that has none.

## Limitations

- Memories become RAM blocks only with two read ports and one write port, or four read ports and two write ports, and without partial (byte-enable) writes. Other memories are built from flip-flops. 2-read/1-write blocks can't start out filled. A RAM input tied to constant 1 (say, a write enable that's always on) fails the build; use `--no-tiled-ram` or drive it from logic.
- `--flat` doesn't route output pin targets; use it with `--no-pin-extensions`.

## Development

Run the test suite with:

```sh
uv run python -m pytest -q tests
```

Tests that need `yosys` or `mchprs_headless` are skipped when those aren't found. `uv run` recompiles the router when a `.pyx` file changes.

Developer build options:

- `--synth-only`: stop after writing the netlist.
- `--artifacts-dir <dir>`: where the netlist and build files go (default: `build/artifacts`, or `$MINECRAFT_V_ARTIFACTS_DIR`). Give each build its own to run builds in parallel.
- `--resume-dir <dir>`, `--resume-reroute <nets>`: save the placement and wires and reuse them on the next build, rerouting only the listed nets.
- `--route-deadline <s>`: stop routing after that many seconds and write a partial result.
- `--allow-routing-failures`: write the schematic even when some wires fail.
- `--no-settle`: save blocks as built instead of in the states they settle to. Without settling, the pulses at paste time can overwrite flip-flops.
- `--debug`: log progress and slow searches, save routing snapshots to `<artifacts>/snapshots` (slow), and print router counters.

Scripts in `scripts/` ([`scripts/README.md`](scripts/README.md) has the full list):

- `bench_examples.py`: builds several examples in parallel and prints failures and `[verify]` lines.
- `add_clock.py`: adds a built-in clock to a build you've already tested, without rebuilding.
- `compare_gate_rtl.py`: checks a built schematic wire by wire against the netlist and the Verilog.
- `dev/`: acceptance checks for single designs, built from `examples/` and `tests/designs/`.
