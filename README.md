![image](assets/logo.png)

# MineRTL: Verilog to Minecraft Redstone Synthesis

**MineRTL** is a hardware synthesis toolchain that takes standard Verilog HDL code and compiles it into functional 3D Minecraft redstone circuits.

## Features

- **Full Synthesis Pipeline:** Parses, elaborates, and optimizes Verilog using Yosys.
- **Custom Cell Library:** Standard logic gates (AND, OR, XOR, NOT) and D-Flip Flops seamlessly implemented in Minecraft.
- **Automated 3D Routing:** Custom layout engine that handles gate placement and 3D redstone wiring.
- **Litematica Integration:** Outputs `.litematic` files for easy importing into Minecraft worlds or fast simulation via MCHPRS.

## Prerequisites

1. Install [yosys](https://github.com/yosyshq/yosys) (Required for Verilog synthesis)
2. Install [uv](https://github.com/astral-sh/uv) (Fast Python package manager)

## Getting Started

1. **Install Python dependencies:**

```sh
uv sync
```

2. **Generate a schematic:**
   We provide several examples in the `examples/` directory (e.g., `counter`, `full_adder`, `helloworld`). Synthesize the counter example with:

```sh
./synth.sh examples/counter
```

The resulting schematic will be saved to `build/result.litematic`. You can import this directly into a Minecraft world using the Litematica mod.

3. **Synthesis Flags & Configuration:**
   The `synth.sh` script automatically wraps our Python-based layout engine (`main.py`) alongside Yosys. It supports several flags that you can pass directly:
   - `--out-litematic <path>`: Specifies custom output path for the `.litematic` file (Default: `build/result.litematic`).
   - `--module <name>`: Sets the top-level Verilog module to synthesize (Default: `main`).
   - `--schematics-dir <dir>`: Sets the directory containing our `.litematic` cell library objects (Default: `schematics`).
   - `--schematic-name <name>`: The name attached to the generated schematic in-game (Default: matches `--module`).
   - `--allow-routing-failures True`: Keeps generating the structure even if some 3D routes fail to connect (useful for debugging physical placement).
   - `--output-pin-targets-json <file.json>`: Path to a JSON file explicitly dictating output coordinates. Note: `synth.sh` will _automatically_ use `pin_targets.json` if it finds it in your specific project directory.

   Example usage:

```sh
./synth.sh examples/counter --module my_counter --out-litematic "counter.litematic" --allow-routing-failures True
```

## Development & Testing

Run our test suite with:

```
uv run pytest
```
