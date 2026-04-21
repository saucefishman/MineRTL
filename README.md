1. Install [yosys](https://github.com/yosyshq/yosys)
2. Install [uv](https://github.com/astral-sh/uv)
3. Install Python dependencies
```sh
uv sync
```
4. Generate the netlist with
```sh
./synth.sh examples/counter
```
