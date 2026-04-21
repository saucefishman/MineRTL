#### Prerequisites
1. Install [yosys](https://github.com/yosyshq/yosys)
2. Install [uv](https://github.com/astral-sh/uv)
#### Getting started

1. Install Python dependencies
```sh
uv sync
```
2. Generate the netlist with
```sh
./synth.sh examples/counter
```
The netlist will be generated in `build/artifacts/netlist.json`.


### Development
#### Testing

Run tests with
```sh
uv run pytest
```
