![image](assets/logo.png)

#### Prerequisites
1. Install [yosys](https://github.com/yosyshq/yosys)
2. Install [uv](https://github.com/astral-sh/uv)
#### Getting started

1. Install Python dependencies
```sh
uv sync
```
2. Generate the schematic with
```sh
./synth.sh examples/counter
```
The resulting schematic will be in `build/result.litematic`.

You can change the output location by passing the --out-litematic flag to the synth.sh script, for example:
```sh 
./synth.sh examples/counter --out-litematic "/Users/saucefishman/Documents/curseforge/minecraft/Instances/mchprs fabric/schematics/counter.litematic"
```


### Development
#### Testing

Run tests with
```sh
uv run pytest
```
