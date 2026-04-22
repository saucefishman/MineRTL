#!/usr/bin/env bash
set -euo pipefail

mkdir -p build/artifacts

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <project-dir>" >&2
    exit 1
fi

dir="$1"

yosys -p "
read_verilog -sv ${dir}/*.v
hierarchy -top main
proc
opt
techmap -map techmap/sdff_decompose.v -map techmap/sdffe_decompose.v
dfflegalize -cell \$_DFF_P_ 0 -cell \$_DFFE_PP_ 0
techmap -map techmap/fa_map.v
techmap
write_json build/artifacts/netlist.json
stat
"

uv run python -m minecraft_v --netlist build/artifacts/netlist.json "${@:2}"
