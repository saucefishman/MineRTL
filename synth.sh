#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <project-dir>" >&2
    exit 1
fi

dir="$1"

yosys -p "
read_verilog -sv ${dir}/*.v
hierarchy -top main
proc
flatten
opt
memory
opt
write_json netlist.json
stat
"
