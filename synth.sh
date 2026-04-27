#!/usr/bin/env bash
set -euo pipefail

mkdir -p build/artifacts

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <project-dir>" >&2
    exit 1
fi

dir="$1"
shift

has_output_pin_targets_flag=false
for arg in "$@"; do
    case "$arg" in
        --output-pin-targets-json|--output-pin-targets-json=*)
            has_output_pin_targets_flag=true
            break
            ;;
    esac
done

default_pin_targets_json="${dir%/}/pin_targets.json"
if [[ -f "$default_pin_targets_json" && "$has_output_pin_targets_flag" == false ]]; then
    set -- "$@" --output-pin-targets-json "$default_pin_targets_json"
fi

yosys -p "
read_verilog -sv ${dir}/*.v
hierarchy -top main
proc
memory_map
opt
techmap -map techmap/sdff_decompose.v -map techmap/sdffe_decompose.v
dfflegalize -cell \$_DFF_P_ 0 -cell \$_DFFE_PP_ 0
techmap -map techmap/fa_map.v
techmap
opt
dfflegalize -cell \$_DFF_P_ 0 -cell \$_DFFE_PP_ 0
opt_clean
write_json build/artifacts/netlist.json
stat
"

uv run python -m minecraft_v --netlist build/artifacts/netlist.json "$@"
