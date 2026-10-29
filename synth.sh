#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <project-dir>" >&2
    exit 1
fi

dir="$1"
shift

# --out-litematic-dir <d> → --out-litematic <d>/<project-dir basename>.litematic
# --artifacts-dir <d>     → netlist + build artifacts go to <d> (default build/artifacts);
#                           give each parallel build its own
args=()
out_litematic_dir=""
artifacts_dir="${MINECRAFT_V_ARTIFACTS_DIR:-build/artifacts}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --artifacts-dir)
            artifacts_dir="$2"
            shift 2
            ;;
        --artifacts-dir=*)
            artifacts_dir="${1#*=}"
            shift
            ;;
        --out-litematic-dir)
            out_litematic_dir="$2"
            shift 2
            ;;
        --out-litematic-dir=*)
            out_litematic_dir="${1#*=}"
            shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done
if [[ -n "$out_litematic_dir" ]]; then
    project_name="$(basename "${dir%/}")"
    args+=(--out-litematic "${out_litematic_dir%/}/${project_name}.litematic")
fi
set -- ${args[@]+"${args[@]}"}
artifacts_dir="${artifacts_dir%/}"
mkdir -p "$artifacts_dir"
export MINECRAFT_V_ARTIFACTS_DIR="$artifacts_dir"

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

yosys -q -p "
read_verilog -sv ${dir}/*.v
hierarchy -top main
proc
memory_map
dfflegalize -cell \$_DFF_P_ 0 -cell \$_DFFE_PP_ 0 -cell \$_DLATCH_P_ 0
techmap -map techmap/dlatch_map.v
opt
techmap -map techmap/sdff_decompose.v -map techmap/sdffe_decompose.v
techmap -map techmap/fa_map.v
techmap
dfflegalize -cell \$_DFF_P_ 0 -cell \$_DFFE_PP_ 0 -cell \$_DLATCH_P_ 0
techmap -map techmap/dffe_pn_decompose.v
opt -nosdff
techmap -map techmap/sdff_decompose.v -map techmap/sdffe_decompose.v
techmap -map techmap/dffe_pn_decompose.v
opt_clean
write_json ${artifacts_dir}/netlist.json
stat
"

uv run python -m minecraft_v --netlist "${artifacts_dir}/netlist.json" "$@"
