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
out_litematic="build/result.litematic"
settle=true
synth_only=false
debug_prints=auto
macros=auto
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
        --synth-only)  # stop after writing the netlist
            synth_only=true
            shift
            ;;
        --debug-prints)  # also passed on to minecraft_v
            debug_prints=true
            args+=("$1")
            shift
            ;;
        --no-debug-prints)
            debug_prints=false
            args+=("$1")
            shift
            ;;
        --macros)  # on by default; prints can't tap inside macros, so they're off unless asked for
            macros=true
            args+=("$1")
            shift
            ;;
        --no-macros)
            macros=false
            args+=("$1")
            shift
            ;;
        --no-settle)
            settle=false
            shift
            ;;
        --out-litematic)
            out_litematic="$2"
            args+=("$1" "$2")
            shift 2
            ;;
        --out-litematic=*)
            out_litematic="${1#*=}"
            args+=("$1")
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
    out_litematic="${out_litematic_dir%/}/${project_name}.litematic"
    args+=(--out-litematic "$out_litematic")
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

# An MCHPRS with clock support can drive the clock, which saves the generated clock's space. An explicit clock flag wins.
headless="${MCHPRS_HEADLESS:-${MCHPRS_DIR:-$HOME/Projects/MCHPRS}/target/release/mchprs_headless}"
has_clock_flag=false
for arg in "$@"; do
    case "$arg" in
        --server-clock|--server-clock=*|--no-server-clock|--generate-clock|--generate-clock=*)
            has_clock_flag=true
            break
            ;;
    esac
done
if [[ -x "$headless" && "$has_clock_flag" == false ]] && [[ "$("$headless" test --help 2>/dev/null || true)" == *--run-clock* ]]; then
    set -- "$@" --server-clock
    echo "[clock] MCHPRS drives the clock. If it stays still: rebuild and restart the server, and stand on the plot (plots only tick with a player)." >&2
fi

# Spinner with the current yosys pass, until killed.
yosys_log="${artifacts_dir}/yosys.log"
yosys_err="${artifacts_dir}/yosys.err"
yosys_spinner() {
    local frames=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏) i=0 start=$SECONDS pass
    while true; do
        pass="$(grep -E '^[0-9]+\. Executing' "$yosys_log" 2>/dev/null | tail -n 1 | cut -c1-60 || true)"
        printf '\r%s [yosys] %ds %s\033[K' "${frames[i++ % 10]}" "$((SECONDS - start))" "$pass" >&2
        sleep 0.1
    done
}
spinner_pid=""
stop_spinner() {
    if [[ -n "$spinner_pid" ]]; then
        kill "$spinner_pid" 2>/dev/null || true
        wait "$spinner_pid" 2>/dev/null || true
        spinner_pid=""
        printf '\r\033[K' >&2
    fi
}
trap stop_spinner EXIT
: > "$yosys_log"
if [[ -t 2 ]]; then
    yosys_spinner &
    spinner_pid=$!
fi

# Prints keep whatever logic they read, so builds without them drop that too. Asserts never build.
strip_debug='delete t:$check'
if [[ "$debug_prints" == false || ( "$debug_prints" == auto && "$macros" != false ) ]]; then
    strip_debug='delete t:$print t:$check'
fi
yosys_status=0
yosys -q -l "$yosys_log" -p "
read_verilog -sv ${dir}/*.v
hierarchy -check -top main
flatten -noscopeinfo
proc
${strip_debug}
memory_map
dfflegalize -cell \$_DFF_P_ 01 -cell \$_DFFE_PP_ 01 -cell \$_DLATCH_P_ 0
techmap -map techmap/dlatch_map.v
opt
techmap -map techmap/sdff_decompose.v -map techmap/sdffe_decompose.v
techmap -map techmap/fa_map.v
techmap
dfflegalize -cell \$_DFF_P_ 01 -cell \$_DFFE_PP_ 01 -cell \$_DLATCH_P_ 0
techmap -map techmap/dffe_pn_decompose.v
opt -nosdff
techmap -map techmap/sdff_decompose.v -map techmap/sdffe_decompose.v
techmap -map techmap/dffe_pn_decompose.v
opt_clean
write_json ${artifacts_dir}/netlist.json
stat
" > "$yosys_err" 2>&1 || yosys_status=$?
stop_spinner
cat "$yosys_err"
if [[ $yosys_status -ne 0 ]]; then
    exit "$yosys_status"
fi

if [[ "$synth_only" == true ]]; then
    exit 0
fi

uv run python -m minecraft_v --netlist "${artifacts_dir}/netlist.json" "$@"

# Paste already settled, so pulses while blocks settle can't clock flip-flops (--no-settle skips it).
# Optional: needs MCHPRS's mchprs_headless.
if [[ "$settle" == true ]]; then
    if [[ -x "$headless" ]]; then
        MCHPRS_HEADLESS="$headless" uv run python scripts/settle_schematic.py "$out_litematic" "${artifacts_dir}/component_layout.json" "$out_litematic"
    else
        echo "[settle] skipped: no mchprs_headless at $headless. The build works, but flip-flops can lose their values at power-up." >&2
    fi
fi
