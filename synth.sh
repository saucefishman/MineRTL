#!/usr/bin/env bash
set -euo pipefail
# The build's start, for the summary line at the end (bash 3.2 has no EPOCHREALTIME).
MINECRAFT_V_START="$(perl -MTime::HiRes=time -e 'printf "%.3f", time' 2>/dev/null || date +%s)"
export MINECRAFT_V_START

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # techmap/ and the Python project, wherever this runs from

if [[ $# -lt 1 || "$1" == -h || "$1" == --help ]]; then
    cat <<EOF
Usage: $0 <project-dir> [options]

Synthesizes the Verilog in <project-dir> (top module main) and builds build/result.litematic.

Synthesis options:
  --out-litematic-dir DIR
                        Write DIR/<project-dir name>.litematic
  --artifacts-dir DIR   Where the netlist and build files go (default: build/artifacts)
  --synth-only          Stop after writing the netlist
  --no-tiled-ram        Build memories from flip-flops instead of RAM blocks
  --no-synth-cache      Run yosys even if build/synth_cache has this netlist (same sources, script and yosys)

Build options:
EOF
    # minecraft_v's own options, minus -h and --netlist, which this script fills in
    uv run --project "$here" python -m minecraft_v --help | sed '1,/^options:/d' \
        | awk '/^  -/ { skip = ($1 == "-h," || $1 == "--netlist") } !skip'
    [[ $# -ge 1 ]]; exit
fi

dir="$1"
shift

# --out-litematic-dir <d> → --out-litematic <d>/<project-dir basename>.litematic
# --artifacts-dir <d>     → netlist + build artifacts go to <d> (default build/artifacts);
#                           give each parallel build its own
args=()
out_litematic_dir=""
out_litematic="build/result.litematic"
synth_only=false
use_synth_cache=true
synth_cache_dir="build/synth_cache"
debug_prints=true
tiled_ram=true
top=main
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
        --tiled-ram)  # memories with a supported port layout become tiled arrays (the default)
            tiled_ram=true
            shift
            ;;
        --no-tiled-ram)  # every memory becomes flip-flops and muxes; also skips the ABC pass below
            tiled_ram=false
            shift
            ;;
        --no-synth-cache)  # run yosys even when build/synth_cache has the netlist
            use_synth_cache=false
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
        --no-debug-prints)  # also passed on to minecraft_v
            debug_prints=false
            args+=("$1")
            shift
            ;;
        --module)  # also passed on to minecraft_v
            top="$2"
            args+=("$1" "$2")
            shift 2
            ;;
        --module=*)
            top="${1#*=}"
            args+=("$1")
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
if [[ -n "${MCHPRS_HEADLESS:-}" ]]; then
    headless="$MCHPRS_HEADLESS"
elif [[ -n "${MCHPRS_DIR:-}" ]]; then
    headless="$MCHPRS_DIR/target/release/mchprs_headless"
else
    headless="$(command -v mchprs_headless || true)"
fi
has_clock_flag=false
for arg in "$@"; do
    case "$arg" in
        --server-clock|--server-clock=*|--no-server-clock|--generate-clock|--generate-clock=*)
            has_clock_flag=true
            break
            ;;
    esac
done
if [[ -n "$headless" && -x "$headless" && "$has_clock_flag" == false ]] && [[ "$("$headless" test --help 2>/dev/null || true)" == *--run-clock* ]]; then
    set -- "$@" --server-clock
    echo "[clock] MCHPRS drives the clock. If it stays still: rebuild and restart the server, and stand on the plot (plots only tick with a player)." >&2
fi

# Spinner with the current yosys pass, until killed.
yosys_log="${artifacts_dir}/yosys.log"
yosys_err="${artifacts_dir}/yosys.err"
yosys_spinner() {
    local frames=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏) i=0 start=$SECONDS pass= latest
    while true; do
        # Only the log's end: a multi-GB netlist's log is tens of MB, and grepping all of it every 0.1 s costs a core.
        # A pass header further back than that is not found, so the last one seen stays.
        latest="$(tail -c 65536 "$yosys_log" 2>/dev/null | grep -E '^[0-9]+\. Executing' | tail -n 1 | cut -c1-60 || true)"
        [[ -z "$latest" ]] || pass="$latest"
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
if [[ "$debug_prints" == false ]]; then
    strip_debug='delete t:$print t:$check'
fi
map_tiled_ram=""
if [[ "$tiled_ram" == true ]]; then
    # Index and width rewrites first: they remove most barrel shifters from register arrays.
    map_tiled_ram="wreduce
opt_expr -fine
opt_clean
peepopt
opt -full -purge
wreduce
opt_expr -fine
opt_clean
peepopt
opt -full -purge
memory_collect
techmap -map \"${here}/techmap/ram_2r1w.v\" -map \"${here}/techmap/ram_4r2w.v\" -map \"${here}/techmap/ram_rom.v\" t:\$mem_v2"
fi
abc_step=""
if [[ "$tiled_ram" == true ]]; then
    # A full adder with a tied input folds to gates first, then &syn2 restructures: 18% fewer nets, about half the gate delay.
    abc_step="techmap -map \"${here}/techmap/fa_const.v\"
techmap
opt -nosdff
opt_clean
abc -g AND,OR,XOR,MUX -script +strash;&get,-n;&syn2;&dch,-f;&nf;&put
opt_clean"
fi
yosys_script="
read_verilog -sv \"${dir}/*.v\"
read_verilog -lib \"${here}/techmap/ram_cells.v\"
hierarchy -check -top ${top}
flatten -noscopeinfo
proc
${strip_debug}
${map_tiled_ram}
memory_map
dfflegalize -cell \$_DFF_P_ 01 -cell \$_DFFE_PP_ 01 -cell \$_DLATCH_P_ 0
techmap -map \"${here}/techmap/dlatch_map.v\"
opt
techmap -map \"${here}/techmap/sdff_decompose.v\" -map \"${here}/techmap/sdffe_decompose.v\"
techmap -map \"${here}/techmap/fa_map.v\"
techmap
dfflegalize -cell \$_DFF_P_ 01 -cell \$_DFFE_PP_ 01 -cell \$_DLATCH_P_ 0
techmap -map \"${here}/techmap/dffe_pn_decompose.v\"
opt -nosdff
techmap -map \"${here}/techmap/sdff_decompose.v\" -map \"${here}/techmap/sdffe_decompose.v\"
techmap -map \"${here}/techmap/dffe_pn_decompose.v\"
opt_clean
${abc_step}"

# Names in the netlist carry the paths yosys read files by and, for $write cells, a memory address. Without them the
# netlist is the same from any checkout and any run, and so is the build, as names feed hashing.
unplace() {
    DIR="${dir%/}/" HERE="${here}/" perl -p -e 's/\Q$ENV{DIR}\E//g; s/\Q$ENV{HERE}\E//g; s/\$0x[0-9a-f]+:/\$:/g' "$@"
}

# The same sources, script, yosys and abc give the same netlist: keep it, keyed by all of them. Sources that pull in
# other files (`include, $readmem) aren't cached, as those files aren't in the key.
synth_cache=""
if [[ "$use_synth_cache" == true ]] && ! grep -qE '`include[[:space:]]|\$readmem[hb][[:space:]]*\(' "${dir%/}"/*.v 2>/dev/null; then
    if command -v sha256sum >/dev/null; then sha=(sha256sum); else sha=(shasum -a 256); fi
    yosys_bin="$(command -v yosys)"
    datdir="$(yosys-config --datdir 2>/dev/null || echo "$(dirname "$yosys_bin")/../share/yosys")"  # its own techmap.v
    # The binaries' hash, kept by their size and modification time: reading 40 MB of them took longer than the rest.
    bins=("$yosys_bin" "$(dirname "$yosys_bin")/yosys-abc")
    stamp="$(stat -L -c '%n %s %Y' "${bins[@]}" 2>/dev/null || stat -L -f '%N %z %Fm' "${bins[@]}" 2>/dev/null || true)"  # GNU, BSD
    stamp_file="${synth_cache_dir%/}/binaries/$(printf '%s\n' "$stamp" | "${sha[@]}" | cut -c1-24)"
    if [[ -n "$stamp" && -s "$stamp_file" ]]; then
        bin_hash="$(cat "$stamp_file")"
    else
        bin_hash="$(cat "${bins[@]}" 2>/dev/null | "${sha[@]}" | cut -c1-64)"
        if [[ -n "$stamp" ]]; then
            mkdir -p "${stamp_file%/*}"
            printf '%s\n' "$bin_hash" > "$stamp_file.tmp$$" && mv "$stamp_file.tmp$$" "$stamp_file"
        fi
    fi
    key="$( { yosys -V; printf '%s\n' "$bin_hash"; cat "$datdir"/*.v 2>/dev/null;
              cat "${BASH_SOURCE[0]}";
              printf '%s\n' "$yosys_script" | unplace; for f in "${dir%/}"/*.v "${here}"/techmap/*.v; do
                  printf '%s\n' "${f##*/}"; cat "$f"; done; } | "${sha[@]}" | cut -c1-24)"
    synth_cache="${synth_cache_dir%/}/${key}"
fi
# Entries are gzipped (a netlist shrinks ~15x). The newest stay: a hit touches its entry, and a store drops the oldest
# entries while the cache is over MINECRAFT_V_SYNTH_CACHE_MAX_GB (default 5), keeping the one just used.
if command -v pigz >/dev/null; then gz=(pigz -1); else gz=(gzip -1); fi
evict_synth_cache() {
    local max_kb used_kb=0 entry kb sizes=() entries=()
    max_kb="$(awk -v gb="${MINECRAFT_V_SYNTH_CACHE_MAX_GB:-5}" 'BEGIN { printf "%d", gb * 1048576 }')"
    while IFS= read -r entry; do  # oldest first
        entry="${entry%/}"
        case "$entry" in */binaries|*.tmp*) continue ;; esac
        kb="$(du -sk "$entry" | cut -f1)"
        entries+=("$entry")
        sizes+=("$kb")
        used_kb=$((used_kb + kb))
    done < <(ls -1dtr "${synth_cache_dir%/}"/*/ 2>/dev/null)
    local i
    for ((i = 0; i < ${#entries[@]} - 1 && used_kb > max_kb; i++)); do
        rm -rf "${entries[i]}"
        used_kb=$((used_kb - sizes[i]))
    done
}
if [[ -n "$synth_cache" && -f "$synth_cache/netlist.json.gz" ]]; then
    stop_spinner
    touch "$synth_cache"
    "${gz[0]}" -dc "$synth_cache/netlist.json.gz" > "$artifacts_dir/netlist.json"
    "${gz[0]}" -dc "$synth_cache/yosys.log.gz" > "$yosys_log"
    cp "$synth_cache/yosys.err" "$artifacts_dir/"
    echo "[yosys] netlist from $synth_cache (--no-synth-cache to run yosys)" >&2
    cat "$yosys_err"
else
    # yosys writes the netlist into a pipe that unplace reads: it is on disk once, not as a raw file and a rewritten copy.
    netlist_pipe="${artifacts_dir}/netlist.pipe"
    rm -f "$netlist_pipe"
    mkfifo "$netlist_pipe"
    unplace "$netlist_pipe" > "${artifacts_dir}/netlist.json.tmp" &
    unplace_pid=$!
    yosys_status=0
    yosys -q -l "$yosys_log" -p "${yosys_script}
write_json \"${netlist_pipe}\"
stat
" > "$yosys_err" 2>&1 || yosys_status=$?
    stop_spinner
    cat "$yosys_err"
    exec 3<>"$netlist_pipe" 3>&-  # unplace is still waiting to open the pipe if yosys failed before write_json: end its input
    unplace_status=0
    wait "$unplace_pid" || unplace_status=$?
    rm -f "$netlist_pipe"
    if [[ $yosys_status -ne 0 ]]; then
        rm -f "${artifacts_dir}/netlist.json.tmp"
        exit "$yosys_status"
    fi
    if [[ $unplace_status -ne 0 ]]; then
        exit "$unplace_status"
    fi
    mv "${artifacts_dir}/netlist.json.tmp" "${artifacts_dir}/netlist.json"
    if [[ -n "$synth_cache" ]]; then
        mkdir -p "$synth_cache.tmp$$"
        "${gz[@]}" -c "$artifacts_dir/netlist.json" > "$synth_cache.tmp$$/netlist.json.gz"
        "${gz[@]}" -c "$yosys_log" > "$synth_cache.tmp$$/yosys.log.gz"
        cp "$yosys_err" "$synth_cache.tmp$$/"
        if [[ -e "$synth_cache" ]] || ! mv "$synth_cache.tmp$$" "$synth_cache" 2>/dev/null; then
            rm -rf "$synth_cache.tmp$$"  # another build got there first
        fi
        evict_synth_cache
    fi
fi

if [[ "$synth_only" == true ]]; then
    exit 0
fi

uv run --project "$here" python -m minecraft_v --netlist "${artifacts_dir}/netlist.json" "$@"
