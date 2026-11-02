"""timed.py <out dir> [build args]: a build with wall times per post-route pass."""
import os, sys, time, atexit, resource
T0 = time.monotonic()
d = sys.argv[1]
os.environ["MINECRAFT_V_ARTIFACTS_DIR"] = d
from litemapy import Schematic
from minecraft_v import main as M
from minecraft_v.placement_engine import builder as B
times = {}
ends = {}
def wrap(mod, name, label=None):
    fn = getattr(mod, name, None)
    if fn is None: return
    label = label or name
    def inner(*a, **k):
        t = time.monotonic()
        try: return fn(*a, **k)
        finally:
            times[label] = times.get(label, 0) + time.monotonic() - t
            ends[label] = time.monotonic()
    setattr(mod, name, inner)
wrap(M, "group_block"); wrap(M, "module_to_component_list")
for n in ("_place_gate_templates", "_route_all_nets", "fix_hold", "_report_flat_violations", "_report_unsupported_blocks", "_compute_critical_path", "apply_dust_shapes", "settle_blocks", "_hidden_cells", "_poisson_placement"):
    wrap(B, n)
wrap(Schematic, "save")
place_io = B._place_io_repeaters; signs = []
def capture(*a, **k):
    r = place_io(*a, **k); signs[:] = r[2]; return r
B._place_io_repeaters = capture
def report():
    end = time.monotonic()
    print("=== times")
    for k, v in sorted(times.items(), key=lambda kv: ends[kv[0]]): print(f"{k:32s}{v:8.1f}")
    if "_route_all_nets" in ends: print(f"{'after routing (wall)':32s}{end - ends['_route_all_nets']:8.1f}")
    print(f"{'total':32s}{end - T0:8.1f}  peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.1f} GB")
atexit.register(report)
sys.argv = [sys.argv[0], "--netlist", d + "/netlist.json", "--route-passes", "1", "--server-clock", "--place-seed", "1", "--out-litematic", d + "/out.litematic", *sys.argv[2:]]
M.main()
with open(d + "/pins", "w") as f:
    for (x, y, z), (nx, nz), name in signs: f.write(f"{name} {x} {y} {z} {nx} {nz}\n")
