# Large designs: scaling synth + build

Branch `claude/verilog-synthesis-memory-disk-b9eff2`. Machine: 16 GB RAM, ~30 GB free disk.

## Findings summary

Goal (user, 2026-11-07; time dropped 2026-11-08):
- cpu x512 (~150K cells): full synth + build under 12 GB peak, 0 routing failures, on 3 seeds. No time limit, but no
  significant time regression.
- A 1 GB netlist through synth + load under 12 GB.
- Memory and disk only. Time work only as a side effect.

Done so far (all on this branch, unmerged):
- 3e5d482: lean netlist load (20x -> 8.5x netlist bytes), gzipped capped synth cache, fifo yosys -> unplace, tail-only
  spinner. Output identical.
- 7601f17: legalizer pair array 48 B -> 1 B per pair. Output identical. cpu x64 3.4 GB (was killed at 16 GB).
- 1ef06ef: legalizer sub-quadratic (neighbour grid + overlap re-check). 1M boxes 208 s / 2.3 GB. Placement changes
  slightly; cpu x128 routing failures 3 -> 4 (one seed).

Current numbers (one seed, cpu xN = N copies of examples/cpu's core, scratch generator `gen_cpu.py`):

| design | cells | peak RSS | wall | routing failures |
|---|---|---|---|---|
| cpu x32 | 9.4K | ~2.5 GB | 112 s | 0 |
| cpu x128 | 37.6K | 5.5 GB | 875 s | 4 (needs --allow-routing-failures) |
| cpu x512 | ~150K | not run | | |

Known limits: yosys ~12.7x netlist bytes (16 GB at ~1.2 GB netlist); Python load ~8.5x; layout stamp ~70 B per world
cell; router A* ~1 GB at x32; `_compact` O(n^2) time.

## User's rules

- Done = cpu x512 under 12 GB, 0 routing failures, 3 seeds; plus 1 GB netlist synth + load under 12 GB.
- "do not focus on time, that is a separate matter. if time improvement is a side effect of memory improvements, ok,
  but otherwise do not consider time (as long there is no significant regression in time)" (2026-11-08)
- May change: placement (if routing failures and wire length don't worsen), defaults, add dependencies, synth flow.
- Git: commit to this branch only; no merge to main, no push until asked.
- "why do you need to evaluate like this? can't you map out O of the various steps" (2026-11-07): derive complexity
  from the code first; measure only to fit constants and check the map.
- Standing (memory): all builds via `~/.cache/mcv-jobs/job.py`; one seed while iterating; Opus plans, Sonnet codes
  (kept-alive workers); never idle when SPARE.

## State log

- 2026-11-07: synth/load scaling measured (synthetic + cpu xN); fixes 3e5d482, 7601f17, 1ef06ef committed.
- 2026-11-08 cpu x256 (in progress, 9b672d6): repeats cover 19956/74595 gates (27%) -> no macros, 75K flat; global
  place 23 s, legalize+compact ~5.6 min, world 539x601x546 (177M); wire >28 min, mostly net_2 (clock?) hitting the
  5000-node cap per sink and ripping; RSS only ~1.6 GB. Time, not memory, is the x256 wall. worker3 investigating
  why repeats don't form.
- 2026-11-08 x256 killed at ~40 min (9G reservation blocked 5 queued jobs). net_2 = clk, 6,400 sinks (next largest
  18), 32 node-cap failures in 8 min, each ripping 4-8 nets. Fits from x32/x64/x128 (time exponent): global place 0.5,
  legalize+compact 1.76, layout+route 1.43 (x256 already over its prediction), hold+ext+verify 2.08. x512 estimate
  >=98 min (wire >=65, hold/verify ~20, legalize+compact ~17); peak RSS ~5 GB. Time is the wall, not memory.
  Clock tree agent started (branch claude/clock-tree).

## Complexity map (from code, 2026-11-08; constants still to fit)

Variables: n components, m nets, p pins, k fanout, V world volume (~825 blocks per gate when flat), E A* expansions.
- Macros decide scale: cpu xN has one ROM per core, so macros are single tiles; if find_repeated covers <50% of gates
  (or hits its 30 s cap, uncached), all 150K go flat. Hold and verify always run on the flat netlist.
- Super-linear: `_compact` O(n^2) x6 calls (being fixed); refine/mirror `Nets::of` O(sum k^2) (clock net);
  `_TreeCells.nearest` O(k*T) per net; `_pin_side_pass` O(n*W); `_wall_vias` O(m*W^2) when nets cross walls;
  `HiddenBoxes.__contains__` O(Q*R); hold margins O(B*(n+p)); yosys `&dch` on one flat AIG.
- V-dense: Region uint32 4 B, CellGrid uint16 2 B, mask 1 B, verify/hold grids 1 B each, write ~3-4 B transient.
- Python-object heavy: wire cells (solid, dust_owner, ripup journal, rec.writer, paths, laid) and gate_cells frozensets
  (~150 B per cell); verify fork allocates its own owner dict + hidden set (not shared pages).

Predicted break order at x512 flat: routing-state memory > verify fork copy > `_compact` time > V arrays > clock k^2.

**Correction 2026-11-08:** RSS undercounts here (memory compressor). cpu x512 mid-routing: RSS 1.3 GB, physical
footprint 11.6 GB, peak 12.4 GB (`vmmap --summary`). All earlier "peak RSS" numbers are lower bounds under memory
pressure; the ~5 GB x512 estimate was wrong. Goal is judged on physical footprint.
- 2026-11-08 cpu x512 acceptance (base 9a92dcc): KILLED by job.py at 14.0 GB footprint in routing, 4349 s. No macros
  (13% coverage), 150,896 components, world 694x784x704 (383M, density 0.47). Place 1744 s (old _compact). At 12.3 GB:
  MALLOC_LARGE 5.8 + VM_ALLOCATE 4.4 + LARGE_REUSABLE 0.8 + small/medium 0.9 GB -> big buffers (~27 B/world cell)
  dominate, not small Python objects. Artifacts 290 MB (pre-gzip). Goal not met.

## TODO (ranked, memory/disk only)

0. Routing-phase memory at x512 (14+ GB). x32 inventory (worker2): world-sized arrays only ~7 B/world cell (Region
   uint32 4 B, CellGrid uint16 2 B, CellMask 1 B; ~2.7 GB at x512) + CellGrid.__init__ transients (np.nonzero int64
   x3 per non-air cell, astype copies; 2-3 GB spike at x512). Routing growth ~2.7 KB per laid path cell: ~1 KB Python
   (tuples/journal entries wire.py:352/360/388, builder.py:1342 _cells_since) + ~1.3 GB untracked C++ (FlatMap/NodeTable
   pools up to 32 x 1M slots x 16 B?). Doing: worker2 = grid init slabs, pool caps, CellMask drop; worker3 = memray
   of route + fork, then packed int64 path-cell storage.

1. yosys memory. Measured 2026-11-08 (cpu/synthetic x512-x2048): current flow marginal 6.3-9.6 B RSS per netlist byte
   (earlier 12.7x overstated); hierarchical (per-module abc, flatten last) 0.5-0.7x of that, but +9-20% gates on the
   synthetic design (no cross-copy merging), -2% on cpu; hierarchical JSON (no final flatten) saves disk only with
   shared modules. Next: direct 1 GB netlist synth + load acceptance test on the current flow. (worker, queued)
2. DONE af66228: repeat finder kept blocks out of instances (different-ROM cores could share a tile); coverage counts
   gates only. Parked: forced small macros (x4/x8, --macro-min-instances 2) fail routing/verify (separate issue).
2b. Disk per build: 89fa580 gzips component_list/component_layout/repeats cache (x64 artifacts 42.5 -> ~19 MB now,
    -> 7.6 MB once synth.sh writes netlist.json.gz + yosys.log.gz; patch ready in scratch repeats/disk/patch_synth.py,
    held until the acceptance runs leave the queue). 1 GB netlist: ~3.8 GB -> ~0.7 GB artifacts + litematic.
2d. Forked verify memory (owner dict / hidden set in the child, CoW from parent writes). (worker3, running)
2c. cpu x512 acceptance build, base branch. (worker2, running)
3. Python load ~8.5x netlist: slot IR models (Component/Net ~3-5 KB each); or load hierarchical JSON.
4. Routing-state memory: wire cells as Python tuples in ~6 structures; verify fork's own owner dict / hidden set.
5. V-dense arrays (~7 B persistent per world cell + transients).
6. Disk per build: component_list.json (1.4x netlist), component_layout.json, resume pickles, snapshots.
7. Macros at scale (hierarchical abc gives 82-88% coverage) shrink flat state; finder needs >30 s at x128+.

Stopped (time only, user 2026-11-08): clock tree (branch claude/clock-tree), `_compact` sub-quadratic
(claude/compact-subquadratic 1ccf530: finished, output identical, x128 place 95 s -> 25 s; unmerged, its dict grid memory at 1M unmeasured), hold/verify time exponent.
Clock findings (claude/clock-tree 1e0dee3, for the separate time project): first clock route is fast (131 s at x256);
the blowup is a ripped clock branch sent to the queue's end, then re-routed in a full world; skew 6-220 ticks at x128
(hold added 2,462 repeaters). Proposed: unrippable balanced trunk + per-group stub jobs; re-queue ripped fanout nets next.

## Waiting on user

(none)
