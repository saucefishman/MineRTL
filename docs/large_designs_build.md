# Large designs: scaling synth + build

Branch `claude/verilog-synthesis-memory-disk-b9eff2`. Machine: 16 GB RAM, ~30 GB free disk.

## Findings summary

Goal (user, 2026-11-07):
- cpu x512 (~150K cells): full synth + build under 12 GB peak, within 60 min, 0 routing failures, on 3 seeds.
- A 1 GB netlist through synth + load under 12 GB.

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

- Done = cpu x512 under 12 GB, 60 min, 0 routing failures, 3 seeds; plus 1 GB netlist synth + load under 12 GB.
- May change: placement (if routing failures and wire length don't worsen), defaults, add dependencies, synth flow.
- Git: commit to this branch only; no merge to main, no push until asked.
- "why do you need to evaluate like this? can't you map out O of the various steps" (2026-11-07): derive complexity
  from the code first; measure only to fit constants and check the map.
- Standing (memory): all builds via `~/.cache/mcv-jobs/job.py`; one seed while iterating; Opus plans, Sonnet codes
  (kept-alive workers); never idle when SPARE.

## State log

- 2026-11-07: synth/load scaling measured (synthetic + cpu xN); fixes 3e5d482, 7601f17, 1ef06ef committed.

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

## TODO (ranked)

1. Baseline cpu x256 full build + per-phase exponents from x32/x64/x128. (worker, running)
2. `_compact` sub-quadratic. (worker2, running)
3. Routing-state memory: wire cells as Python tuples in several structures; move to compact arrays keyed by cell id.
   Needs memray of the route phase first. (child session once the profile is in)
4. Verify fork: owner dict / hidden set allocated in the child; share or build compactly.
5. Check find_repeated at x512: finishes in 30 s? coverage >=50%? (decides flat vs macro)
6. Correctness: repeat finder labels by type, so different-content ROMs may share a tile (subgraphs.py:61); check
   macro_netlist keeps per-instance ROMs, e.g. simulate two cores with different programs.
7. Clock-net k^2 terms (refine/mirror Nets::of, _TreeCells.nearest).
8. V-dense arrays: fewer persistent full-volume grids.
9. Python IR models slotted (~3-5 KB per component).
10. yosys: per-module synthesis / ABC before flatten for the 1 GB netlist target.
11. Disk per build: component_list.json, resume pickles, snapshots.

## Waiting on user

(none)
