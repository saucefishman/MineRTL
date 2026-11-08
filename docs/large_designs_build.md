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
- Standing (memory): all builds via `~/.cache/mcv-jobs/job.py`; one seed while iterating; Opus plans, Sonnet codes
  (kept-alive workers); never idle when SPARE.

## State log

- 2026-11-07: synth/load scaling measured (synthetic + cpu xN); fixes 3e5d482, 7601f17, 1ef06ef committed.

## TODO (ranked)

1. Baseline cpu x256 full build: per-phase time and peak RSS, memray of the peak phase. (worker)
2. `_compact` sub-quadratic (O(n^2) time; x128 ~95 s placing). (worker2)
3. Layout stamp / world-volume arrays (~70 B per world cell): dtypes, sparse storage.
4. Router: A* memory and routing time at scale; routing failures at x128+.
5. Python IR models slotted (~3-5 KB per component).
6. yosys: hierarchical / per-module synthesis and ABC to get a 1 GB netlist under 12 GB.
7. Disk per build: component_list.json, resume pickles, snapshots.

## Waiting on user

(none)
