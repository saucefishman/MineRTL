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
- "note that the cpu design with the multi gb netlist will be radically different from the extremely basic cpu example
  in the repo; avoid overly specific optimizations (everything so far seems fine)" (2026-11-08): keep changes
  general; measure on a second, structurally different design (synthetic datapath, video) besides cpu xN.
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

- 2026-11-08 1 GB acceptance, synthetic N=5600 (1,014 MB netlist, 1.65M cells), current flow: yosys footprint peak
  11.26 GB, 81 min (1,971 s sys: paging); Python load (read, parse, to components, save) footprint peak 7.8 GB, 220 s;
  synth disk peak 1.14 GB. PASS under 12 GB (one run). cpu N=5900 (1.061 GB, 1.74M gates): yosys 10.85 GB, 67 min;
  load 7.89 GB, 132 s; disk peak 1.21 GB. PASS. Gap: yosys-abc child footprint not sampled (worker measuring).
  Earlier hier-vs-current yosys comparison was RSS-only: absolute numbers ~half, ratio likely holds.
  Feasibility of several-GB netlists on this machine: see the reply of 2026-11-08 (synth+load ~1-1.5 GB ceiling now;
  full build ceiling below cpu x512, ~90 MB netlist).

- 2026-11-08 synthetic datapath x48 (worker3's route-state branch) reached >=13.8 GB footprint while declaring 5G;
  job.py missed it (build in its own session, outside the job's process group). Killed by hand (swap 0.8 GB free).
  Fixed ~/.cache/mcv-jobs/job.py to measure/kill descendants too (new clients only). The x48 base run (before
  route-state) also passed 16.6 GB and was killed by hand: the synthetic datapath outgrows 12 GB far earlier than
  cpu xN (x48 is ~150-200K cells? to confirm), so it is the harder test design.

- 2026-11-08 3adaaed merged route-keys: solid/dust_owner bits and owners live in the CellGrid, writer map is a packed
  PosMap. cpu x64 parent peak 6426 (original) -> 5375 MB; x64 routing end 4203 -> 3719. Datapath x48: routing done at
  10.6 GB (~7 KB per path cell, mostly A*/grid, not Python), then repair/hold/timing climbed past 16.6-18 GB unfinished.

- 2026-11-08 18e68c3 merged ir-slots: IR as slotted dataclasses (Component 1264 -> 280 B, PinRef 480 -> 80, NetEndpoint
  480 -> 56). Load at ~300K components: 1.5 GB -> 0.87 GB footprint; load peak now the JSON parse (~4.5x netlist).
  yosys-abc adds ~0.24 GB at the combined peak (N=2000); hier flow footprint 0.67x of current, +19% gates (synthetic).

- 2026-11-08 5a1f266 merged grid-mem 04eb2b5: the failed-net list kept exceptions with tracebacks, holding each failed
  search's tables/cell lists for the rest of the build. Datapath gen.py 16: peak 5.09 -> 2.43 GB, identical results.
  Likely a big part of x512's 14 GB (hundreds of failed searches). x512 acceptance rerun on 5a1f266 (worker2).

- 2026-11-08 cpu x512 acceptance rerun on 5a1f266: COMPLETED, 7,773 s. Wire 5,081 s, 166,254 nets, 15 failed -> 10
  after repair (incl. net_2 clock: repeater extension), verify 0 violations, critical path 234 ticks. Parent footprint:
  place 0.54 GB, wire 9.98 GB, repair 10.24 GB (+2 forks 1.2 GB), timing/hold/verify 11.3 GB (+5.2 GB verify fork +
  1.1-1.5 GB forks), save 12.3 GB. Litematic 46 MB, world 694x784x769, artifacts 118 MB. Was killed at 14 GB before.
  Goal not yet met: parent 12.3 GB at save, verify fork private memory on top, 10 routing failures.

- 2026-11-08 a3e6fa2 merged post-route-mem: verify fork hides gate interiors as packed keys, verify workers capped at 4
  (MINECRAFT_V_VERIFY_WORKERS), chunked settle neighbourhood. Signal child -6..-20%; output same. Post-wire phases at
  datapath x24: repair +1.6 GB native A* transient, hold +93, timing +40, settle +60 (+200 transient), fork ~30% of parent.

- 2026-11-08 a03a3be merged yosys-mem: `setattr -unset src t:$print %n` after proc. yosys+abc peak at N=2000:
  synthetic 3,994 -> 3,091 MB, cpu 3,584 -> 2,867 MB; gates identical. Peak pass: opt/techmap just before abc.
  Rejected: opt_clean -purge before abc (changes gates +-5% on small examples), no &dch (+5-10% gates, no peak gain).
  Hier flow (+src strip): -64%/-46% memory, +19% gates on synthetic -> opt-in flag next. job.py: bare --mem = GB.

## TODO (ranked, memory/disk only)

0c. Routing failures at x512 (10 nets incl. clock): classify + smallest repro. (worker4, prio -10)
0b. Save step streaming (parent +1 GB at x512 save). (worker2)
0a. Post-routing phases (repair/hold/timing/verify) memory: datapath x48 grew 10.6 -> 18 GB after routing. (worker3,
    claude/post-route-mem)
0. Routing-phase memory at x512 (14+ GB). x32 inventory (worker2): world-sized arrays only ~7 B/world cell (Region
   uint32 4 B, CellGrid uint16 2 B, CellMask 1 B; ~2.7 GB at x512) + CellGrid.__init__ transients (np.nonzero int64
   x3 per non-air cell, astype copies; 2-3 GB spike at x512). Routing growth ~2.7 KB per laid path cell: ~1 KB Python
   (tuples/journal entries wire.py:352/360/388, builder.py:1342 _cells_since) + ~1.3 GB untracked C++ (FlatMap/NodeTable
   pools up to 32 x 1M slots x 16 B?). memray x32 (worker3): A* scratch on failing/repair searches = 2.14 of 3.2 GB peak
   (~1 KB per expansion: cached moves, 40 B Move, Python mirror lists); persistent wire ~760 B per path cell; gate_cells
   142 MB. Verify fork +22-30% at its moment, child memory private; not the overall peak (repair-of-failed-nets is).
   Doing: worker2 (claude/grid-mem) = _astar.pyx scratch + grid init slabs + pools + CellMask; worker3
   (claude/route-state) = packed int64 solid/dust_owner/journals/gate_cells.
   - c54d685 merged grid-mem 5a84d13: CellGrid filled per x slab (no whole-volume nonzero/astype temporaries), search
     pools keep <=8 idle vectors. x32 peak 1841 -> 1787 MB; x512 transient saving (2-3 GB) not measured. Output same.
     CellMask kept (each router rebuilds its grid from it). Next (worker2): attribute the x64 repair-phase peak (6.4 GB).

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
3. DONE 18e68c3 (IR slots). Next load lever: streaming JSON parse. yosys per-pass memory (worker, claude/yosys-mem).
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
