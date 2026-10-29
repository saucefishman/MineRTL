# Cython A* migration plan

## Status

- **Phase 1 done.** `_astar.pyx` runs the A* loop and `_static_moves` stays in Python.
  Greedy output is identical on all examples (`base_ce6164e.json`), and `MINECRAFT_V_ASTAR=check`
  finds no differences search by search. Sequential builds: regs 7.2s → 3.5s, sevenseg
  5.7 → 3.8, runner 13.5 → 8.6, sevensegplus 12.5 → 7.8. video at 60s: ~407 → ~488 of 1107
  nets, so phase 2 is needed.
- Deviation: instead of the replay harness, `MINECRAFT_V_ASTAR=check` runs both loops on every
  search of a real build, raises on any difference (result, best node, pruning count, stop
  reason, came_from chain) and prints both loops' time. In check mode the Python loop runs
  first and fills the shared move cache, so the kernel's time there excludes move generation.
- Switch: `MINECRAFT_V_ASTAR=kernel|py|check` (default kernel; falls back to Python with a
  warning if the extension isn't built).

Goal: get `video` under the 60s cap (now ~407/1107 nets by 60s) and cut mid-size builds further,
with greedy output **byte-identical** to the Python A* at `ce6164e`
(`scripts/check_same_output.py --compare build/same_output/base_ce6164e.json`).

## Where the time is (ce6164e structure, video, 60s)

- Search ≈ 92% of wall time; ~1.9M heap pops at ~29µs each.
- Main loop (heap tuples, `g_score`/`came_from` dicts, snapshot re-centring, gating): ~65%.
- Static move generation (`_static_moves`, `_wire_walkable`, tower/-4 candidates), cached per
  cell: ~25%. Cells are expanded ~18x on average, so this is paid once per ~18 pops.

So there are two layers, and the plan ports them in that order: the per-pop loop first (biggest,
self-contained), then the per-cell world checks (needs a dense world representation).

## Design

### Module layout

- `minecraft_v/placement_engine/_astar.pyx` (Cython, `language = c++`): the kernel.
- `pathfinding.py` keeps `_find_wire_path` as the entry point. It does the per-search setup in
  Python (goal relief, keepout views, seeds, groups), then calls the kernel and does the epilogue
  in Python (lamp trail on failure, path reconstruction checks, `_check_tower_self_overlap`).
- The current Python loop stays as the reference implementation behind
  `MINECRAFT_V_PURE_ASTAR=1` (also the fallback when the extension isn't built). It is the
  oracle for equivalence checks, so it is not deleted.

### Kernel state (phase 1)

- Node = `uint64` (cell id << 25 | tower flag | 24-bit snapshot), as now.
- Heap: `std::vector<Entry>` + `std::push_heap/pop_heap`. `Entry {double f; int neg_h;
  double g; int64 counter; double g_wire; uint64 node; uint64 parent; int32 group;}`, compared
  lexicographically on `(f, neg_h, g, counter)`. counter is unique, so the pop order is total:
  any correct min-heap gives the same sequence as `heapq`.
- `g_score`, `came_from`, `explored`: one `std::unordered_map<uint64, NodeInfo>` (or a flat
  open-addressing table). `explored_xyz`: a byte per cell id.
- Move table per cell id: `std::vector<Move>` with `Move {int32 dest; double cost; uint32
  forbid; uint8 gate; int32 extra_cell; uint8 shift_idx; int8 flat_dx, flat_dz; bool has_flat;
  int32 h; bool near; uint32 arrive_flag; double congestion;}`, plus the -4 alternatives.
  Snapshot shift tables become a small static array indexed by `shift_idx` (one per move delta).
- `g` and costs are `double`. Values are integers except congestion; doubles add them exactly
  in the same order as Python, so `f`, the f-cap and the h-radius compare identically.
- Congestion cost depends only on the destination cell within a search (congestion is frozen
  per net), so it is computed once per move entry instead of once per push.

### Phase 1 boundary

The kernel asks Python for a cell's moves the first time it expands that cell (a callback into
the existing `_static_moves`, returning flat tuples it copies into `Move`s). That's ~80k calls per
60s of video, cheap compared with 1.9M pops. All world data structures (sets, `_ForeignKeepout`,
overlay/journaled views) stay untouched.

### Phase 2: world checks in C

Port `_static_moves`, `_wire_walkable`, `_tower_candidates`, `_powered_minus4_move`,
`m4_cell_clear` and the support/torch/powered-block checks over dense per-search arrays:

- `flags: uint16[cells]` over the search box (bounds + margin of 2 in X/Z, 5 in Y for the
  tower/-4/slope-down reach). Bits: in-bounds, solid, non-air, wire, torch, repeater, glass,
  powered-block (solid with an input repeater; replaces the `workspace[...]['facing']` read),
  footprint (raw and goal-relieved), protected (raw and goal-relieved), inverted, terminal.
- `owner: int32[cells]`: 0 none, 1 this net, else another net's index (`breaks_slope` compares
  two owners, so a plain own/foreign bit isn't enough).
- Filling these per search from Python sets is O(placed cells) per search, too slow on video
  (~20k cells x ~2000 searches). So the grids are maintained incrementally, the way
  `IndexedWorkspace` already maintains `_BlockIndex`:
  - `IndexedWorkspace` also keeps numpy `kind`/`glass` grids.
  - `solid` and `dust_owner` get grid-mirroring wrappers on the real workspace (they're
    already wrapped in `JournaledSet`/`JournaledDict`, so the hook goes there).
  - Keepout counts (`_ForeignKeepout._count`) become a grid as well.
  - Per search: one numpy slice copy of the box, then patch the per-search parts (goal relief,
    own-net keepout, effective footprint) from their small Python sets.
  - Negotiated mode: `OverlayWorkspace`/`OverlaySet`/`OverlayDict` patch the copied box with
    their overlay (small), as `_BlockIndex.layered` does now.
- First step of phase 2: measure region and box volumes on video, and the cost of the
  per-search copy, before committing to this layout.

### Build

- Add `[build-system]` (setuptools + Cython) and a `setup.py` declaring the extension.
- `[tool.uv] cache-keys` includes `minecraft_v/**/*.pyx` and `setup.py`, so `uv run` rebuilds the
  extension when the kernel changes.
- Check early that the editable install builds the `.so` in place and that `synth.sh` /
  `bench_examples.py` pick it up. Otherwise fall back to `build_ext --inplace` in a script.
- `pathfinding.py` imports the kernel inside `try/except ImportError` and falls back to Python
  with a one-line warning. Any other mismatch is a hard error, never a silent fallback.

## Steps

1. **Baseline**: `build/same_output/base_ce6164e.json` (running), plus sequential timings for
   regs/sevenseg/runner/sevensegplus and the video net count at 60s.
2. **Replay harness**: `scripts/astar_replay.py` dumps the inputs of chosen searches (the top
   video searches by time, plus a few regs/runner ones) and replays them under Python and the
   kernel. It compares paths and error reasons and times them. This is faster to iterate on
   than full builds.
3. **Build plumbing**: trivial extension, uv rebuild on edit, import fallback, env switch.
4. **Phase 1 kernel**: main loop in C++ with the Python move callback. Gate: replay identical,
   then `check_same_output --compare` SAME on all examples, then timings.
5. **Phase 2a**: incremental world grids (no kernel change yet). Check the grids against the
   Python sets on every search in a debug mode.
6. **Phase 2b**: port the world checks into the kernel. Gate as in step 4.
7. Update `CLAUDE.md` (`placement_engine` layout and the pathfinding sections) and the
   negotiated-mode notes.

## Must stay identical

- Move generation order (horizontal: flat, up, down per direction; 2x slopes; tower +1Y; tower
  candidates; -4 last). It feeds the push order, and through it the counter tie-break.
- `came_from` first-expansion-wins; nodes carry the tower flag; per-group f-cap (`continue`
  under the group budget, `break` above the overall max) and per-group h-radius.
- Stop conditions and messages: f-cap, stagnation (`_ROUTE_STAGNATION`), node cap
  (`_ROUTE_MAX_NODES` unique cells), deadline checked every 256 pops (`time.monotonic`
  semantics), raised as `RouteTimeout` vs `ValueError` as now.
- On failure, `best_node`'s `came_from` chain is still painted with lamps by Python, so the
  kernel returns `came_from` lookups for that chain.
- Negotiated mode can't be hash-checked with timeouts on (they depend on speed). Check it with
  `_ROUTE_SEARCH_TIMEOUT`/`_ROUTE_FINAL_SEARCH_TIMEOUT` raised, as was done for 7601c9b.

## Expected gains (estimates, to be replaced with measurements)

- Phase 1: per-pop cost ~18µs → ~1µs. Video is then bound by Python move generation
  (~25% today), so ~3x overall: borderline for 60s.
- Phase 2: move generation ~10–30x faster. Video A* ≈ 10x+, so the build is bounded by
  non-search work (~8% today: setup, laying, NBT save). Mid examples land at ~1–2s, mostly
  yosys and setup.

## Risks

- Python fallback and kernel drift apart. Mitigation: replay harness and `check_same_output`
  in the gate for every kernel change; the fallback stays the reference.
- Grid maintenance misses a write path (some code writing `solid`/`dust_owner` without the
  wrapper). Mitigation: the step-5 debug cross-check.
- Build friction (compiler, uv cache). Mitigation: the import fallback keeps builds working
  without the extension.
