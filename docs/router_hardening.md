# Router hardening: failure report and hand-off

Status as of 2026-10-28, commit 911556e. Read CLAUDE.md first. It covers the routing model, conventions, and the 60s build cap.

## Why this matters

The router was debugged against the legacy placement of the bundled examples. On those exact layouts it routes everything. Change the layout and it breaks:

| Placement | Builds with ≥1 failed/open net |
|---|---|
| legacy (canonical gate order) | 0 / 4 |
| legacy with **shuffled** gate order | 20 / 40 |
| `--placer sa`, seeds 1–8 | 9 / 32 |

Examples: regs, runner, sevenseg, sevensegplus (+ counteradvanced in the shuffled run).

The new SA placer (`--placer sa`, `minecraft_v/placement_engine/anneal.py`) cuts estimated wirelength 25–36% and makes layouts about 40% narrower. It can't become the default while the router fails on about 28% of its layouts. **Goal: SA layouts route cleanly**, i.e. 0 failed nets and 0 `[verify]` violations across a seed sweep, with no regression on legacy layouts.

## How to reproduce and measure

Every build needs a hard 60s wall-clock cap. `bench_examples.py` enforces it and kills the whole process group. Never run `synth.sh` / `python -m minecraft_v` without a cap. Exclude `video` unless you're testing it on purpose.

Routing is **deterministic per placement**. `PYTHONHASHSEED` / `--hash-seed` does not change results, so to sample, vary the placement seed:

```bash
# one failing case (deterministic, re-verified)
uv run python scripts/bench_examples.py --out build/repro regs -- --placer sa --place-seed 5

# the sweep that defines "done" (8 seeds x 4 examples, ~90s total)
for s in 1 2 3 4 5 6 7 8; do uv run python scripts/bench_examples.py --out build/sweep_$s regs runner sevenseg sevensegplus -- --placer sa --place-seed $s; done

# legacy regression check (must stay 0 failed / 0 violations)
uv run python scripts/bench_examples.py --out build/legacy
```

Each build writes `build/<out>/<example>/log.txt` with lines `[error] skipped net ...`, `[verify] open|short ...`, `[verify] unsupported ...`. Add `--snapshots` to the extra args for per-10-net litematic snapshots. They're slow; use them only on a single repro.

## Headline findings

1. **High-fanout nets fail.** Nets with ≥6 sinks are 9% of all nets but **80% of failing nets** (50 failing nets across 152 SA builds). Typical failing nets are `clk` (8 sinks) and flip-flop `Q` outputs (6–12 sinks). That points at multi-sink tree routing: `tree_seeds` in `_route_all_nets` and the per-net repeater pass `_place_repeaters_for_net`.
2. **Almost every "No route" stops 1–2 cells from the goal.** Examples: (19,12,23)→(19,12,22), (37,5,11)→(37,5,10), (43,20,56)→(43,21,56). Search reaches the pin and can't make the last move. This is pin access, not global search.
3. **The default mode never rips up a failed net.** `route_passes=1` (greedy, `_route_all_nets`, `builder.py` ~441–510) catches the exception and leaves the half-laid path in the world. That debris shows up as `[verify] unsupported` dust and blocks later nets. The negotiated mode already has journaling + `ripup.rip_up`.
4. **Laying disagrees with search legality.** Two failure classes are "A* accepted a path that `wire.py` then can't lay". Those are kernel/laying mismatches: rules in `_astar.pyx` `World` (and `_wire_walkable`) that don't match `wire.py`.

## Failure classes

Counts are over 152 SA builds. Repro commands use the bench line above with the given example and seed.

### 0. No rip-up of failed nets in greedy mode — fix first
- Where: `builder.py` `_route_all_nets`, the `except Exception` around laying (~line 500).
- Effect: partial wire, support and repeaters from a failed net stay in `workspace` / `solid` / `dust_owner`. Seen as `[verify] unsupported minecraft:redstone_wire` right after a support failure, e.g. sevensegplus seed 1 `net_2` and seed 3 `net_32`.
- Fix: record each net with the existing `Recorder` (`ripup.py`) and `rip_up` on failure. `solid` and `dust_owner` are already `JournaledSet` / `JournaledDict` in `build_litematic_from_component_list`. Workspace writes would need to go through `JournalingWorkspace` (as `_commit_route` does), stacked with the `IndexedWorkspace` wrapper `_route_all_nets` uses so the block index stays current. That's cheap. It also enables the next step: a failed net can be retried after its blocker is ripped, or with a different sink order.

### 1. Routed high-fanout net is open (signal never reaches sinks) — 3 builds
- Repros:
  - regs seed 5: `net_2` = `clk`, 8 sinks, **7 open**, with 0 failed nets. Every route laid, but signal doesn't arrive.
  - sevensegplus seed 2: `net_38`, a DFF Q with 9 sinks, 6 open.
- Opens on most sinks of one net mean the signal dies near the root or at a shared trunk segment.
- Hypotheses to check, in order:
  1. A **tree-seed join** lands on a cell that doesn't propagate signal into the branch. Candidates: a repeater's side or output face, a tower or slope-2 intermediate, a glass ramp cell, an inverted (powered −4) cell, or dust whose connection shape changes when the branch is added. `_find_wire_path` admits every own-net path cell as a seed (`tree_seeds` filter only drops `solid` and out-of-bounds).
  2. `_place_repeaters_for_net` puts a repeater on a trunk cell that a **later-joined branch** needs to read from the side or upstream. `branch_points` is computed from `ordered_paths`, but a branch can join at a cell adjacent to the repeater, not on it.
  3. A later path's support block or dust changes an earlier slope connection, e.g. a solid block placed diagonally above a slope breaks the dust step.
- Debug approach: in `verify.verify_signals`, dump the reached set for the open net and find the frontier, the last reached cell next to unreached own-net cells. Look at that block in a `--snapshots` litematic.
- Better long-term: verify each net right after laying it (signal reach from source to that net's sinks) and treat an open as a routing failure. That makes it rip-up-and-retry instead of a silent bad build.

### 2. No viable repeater position — 17 occurrences (most common core-net failure)
- Repro: sevenseg seed 2, `net_16`: "signal dist 15 >= 15, no viable position in [...]". The listed cells are all slope or turn cells.
- Cause: A* doesn't model signal strength. Repeaters are placed afterwards (`wire.py` `_place_repeaters_for_net`). `is_viable` needs a straight, flat, non-branch cell with a matching next step. A 15-block stretch made of slopes and turns has no slot.
- Fix options, best first:
  - Track blocks-since-reset in the A* state (or as a cost term) and forbid or penalize paths that reach 15 without passing a repeater-viable cell. Towers and powered −4 already reset. The kernel is `_astar.pyx`; `docs/cython_astar_plan.md` has the layout.
  - Let the repeater pass use more positions, e.g. the flat cell before a slope, or the first cell after a turn if the facing works.
  - On failure, re-route just that path with a penalty on the offending stretch (needs class 0's rip-up).

### 3. Path tries to lay dust over an existing block (wool) — 8 occurrences
- Repros: runner seed 3 (`net_67` over gray_wool at (53,5,100), `net_49` over black_wool at (34,7,36)), sevenseg seed 4 (`net_27` over cyan_wool at (27,37,75)).
- Wool is per-net **support** (`WOOLS[net_idx % len(WOOLS)]`, `builder.py` ~492). So the search walked through a cell that holds a support block by the time laying reaches it.
- First check: is that wool **this net's own** (same color index), placed earlier in the same path or by an earlier sink path of the same net? Or another net's?
  - **Own:** in-path self-collision. The path snapshot / `move_cells` in the kernel doesn't cover the support blocks that earlier moves of the same path (or earlier tree paths) will place. This is like the earlier zigzag double-expansion bug.
  - **Foreign:** stale world view. `IndexedWorkspace` / `_BlockIndex` isn't updated for support writes, or `solid` lacks the block.

### 4. Support cell under new dust is occupied by wire — 8 occurrences
- Repros: sevensegplus seed 1 `net_2` (support at (44,44,79)), seed 3 `net_32` (38,44,67), seed 8 `net_29` (49,52,67). All on high-fanout nets.
- `_place_support` (`wire.py:11`) raises when `(x, y−1, z)` is wire. Walkability check 3 forbids **foreign** wire directly below, but own-net wire below is allowed. Dust can't stand on dust, so laying fails.
- Fix: in the kernel `wire_walkable` / move generation and in `_wire_walkable`, a flat or slope destination needs support to be placeable. `(x, y−1, z)` must be solid or air, not own-net wire. The one exception is the alternating down-staircase glass case already handled in `_lay_redstone_path`. Keep `_wire_walkable` in sync with `World.wire_walkable`.

### 5. Pin access: goal unreachable or not walkable — 8 "no route" + 3 "goal not walkable"
- Repros: regs seed 8 `net_2` (`clk`): closest (19,12,23), goal (19,12,22), one step. runner seed 3 `net_19`: "No walkable cell near goal (34,29,53)".
- Cause: earlier nets put wire, support or halo next to a later net's terminal, so the goal or its only approach is illegal. Keepout (`_compute_net_protected` / `_KeepoutMap`) covers terminals + cardinal neighbors at dy −2..+2 + diagonals at the same Y. That doesn't reserve the **approach corridor**. Goal relief in `_find_wire_path` only admits the goal cell itself (`pathfinding.py` ~717–723).
- Fix options:
  - Reserve a 2–3 cell approach corridor in each pin's side direction in `protected` for every other net, mirroring the I/O repeater cell reservations.
  - With class 0's rip-up in place: on a goal failure, rip the net occupying the approach and re-queue both.
  - Route high-fanout nets first; they suffer most from late blocking.

### 6. Pin-target extension failures — 30 occurrences (lower priority)
- runner fails `IGNORING - Pin target routing failed` **under every placer, legacy included** (legacy: `net_11` to (56,1,0)). Goals sit on the z=0 edge, and search ends diagonally adjacent at z=1. Same pin-access shape as class 5, in `_route_output_pin_extensions`. bench doesn't count these as failed nets. Fix after the core classes.

## Suggested order

1. Class 0: rip-up in greedy mode (small, unblocks retries).
2. Class 4 then class 3: kernel/laying legality mismatches. These are local rules, and the repros are deterministic.
3. Class 5: pin-access corridor, plus retry with rip-up of the blocker.
4. Class 2: signal-strength-aware search.
5. Class 1: per-net verify after laying; investigate the tree joins.
6. Class 6.

After each step, run the 8-seed SA sweep and the legacy check. Record bad builds / 32, failed nets, and total time in the commit message. Commit straight to main and push (repo convention). Don't commit the user's uncommitted `constants.py` change (`_ROUTE_STAGNATION`) or their staged `examples/video/*`; commit with explicit paths.

## Code map

- `builder.py`: `_route_all_nets` (greedy), `_route_all_nets_negotiated`, `_commit_route`, `_route_output_pin_extensions`, `_report_signal_violations` / `_report_unsupported_blocks`.
- `pathfinding.py`: `_find_wire_path` (seeds, goal relief, path reconstruction) and `_wire_walkable` (must match the kernel).
- `_astar.pyx`: search loop and `World` move generation / walkability. **Move-rule changes go here.** The Python `_search_py` is deprecated.
- `wire.py`: `_place_support`, `_lay_dust_cell`, `_lay_slope2_move`, `_lay_tower_move`, `_lay_redstone_path`, `_place_repeaters_for_net`.
- `ripup.py`: `Recorder`, journaled containers, `rip_up`.
- `verify.py`: `verify_signals` (opens/shorts), `verify_support`.
- Tests: `tests/test_ripup.py`, `tests/test_verify.py`, `tests/test_negotiate.py`. `tests/test_placement_litematic.py` currently fails to import (`_io_repeater_facing` isn't exported from `placement_engine/__init__.py`), and 3 tests in `test_module_to_component_list.py` expect old footprint sizes. Both were already broken at HEAD.
