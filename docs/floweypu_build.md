# floweypu build task

Goal: `examples/floweypu` builds in under 20 minutes, faster is better.
Done means: routes with 0 failures, 0 verify violations, and a test program gives correct output in mchprs_headless.

## User's rules
- Source edits allowed if proven equivalent. Hist keeps all bits, behaviour unchanged: no trimming, no bug fix.
- Stay on branch claude/floweypu-example-checklist-ca457a. Commit here, merge children here, merge main in when useful. No push, no merge to main.
- Large tasks: child sessions in their own worktrees, then merge here. Small tasks: subagents, sonnet when clearly defined.
- Idea generation: agent court of different personalities. Borrow from EDA where possible.
- Every test has a timeout: guess it, double on timeout. Debug with a minimal repro first.
- Heartbeat every 5 minutes: check nothing is frozen.
- World height 376, can be optimised later. Server clock lever, no generated clock.

- Stay faithful to the original; adjust only for legitimate impossibilities in synthesis.
- Test program: prints 1, 2, 3 one at a time, then "hi" together, lowercase, on one clock. Output on the CPU's character ports (e0/d0, e1/d1).
- Footprint should be closer to square, not as wide (baseline 2965x380x499).
- Owner: verify is sometimes wrong. Once a build has 0 failed nets, simulate; acceptance decides, a verify-only fault isn't a blocker.
- Owner: delegate liberally to subconversations and subagents. At each progress check, reassess a list of ideas to try and run them in parallel where possible (one full build at a time for memory).

## State
- Branch at beef260: main merged. RAM work in minecraft_v/placement_engine/ram.py, techmap/ram_*.v, flag --tiled-ram.
- All stacks in a design become one `ram_group` block: lanes for shared nets, pins all on south face, gates south of it.
- Predictor alone (examples/bpmem) routed 690 nets, 0 failed, in about 16 min before main was merged; finish and logic never seen.
- Repros: scripts/ram_pin_repro.py (pin faces). Scratch model tests: bpcheck.py, histcheck.py in scratchpad.
- CPU memory is external (ra0/rv0...): test program needs a memory wrapper or a driven bus. Register file is 16x16, 4R2W.
- Rebuild A* after merges touching _astar.pyx: `.venv/bin/python setup.py build_ext --inplace`.
- examples/floweypu_hi = CPU + ROM (main.v copied from floweypu/test/main_rom.v; cpu.v a symlink). Test bench `sh examples/floweypu/test/run.sh` passes in RTL. Program pads slot 1 with nops for 1 2 3 on e0/d0; h on e0 and i on e1 share a clock. rst needs at least one clock high (sync); no-reset power-up doesn't work. rv(t+2)=mem[ra(t)].
- Fixed: macro classes whose instance outputs feed another instance's input crashed check_equivalent; now skipped (_chains_within).
- Predictor on merged code: wire done 690 nets 0 failed in 1036s, 1.5s per net, far too slow per net; profile.
- Baseline examples/floweypu full build: 17,214 top components, 18,587 nets, 2965x380x499. Failed at 1307s on net_6948, a 15-block net at a macro bank (f-cap), only 1 failed search before it. Progress unknown: add a nets-done count.
- Latch walk through RAM blocks was 93% of routing time (find_latch through own sinks). Stopped at block inputs: hist routing 145s -> 5s. Rest of hist build (190s) is post-route passes walking 1.1M block cells: settle 59s, dust shapes 38s, stamp 20s, hold 15s. Scales with block count, so predictor's 8.5M would be 20+ min.
- STACK_DUMP_EVERY=<s> in scripts/build_with_pins.py dumps stacks to artifacts/stacks.txt. py-spy needs root here.
- With synth script shrink: 14,936 cells. Short nets fail walled in (budget floor 40 didn't help, reverted). Macros on: top 13.5k cells, 15 instances, 1545x367x488, but banks disable south-pin placement, gates spread east. Macros off: 1740x309x602. Both 60-110 failed nets within 15 minutes; final count pending.
- Post-route block-scale work: background agent post-route-speed in its own worktree.
- Decision: --no-macros with gates south of RAM (variant fb, scratchpad/fb). Macro banks disable south_pins placement and squeeze gates into two planes at the east end; 61/63 fa failures there. fb: 120/122 failed searches recovered, remaining 'gate contact' errors.
- Idea court findings to try, in order: (a) skip wide retry when failure was f-cap or rip list empty: it can't change the result, and burns 12M expansions (builder ~1262); (b) fail short nets early when closest stops improving; (c) reserve pin access stubs at south face, route long nets first; (d) coarse global routing corridors (tile demand vs capacity) as job.bounds, PathFinder on failed set only; (e) regfile/tag_rf and high-fanout selects as prebuilt lanes; ABC per module to keep slices; stage-parallel hierarchy.
- Shared .venv editable install can point to another worktree: always set PYTHONPATH=<worktree> on builds.
- RAM blocks are settled alone once, cached in build/macro_cache/ram by contents and code, and pasted in bulk. Post-route passes treat their boxes as opaque and go in only if wire touches them or a pin is driven high. hist post-route 62s -> 3s; predictor 56s. First build of a block: 30s hist, 325s predictor.
- Comparators save their settled signal in a tile entity: without it a preloaded row read at rest left bitlines stuck until next read.
- Open: a preloaded group with one stack's read address not shared with the other (a0 = m[~addr0] in examples/preload) reads a0 and a2 stuck, same on old code. Not looked into.
- Acceptance: scripts/check_floweypu.py <lit> <pins> <half> (PYTHONPATH=scripts). Fanout-first (ROUTE_FIRST_FANOUT) experiment: scratchpad f64, f16.
- Native sample: 83% of router in A*, half of that _Cells_load via Python set probes. Agent astar-speed working on it. STACK_DUMP_EVERY's watchdog spins a core: don't use. f64 no faster than baseline by 30 min; failures a bit fewer.
- Density: RAM_GATE_DENSITY 0.15 reaches rank 15036 in under 15 min vs 35 at 0.3. Testing 0.10, 0.07.
- Full routes hit 60 min limit at last rip-ups, 0 errors: d15, d10, d07 all. Run g1 (scratchpad/g1) with 2h limit, density 0.15, fanout 64.
- Machine has 17GB: run ONE full build at a time, memory pressure wrecks timings (rank at 15 min: 15,178 alone vs 13,900 with three). A* grid (CellGrid) merged: 1.9x on examples/cpu, identical paths. MINECRAFT_V_ASTAR_GRID=0 off, =check verifies.
- Remaining router gain: verify._World latch and inverted walks in kernel, 1.3-1.5x guess.
- h1 (A* grid, density .15, fanout-first 64, big nets not ripped): first full route: 15,230 nets, 4 failed, wire 1200s, verify 1 short, critical 2703 ticks. Next w5: RAM_GATE_WIDTH=.5 centered compact gates.
- h1 PASSES scripts/check_floweypu.py in redstone (half 3000): 1 2 3 h+i. Total 1828s, 1191x306x724, 4 failed nets, 1 short (net_1069/net_5485). Saved at scratchpad/h1. Goal left: 0 failures/shorts, under 1200s total.
- w35 (RAM_GATE_WIDTH .35, density .15, fanout first 64, big nets not ripped): wire 562s, 15,230 nets, 0 failed, 1094x263x1178. Verify 3 violations. w5 post-route hung 25+ min in settle: suspect oscillation from shorts.
- Defaults now: RAM_GATE_WIDTH .35, RAM_GATE_DENSITY .15, ROUTE_FIRST_FANOUT 64 (env overrides). Build needs --no-macros and --tiled-ram. Agent hold-faults owns full builds: post-hold verify faults, phase speed, acceptance.
- hf3: 0 failed, 0 verify, 1209s, but FAILS acceptance simulation; h1 still passes, harness fine. Verify misses it. Hunt: gate vs RTL on hf3 (verify-vs-sim session), settle cap suspect.
- hf3 root cause (verify-vs-sim session): net_1820 stuck high on 110/114 sinks, held on through a double-torch drop looping onto its own wire. Same polarity so `inverted` silent; `_latch_cycle` only counted cycles with a repeater. Fix in progress. Gate vs RTL helper at that session's branch (scripts/compare_gate_rtl.py).
- hf3 (fixes merged): 0 failed, 0 verify, 1209s warm RAM cache (cold +237s). Acceptance: 2+ reset clocks PASS at half 3000, 1456?, 1000, 750, 500 in redstone. 1 reset clock FAILS (`2 3 h i`, loses 1) though RTL passes with 1 and h1 passes with 1: unexplained. redpiler pending.
- Mac slept about 70 min, API errors killed agents; keep-awake requested. Load hit 36: final timing needs a quiet machine.
- hf3 fully explained: net_1820 torch latch and net_2367 repeater across a gate pin cell (pass-by open), both real, verify now reports both, router retries latch. scripts/compare_gate_rtl.py compares gates with netlist and RTL: use it, stronger than acceptance. Low-phase samples: 0 mismatches netlist vs RTL on h1, hf1-3. Open: full adder outputs net_1485, net_1515, net_15224 in hf1, hf2.
- Bar: 1-reset acceptance plus clean scripts/compare_gate_rtl.py (low-phase samples) in redstone. hf3's 2-reset pass is NOT clean: net_1820 stays stuck on 110 sinks; first reset clock leaves net_25 wrong, second clears it. h1 also has net_1533 latch and net_1069 short.
- Cold RAM cache: 491s -> 52-86s, per-bank cache, RAM_WORKERS fork pool, same cells.
- k1 (latch fix, no repeater-pin fix, no postroute kernel): 15,230 nets, 0 failed, 0 verify. FIRST CLEAN ACCEPTANCE: 1-reset PASS at half 1500 in redstone and redpiler, compare_gate_rtl wiring/gates/flops all 0 over 64 samples. 2049s cold under load (RAM 684, route 935, post-route 291). Installed to MCHPRS claude.litematic with half 1500; not tried in game. Redpiler test took ~15 min on this build.
- Merged postroute-time (post-route replay 357s to 28s; `_signals.pyx`, rebuild after merge). Its has_latch predated torch latches: fixed 210f4dc. Next: merge net-check-speed, one quiet cold full build, acceptance, that decides the goal.
- q1, GOAL MET (dd8cc68, quiet machine, empty RAM cache, one seed): synth 8s + build 703s = 711s. 15,230 nets, 0 failed, 0 verify. 1-reset PASS at half 1500 in redstone and redpiler, compare_gate_rtl wiring/gates/flops 0 over 30 clocks. Times: RAM group 10.6, placement 57.6, routing 576.7, post-route 36.7. Peak RSS 6.3 GB. 1177x305x1164. Installed to MCHPRS with half 1500; not tried in game. True minimum clock unknown (generated half 1308 is an estimate, not tested on q1).
- User target (2026-11-02): plot XZ is 1024x1024, footprint must fit under that (q1 is 1177x1164). If a sub-20-minute build is impossible at those dims, the plot can be expanded. Height limit stays 376.
- Paused 2026-11-02 for machine sleep. Open agents, each with WIP commit and WIP_NOTES.md on its branch: clock-timing (worktree-agent-ac17b89ea74c4c619), regression-sweep (worktree-agent-ab2ac90f1a6888808), density (worktree-agent-a704db239106ebe52), ram-stack (worktree-agent-a5f131867af3f0c77). None reviewed or merged here. Resume them, then merge, then quiet full build plus acceptance.
- RAM_FOLD=1 (write lines under the first read decoder) takes the floweypu group from 1086x252x405 to 808x252x408. RAM_STACK_BANKS=12 at synth (scripts/ram_measure.py to see) adds 704x316x408. Both off by default. A full build needs RAM_GATE_WIDTH near .6: at .35 the narrower block makes gates deeper. Open: bpmem at 8 banks a stack fails its model check on the first read after power-up, with or without fold, both seeds; 12 and 16 pass.
- Regression check against main (all examples, every flag, random inputs against RTL in redstone): only examples/cpu had broken, on seeds 0 and 1. Two causes. A: the 190-high stack limit also split routed banks: now RAM blocks only. B: no repeater at a slope's foot walls cpu's nets off: now on only for designs with RAM blocks (pathfinding.set_foot_repeaters, from main.py), so floweypu's path is the same code as before, not rebuilt. Without RAM, foot repeaters are back as on main; verify and repair catch their faults. Nothing deprecated. Broken on main too: --debug, pinext and video pin targets, sevenseg --flat, halfvideo --route-passes 3. New and open: --tiled-ram memread reads wrong words at any clock tested (7 of 1280 bits, half 428 and 3000).

- synth-nets (`worktree-agent-ab09b6e9e2db63bc5`, f7b1e27, not merged): adder fold (`techmap/fa_const.v`) plus `&syn2;&dch,-f;&nf`: nets 15,230 to 12,621, tiles down 10%, gate levels 102 to 82, ABC cec equivalent. Never finished routing; muxes double, a routing risk. Fallback: tile-cost library, fewer muxes. density tests it after t1.

## TODO: investigate when the machine has room
Rule: if the machine is strained (swap filling, stalled processes), write ideas here instead of starting them.
- Longer test program exercising branches and the predictor; `hi` is too short to clear it. Wait for preload-fault.
- Footprint: fold + `RAM_STACK_BANKS=12` with gate width about .55 if fold alone can't route under 1024x1024.
- Footprint: two RAM tiers with lanes in two bundles (estimate 432x328x413, not prototyped).
- Clock: false-path handling, once clock-timing reports how much pessimism they cause.
- Routing: coarse global routing corridors for long edges, tag_rf and select lanes. Failed searches track edge length: under 100 blocks about 0.2% fail, 100-300 blocks 12-16%, 300+ almost all (h1). They don't cluster at RAM pins, so pin-access stubs are not worth trying.
- Congestion: RUDY can't rank settings or locate failures (`scripts/predict_congestion.py` on worktree-agent-ae2dffb15a5e5ae50, not merged). Unrouted guesses for 1024x1024: fold with width .5 channel 250 (863x305x905), or width .6 channel 250 density .20 (863x306x839).
- Routing: failed-set PathFinder for hard nets; per-module ABC; stage hierarchy.
- Open: `--tiled-ram` on `memread` wrong words, `bpmem` `next_pc1` (preload-fault running).
- Open: `hist` 1 failure at half 500 and 350, folded and current, not looked into.
- Open: full-adder opens in hf1 and hf2, `net_9622` inverted, unexplained.
- Open: RAM group with lone lower read pin refused, address pitch 5 broken.

## Job queue
All builds and simulations go through `python3 scripts/job.py run --mem 7G --name floweypu-build --timeout 2400 -- <command>`; it blocks until there's room. Rough memory: floweypu build 7G, floweypu simulation 3G, small example 1G. `--exclusive` for builds whose time matters, `--priority 10` for final checks, `-10` for sweeps. `python3 scripts/job.py status` shows the queue. Waiting raises priority one level per minute, so nothing starves. Timeouts don't count sleep. Jobs may swap: admission allows 10G over what's free (`JOB_OVERCOMMIT_GB`), up to 16G of swap in use, because most jobs aren't time critical. Timings from shared runs mean nothing; use `--exclusive`.

- t1, PLOT TARGET MET (density's branch worktree-agent-a704db239106ebe52, one seed, quiet machine): `RAM_FOLD=1 RAM_GATE_CHANNEL=380 RAM_GATE_WIDTH=.6`, synth 7s + build 361s = 368s (wire 227, placement 60). 863x306x989, 15,230 nets, 0 failed, 0 verify, peak 6.2 GB. 1-reset PASS at half 1500 in redstone and redpiler, compare_gate_rtl 0. Installed to MCHPRS with half 1500; not tried in game. Those settings aren't merged here or the default yet.
- Generated clock now from edge-aware static timing (hold.py clock_requirement) with RAM arcs measured by tick_sim. Small examples safe at generated half: mem 197 of 197 needed, bphist 599 (fails 594), ramtile 36 (28), regs 16 (14), counter 29 (22), cpu no RAM 353 (177, weak traffic). floweypu value not yet built or swept; scripts/clock_sweep.py measures one.

- Faster `mchprs_headless` from MineRTL session, uncommitted on MCHPRS master, copy at `~/.cache/mcv-jobs/mchprs_headless` (use `MCHPRS_HEADLESS`): t1 PASS in both modes, about 140s total; hf3 redstone still FAIL.
- s1 (new synthesis recipe, density's branch): 8+501s, 863x305x954, 12,621 nets, passes the bar after a repair pass on net_2 (1086 sinks). t1 is still faster.
- Predictor fault found (preload-fault): not lost preload. Live pins with no net (unused block outputs, unused inputs) drive the cell in front of them; router treated it as free, verify can't see it. bpmem `next_pc1` and wpc1 shorts fixed by keepouts (`_unused_output_terminals`). q1 has it on `s2_rc[0]`; t1 not checked yet. Earlier notes blaming preload or power-up are wrong. `memread` tiled was separate: hold ignored gated clocks, fixed.
- Longer predictor test `examples/floweypu_loop` (RTL prints `aaaanttnttnttcccccccxyxy`, halts edge 212) FAILS on a fixed build: flip-flop `slice$25910` hold margin -14 logged as `[hold] still short` and ignored, wrong from clock 35, not speed. So `still short` lines are real faults and t1, p200, p100 are unproven until hold-leftovers finishes. New `live_pin` verify violation; `scripts/check_bpmem.py`.
- Best footprint so far p100: 863x305x684, passes hi at 1500 and generated half, compare 0, live pins and hold leftovers (0, 0, 4) still to clear.
- Queue bug fixed (bf28fe9): stale clients at the head stalled everyone. A client runs the code it started with.
- Predictor proven in gates once: `floweypu_loop` with hold detours (hs/lp2 in scratchpad) prints `aaaanttnttnttcccccccxyxy`, halts edge 212, flip-flops 0, gate vs RTL 0, redpiler at half 2000 only. Real hold limits measured: DFF safe after 2 ticks, DFFE after 5; `[hold] still short` now prints slack and a real shortfall is a `hold` violation.
- hi1 (density's defaults 6360805 plus hold, live pin and clock fixes): 863x305x585, synth 7 + build 385 = 392s, 0 failed, 0 verify, hold detours left no shortfall, generated half 1591. hi PASS 1 reset in redstone and redpiler at 1591, compare 0. Installed to MCHPRS at its generated half; not tried in game. loop1 (predictor program) pending.

