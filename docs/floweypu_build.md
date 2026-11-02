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
