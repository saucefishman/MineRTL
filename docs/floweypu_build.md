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

## State
- Branch at beef260: main merged. RAM work in minecraft_v/placement_engine/ram.py, techmap/ram_*.v, flag --tiled-ram.
- All stacks in a design become one `ram_group` block: lanes for shared nets, pins all on south face, gates south of it.
- Predictor alone (examples/bpmem) routed 690 nets, 0 failed, in about 16 min before main was merged; finish and logic never seen.
- Repros: scripts/ram_pin_repro.py (pin faces). Scratch model tests: bpcheck.py, histcheck.py in scratchpad.
- CPU memory is external (ra0/rv0...): test program needs a memory wrapper or a driven bus. Register file is 16x16, 4R2W.
- Rebuild A* after merges touching _astar.pyx: `.venv/bin/python setup.py build_ext --inplace`.
