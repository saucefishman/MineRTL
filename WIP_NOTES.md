# WIP: branch vs main regression check (stopped early: machine sleeping)

Base: this worktree started at main, not the floweypu branch; fast-forwarded to claude/floweypu-example-checklist-ca457a (07f6c82).
Scripts and logs: /private/tmp/claude-501/-Users-saucefishman-Projects-minecraft-v--claude-worktrees-floweypu-example-checklist-ca457a/ab94c7fc-cc16-42ef-8f67-09079afc2e0c/scratchpad (diffsim.py, features.sh, mktree.sh, results/*.tsv). diffsim drives builds with random inputs in mchprs_headless (redstone mode only) against Yosys-proc'd RTL in iverilog.

## Results so far
- Same on main and branch, 0 failed, 0 verify, 0 sim mismatches: and_gate counter counteradvanced helloworld helloworld2 mem memread multibit multimodule multionebit mux or_gate powerup powerup_inverted printing regs runner sevenseg sevensegplus writeback halfvideo. Branch was faster on every one of these; timings not rerun on a quiet machine.
- pinext, video: fail pin target routing on BOTH trees (not a regression). The memory figure of video about 127s was not reproduced.
- cpu: REGRESSION. main clean on seeds 0 and 1 (270s, 375s; 0 sim mismatches). Branch fails routing on both seeds.
  1. Cause A, fixed here, uncommitted until this WIP: _BANK_STACK_HEIGHT split routed banks into two stacks (domain 574 wide vs 356). Now only south_pins RAM blocks split. Tests in tests/test_poisson.py. Not enough alone: cpu then fails on net_1565.
  2. Cause B, likely, one seed only: fe47654 slope-foot repeater ban. With it reverted on top of A (trees/slope), cpu seed 0 is clean: 124s, 0 failed, 0 verify, 320 bits, 0 mismatches. NOT a fix to apply as is: that rule fixes real opens and shorts on floweypu. ROUTE_FIRST_FANOUT=0 did not help.
- Features, regs and runner, both trees same and clean in simulation: --placer legacy/bary/sa (with seed, workers, no-y/z/flow/mirror), --flat, --route-passes 3, --route-workers 2, --wire-timeout, --route-deadline, --no-macros, --macro-min-instances, --no-settle, spacing flags, --no-debug-prints, --generate-clock none, --server-clock N. Also halfvideo and memread with sa, bary as listed in results/feat.tsv.
- Broken on both trees, not to fix: --debug (IndexError in rip_up; on branch runner segfaults, rc=-11), sevenseg --flat (pin target), halfvideo --route-passes 3 (support block).
- --placer cluster does not exist on main or branch.
- mem --no-macros: main ok; branch once failed verify (open net_1437, raises after repair), but passed 0 verify, 0 mismatches when rerun with --allow-routing-failures and --resume-dir. Unresolved: rerun plain, twice.
- Tests at the stack fix: 662 passed, 1 xfailed, 1 xpassed.

## Not run
Physical clock (--no-server-clock, add_clock.py), --tiled-ram, --resume-reroute, branch-only examples (bphist bpmem preload ramtile ramwalk), check_memread, check_powerup, check_ram, check_preload, compare_gate_rtl, redpiler mode, floweypu (by instruction). No deprecations made.

## Next
1. Decide cpu fix: keep the slope-foot ban for safety but give long crossings a way through (e.g. allow the repeater where the block under the slope is safe), confirm on cpu seeds 0 and 1 and on floweypu acceptance. Bisect step at 60b15e2 was killed unfinished.
2. Rerun the Not run list and mem --no-macros.
3. Remove the scratch main worktree: git worktree remove <scratchpad>/main_tree. Still registered.
