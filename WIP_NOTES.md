# Floweypu footprint: paused for machine sleep

Branch worktree-agent-a704db239106ebe52, fast-forwarded to 07f6c82 (it had been cut at 97db436, 77 commits behind).
Target from the user: fit 1024x1024 XZ, height 376, under 1200s, whole bar. Scope: gate placement, clearances, margins,
legalization in poisson.py. Agent ram-stack owns splitting the RAM group; my fold edit to ram.py was reverted.

## Measured, q1 (1177x305x1164, 417.9M volume, 11.77M non-air, density 0.0282, 39% of XZ columns used)
- RAM group: 1086x252x405 at origin (7,5,11), 110.8M, 26.5% of volume. Y below 118 is lanes, nearly empty.
- X: 7 margin + 1086 RAM + 4 clear + 19 outputs at pitch 4 (x 1097..1169) + about 7. X cannot reach 1024 without a narrower
  RAM: the block alone is 1093 to its east end. Removing outputs from X entirely still leaves about 1100.
- Z: gates (14,929) sit at Z 935..1144 (1st pct 935, X 359..727). Z 430..930 is a void of long wires: the block's charge
  presses gates to the far end of the domain. So RAM_GATE_DENSITY mostly sets void length, not gate density.

## Runs (none passed the bar yet; none reached acceptance)
| run | settings | footprint | failed | routing | total | acceptance |
|---|---|---|---|---|---|---|
| q1 | density .15, width .35 | 1177x305x1164 | 0 | 577s | 711s | pass (earlier) |
| d30 | RAM_GATE_DENSITY=.30 | legal 1094x276x836, workspace 1177x317x824 | n/a | not reached | killed at 1300s | n/a |
| c250 | RAM_GATE_CHANNEL=250, pitch 4 | legal 1094x263x929 | unknown | unfinished, stopped for sleep | placement 65.9s | n/a |

d30 spent 1264.6s of 1300s in placement: legalization slows badly on a shorter domain. c250 avoids it (65.9s) by moving gates
as one after global placement. c250 was still routing, with 'No route' retries, when stopped; its time is unknown.

## Uncommitted idea now in this commit, untested
- RAM_GATE_CHANNEL (poisson.py), off by default: tested only by c250, unfinished.
- _SOUTH_PINS_OUTPUT_PITCH default changed from 4 to 2 (RAM_OUTPUT_PITCH), NOT built or tested. It would save about 38 X.
  Set it back to 4 or test it before relying on it; this also changes smaller south-pin designs.
- tests/ not run on this commit (test_ram.py passed 18, while a since-reverted ram.py edit was in place).

## Next
1. Rerun c250 (scratchpad b.sh, needs .venv symlink and kernels), then channels 150, 50, to find the Z floor and route time.
2. Test output pitch 2, then 3; try trimming margins.
3. Report XZ saved without changing the RAM block, then combine with ram-stack's narrower block for 1024x1024.
4. Acceptance on any candidate: scratchpad acc.sh. Run tests/ before making anything default.

RAM narrowing notes for ram-stack: regs stack s6 has no lanes, so it could sit on top of others (saves about 150 X). A high
address bit needs one line per bank, not two (about 156 X over the 6 big stacks), but the x + 2 'around' tap needs rework.
Scripts: scratchpad/{b.sh,acc.sh,measure.py} under the session's scratchpad directory.
