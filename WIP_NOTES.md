# RAM vertical stacking: work in progress

Worktree was at 97db436; fast-forwarded to claude/floweypu-example-checklist-ca457a (e51d4f0) to get ram.py.

## Measured (scripts/ram_measure.py, floweypu_hi netlist)
- Group 1086 x 252 x 405, 7 stacks, 58 lanes (36 shared by 6 stacks, 22 by 4). Free height under 376: 124 before gates and wire; q1 uses 53 above the group.
- X: bit columns 266 (24%), decoders about 806 (74%), rest gaps. Stack X: btb 210, hist 126, takes 150 x 4, regfile 150.
- Y: lanes 118 (47%), banks 126, exits 8. Z: rows 388, north 5, south 7, lanes 5.
- q1 gate domain (380, 256, 1167).

## Estimates (not built)
- Decoder fold (write lines on plane 5 under read a/c decoder): X about 816, Y 252, Z about 408.
- Two tiers as is: 118 + 136 + 134 = 388 high: over 376. Needs lanes reduced.
- Lanes in two bundles, north and south of towers, sharing Ys: lift about 60. Pins from the north bundle go round the end of the south lanes; pair lanes so same-column towers don't clash.
- With that: takes as 2 stacks of 16 banks: X about 604 folded, Y about 322. Two tiers: X about 432, Y about 328. Upper levels add 64-67 tower ticks; narrower lanes and lower lift give back about 70 by arithmetic. Check the 240-tick gap in _DELAY_TAPS.
- Fold alone may fit the plot: gate band about 471 wide, 585 deep at density .15 and Y 256. Unproven.

## Code state
- RAM_FOLD=1 in ram.py, default off, bare stack harness only so far (check_ram.py array, 4 reads, stacks have no lanes): 4x2 and 8x3 with 2 banks: 0 failures in redstone and redpiler. 64x2 with 8 banks: NO RESULT, printed its size (87x134x403, versus 125 wide unfolded) but no pass lines, so it most likely hit the 500s alarm: rerun with 1000s. The group with lanes and the dynamic tower order on lanes are untested.
- tests/ not run on this diff. bphist and bpmem not routed. No cold time, routing or timing comparison yet.
- scripts/ram_measure.py added.

## Next
1. Rerun the arrays with RAM_FOLD=1, then tests/test_ram.py and tests/.
2. Add a fold test; bphist then bpmem route with model check (scratchpad bpcheck.py), one seed, against current.
3. Lane bundles, then tall takes or tiers via a -D STACK_BANKS option in the techmap.
4. Report table; ask parent for a full floweypu build.

Prior work: Puttaswamy and Loh bit, word and port partitioning of 3D SRAM: port partitioning best for multiported; the fold is a similar idea. https://courses.grainger.illinois.edu/CS533/sp2023/notes/stack2.pdf
