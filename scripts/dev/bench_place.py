"""Rerun placement alone from a saved placer input, for timing and profiling the placer.

Save the input with a build run under MINECRAFT_V_PLACE_DUMP=<file>.pkl, then:

    uv run python scripts/dev/bench_place.py <file>.pkl [--seed N] [--profile out.prof] [--out placed.pkl]
"""
from __future__ import annotations

import argparse
import cProfile
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minecraft_v.placement_engine.poisson import _poisson_placement  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--seed", type=int, default=None, help="override the saved place seed")
    ap.add_argument("--profile", type=Path, default=None, help="write cProfile stats here")
    ap.add_argument("--out", type=Path, default=None, help="pickle the placed origins here")
    args = ap.parse_args()
    with open(args.dump, "rb") as f:
        components, nets, kwargs = pickle.load(f)
    if args.seed is not None:
        kwargs["seed"] = args.seed
    prof = cProfile.Profile() if args.profile else None
    t0 = time.monotonic()
    if prof:
        prof.enable()
    placed = _poisson_placement(components, nets, **kwargs)
    if prof:
        prof.disable()
        prof.dump_stats(args.profile)
    took = time.monotonic() - t0
    hi = [max(p.origin[a] + (p.component.footprint.width, p.component.footprint.height,
                             p.component.footprint.depth)[a] for p in placed) for a in range(3)]
    print(f"placed {len(placed)} components in {took:.1f}s, extent {tuple(hi)}")
    if args.out:
        with open(args.out, "wb") as f:
            pickle.dump([(p.component.id, p.origin) for p in placed], f)


if __name__ == "__main__":
    main()
