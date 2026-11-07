"""Rerun placement alone from a saved placer input, for timing and profiling the placer.

Save the input with a build run under `record`, which takes the same arguments as `python -m minecraft_v`:

    uv run python scripts/dev/bench_place.py record <file>.pkl --netlist build/artifacts/netlist.json --place-seed 2 ...

then rerun placement alone:

    uv run python scripts/dev/bench_place.py <file>.pkl [--seed N] [--profile out.prof] [--out placed.pkl] [--no-kernel]
"""
from __future__ import annotations

import argparse
import cProfile
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minecraft_v.placement_engine import poisson  # noqa: E402


def record(dump: Path, build_args: list[str]) -> None:
    """Run a build that saves the placer's input to dump."""
    import runpy

    from minecraft_v.placement_engine import builder

    place = builder._poisson_placement

    def saving(components, nets, **kwargs):
        with open(dump, "wb") as f:
            pickle.dump((components, nets, kwargs), f)
        return place(components, nets, **kwargs)

    builder._poisson_placement = saving
    sys.argv = ["minecraft_v", *build_args]
    runpy.run_module("minecraft_v", run_name="__main__")


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "record":
        record(Path(sys.argv[2]), sys.argv[3:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--seed", type=int, default=None, help="override the saved place seed")
    ap.add_argument("--profile", type=Path, default=None, help="write cProfile stats here")
    ap.add_argument("--out", type=Path, default=None, help="pickle the placed components and origins here")
    ap.add_argument("--no-kernel", action="store_true", help="the Python placer, without the C++ kernel")
    args = ap.parse_args()
    with open(args.dump, "rb") as f:
        components, nets, kwargs = pickle.load(f)
    if args.no_kernel:
        poisson._place = None
    if args.seed is not None:
        kwargs["seed"] = args.seed
    prof = cProfile.Profile() if args.profile else None
    t0 = time.monotonic()
    if prof:
        prof.enable()
    placed = poisson._poisson_placement(components, nets, **kwargs)
    if prof:
        prof.disable()
        prof.dump_stats(args.profile)
    took = time.monotonic() - t0
    hi = [max(p.origin[a] + (p.component.footprint.width, p.component.footprint.height,
                             p.component.footprint.depth)[a] for p in placed) for a in range(3)]
    print(f"placed {len(placed)} components in {took:.1f}s, extent {tuple(hi)}")
    if args.out:
        with open(args.out, "wb") as f:
            pickle.dump([(p.component.model_dump_json(), p.origin) for p in placed], f)


if __name__ == "__main__":
    main()
