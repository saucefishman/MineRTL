"""Check that a refactor leaves build output unchanged: hash each example's placed blocks.

    uv run python scripts/check_same_output.py --save build/same_output.json     # record baseline
    uv run python scripts/check_same_output.py --compare build/same_output.json  # after a change
    uv run python scripts/check_same_output.py --compare base.json regs -- --route-passes 3

Builds run through bench_examples.run_example (same caps, PYTHONHASHSEED=0, per-build dirs).
The hash covers every region's block array and palette, not the file bytes (those carry
timestamps). Builds that were killed or failed are reported, not hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from litemapy import Schematic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_examples import EXCLUDED_BY_DEFAULT, ROOT, run_example  # noqa: E402


def litematic_hash(path: Path) -> str:
    schem = Schematic.load(str(path))
    h = hashlib.sha256()
    for name in sorted(schem.regions):
        region = schem.regions[name]
        h.update(name.encode())
        h.update(repr((region.x, region.y, region.z, region.width, region.height, region.length)).encode())
        h.update(repr([str(b) for b in region._Region__palette]).encode())
        h.update(region._Region__blocks.tobytes())
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("examples", nargs="*", help="example names (default: all except video)")
    parser.add_argument("--cap", type=float, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "same_output")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", type=Path, help="write hashes to this JSON file")
    group.add_argument("--compare", type=Path, help="compare hashes against this JSON file")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    examples = args.examples or sorted(
        p.name for p in (ROOT / "examples").iterdir() if p.is_dir() and p.name not in EXCLUDED_BY_DEFAULT)

    def build(ex: str) -> tuple[str, str, str | None]:
        line = run_example(ex, args.cap, extra, args.out)
        result = args.out / ex / "result.litematic"
        ok = line.split()[1] == "ok" and result.exists()
        return ex, line, litematic_hash(result) if ok else None

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(build, examples))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({ex: h for ex, _, h in results}, indent=1))
        for _, line, h in results:
            print(f"{line}  {h or '-'}")
        return

    baseline = json.loads(args.compare.read_text())
    mismatches = 0
    for ex, line, h in results:
        expected = baseline.get(ex)
        same = h == expected
        mismatches += not same
        print(f"{'SAME' if same else 'DIFF'}  {line}  {h or '-'} (baseline {expected or '-'})")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
