"""Build examples in parallel with a hard per-build time limit; report routing + verify results.

    uv run python scripts/bench_examples.py                      # all examples except video
    uv run python scripts/bench_examples.py regs sevenseg -- --route-passes 4
    uv run python scripts/bench_examples.py --cap 30 --workers 4 counter

Each build gets its own artifacts dir and output under build/bench/<example>/, so builds
don't clobber each other. Builds over --cap seconds are killed (whole process group).
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_BY_DEFAULT = {"video"}


def run_example(example: str, cap: float, extra: list[str], out_root: Path) -> str:
    out_dir = out_root / example
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    cmd = ["./synth.sh", f"examples/{example}", "--artifacts-dir", str(out_dir),
           "--out-litematic", str(out_dir / "result.litematic"), "--allow-routing-failures", "True", *extra]
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=cap)
            status = "ok" if rc == 0 else f"rc={rc}"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            status = "KILLED"
    elapsed = time.monotonic() - t0
    text = log_path.read_text(errors="replace")
    done = re.findall(r"\[wire\] done \((\d+) nets, (\d+) failed", text)
    verify = re.findall(r"\[verify\] (\d+) violation", text)
    unsupported = re.findall(r"\[verify\] (\d+) unsupported", text)
    routed = f"{done[-1][1]}/{done[-1][0]} failed" if done else "-"
    # verify summaries are printed only when nonzero
    violations = f"{verify[-1] if verify else 0} verify violations" if done else ""
    if unsupported and unsupported[-1] != "0":
        violations += f", {unsupported[-1]} unsupported"
    error = ""
    if status not in ("ok", "KILLED"):
        lines = [line for line in text.splitlines() if line.strip()]
        error = lines[-1][:100] if lines else ""
    return f"{example:<16} {status:<7} {elapsed:6.1f}s  {routed:<12} {violations:<20} {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("examples", nargs="*", help="example names (default: all except video)")
    parser.add_argument("--cap", type=float, default=60, help="per-build time limit in seconds (default 60)")
    parser.add_argument("--workers", type=int, default=4, help="parallel builds (default 4)")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "bench", help="output root")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    examples = args.examples or sorted(
        p.name for p in (ROOT / "examples").iterdir() if p.is_dir() and p.name not in EXCLUDED_BY_DEFAULT)
    with ThreadPoolExecutor(args.workers) as pool:
        for line in pool.map(lambda ex: run_example(ex, args.cap, extra, args.out), examples):
            print(line, flush=True)


if __name__ == "__main__":
    main()
