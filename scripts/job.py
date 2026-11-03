#!/usr/bin/env python3
"""A machine-wide job queue, so builds and simulations don't run the machine out of memory.

    job.py run --mem 7G --name floweypu-build --timeout 2400 -- ./long_command args
    job.py status

`run` blocks until the job gets a slot, then runs it and exits with its code. There's no daemon: every waiting client
takes the lock and works out who may start. A job starts when its memory fits in what's free right now and a core is
spare. Memory is what's measured free plus an allowance for swapping, less what running jobs have declared but not
touched yet. Peak memory per --name is remembered, so a job that declared too little declares enough next time.

Standard library only, so any worktree can call it with any python3. State is in $JOB_DIR, by default ~/.cache/mcv-jobs.
"""
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_DIR = Path(os.environ.get("JOB_DIR", Path.home() / ".cache" / "mcv-jobs"))
_GB = 1 << 30
_DEFAULT_MEM = 2 * _GB
_RESERVE = 1 * _GB  # kept free for everything not in the queue
# Most jobs aren't urgent, so running slowly on swap beats waiting. --exclusive is for the ones whose time matters.
_OVERCOMMIT = int(float(os.environ.get("JOB_OVERCOMMIT_GB", 10)) * _GB)
_SWAP_CAP = 16 * _GB  # the system grows swap as needed, so the limit is on how much is in use
_SWAP_LOW = 1 * _GB  # less room than this under the cap: nothing new starts, and the lowest priority job pauses
_SWAP_OK = 2 * _GB  # paused jobs carry on once this much is free
_PEAK_MARGIN = 1.1
_MAX_BACKFILL_WAIT = 600  # seconds the head of the queue waits before smaller jobs stop passing it
_VERSION = 4
_AGING = 60.0  # seconds of waiting worth one level of priority
_POLL = 2.0
_SAMPLE = 5.0


def _parse_mem(text: str) -> int:
    match = re.fullmatch(r"([\d.]+)([KMG]?)B?", text.strip().upper())
    if not match:
        raise argparse.ArgumentTypeError(f"bad memory size {text!r}")
    return int(float(match[1]) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": _GB}[match[2]])


def _gb(n: float) -> str:
    return f"{n / _GB:.1f}G"


def available_memory() -> int:
    """Bytes the system can hand out without swapping: free pages and ones it can drop."""
    if "JOB_FAKE_FREE" in os.environ:
        return int(os.environ["JOB_FAKE_FREE"])
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+)", out)[1])
    pages = {m[1]: int(m[2]) for m in re.finditer(r"^(.+?):\s+(\d+)\.$", out, re.M)}
    return page * sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))


def swap_free() -> int:
    """Bytes of swap still usable, under the cap."""
    if "JOB_FAKE_SWAP" in os.environ:
        return int(os.environ["JOB_FAKE_SWAP"])
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    return _SWAP_CAP - int(float(re.search(r"used = ([\d.]+)M", out)[1]) * (1 << 20))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _cpu_seconds(text: str) -> float:
    days, _, rest = text.rpartition("-")
    total = 0.0
    for part in rest.split(":"):
        total = total * 60 + float(part)
    return total + (int(days) * 86400 if days else 0)


def _group_usage(pgid: int) -> tuple[int, dict[int, float]]:
    """Resident bytes of a process group, and CPU seconds per process in it."""
    out = subprocess.run(["ps", "-Ao", "pid=,pgid=,rss=,time="], capture_output=True, text=True).stdout
    rss = 0
    cpu = {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 4 and int(fields[1]) == pgid:
            rss += int(fields[2]) * 1024
            cpu[int(fields[0])] = _cpu_seconds(fields[3])
    return rss, cpu


@contextmanager
def _state():
    """The queue's state, under the lock, written back afterwards."""
    _DIR.mkdir(parents=True, exist_ok=True)
    with open(_DIR / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = _DIR / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {"jobs": {}, "peaks": {}}
        for key, job in list(state["jobs"].items()):
            if not _alive(job["client"]) and not (job.get("child") and _alive(job["child"])):
                del state["jobs"][key]
        yield state
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(path)


def _need(job: dict, peaks: dict) -> int:
    """What it will use: its last peak once there is one, as a declared size is usually a guess on the safe side."""
    peak = peaks.get(job["name"], 0)
    return int(peak * _PEAK_MARGIN) if peak else job["mem"]


def _order(job: dict) -> tuple:
    """Who goes first. Waiting counts as priority, so low priority jobs get their turn."""
    return (-job["priority"] - (time.time() - job["queued"]) / _AGING, job["queued"])


def may_start(key: str, state: dict, free: int, swap: int, cores: int, now: float) -> bool:
    """Whether the waiting job `key` may start now. A machine with nothing queued running always starts something."""
    jobs = state["jobs"]
    peaks = state["peaks"]
    running = [j for j in jobs.values() if j["state"] != "waiting"]
    # A client keeps the code it started with, so one from an older version may never go when this one thinks it should.
    waiting = sorted((j for j in jobs.values() if j["state"] == "waiting" and j.get("version") == _VERSION), key=_order)
    me = jobs[key]
    head = waiting[0]
    if not running:
        return me is head
    if me["exclusive"] or any(j["exclusive"] for j in running) or swap < _SWAP_LOW:
        return False
    if me is not head and head["exclusive"]:
        return False
    budget = free + _OVERCOMMIT - _RESERVE - sum(max(0, _need(j, peaks) - j.get("high", 0)) for j in running)
    if sum(j["cores"] for j in running) + me["cores"] > cores:
        return False
    # Anyone ahead who fits goes first, so don't take their room. Nor that of the first in line once it has waited long,
    # fitting or not: smaller jobs may pass it, but not forever.
    for job in waiting:
        if job is me:
            break
        if _need(job, peaks) <= budget or (job is head and now - job["queued"] > _MAX_BACKFILL_WAIT):
            budget -= _need(job, peaks)
    return _need(me, peaks) <= budget


def _pause_choice(state: dict) -> str | None:
    """The job to pause when swap runs out: lowest priority, newest. Never the only one still running."""
    active = [(k, j) for k, j in state["jobs"].items() if j["state"] == "running"]
    if len(active) < 2:
        return None
    return max(active, key=lambda item: (_order(item[1]), item[0]))[0]


def run(args: argparse.Namespace) -> int:
    key = f"{os.getpid()}-{time.time_ns()}"
    name = args.name or Path(args.command[0]).name
    job = {"name": name, "mem": args.mem, "cores": args.cores, "priority": args.priority, "exclusive": args.exclusive,
           "queued": time.time(), "version": _VERSION, "client": os.getpid(), "state": "waiting", "command": " ".join(args.command)[:200]}
    with _state() as state:
        state["jobs"][key] = job
    child = None

    def stop(signum, _frame):
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGCONT)
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        with _state() as state:
            state["jobs"].pop(key, None)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    cores = os.cpu_count() or 1
    while True:
        free, swap = available_memory(), swap_free()
        with _state() as state:
            if may_start(key, state, free, swap, cores, time.time()):
                state["jobs"][key]["state"] = "running"
                state["jobs"][key]["started"] = time.time()
                waited = time.time() - job["queued"]
                break
        time.sleep(_POLL)
    if waited > 1:
        print(f"[job] {name} started after waiting {waited:.0f}s", file=sys.stderr, flush=True)
    child = subprocess.Popen(args.command, start_new_session=True)
    with _state() as state:
        state["jobs"][key]["child"] = child.pid
    # time.monotonic doesn't count sleep here, so a closed lid can't use up a timeout.
    start = time.monotonic()
    peak = 0
    cpu: dict[int, float] = {}
    paused = False
    timed_out = None
    while child.poll() is None:
        try:
            child.wait(timeout=_SAMPLE)
            break
        except subprocess.TimeoutExpired:
            pass
        rss, seen = _group_usage(child.pid)
        peak = max(peak, rss)
        cpu.update(seen)
        swap = swap_free()
        with _state() as state:
            mine = state["jobs"][key]
            mine["rss"] = rss
            mine["high"] = peak  # idle pages get compressed, so what it holds now can read far below what it needs
            if not paused and swap < _SWAP_LOW and _pause_choice(state) == key:
                paused = True
                mine["state"] = "paused"
                os.killpg(child.pid, signal.SIGSTOP)
            elif paused and (swap >= _SWAP_OK or not any(j["state"] == "running" for j in state["jobs"].values())):
                paused = False
                mine["state"] = "running"
                os.killpg(child.pid, signal.SIGCONT)
        if args.timeout and time.monotonic() - start > args.timeout:
            timed_out = f"{args.timeout}s awake"
        elif args.cpu_timeout and sum(cpu.values()) > args.cpu_timeout:
            timed_out = f"{args.cpu_timeout}s CPU"
        if timed_out:
            print(f"[job] {name} over its limit of {timed_out}, stopping", file=sys.stderr, flush=True)
            os.killpg(child.pid, signal.SIGCONT)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            break
    code = child.wait()
    with _state() as state:
        state["jobs"].pop(key, None)
        state["peaks"][name] = max(state["peaks"].get(name, 0), peak)
    return 124 if timed_out else code if code >= 0 else 128 - code


def status() -> int:
    free, swap = available_memory(), swap_free()
    with _state() as state:
        jobs = sorted(state["jobs"].values(), key=lambda j: (j["state"] == "waiting", _order(j)))
        peaks = state["peaks"]
    print(f"free {_gb(free)}, swap free {_gb(swap)}, {sum(j['state'] != 'waiting' for j in jobs)} running, "
          f"{sum(j['state'] == 'waiting' for j in jobs)} waiting")
    now = time.time()
    for j in jobs:
        since = now - j.get("started", j["queued"])
        flags = (" exclusive" if j["exclusive"] else "") + (f" prio {j['priority']}" if j["priority"] else "")
        print(f"  {j['state']:8} {since:6.0f}s  {_gb(j.get('rss', 0)):>6} of {_gb(_need(j, peaks)):>6}  "
              f"{j['name']}{flags}: {j['command'][:60]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    go = sub.add_parser("run", help="wait for a slot, then run a command")
    go.add_argument("--mem", type=_parse_mem, default=_DEFAULT_MEM, help="memory it will use, like 7G (default 2G)")
    go.add_argument("--cores", type=int, default=1)
    go.add_argument("--name", help="job kind, for remembering its peak memory (default: the command)")
    go.add_argument("--priority", type=int, default=0, help="higher goes first: 10 final checks, -10 exploratory sweeps")
    go.add_argument("--exclusive", action="store_true", help="alone on the machine, for builds whose time matters")
    go.add_argument("--timeout", type=float, help="seconds running, not counting machine sleep or queueing")
    go.add_argument("--cpu-timeout", type=float, help="CPU seconds across the job's processes")
    go.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("status", help="what's running and waiting")
    args = parser.parse_args()
    if args.action == "status":
        return status()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("run needs a command after --")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
