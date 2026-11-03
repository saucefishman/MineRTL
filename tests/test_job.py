"""scripts/job.py against real clients, with memory and swap faked through the environment."""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "job.py"
GB = 1 << 30


def _env(tmp_path, free=64, swap=64, overcommit=0):
    return {**os.environ, "JOB_DIR": str(tmp_path), "JOB_FAKE_FREE": str(free * GB), "JOB_FAKE_SWAP": str(swap * GB),
            "JOB_OVERCOMMIT_GB": str(overcommit)}


def _start(env, log, tag, seconds, *flags):
    body = f"echo start {tag} >> {log}; sleep {seconds}; echo end {tag} >> {log}"
    return subprocess.Popen([sys.executable, SCRIPT, "run", *flags, "--", "sh", "-c", body], env=env)


def _events(log):
    return log.read_text().split("\n")[:-1]


def test_passes_on_the_exit_code(tmp_path):
    env = _env(tmp_path)
    assert subprocess.run([sys.executable, SCRIPT, "run", "--", "sh", "-c", "exit 3"], env=env).returncode == 3


def test_jobs_that_fit_run_together(tmp_path):
    env, log = _env(tmp_path), tmp_path / "log"
    jobs = [_start(env, log, i, 3, "--mem", "1G", "--name", f"t{i}") for i in range(2)]
    assert [j.wait(30) for j in jobs] == [0, 0]
    assert sorted(_events(log)[:2]) == ["start 0", "start 1"]


def test_jobs_that_dont_fit_take_turns(tmp_path):
    env, log = _env(tmp_path, free=8), tmp_path / "log"
    first = _start(env, log, "a", 3, "--mem", "6G", "--name", "a")
    time.sleep(1)
    second = _start(env, log, "b", 0, "--mem", "6G", "--name", "b")
    assert [first.wait(30), second.wait(30)] == [0, 0]
    assert _events(log) == ["start a", "end a", "start b", "end b"]


def test_overcommit_lets_jobs_swap(tmp_path):
    env, log = _env(tmp_path, free=8, overcommit=6), tmp_path / "log"
    first = _start(env, log, "a", 3, "--mem", "6G", "--name", "a")
    time.sleep(1)
    second = _start(env, log, "b", 0, "--mem", "6G", "--name", "b")
    assert [first.wait(30), second.wait(30)] == [0, 0]
    assert _events(log).index("start b") < _events(log).index("end a")


def test_a_job_bigger_than_the_machine_still_runs_alone(tmp_path):
    env, log = _env(tmp_path, free=4), tmp_path / "log"
    assert _start(env, log, "big", 0, "--mem", "100G").wait(30) == 0


def test_exclusive_runs_alone(tmp_path):
    env, log = _env(tmp_path), tmp_path / "log"
    first = _start(env, log, "a", 3, "--mem", "1M", "--name", "a")
    time.sleep(1)
    solo = _start(env, log, "solo", 3, "--exclusive", "--mem", "1M", "--name", "solo")
    time.sleep(1)
    last = _start(env, log, "z", 0, "--mem", "1M", "--name", "z")
    assert [first.wait(40), solo.wait(40), last.wait(40)] == [0, 0, 0]
    assert _events(log) == ["start a", "end a", "start solo", "end solo", "start z", "end z"]


def test_priority_goes_first(tmp_path):
    env, log = _env(tmp_path, free=8), tmp_path / "log"
    first = _start(env, log, "a", 4, "--mem", "6G", "--name", "a")
    time.sleep(1)
    low = _start(env, log, "low", 0, "--mem", "6G", "--name", "low", "--priority", "-10")
    time.sleep(0.5)
    high = _start(env, log, "high", 0, "--mem", "6G", "--name", "high", "--priority", "10")
    assert [first.wait(40), low.wait(40), high.wait(40)] == [0, 0, 0]
    assert _events(log).index("start high") < _events(log).index("start low")


def test_timeout_stops_the_job_and_its_children(tmp_path):
    env = _env(tmp_path)
    marker = tmp_path / "survived"
    body = f"(sleep 12; touch {marker}) & wait"
    done = subprocess.run([sys.executable, SCRIPT, "run", "--timeout", "1", "--", "sh", "-c", body], env=env, timeout=30)
    assert done.returncode == 124
    time.sleep(8)
    assert not marker.exists()


def test_a_killed_client_frees_its_slot(tmp_path):
    env, log = _env(tmp_path, free=8), tmp_path / "log"
    first = _start(env, log, "a", 2, "--mem", "6G", "--name", "a")
    time.sleep(1)
    first.kill()  # the client goes without cleaning up; its job ends soon after
    first.wait()
    second = _start(env, log, "b", 0, "--mem", "6G", "--name", "b")
    assert second.wait(30) == 0
    assert _events(log)[-1] == "end b"


def test_low_swap_stops_new_jobs_until_it_recovers(tmp_path):
    env, log = _env(tmp_path, swap=0), tmp_path / "log"
    first = _start(env, log, "a", 2, "--mem", "1M", "--name", "a")
    time.sleep(1)
    second = _start(env, log, "b", 0, "--mem", "1M", "--name", "b")
    assert [first.wait(30), second.wait(30)] == [0, 0]
    assert _events(log) == ["start a", "end a", "start b", "end b"]


def test_status_lists_jobs(tmp_path):
    env, log = _env(tmp_path), tmp_path / "log"
    job = _start(env, log, "a", 3, "--mem", "1G", "--name", "listed")
    time.sleep(1.5)
    out = subprocess.run([sys.executable, SCRIPT, "status"], env=env, capture_output=True, text=True).stdout
    assert "listed" in out and "running" in out
    assert job.wait(30) == 0


def test_waiting_overtakes_priority():
    spec = importlib.util.spec_from_file_location("job", SCRIPT)
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    now = time.time()
    fresh_high = {"priority": 10, "queued": now}
    old_low = {"priority": -10, "queued": now - 21 * job._AGING}
    assert job._order(old_low) < job._order(fresh_high)
    assert job._order({"priority": 10, "queued": now}) < job._order({"priority": -10, "queued": now})


def _job_module():
    spec = importlib.util.spec_from_file_location("job", SCRIPT)
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    return job


def _queued(job, state, mem, waited=0.0):
    return {"name": "n", "mem": mem * GB, "cores": 1, "priority": 0, "exclusive": False, "state": state,
            "queued": time.time() - waited, "version": job._VERSION}


def test_a_long_waiter_that_doesnt_fit_only_holds_its_own_room():
    job = _job_module()
    jobs = {"run": _queued(job, "running", 1) | {"high": GB}, "big": _queued(job, "waiting", 20, 9999),
            "small": _queued(job, "waiting", 1, 1)}
    state = {"jobs": jobs, "peaks": {}}
    assert not job.may_start("small", state, 10 * GB, 8 * GB, 8, time.time())
    assert job.may_start("small", state, 30 * GB, 8 * GB, 8, time.time())


def test_a_client_on_an_older_version_blocks_nobody():
    job = _job_module()
    stale = _queued(job, "waiting", 20, 9999) | {"priority": 20}
    del stale["version"]
    jobs = {"run": _queued(job, "running", 1) | {"high": GB}, "stale": stale, "new": _queued(job, "waiting", 1, 1)}
    assert job.may_start("new", {"jobs": jobs, "peaks": {}}, 8 * GB, 8 * GB, 8, time.time())


def test_a_known_peak_replaces_the_declared_size():
    job = _job_module()
    assert job._need({"name": "a", "mem": 7 * GB}, {"a": GB}) == int(GB * job._PEAK_MARGIN)
    assert job._need({"name": "b", "mem": 7 * GB}, {"a": GB}) == 7 * GB
