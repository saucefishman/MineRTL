import os

import pytest

from minecraft_v.placement_engine import forks
from minecraft_v.placement_engine.forks import Forked


@pytest.mark.parametrize("workers", ["1", "2"])
def test_forked_sees_the_state_at_construction(monkeypatch, workers):
    monkeypatch.setenv("MINECRAFT_V_WORKERS", workers)
    state = [1]
    job = Forked(lambda: state[0])
    state[0] = 2  # after construction: the job must not see it, forked or not
    assert job.result() == 1


@pytest.mark.parametrize("workers", ["1", "2"])
def test_forked_raises_from_result(monkeypatch, workers):
    monkeypatch.setenv("MINECRAFT_V_WORKERS", workers)

    def boom():
        raise ValueError("bad")

    job = Forked(boom)  # constructing never raises
    with pytest.raises(ValueError, match="bad"):
        job.result()


def test_children_only_as_many_as_the_free_memory_takes(monkeypatch):
    monkeypatch.setenv("MINECRAFT_V_WORKERS", "4")
    here = os.getpid()
    monkeypatch.setattr(forks, "parent_bytes", lambda: 1000)
    items = list(range(640))
    monkeypatch.setattr(forks, "free_bytes", lambda: 10 ** 9)
    assert len(set(forks.fork_map(lambda i: os.getpid(), items, 4))) == 4
    monkeypatch.setattr(forks, "free_bytes", lambda: 700)  # 490 usable: two children of 200 fit
    assert len(set(forks.fork_map(lambda i: os.getpid(), items, 4))) == 2
    monkeypatch.setattr(forks, "free_bytes", lambda: 1)
    assert set(forks.fork_map(lambda i: os.getpid(), items, 4)) == {here}
    assert forks.fork_map(lambda i: i * 2, items, 4) == [i * 2 for i in items]
    assert forks.Forked(lambda: os.getpid()).result() == here  # no room for a child of 750: runs here
    monkeypatch.setattr(forks, "free_bytes", lambda: 10 ** 9)
    assert forks.Forked(lambda: os.getpid()).result() != here
