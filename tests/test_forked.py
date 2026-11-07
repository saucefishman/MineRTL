import pytest

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
