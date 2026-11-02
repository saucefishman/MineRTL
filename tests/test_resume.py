from types import SimpleNamespace

from minecraft_v.placement_engine.resume import ResumeStore


def test_second_ask_for_a_laid_target_keeps_its_path(tmp_path):
    a, b, c = (0, 0, 0), (1, 0, 0), (2, 0, 0)
    store = ResumeStore(tmp_path)
    store.remember("n", c, [a, b, c])
    store.remember("n", c, [c])  # already on the wire: this search has nothing to lay
    store.save_paths()
    assert ResumeStore(tmp_path).path_for("n", c) == [a, b, c]


def test_new_route_to_a_target_replaces_its_path(tmp_path):
    a, b, c = (0, 0, 0), (1, 0, 0), (2, 0, 0)
    store = ResumeStore(tmp_path)
    store.remember("n", c, [a, c])
    store.remember("n", c, [a, b, c])
    store.save_paths()
    assert ResumeStore(tmp_path).path_for("n", c) == [a, b, c]


def test_jobs_replay_in_the_order_they_last_finished(tmp_path):
    store = ResumeStore(tmp_path)
    for key in ("a", "b", "c", "a"):  # a was ripped and routed again
        store.finished(key)
    store.save_paths()
    jobs = [SimpleNamespace(key=key) for key in ("a", "new", "b", "c")]
    assert [job.key for job in ResumeStore(tmp_path).in_saved_order(jobs)] == ["b", "c", "a", "new"]


def test_targets_replay_in_the_order_they_were_laid(tmp_path):
    a, b, c, d = (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)
    store = ResumeStore(tmp_path)
    store.remember("n", b, [a, b])
    store.remember("n", c, [a, c])
    store.remember("n", b, [c, b])  # ripped and routed again, after c
    store.save_paths()
    assert ResumeStore(tmp_path).targets_in_saved_order("n", [b, d, c]) == [c, b, d]


def test_waypoints_keep_their_cells_unless_rerouted(tmp_path):
    wanted, cell, other = (0, 0, 0), (1, 0, 0), (2, 0, 0)
    store = ResumeStore(tmp_path)
    assert store.waypoint_cell("n", wanted, lambda: cell) == cell
    assert store.waypoint_cell("gone", wanted, lambda: None) is None
    store.save_paths()
    replay = ResumeStore(tmp_path)
    assert replay.waypoint_cell("n", wanted, lambda: other) == cell
    assert replay.waypoint_cell("gone", wanted, lambda: other) is None
    assert ResumeStore(tmp_path, {"n"}).waypoint_cell("n", wanted, lambda: other) == other
