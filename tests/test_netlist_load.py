import gzip
import hashlib
import json

from minecraft_v.build_utils import load_artifact_json, save_build_artifacts
from minecraft_v.main import module_to_component_list
from minecraft_v.models import Netlist

NETLIST = {
    "creator": "yosys",
    "modules": {"main": {
        "ports": {"a": {"direction": "input", "bits": [2]}, "y": {"direction": "output", "bits": [3]}},
        "cells": {
            "n": {"hide_name": 1, "type": "$_NOT_", "parameters": {}, "attributes": {"src": "x.v:1"},
                  "port_directions": {"A": "input", "Y": "output"}, "connections": {"A": [2], "Y": [3]}},
            "p": {"hide_name": 1, "type": "$print", "parameters": {}, "attributes": {"src": "x.v:2"},
                  "connections": {}},
        },
        "netnames": {"a": {"hide_name": 0, "bits": [2], "attributes": {"src": "x.v:1"}},
                     "r": {"hide_name": 0, "bits": [3], "attributes": {"init": "1"}}},
    }},
}


def test_load_keeps_what_a_build_reads():
    module = Netlist.from_json(json.dumps(NETLIST)).modules["main"]
    assert module.cells["n"].connections == {"A": [2], "Y": [3]}
    assert module.cells["n"].attributes == {}  # source locations of other cells are dropped
    assert module.cells["p"].attributes == {"src": "x.v:2"}
    assert list(module.netnames) == ["r"]  # only the init


def test_consume_gives_the_same_components_and_empties_the_cells():
    kept = Netlist.from_json(json.dumps(NETLIST)).modules["main"]
    eaten = Netlist.from_json(json.dumps(NETLIST)).modules["main"]
    assert module_to_component_list(eaten, consume=True) == module_to_component_list(kept)
    assert eaten.cells == {} and kept.cells


def test_saved_component_list_is_json_of_the_list_and_returns_its_digest(tmp_path):
    comp = module_to_component_list(Netlist.from_json(json.dumps(NETLIST)).modules["main"])
    digest = save_build_artifacts(comp, tmp_path)
    data = gzip.decompress((tmp_path / "component_list.json.gz").read_bytes())
    assert load_artifact_json(tmp_path / "component_list.json") == json.loads(data)  # either name reads the .gz
    assert hashlib.sha256(data).hexdigest() == digest
    assert json.loads(data) == json.loads(comp.model_dump_json())


def test_artifacts_read_by_either_name_and_a_saved_one_replaces_an_older_plain_file(tmp_path):
    from minecraft_v.build_utils import artifact_exists, open_artifact, save_artifact
    (tmp_path / "component_list.json").write_text("stale")
    comp = module_to_component_list(Netlist.from_json(json.dumps(NETLIST)).modules["main"])
    save_build_artifacts(comp, tmp_path)
    assert not (tmp_path / "component_list.json").exists()  # else a reader would take the stale one
    assert artifact_exists(tmp_path / "component_list.json")
    path = save_artifact("layout.json", [{"a": 1}], tmp_path, indent=None, compress=True)
    assert path.name == "layout.json.gz" and load_artifact_json(tmp_path / "layout.json") == [{"a": 1}]
    (tmp_path / "plain.json").write_text("[1]")
    with open_artifact(tmp_path / "plain.json") as f:
        assert json.load(f) == [1]
