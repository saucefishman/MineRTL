import hashlib
import json

from minecraft_v.build_utils import save_build_artifacts
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
    data = (tmp_path / "component_list.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    assert json.loads(data) == json.loads(comp.model_dump_json())
