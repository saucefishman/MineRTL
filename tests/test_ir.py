import json

import pytest

from minecraft_v.placement_engine.ir import (
    CardinalDirection, Component, ComponentList, ComponentType, Direction, Footprint, NetConnection, NetEndpoint,
    PinRef, PlacementHints, load_component_list,
)


def _list() -> ComponentList:
    gate = Component(id="g", type=ComponentType.NOT, footprint=Footprint(width=4, height=2, depth=3, layer="a"),
                     pins=[PinRef(name="A", direction=Direction.IN, side=CardinalDirection.SOUTH, offset=(0, 0, 0)),
                           PinRef(name="Y", direction=Direction.OUT, side=CardinalDirection.NORTH, offset=(1, 0, 2),
                                  const_value="0")],
                     params={"W": "1"}, metadata={"init": 1}, hints=PlacementHints(preferred_pos=(1, 2, 3), keepout=2))
    src = Component(id="in", type=ComponentType.INPUT_PIN, pins=[PinRef(name="in", direction=Direction.OUT)])
    return ComponentList(schema_version="0.1", components=[src, gate], nets=[
        NetConnection(net_id="net_1", source=NetEndpoint(component_id="in", pin_name="in"),
                      sinks=[NetEndpoint(component_id="g", pin_name="A")])])


def test_json_round_trip_is_equal():
    comp = _list()
    again = ComponentList.from_json(comp.model_dump_json())
    assert again == comp and again.components[1].pins[1].offset == (1, 0, 2)
    assert isinstance(again.components[1].type, ComponentType) and again.components[1].hints.preferred_pos == (1, 2, 3)


def test_dump_modes():
    pin = _list().components[1].pins[0]
    assert pin.model_dump() == {"name": "A", "direction": Direction.IN, "side": CardinalDirection.SOUTH,
                                "offset": (0, 0, 0), "const_value": None}
    assert pin.model_dump(mode="json") == {"name": "A", "direction": "IN", "side": "south", "offset": [0, 0, 0],
                                           "const_value": None}


def test_model_copy_updates_only_what_is_given():
    pin = _list().components[1].pins[0]
    moved = pin.model_copy(update={"offset": (1, 1, 1)})
    assert moved.offset == (1, 1, 1) and moved.name == "A" and pin.offset == (0, 0, 0)


def test_the_graph_is_checked():
    comp = _list()
    with pytest.raises(ValueError, match="bad pin"):
        ComponentList(schema_version="0.1", components=comp.components, nets=[
            NetConnection(net_id="n", source=NetEndpoint(component_id="in", pin_name="in"),
                          sinks=[NetEndpoint(component_id="g", pin_name="nope")])])
    with pytest.raises(ValueError, match="sink not IN"):
        ComponentList(schema_version="0.1", components=comp.components, nets=[
            NetConnection(net_id="n", source=NetEndpoint(component_id="in", pin_name="in"),
                          sinks=[NetEndpoint(component_id="g", pin_name="Y")])])
    with pytest.raises(ValueError, match="duplicate component id"):
        ComponentList(schema_version="0.1", components=comp.components * 2, nets=[])
    with pytest.raises(ValueError, match="dup pin names"):
        Component(id="x", type=ComponentType.AND, pins=[PinRef(name="A", direction=Direction.IN)] * 2)
    with pytest.raises(ValueError, match="bad schema"):
        ComponentList(schema_version="9", components=[], nets=[])


def test_load_component_list_from_a_file(tmp_path):
    path = tmp_path / "component_list.json"
    path.write_text(_list().model_dump_json())
    assert load_component_list(path) == _list()
    assert load_component_list(json.loads(path.read_text())) == _list()


def test_wide_components_are_looked_up_by_name():
    pins = [PinRef(name=f"p{i}", direction=Direction.IN) for i in range(20)]
    wide = Component(id="w", type=ComponentType.CUSTOM, pins=pins + [PinRef(name="o", direction=Direction.OUT)])
    src = Component(id="s", type=ComponentType.INPUT_PIN, pins=[PinRef(name="s", direction=Direction.OUT)])
    ComponentList(schema_version="0.1", components=[src, wide], nets=[
        NetConnection(net_id="n", source=NetEndpoint(component_id="s", pin_name="s"),
                      sinks=[NetEndpoint(component_id="w", pin_name="p17")])])


def test_a_pin_comes_back_from_its_json_dump():
    for pin in _list().components[1].pins + [PinRef(name="x", direction=Direction.IN)]:
        assert PinRef.model_validate(json.loads(json.dumps(pin.model_dump(mode="json")))) == pin
