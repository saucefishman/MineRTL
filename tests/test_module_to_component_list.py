import pytest

from minecraft_v.main import module_to_component_list
from minecraft_v.models import Cell, Module, Port
from minecraft_v.placement_ir import (
    CardinalDirection,
    ComponentType,
    Direction,
)


def _make_module(ports=None, cells=None):
    return Module(ports=ports or {}, cells=cells or {})


def _make_cell(type_, port_directions, connections, parameters=None):
    return Cell(
        hide_name=1,
        type=type_,
        parameters=parameters or {},
        port_directions=port_directions,
        connections=connections,
    )


# --- helpers ---

def _comp(result, id_):
    return next(c for c in result.components if c.id == id_)

def _net(result, net_id):
    return next(n for n in result.nets if n.net_id == net_id)

def _pin(comp, name):
    return next(p for p in comp.pins if p.name == name)


# --- port mapping ---

class TestPortMapping:
    def test_input_port_creates_input_pin_component(self):
        mod = _make_module(ports={"clk": Port(direction="input", bits=[2])})
        result = module_to_component_list(mod)
        comp = _comp(result, "clk")
        assert comp.type == ComponentType.INPUT_PIN
        assert len(comp.pins) == 1
        assert comp.pins[0].direction == Direction.OUT

    def test_output_port_creates_output_pin_component(self):
        mod = _make_module(ports={"q": Port(direction="output", bits=[3])})
        result = module_to_component_list(mod)
        comp = _comp(result, "q")
        assert comp.type == ComponentType.OUTPUT_PIN
        assert comp.pins[0].direction == Direction.IN

    def test_multi_bit_port_indexed_pin_names(self):
        mod = _make_module(ports={"count": Port(direction="output", bits=[5, 6, 7, 8])})
        result = module_to_component_list(mod)
        comp = _comp(result, "count")
        names = [p.name for p in comp.pins]
        assert names == ["count[0]", "count[1]", "count[2]", "count[3]"]

    def test_single_bit_port_no_index(self):
        mod = _make_module(ports={"en": Port(direction="input", bits=[10])})
        result = module_to_component_list(mod)
        comp = _comp(result, "en")
        assert comp.pins[0].name == "en"

    def test_constant_bits_skipped(self):
        # "0" / "1" are constant strings, not net IDs — no pins emitted
        mod = _make_module(ports={"tied": Port(direction="input", bits=["0", "1"])})
        result = module_to_component_list(mod)
        comp = _comp(result, "tied")
        assert comp.pins == []


# --- cell mapping ---

class TestCellMapping:
    def test_and_cell_type_and_footprint(self):
        mod = _make_module(cells={
            "u1": _make_cell(
                "$_AND_",
                {"A": "input", "B": "input", "Y": "output"},
                {"A": [2], "B": [3], "Y": [4]},
            )
        })
        result = module_to_component_list(mod)
        comp = _comp(result, "u1")
        assert comp.type == ComponentType.AND
        assert comp.footprint.width == 3
        assert comp.footprint.depth == 3

    def test_and_pin_sides(self):
        mod = _make_module(cells={
            "u1": _make_cell(
                "$_AND_",
                {"A": "input", "B": "input", "Y": "output"},
                {"A": [2], "B": [3], "Y": [4]},
            )
        })
        result = module_to_component_list(mod)
        comp = _comp(result, "u1")
        assert _pin(comp, "A").side == CardinalDirection.SOUTH
        assert _pin(comp, "B").side == CardinalDirection.SOUTH
        assert _pin(comp, "Y").side == CardinalDirection.NORTH

    def test_not_cell(self):
        mod = _make_module(cells={
            "inv": _make_cell(
                "$_NOT_",
                {"A": "input", "Y": "output"},
                {"A": [1], "Y": [2]},
            )
        })
        result = module_to_component_list(mod)
        comp = _comp(result, "inv")
        assert comp.type == ComponentType.NOT
        assert comp.footprint.width == 1

    def test_dff_cell(self):
        mod = _make_module(cells={
            "ff": _make_cell(
                "$_DFF_P_",
                {"C": "input", "D": "input", "Q": "output"},
                {"C": [1], "D": [2], "Q": [3]},
            )
        })
        result = module_to_component_list(mod)
        comp = _comp(result, "ff")
        assert comp.type == ComponentType.DFF
        assert _pin(comp, "C").side == CardinalDirection.EAST
        assert _pin(comp, "Q").side == CardinalDirection.NORTH

    def test_dffe_cell(self):
        mod = _make_module(cells={
            "ffe": _make_cell(
                "$_DFFE_PP_",
                {"C": "input", "D": "input", "E": "input", "Q": "output"},
                {"C": [1], "D": [2], "E": [3], "Q": [4]},
            )
        })
        result = module_to_component_list(mod)
        comp = _comp(result, "ffe")
        assert comp.type == ComponentType.DFFE
        assert comp.footprint.width == 7
        assert comp.footprint.depth == 3
        assert _pin(comp, "E").side == CardinalDirection.EAST

    def test_mux_cell(self):
        mod = _make_module(cells={
            "m": _make_cell(
                "$_MUX_",
                {"A": "input", "B": "input", "S": "input", "Y": "output"},
                {"A": [1], "B": [2], "S": [3], "Y": [4]},
            )
        })
        result = module_to_component_list(mod)
        comp = _comp(result, "m")
        assert comp.type == ComponentType.MUX
        assert _pin(comp, "S").side == CardinalDirection.WEST

    def test_fulladder_no_port_directions(self):
        # techmap cells may omit port_directions entirely
        mod = _make_module(cells={
            "fa": Cell(
                hide_name=1,
                type="fulladder",
                parameters={},
                connections={"A": [1], "B": [2], "Cin": [3], "S": [4], "Cout": [5]},
            )
        })
        result = module_to_component_list(mod)
        comp = _comp(result, "fa")
        assert comp.type == ComponentType.FULL_ADDER
        # no port_directions → no pins emitted, but component still present
        assert comp.pins == []

    def test_unknown_cell_type_raises(self):
        mod = _make_module(cells={
            "bad": _make_cell("$_UNKNOWN_", {"A": "input"}, {"A": [1]})
        })
        with pytest.raises(ValueError, match="Unknown cell type"):
            module_to_component_list(mod)

    def test_cell_params_preserved(self):
        mod = _make_module(cells={
            "u1": _make_cell(
                "$_AND_",
                {"A": "input", "B": "input", "Y": "output"},
                {"A": [2], "B": [3], "Y": [4]},
                parameters={"WIDTH": "00000001"},
            )
        })
        result = module_to_component_list(mod)
        assert _comp(result, "u1").params == {"WIDTH": "00000001"}


# --- net mapping ---

class TestNetMapping:
    def _simple_inverter_module(self):
        return _make_module(
            ports={
                "in": Port(direction="input", bits=[2]),
                "out": Port(direction="output", bits=[3]),
            },
            cells={
                "inv": _make_cell(
                    "$_NOT_",
                    {"A": "input", "Y": "output"},
                    {"A": [2], "Y": [3]},
                )
            },
        )

    def test_net_count(self):
        result = module_to_component_list(self._simple_inverter_module())
        assert len(result.nets) == 2

    def test_input_to_cell_net(self):
        result = module_to_component_list(self._simple_inverter_module())
        net = _net(result, "net_2")
        assert net.source.component_id == "in"
        assert net.source.pin_name == "in"
        assert any(s.component_id == "inv" and s.pin_name == "A" for s in net.sinks)

    def test_cell_to_output_net(self):
        result = module_to_component_list(self._simple_inverter_module())
        net = _net(result, "net_3")
        assert net.source.component_id == "inv"
        assert net.source.pin_name == "Y"
        assert any(s.component_id == "out" for s in net.sinks)

    def test_no_net_for_unconnected_bits(self):
        # port with only constant bits — no net emitted
        mod = _make_module(ports={"tied": Port(direction="input", bits=["0"])})
        result = module_to_component_list(mod)
        assert result.nets == []

    def test_fanout_net_multiple_sinks(self):
        # one source drives two AND inputs
        mod = _make_module(
            ports={"a": Port(direction="input", bits=[2])},
            cells={
                "g1": _make_cell("$_AND_", {"A": "input", "B": "input", "Y": "output"}, {"A": [2], "B": [2], "Y": [5]}),
            },
        )
        result = module_to_component_list(mod)
        net = _net(result, "net_2")
        assert len(net.sinks) == 2  # g1.A + g1.B (source is "a" port)
