from litemapy import BlockState, Region

from minecraft_v.placement_engine import propagate
from minecraft_v.placement_engine.propagate import propagate_states, settle


def _states(blocks: dict[tuple[int, int, int], BlockState]) -> dict[tuple[int, int, int], list[str]]:
    region = Region(0, 0, 0, 12, 6, 6)
    for pos, block in blocks.items():
        region[pos] = block
    lines = (line.split() for line in propagate_states(region))
    return {(int(x), int(y), int(z)): rest for x, y, z, *rest in lines}


STONE = BlockState("minecraft:stone")
DUST = BlockState("minecraft:redstone_wire")
TORCH = BlockState("minecraft:redstone_torch", lit="true")
REDSTONE_BLOCK = BlockState("minecraft:redstone_block")


def repeater(facing: str, **props: str) -> BlockState:
    return BlockState("minecraft:repeater", facing=facing, **props)


def test_dust_loses_a_level_per_step_and_ends_at_nothing_past_range():
    blocks = {(0, 1, 0): REDSTONE_BLOCK}
    for x in range(1, 12):
        blocks[(x, 0, 0)] = STONE
        blocks[(x, 1, 0)] = DUST
    states = _states(blocks)
    assert states[(1, 1, 0)] == ["power", "15"]
    assert states[(11, 1, 0)] == ["power", "5"]


def test_stacked_torches_invert_each_other():
    states = _states({(1, 0, 0): STONE, (1, 1, 0): TORCH, (1, 2, 0): STONE, (1, 3, 0): TORCH})
    assert states[(1, 1, 0)] == ["lit", "1"]
    assert states[(1, 3, 0)] == ["lit", "0"]


def test_locked_repeater_keeps_its_saved_state():
    # (1, 1, 1) reads nothing, and the powered repeater at its side locks it.
    blocks = {
        (1, 0, 1): STONE,
        (1, 1, 1): repeater("west", powered="true"),
        (1, 0, 2): STONE,
        (1, 1, 2): repeater("south", powered="true"),
        (1, 1, 3): REDSTONE_BLOCK,
    }
    assert _states(blocks)[(1, 1, 1)] == ["powered", "1", "1"]


def test_unlocked_repeater_follows_its_input():
    blocks = {(1, 0, 1): STONE, (1, 1, 1): repeater("west", powered="true")}
    assert _states(blocks)[(1, 1, 1)] == ["powered", "0", "0"]


def test_dust_passes_signal_up_over_glass_but_not_down():
    lower, upper = (1, 1, 0), (2, 2, 0)
    blocks = {(1, 0, 0): STONE, lower: DUST, (2, 1, 0): BlockState("minecraft:glass"), upper: DUST}
    assert _states({**blocks, (0, 1, 0): REDSTONE_BLOCK})[upper] == ["power", "14"]
    assert _states({**blocks, (3, 2, 0): REDSTONE_BLOCK})[lower] == ["power", "0"]


def test_settle_writes_states_into_region():
    region = Region(0, 0, 0, 3, 4, 1)
    for pos, block in {(1, 0, 0): STONE, (1, 1, 0): TORCH, (1, 2, 0): STONE, (1, 3, 0): TORCH}.items():
        region[pos] = block
    settle(region)
    assert region[1, 1, 0]["lit"] == "true"
    assert region[1, 3, 0]["lit"] == "false"


def test_oscillator_stops_instead_of_hanging(monkeypatch):
    # The torch lights dust that runs into a repeater, which powers the torch's own block.
    messages = []
    monkeypatch.setattr(propagate, "log", messages.append)
    _states({(0, 0, 0): STONE, (0, 1, 0): TORCH, (1, 1, 0): DUST, (1, 0, 0): repeater("east"), (2, 0, 0): DUST})
    assert "oscillator" in messages[0]
