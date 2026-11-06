"""_build_footprint_blocked's CellMask holds exactly the cells the set it replaced did."""
import random

from minecraft_v.placement_engine.builder import (
    _DIRS_6, _HORIZ_DIRS, CellMask, _build_footprint_blocked, _cells_in, _io_envelope_blocked, _shell_slabs,
)
from minecraft_v.placement_engine.ir import Component, ComponentType, Footprint
from minecraft_v.placement_engine.layout import _Placed


def _reference(placed, pin_terminal, io_repeater_cells, const_positions, io_fixtures, block_output_lamps, bounds):
    """The frozenset version, as it was."""
    boxes = []
    for item in placed:
        if item.component.type == ComponentType.CUSTOM:
            boxes += _shell_slabs(item, (1, 1, 2, 0))
    shifts = ((0, 0, 0), *_DIRS_6, (0, 2, 0), *((dx, dy, dz) for dx, dz in _HORIZ_DIRS for dy in (1, -1)))
    for item in placed:
        if item.component.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN, ComponentType.CUSTOM):
            continue
        fp = item.component.footprint
        ox, oy, oz = item.origin
        boxes += [((ox + dx, oy + dy, oz + dz), (ox + dx + fp.width, oy + dy + fp.height, oz + dz + fp.depth))
                  for dx, dy, dz in shifts]
    cells = _cells_in(boxes)
    for rx, ry, rz in io_repeater_cells:
        cells.add((rx, ry + 1, rz))
    for (rx, ry, rz), (nx, nz), is_input in io_fixtures:
        if not is_input and not block_output_lamps:
            continue
        for fx, fy, fz in [(rx + nx, ry, rz + nz)] + ([(rx + nx, ry + 1, rz + nz)] if is_input else []):
            cells.add((fx, fy, fz))
            cells.update((fx + dx, fy + dy, fz + dz) for dx, dy, dz in _DIRS_6)
    for x, y, z in const_positions:
        cells.update((x + dx, y + dy, z + dz) for dx, dy, dz in _DIRS_6)
    xs, zs, top = _io_envelope_blocked(placed, bounds, block_outputs=not block_output_lamps)
    cells |= {(x, y, z) for x, z in zip(xs.tolist(), zs.tolist()) for y in range(top)}
    return cells - set(pin_terminal.values())


def _layout(seed: int):
    rng = random.Random(seed)
    placed = []
    for i in range(30):
        placed.append(_Placed(Component(id=f"g{i}", type=ComponentType.AND, pins=[],
                                        footprint=Footprint(width=3, height=2, depth=4)),
                              (rng.randrange(0, 50), rng.randrange(0, 20), rng.randrange(8, 50))))
    placed.append(_Placed(Component(id="bank", type=ComponentType.CUSTOM, pins=[],
                                    footprint=Footprint(width=20, height=12, depth=15)), (0, 0, 40)))  # at the edge
    for i, x in enumerate((3, 9, 15)):
        placed.append(_Placed(Component(id=f"in{i}", type=ComponentType.INPUT_PIN, pins=[],
                                        footprint=Footprint(width=1, height=1, depth=1)), (x, 5, 2)))
        placed.append(_Placed(Component(id=f"out{i}", type=ComponentType.OUTPUT_PIN, pins=[],
                                        footprint=Footprint(width=1, height=1, depth=1)), (x + 20, 5, 60)))
    terminals = {("g0", "A"): placed[0].origin, ("x", "y"): (1, 1, 1), ("o", "p"): (-1, 0, 0)}
    reps = {(3, 5, 3), (9, 5, 3), (0, 0, 0)}
    consts = {(10, 0, 10), (0, 3, 3)}
    fixtures = [((3, 5, 3), (0, -1), True), ((23, 5, 59), (0, 1), False), ((0, 0, 1), (-1, 0), True)]
    return placed, terminals, reps, consts, fixtures


def test_mask_matches_set():
    for seed in range(4):
        placed, terminals, reps, consts, fixtures = _layout(seed)
        shape = (60, 30, 70)
        bounds = (0, 0, 2, shape[0] - 1, shape[1] - 1, shape[2] - 1)
        for lamps in (True, False):
            mask = _build_footprint_blocked(placed, terminals, reps, consts, fixtures, block_output_lamps=lamps,
                                            envelope=_io_envelope_blocked(placed, bounds, block_outputs=not lamps),
                                            shape=shape)
            want = _reference(placed, terminals, reps, consts, fixtures, lamps, bounds)
            assert set(mask) == want
            assert len(mask) == len(want)
            assert all(p in mask for p in want)
            assert (-1, 0, 0) not in mask and (1, 1, 1) not in mask


def test_cell_mask_set_ops():
    mask = CellMask((4, 4, 4))
    mask.add_box((-1, 0, 0), (2, 1, 1))
    assert set(mask) == {(-1, 0, 0), (0, 0, 0), (1, 0, 0)}
    union = mask | {(3, 3, 3), (9, 9, 9)}
    assert (3, 3, 3) in union and (9, 9, 9) in union and (3, 3, 3) not in mask
    mask.discard((-1, 0, 0))
    mask.discard((0, 0, 0))
    assert set(mask) == {(1, 0, 0)}


def _barrier_reference(columns, gate_far, width):
    import bisect
    from minecraft_v.placement_engine.builder import _ENVELOPE_GATE_GAP, _ENVELOPE_THICKNESS
    if not columns:
        return []
    xs = sorted(columns)
    limit = []
    for x in range(width):
        lane = [columns[c] for c in (x - 1, x, x + 1) if c in columns]
        if lane:
            limit.append(max(lane))
            continue
        i = bisect.bisect(xs, x)
        near = max(columns[c] for c in xs[max(i - 1, 0):i + 1])
        limit.append(max([near, *(far + _ENVELOPE_GATE_GAP for x0, x1, far in gate_far if x0 <= x <= x1)]))
    cells = []
    for x, z in enumerate(limit):
        top = max(limit[max(x - 1, 0):x + 2]) + _ENVELOPE_THICKNESS
        cells.extend((x, far) for far in range(z + 1, top + 1))
    return cells


def test_io_barrier_matches_reference():
    from minecraft_v.placement_engine.builder import _io_barrier
    rng = random.Random(1)
    for _ in range(50):
        width = rng.randrange(5, 80)
        columns = {rng.randrange(0, width): rng.randrange(-40, 40) for _ in range(rng.randrange(0, 6))}
        gates = []
        for _ in range(rng.randrange(0, 20)):
            x0 = rng.randrange(-5, width + 5)
            gates.append((x0, x0 + rng.randrange(0, 10), rng.randrange(-60, 60)))
        assert _io_barrier(columns, gates, width) == _barrier_reference(columns, gates, width)
