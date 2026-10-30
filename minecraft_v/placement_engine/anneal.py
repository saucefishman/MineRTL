"""Gate placement optimisation: barycenter row ordering + simulated annealing.

Works on the row model of ``layout._pack_layout``: gates sit in rows keyed by
``(layer, y_level)``, packed left to right with ``gutter`` and centred; layers
stack in Z (higher layer = nearer the output row).  IO keeps its organisation
(output row / input row, Y by bit index, shared multibit columns) and only
follows the gates through ``_assign_io_x_positions``.

Annealing is VPR-style: cost = weighted net bounding boxes + widest row +
RUDY congestion (each net's wire estimate spread over its XZ bounding box,
gate footprints as blockage, binned, quadratic penalty — HPWL alone packs
rows tight and leaves the router no room to detour), moves
are swaps/shifts inside a row, moves to another Y level, moves to another
layer inside the gate's slack window (every same-level feed-forward sink stays
in a strictly later layer than its driver), and X-mirroring a gate
(``template._mirror_component_x``: swaps which side its pins sit on).
"""
from __future__ import annotations

import math
import multiprocessing
import random
import sys
import time
from collections import defaultdict

from minecraft_v.placement_engine.ir import Component, ComponentType, NetConnection
from minecraft_v.placement_engine.layout import (
    _Placed,
    _assign_gate_layers,
    _assign_io_x_positions,
    _feedforward_nets,
    _io_bit_index,
    _io_y_index,
    _majority_y,
    _pack_layout,
)
from minecraft_v.placement_engine.template import _mirror_component_x

_PLACE_WX = 1.0
_PLACE_WY = 3.0              # Y is expensive to route: tower = cost 8 for +4Y
_PLACE_WZ = 1.0
_PLACE_W_AREA = 0.5          # per block of the widest row (workspace width)
_PLACE_W_CONG = 0.0          # RUDY congestion: w * sum(bin demand^2) / bin area; off — spreads
                             # layouts and slows routing without fewer failures (see bench in git log)
_PLACE_BIN = 6               # congestion bin size in X and Z (blocks)
_PLACE_BARY_SWEEPS = 6
_PLACE_INNER_NUM = 4.0       # moves per temperature = inner_num * N^(4/3), at most _PLACE_MAX_TEMP_MOVES
_PLACE_MAX_TEMP_MOVES = 4000  # keeps enough temperatures inside the budget on big designs (video)
_PLACE_T0_ACCEPT = 0.3       # T0 accepts this fraction of uphill random moves; low — we start from a good seed
_PLACE_MAX_TEMPS = 150       # deterministic stop (time budget is only a safety net)
_PLACE_TIME_BUDGET = 10.0    # seconds, plus _PLACE_TIME_PER_GATE per gate, at most _PLACE_TIME_MAX
_PLACE_TIME_PER_GATE = 0.02
_PLACE_TIME_MAX = 30.0       # leave most of a 60s build cap to routing
_PLACE_MOVE_WEIGHTS = {"swap": 0.35, "shift": 0.25, "ymove": 0.15, "yswap": 0.10, "zmove": 0.15,
                       "mirror": 0.10}

_IO_TYPES = (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)


class _PlaceState:
    """Mutable row placement with incremental cost and one-level undo."""

    def __init__(
            self,
            components: list[Component],
            nets: list[NetConnection],
            layer_of: dict[str, int],
            y_of: dict[str, int],
            *,
            base_y: int,
            gutter: int,
            routing_gutter: int,
            y_moves: bool,
            clock_pin: str | None = None,
            flat: bool = False,
    ):
        self.components = components
        self.nets = nets
        self.comps = {c.id: c for c in components}
        self.gates = [c.id for c in components if c.type not in _IO_TYPES]
        self.inputs = [c for c in components if c.type == ComponentType.INPUT_PIN]
        self.outputs = [c for c in components if c.type == ComponentType.OUTPUT_PIN]
        self.base_y = base_y
        self.gutter = gutter
        self.routing_gutter = routing_gutter
        self.y_moves = y_moves
        self.flat = flat
        self.clock_pin = clock_pin
        self.mirrored: set[str] = set()
        self._mirror_copy: dict[str, Component] = {}

        self.layer = dict(layer_of)
        self.y = dict(y_of)
        self.max_layer = max(layer_of.values(), default=0)

        self.rows: dict[tuple[int, int], list[str]] = {}
        for gid in self.gates:
            self.rows.setdefault((self.layer[gid], self.y[gid]), []).append(gid)
        self.row_w = {k: self._row_width(r) for k, r in self.rows.items()}

        self.levels = sorted(
            {self.y[g] for g in self.gates}
            | {self.y[c.id] for c in components if c.type in _IO_TYPES and _io_y_index(c, clock_pin) is not None}
        )

        pin_offset = {(c.id, p.name): p.offset for c in components for p in c.pins}
        self.net_pins: list[list[tuple[str, int, int, int]]] = []
        self.comp_nets: dict[str, list[int]] = defaultdict(list)
        for ni, net in enumerate(nets):
            pins = []
            for ep in [net.source, *net.sinks]:
                dx, dy, dz = pin_offset[(ep.component_id, ep.pin_name)]
                pins.append((ep.component_id, dx, dy, dz))
                if not self.comp_nets[ep.component_id] or self.comp_nets[ep.component_id][-1] != ni:
                    self.comp_nets[ep.component_id].append(ni)
            self.net_pins.append(pins)

        gate_set = set(self.gates)
        self.preds: dict[str, set[str]] = defaultdict(set)
        self.succs: dict[str, set[str]] = defaultdict(set)
        gate_comps = [self.comps[g] for g in self.gates]
        for net in _feedforward_nets(gate_comps, nets):
            src = net.source.component_id
            for s in net.sinks:
                if s.component_id in gate_set and s.component_id != src:
                    self.succs[src].add(s.component_id)
                    self.preds[s.component_id].add(src)

        self.pos: dict[str, tuple[int, int, int]] = {}
        self.layer_z: dict[int, int] = {}
        self.input_z = 0
        self.net_cost: list[float] = [0.0] * len(nets)
        self.bins: dict[tuple[int, int], float] = {}
        self.net_bins: list[dict] = [{} for _ in nets]
        self.gate_bins: dict[str, dict] = {}
        self.cost = 0.0
        self._undo: dict | None = None
        self.full_recompute(refresh_io=True)

    # ── geometry ────────────────────────────────────────────────────────
    def comp(self, gid: str) -> Component:
        """The component as currently placed (mirrored copy if ``gid`` is mirrored)."""
        if gid not in self.mirrored:
            return self.comps[gid]
        if gid not in self._mirror_copy:
            self._mirror_copy[gid] = _mirror_component_x(self.comps[gid])
        return self._mirror_copy[gid]

    def _toggle_mirror(self, gid: str) -> None:
        self.mirrored ^= {gid}
        w1 = self.comps[gid].footprint.width - 1
        for ni in self.comp_nets[gid]:
            self.net_pins[ni] = [(c, w1 - dx, dy, dz) if c == gid else (c, dx, dy, dz)
                                 for c, dx, dy, dz in self.net_pins[ni]]

    def _row_width(self, row: list[str]) -> int:
        return sum(self.comps[g].footprint.width + self.gutter for g in row)

    def _stack(self) -> tuple[dict[int, int], int]:
        depth: dict[int, int] = {}
        for (layer, _), row in self.rows.items():
            d = max(self.comps[g].footprint.depth for g in row)
            depth[layer] = max(depth.get(layer, 0), d)
        # Empty layers get zero thickness but still a Z, so a gate can move into one
        # without restacking (layer Z is frozen within a temperature).
        z = 1 + self.routing_gutter
        layer_z = {}
        for layer in range(self.max_layer, -1, -1):
            layer_z[layer] = z
            if layer in depth:
                z += depth[layer] + self.routing_gutter
        return layer_z, z

    def _place_row(self, key: tuple[int, int]) -> None:
        layer, level = key
        row = self.rows.get(key)
        if not row:
            return
        x = -(self.row_w[key] // 2)  # virtual coords, centred on x=0
        y = self.base_y + level
        z = self.layer_z[layer]
        for gid in row:
            self.pos[gid] = (x, y, z)
            x += self.comps[gid].footprint.width + self.gutter

    def _place_io(self, refresh: bool) -> None:
        if refresh:
            if self.y_moves:
                for c in self.inputs + self.outputs:
                    if _io_y_index(c, self.clock_pin) is None:
                        ys = [self.y[o] for ni in self.comp_nets[c.id] for (o, *_r) in self.net_pins[ni]
                              if o != c.id and o in self.y and (o in self.layer or _io_y_index(self.comps[o], self.clock_pin) is not None)]
                        if ys:
                            self.y[c.id] = _majority_y(ys)
            gates_placed = [_Placed(component=self.comp(g), origin=self.pos[g]) for g in self.gates]
            self._io_x = {}
            for group in (self.inputs, self.outputs):
                self._io_x.update(_assign_io_x_positions(group, gates_placed, self.nets, 0, self.gutter, 0, self.flat))
        for c in self.outputs:
            self.pos[c.id] = (self._io_x[c.id], self.base_y + self.y[c.id], 1)
        for c in self.inputs:
            self.pos[c.id] = (self._io_x[c.id], self.base_y + self.y[c.id], self.input_z)

    def _net_eval(self, ni: int) -> tuple[float, dict]:
        pos = self.pos
        x0 = y0 = z0 = math.inf
        x1 = y1 = z1 = -math.inf
        for cid, dx, dy, dz in self.net_pins[ni]:
            ox, oy, oz = pos[cid]
            x, y, z = ox + dx, oy + dy, oz + dz
            if x < x0: x0 = x
            if x > x1: x1 = x
            if y < y0: y0 = y
            if y > y1: y1 = y
            if z < z0: z0 = z
            if z > z1: z1 = z
        cost = _PLACE_WX * (x1 - x0) + _PLACE_WY * (y1 - y0) + _PLACE_WZ * (z1 - z0)
        if not _PLACE_W_CONG:
            return cost, {}
        w, d = x1 - x0 + 1, z1 - z0 + 1
        return cost, _rect_bins(int(x0), int(z0), int(x1), int(z1), (w + d) / (w * d))

    def _gate_rect_bins(self, gid: str) -> dict:
        if not _PLACE_W_CONG:
            return {}
        x, _, z = self.pos[gid]
        fp = self.comps[gid].footprint
        return _rect_bins(x, z, x + fp.width - 1, z + fp.depth - 1, 1.0)

    def _shift_bins(self, old: dict, new: dict, changed: dict) -> None:
        bins = self.bins
        for b, a in old.items():
            if b not in changed:
                changed[b] = bins.get(b, 0.0)
            bins[b] = bins.get(b, 0.0) - a
        for b, a in new.items():
            if b not in changed:
                changed[b] = bins.get(b, 0.0)
            bins[b] = bins.get(b, 0.0) + a

    def _cong_delta(self, changed: dict) -> float:
        bins = self.bins
        return _PLACE_W_CONG * sum(bins[b] ** 2 - v ** 2 for b, v in changed.items()) / _PLACE_BIN ** 2

    def _rebuild_bins(self) -> float:
        """Recompute all congestion bins from current positions; returns the congestion cost."""
        self.bins = {}
        self.gate_bins = {g: self._gate_rect_bins(g) for g in self.gates}
        changed: dict = {}
        for g in self.gates:
            self._shift_bins({}, self.gate_bins[g], changed)
        for ni in range(len(self.nets)):
            self._shift_bins({}, self.net_bins[ni], changed)
        return _PLACE_W_CONG * sum(v ** 2 for v in self.bins.values()) / _PLACE_BIN ** 2

    def _area_cost(self) -> float:
        return _PLACE_W_AREA * max(self.row_w.values(), default=0)

    def full_recompute(self, refresh_io: bool = False, restack: bool = True) -> float:
        self.row_w = {k: self._row_width(r) for k, r in self.rows.items()}
        if restack:
            self.layer_z, self.input_z = self._stack()
        for key in self.rows:
            self._place_row(key)
        self._place_io(refresh_io)
        evals = [self._net_eval(ni) for ni in range(len(self.nets))]
        self.net_cost = [c for c, _ in evals]
        self.net_bins = [b for _, b in evals]
        self.cost = sum(self.net_cost) + self._area_cost() + self._rebuild_bins()
        return self.cost

    # ── legality ────────────────────────────────────────────────────────
    def legal(self, gid: str, layer: int, level: int, override: dict[str, tuple[int, int]] | None = None) -> bool:
        """Every same-level feed-forward driver in an earlier layer, every sink in a later one."""
        def at(o: str) -> tuple[int, int]:
            if override and o in override:
                return override[o]
            return self.layer[o], self.y[o]
        for p in self.preds[gid]:
            pl, py = at(p)
            if py == level and pl >= layer:
                return False
        for s in self.succs[gid]:
            sl, sy = at(s)
            if sy == level and sl <= layer:
                return False
        return True

    def layer_window(self, gid: str, level: int) -> tuple[int, int]:
        lo = max((self.layer[p] + 1 for p in self.preds[gid] if self.y[p] == level), default=0)
        hi = min((self.layer[s] - 1 for s in self.succs[gid] if self.y[s] == level), default=self.max_layer)
        return lo, hi

    # ── moves ───────────────────────────────────────────────────────────
    def apply(self, rows_new: dict[tuple[int, int], list[str]], ly_new: dict[str, tuple[int, int]],
              mirror: str | None = None) -> float:
        """Apply a move (``mirror``: toggle that gate's X mirroring); returns the cost delta.
        ``undo()`` or ``commit()`` must follow."""
        undo: dict = {
            "rows": {k: self.rows.get(k) for k in rows_new},
            "row_w": {k: self.row_w.get(k) for k in rows_new},
            "ly": {c: (self.layer[c], self.y[c]) for c in ly_new},
            "cost": self.cost,
            "mirror": mirror,
        }
        if mirror is not None:
            self._toggle_mirror(mirror)
        old_area = self._area_cost()
        for k, lst in rows_new.items():
            if lst:
                self.rows[k] = lst
                self.row_w[k] = self._row_width(lst)
            else:
                self.rows.pop(k, None)
                self.row_w.pop(k, None)
        for c, (layer, level) in ly_new.items():
            self.layer[c], self.y[c] = layer, level

        # Layer Z stays frozen until the next full_recompute, so every move is local.
        gates_moved = [g for lst in rows_new.values() for g in lst]
        undo["pos"] = {g: self.pos[g] for g in gates_moved}
        for key in rows_new:
            self._place_row(key)
        nets = {ni for g in gates_moved for ni in self.comp_nets[g]}
        if mirror is not None:
            nets.update(self.comp_nets[mirror])

        undo["net"] = {ni: (self.net_cost[ni], self.net_bins[ni]) for ni in nets}
        undo["gate_bins"] = {g: self.gate_bins[g] for g in gates_moved}
        changed: dict = {}
        delta = 0.0
        for g in gates_moved:
            new_bins = self._gate_rect_bins(g)
            self._shift_bins(self.gate_bins[g], new_bins, changed)
            self.gate_bins[g] = new_bins
        for ni in nets:
            new, new_bins = self._net_eval(ni)
            delta += new - self.net_cost[ni]
            self.net_cost[ni] = new
            self._shift_bins(self.net_bins[ni], new_bins, changed)
            self.net_bins[ni] = new_bins
        undo["bins"] = changed
        delta += self._cong_delta(changed)
        delta += self._area_cost() - old_area
        self.cost += delta
        self._undo = undo
        return delta

    def undo(self) -> None:
        u = self._undo
        assert u is not None
        if u["mirror"] is not None:
            self._toggle_mirror(u["mirror"])
        for k, lst in u["rows"].items():
            if lst is None:
                self.rows.pop(k, None)
                self.row_w.pop(k, None)
            else:
                self.rows[k] = lst
                self.row_w[k] = u["row_w"][k]
        for c, (layer, level) in u["ly"].items():
            self.layer[c], self.y[c] = layer, level
        self.pos.update(u["pos"])
        for ni, (cost, bins) in u["net"].items():
            self.net_cost[ni] = cost
            self.net_bins[ni] = bins
        self.gate_bins.update(u["gate_bins"])
        for b, v in u["bins"].items():
            self.bins[b] = v
        self.cost = u["cost"]
        self._undo = None

    def commit(self) -> None:
        self._undo = None

    def snapshot(self) -> tuple:
        return {k: list(r) for k, r in self.rows.items()}, dict(self.layer), dict(self.y), frozenset(self.mirrored)

    def restore(self, snap: tuple) -> None:
        rows, layer, y, mirrored = snap
        self.rows = {k: list(r) for k, r in rows.items()}
        self.layer, self.y = dict(layer), dict(y)
        for g in sorted(self.mirrored ^ mirrored):
            self._toggle_mirror(g)
        self.full_recompute(refresh_io=True)

    def _insert_nearest(self, row: list[str], gid: str) -> list[str]:
        gx = self.pos[gid][0]
        idx = sum(1 for o in row if self.pos[o][0] < gx)
        return row[:idx] + [gid] + row[idx:]

    def propose(self, rng: random.Random, kinds: list[str], weights: list[float], rlim: int):
        """Return ``(rows_new, ly_new)`` for a random legal move, or None."""
        gid = self.gates[rng.randrange(len(self.gates))]
        layer, level = self.layer[gid], self.y[gid]
        key = (layer, level)
        row = self.rows[key]
        kind = rng.choices(kinds, weights)[0]

        if kind == "mirror":
            if self.comps[gid].footprint.width < 2:
                return None
            return {}, {}, gid

        if kind in ("swap", "shift"):
            if len(row) < 2:
                return None
            i = row.index(gid)
            j = min(len(row) - 1, max(0, i + rng.randint(-rlim, rlim)))
            if j == i:
                return None
            new = list(row)
            if kind == "swap":
                new[i], new[j] = new[j], new[i]
            else:
                new.insert(j, new.pop(i))
            return {key: new}, {}

        if kind == "ymove":
            others = [lv for lv in self.levels if lv != level]
            if not others:
                return None
            new_level = others[rng.randrange(len(others))]
            if not self.legal(gid, layer, new_level):
                return None
            dst = (layer, new_level)
            src_row = [g for g in row if g != gid]
            return {key: src_row, dst: self._insert_nearest(self.rows.get(dst, []), gid)}, {gid: (layer, new_level)}

        if kind == "yswap":
            cands = [g for (lay, lv), r in self.rows.items() if lay == layer and lv != level for g in r]
            if not cands:
                return None
            other = cands[rng.randrange(len(cands))]
            o_level = self.y[other]
            override = {gid: (layer, o_level), other: (layer, level)}
            if not (self.legal(gid, layer, o_level, override) and self.legal(other, layer, level, override)):
                return None
            okey = (layer, o_level)
            src_row = [other if g == gid else g for g in row]
            dst_row = [gid if g == other else g for g in self.rows[okey]]
            return {key: src_row, okey: dst_row}, override

        # zmove
        lo, hi = self.layer_window(gid, level)
        lo = max(lo, layer - rlim)
        hi = min(hi, layer + rlim)
        choices = [lay for lay in range(lo, hi + 1) if lay != layer]
        if not choices:
            return None
        new_layer = choices[rng.randrange(len(choices))]
        dst = (new_layer, level)
        src_row = [g for g in row if g != gid]
        return {key: src_row, dst: self._insert_nearest(self.rows.get(dst, []), gid)}, {gid: (new_layer, level)}

    def layer_gates(self) -> dict[int, list[Component]]:
        out: dict[int, list[Component]] = defaultdict(list)
        for (layer, _), row in sorted(self.rows.items()):
            out[layer].extend(self.comp(g) for g in row)
        return out


def _rect_bins(x0: int, z0: int, x1: int, z1: int, density: float) -> dict[tuple[int, int], float]:
    """Spread ``density`` per block over the inclusive rect, summed per congestion bin."""
    b = _PLACE_BIN
    out = {}
    for bx in range(x0 // b, x1 // b + 1):
        w = min(x1, bx * b + b - 1) - max(x0, bx * b) + 1
        for bz in range(z0 // b, z1 // b + 1):
            out[(bx, bz)] = density * w * (min(z1, bz * b + b - 1) - max(z0, bz * b) + 1)
    return out


def _barycenter(state: _PlaceState) -> None:
    """Sugiyama barycenter sweeps: sort each row by the mean X its gates connect to."""
    best_cost = state.cost
    best = state.snapshot()
    own_dx = {}
    for ni, pins in enumerate(state.net_pins):
        for cid, dx, _, _ in pins:
            own_dx.setdefault((cid, ni), dx)

    layers = sorted({layer for layer, _ in state.rows})
    for sweep in range(_PLACE_BARY_SWEEPS):
        changed = False
        for layer in (layers if sweep % 2 == 0 else layers[::-1]):
            for key in sorted(k for k in state.rows if k[0] == layer):
                row = state.rows[key]
                bc = {}
                for i, g in enumerate(row):
                    xs = [state.pos[c][0] + dx - own_dx[(g, ni)]
                          for ni in state.comp_nets[g] for c, dx, _, _ in state.net_pins[ni] if c != g]
                    bc[g] = (sum(xs) / len(xs) if xs else state.pos[g][0], i)
                new = sorted(row, key=bc.__getitem__)
                if new != row:
                    changed = True
                    state.rows[key] = new
                    state._place_row(key)
        state.full_recompute(refresh_io=True)
        if state.cost < best_cost:
            best_cost, best = state.cost, state.snapshot()
        if not changed:
            break
    if state.cost > best_cost:
        state.restore(best)


def _greedy_mirror(state: _PlaceState) -> None:
    """Mirror each gate whose mirroring lowers the cost (one pass, gate order)."""
    for g in state.gates:
        if state.comps[g].footprint.width < 2:
            continue
        if state.apply({}, {}, g) < -1e-9:
            state.commit()
        else:
            state.undo()
    state.full_recompute(refresh_io=True)


def _initial_temperature(uphill: list[float]) -> float:
    """Temperature at which the mean Metropolis acceptance of ``uphill`` is ``_PLACE_T0_ACCEPT``."""
    if not uphill:
        return 0.0
    lo, hi = 1e-9, max(uphill) * 100
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if sum(math.exp(-d / mid) for d in uphill) / len(uphill) < _PLACE_T0_ACCEPT:
            lo = mid
        else:
            hi = mid
    return hi


class _ChainCtx:
    """What forked chain workers inherit: set before the pool forks, read-only afterwards."""
    state: _PlaceState
    kinds: list[str]
    weights: list[float]


def _run_chain(state: _PlaceState, rng: random.Random, kinds: list[str], weights: list[float],
               temp: float, rlim: float, moves: int, deadline: float | None,
               best_cost: float, best: tuple) -> tuple[int, int, float, tuple]:
    """One temperature of Metropolis moves; returns (tried, accepted, best_cost, best)."""
    tried = accepted = 0
    for i in range(moves):
        if deadline is not None and i % 256 == 255 and time.monotonic() >= deadline:
            break
        mv = state.propose(rng, kinds, weights, max(1, int(rlim)))
        if mv is None:
            continue
        tried += 1
        delta = state.apply(*mv)
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            accepted += 1
            state.commit()
            if state.cost < best_cost - 1e-9:
                best_cost, best = state.cost, state.snapshot()
        else:
            state.undo()
    return tried, accepted, best_cost, best


def _chain_task(job: tuple) -> tuple[tuple, float, int, int]:
    """Worker: run one chain from ``snap``; returns (final snapshot, true cost, tried, accepted)."""
    snap, temp, rlim, moves, seed = job
    ctx = _ChainCtx
    ctx.state.restore(snap)
    tried, accepted, _, _ = _run_chain(
        ctx.state, random.Random(seed), ctx.kinds, ctx.weights, temp, rlim, moves, None,
        math.inf, snap)
    ctx.state.full_recompute(refresh_io=True)
    return ctx.state.snapshot(), ctx.state.cost, tried, accepted


def _anneal(state: _PlaceState, rng: random.Random, *, y_moves: bool, z_moves: bool,
            mirror_moves: bool = True, workers: int = 1) -> None:
    n = len(state.gates)
    if n < 2:
        return
    kinds = [k for k in _PLACE_MOVE_WEIGHTS
             if (k not in ("ymove", "yswap") or y_moves) and (k != "zmove" or z_moves)
             and (k != "mirror" or mirror_moves)]
    weights = [_PLACE_MOVE_WEIGHTS[k] for k in kinds]
    max_rlim = max(max(len(r) for r in state.rows.values()), state.max_layer, 1)
    budget = min(_PLACE_TIME_MAX, _PLACE_TIME_BUDGET + _PLACE_TIME_PER_GATE * n)
    deadline = time.monotonic() + budget

    print(f"[place] annealing with {budget:.0f}s budget")

    seed_cost = state.cost
    seed = state.snapshot()

    deltas = []
    for _ in range(n):
        mv = state.propose(rng, kinds, weights, max_rlim)
        if mv is None:
            continue
        deltas.append(state.apply(*mv))
        state.undo()
    temp = _initial_temperature([d for d in deltas if d > 0])
    if temp <= 0:
        return

    best_cost, best = state.cost, state.snapshot()
    rlim = float(max_rlim)
    moves_per_temp = min(_PLACE_MAX_TEMP_MOVES, max(n, int(_PLACE_INNER_NUM * n ** (4 / 3))))
    num_nets = max(1, len(state.nets))
    timed_out = False
    pool = None
    if workers > 1:
        # Parallel chains: every worker anneals a share of the temperature's moves from the same
        # state, the cheapest end state wins. Forked workers inherit the state copy-on-write.
        _ChainCtx.state, _ChainCtx.kinds, _ChainCtx.weights = state, kinds, weights
        pool = multiprocessing.get_context("fork").Pool(workers)
    chain_moves = max(1, -(-moves_per_temp // workers))
    live = sys.stdout.isatty()
    t_start = time.monotonic()
    for step in range(_PLACE_MAX_TEMPS):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        if pool is None:
            tried, accepted, best_cost, best = _run_chain(
                state, rng, kinds, weights, temp, rlim, moves_per_temp, deadline, best_cost, best)
        else:
            snap = state.snapshot()
            jobs = [(snap, temp, rlim, chain_moves, rng.getrandbits(62)) for _ in range(workers)]
            results = pool.map(_chain_task, jobs)
            tried = sum(r[2] for r in results)
            accepted = sum(r[3] for r in results)
            winner = min(range(workers), key=lambda k: (results[k][1], k))
            state.restore(results[winner][0])
        alpha = accepted / tried if tried else 0.0
        if alpha > 0.96:
            temp *= 0.5
        elif alpha > 0.8:
            temp *= 0.9
        elif alpha > 0.15:
            temp *= 0.95
        else:
            temp *= 0.8
        rlim = min(float(max_rlim), max(1.0, rlim * (1 - 0.44 + alpha)))
        # IO follows the gates once per temperature.
        state.full_recompute(refresh_io=True)
        if state.cost < best_cost - 1e-9:
            best_cost, best = state.cost, state.snapshot()
        done = temp < 0.005 * state.cost / num_nets
        if live or step % 10 == 9 or done:
            line = (f"[place] T {step + 1}/{_PLACE_MAX_TEMPS}  cost {state.cost:.0f}  best {best_cost:.0f}  "
                    f"accept {alpha:.0%}  {time.monotonic() - t_start:.0f}s/{budget:.0f}s")
            print(f"\r{line}\033[K" if live else line, end="" if live else "\n", flush=True)
        if done:
            break
    if live and step:
        print()
    if pool is not None:
        pool.terminate()
    if timed_out:
        print(f"[place] annealing hit the {budget:.0f}s safety budget")

    state.restore(best)
    if state.cost > seed_cost:
        state.restore(seed)


def _optimize_placement(
        components: list[Component],
        nets: list[NetConnection],
        y_level: dict[str, int],
        *,
        placer: str,
        base_y: int,
        gutter: int,
        io_margin: int,
        routing_gutter: int,
        y_moves: bool = True,
        z_moves: bool = True,
        mirror_moves: bool = True,
        seed: int = 0,
        clock_pin: str | None = None,
        workers: int = 1,
        flat: bool = False,
) -> list[_Placed]:
    """Place gates with barycenter ordering (``placer='bary'``) or barycenter + annealing (``'sa'``)."""
    gates = [c for c in components if c.type not in _IO_TYPES]
    y_of = {c.id: y_level.get(c.id, 0) for c in components}
    layer_of = _assign_gate_layers(gates, nets, {c.id: y_of[c.id] for c in gates})

    t0 = time.monotonic()
    state = _PlaceState(
        components, nets, layer_of, y_of,
        base_y=base_y, gutter=gutter, routing_gutter=routing_gutter, y_moves=y_moves,
        clock_pin=clock_pin, flat=flat,
    )
    seed_cost = state.cost
    if state.gates:
        _barycenter(state)
        if mirror_moves:
            _greedy_mirror(state)
        bary_cost = state.cost
        if placer == "sa":
            _anneal(state, random.Random(seed), y_moves=y_moves, z_moves=z_moves, mirror_moves=mirror_moves,
                    workers=workers)
        print(f"[place] {placer}: cost legacy={seed_cost:.0f} bary={bary_cost:.0f} "
              f"final={state.cost:.0f} ({time.monotonic() - t0:.2f}s)")

    def comp_y(c: Component) -> int:
        return base_y + state.y[c.id]

    return _pack_layout(
        components, nets, state.layer_gates(), comp_y,
        gutter=gutter, io_margin=io_margin, routing_gutter=routing_gutter, flat=flat,
    )
