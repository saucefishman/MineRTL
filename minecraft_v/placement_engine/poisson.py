"""Electrostatic 3D gate placement (ePlace-3D / eDensity-3D, Lu et al., ISPD 2016).

Gates are free boxes in X, Y and Z: no rows, depth layers, flow order or Y levels. Each is a
positive charge the size of its footprint plus spacing, so spacing is enforced by the boxes not
overlapping. Global placement minimises weighted-average wirelength + lambda * electrostatic
energy with Nesterov's method; the potential comes from Poisson's equation with Neumann
boundaries, solved exactly on a uniform bin grid by a 3D DCT. An order-preserving 3D legalizer then
removes what overlap is left, and a compaction + reposition pass tightens the result.

IO is free too, within three rules: all bits of a signal share an X, all inputs share a Z and all
outputs share a Z. Each signal's X and each pin's Y follow its gates (_IoPlacer), and the two rows sit
just outside the gates (_io_rows).
"""
from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict

import numpy as np
from scipy import fft, sparse
from scipy.sparse.linalg import cg

from minecraft_v.placement_engine.ir import Component, ComponentType, NetConnection
from minecraft_v.placement_engine.layout import _BIT_INDEX_RE, _Placed, _assign_io_x_positions, _io_base_name
from minecraft_v.placement_engine.template import _mirror_component_x

_WEIGHTS = np.array([1.0, 2.0, 1.0])  # wirelength per block in X, Y, Z: Y needs slopes and towers
_BINS_PER_BOX = 2            # bins per mean box size along each axis
_TARGET_DENSITY = 0.8        # padded box volume / domain volume
_STOP_OVERFLOW = 0.12        # global placement ends below this share of charge in overfull bins
_MAX_ITERS = 1500
_STALL_ITERS = 250           # or after this many iterations without a lower overflow
_MIN_ITERS = 50
_IO_REFRESH = 40             # iterations between IO X reassignments
_MU_MIN, _MU_MAX = 0.75, 1.1  # per-iteration lambda growth (ePlace)
_MU_REF = 0.002              # wirelength rise, as a share of total, that costs one step of growth
_MAX_STEP_BINS = 1.0         # largest move in one iteration, in bins
_LEGAL_AXIS_COST = np.array([1.0, 1.5, 1.0])  # legalizer: separating along Y costs more than X or Z
_REFINE_PASSES = 3
_REFINE_POPS = 60
_IO_COLUMN_CLEAR = 4         # a gate this close in X is in an IO column's way: nearer, a pin lands in the
                             # keepout of the gate's terminal and its wire can't start
_DENSE_GATES = 700           # at or above: dense spacing (see anneal._AUTO_DENSE), regsbig and video
_LOOSE, _DENSE = (6, 6), (8, 8)

_IO_TYPES = (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)


class _Netlist:
    """Pin arrays over all components; gates first, then IO."""

    def __init__(self, gates: list[Component], io: list[Component], nets: list[NetConnection]):
        self.comps = gates + io
        self.n_gates = len(gates)
        index = {c.id: i for i, c in enumerate(self.comps)}
        comp, net, offset = [], [], []
        n_nets = 0
        for n in nets:
            endpoints = [n.source, *n.sinks]
            if len({ep.component_id for ep in endpoints}) < 2:
                continue
            for ep in endpoints:
                c = self.comps[index[ep.component_id]]
                pin = next(p for p in c.pins if p.name == ep.pin_name)
                comp.append(index[ep.component_id])
                net.append(n_nets)
                offset.append(_pin_offset(c, pin.offset))
            n_nets += 1
        self.pin_comp = np.array(comp, dtype=np.int64)
        self.pin_net = np.array(net, dtype=np.int64)
        self.pin_offset = np.array(offset, dtype=float).reshape(-1, 3)
        self.n_nets = n_nets
        self.degree = np.bincount(self.pin_comp, minlength=len(self.comps)).astype(float)

    def hpwl(self, pos: np.ndarray) -> float:
        p = pos[self.pin_comp] + self.pin_offset
        hi = np.full((self.n_nets, 3), -np.inf)
        lo = np.full((self.n_nets, 3), np.inf)
        np.maximum.at(hi, self.pin_net, p)
        np.minimum.at(lo, self.pin_net, p)
        return float(((hi - lo) * _WEIGHTS).sum()) if self.n_nets else 0.0

    def wa_gradient(self, pos: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Gradient of the weighted-average wirelength model per component."""
        p = pos[self.pin_comp] + self.pin_offset
        grad = np.zeros_like(pos)
        for axis in range(3):
            x = p[:, axis]
            for sign in (1.0, -1.0):
                top = np.full(self.n_nets, -np.inf)
                np.maximum.at(top, self.pin_net, sign * x)
                e = np.exp((sign * x - top[self.pin_net]) / gamma[axis])
                s0 = np.bincount(self.pin_net, e, self.n_nets)
                mean = np.bincount(self.pin_net, x * e, self.n_nets) / s0
                g = (1 + sign * (x - mean[self.pin_net]) / gamma[axis]) * e / s0[self.pin_net]
                grad[:, axis] += sign * _WEIGHTS[axis] * np.bincount(self.pin_comp, g, len(pos))
        return grad


def _pin_offset(c: Component, offset: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pin position relative to the origin; pin Z counts from the back (builder._pin_world)."""
    return offset[0], offset[1], c.footprint.depth - 1 - offset[2]


class _Field:
    """Bin grid, charge deposition and the Poisson solve."""

    def __init__(self, sizes: np.ndarray, dims: np.ndarray):
        self.dims = dims
        mean = sizes.mean(axis=0)
        self.shape = tuple(int(max(1, min(128, round(_BINS_PER_BOX * dims[a] / mean[a])))) for a in range(3))
        self.h = dims / np.array(self.shape)
        # Boxes smaller than a bin are smeared over one, keeping their charge (ePlace local smoothing).
        self.eff = np.maximum(sizes, self.h)
        self.charge = np.prod(sizes, axis=1)
        self.scale = self.charge / np.prod(self.eff, axis=1)
        self.span = [int(math.ceil(self.eff[:, a].max() / self.h[a])) + 1 for a in range(3)]
        eig = [(2 - 2 * np.cos(np.pi * np.arange(m) / m)) / self.h[a] ** 2 for a, m in enumerate(self.shape)]
        self.laplace = eig[0][:, None, None] + eig[1][None, :, None] + eig[2][None, None, :]
        self.laplace[0, 0, 0] = 1.0

    def _overlaps(self, pos: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per axis: the bins each box touches and its overlap length with them."""
        bins, lens = [], []
        for a in range(3):
            lo = pos[:, a]
            hi = lo + self.eff[:, a]
            first = np.floor(lo / self.h[a]).astype(np.int64)
            idx = first[:, None] + np.arange(self.span[a])
            overlap = np.minimum(hi[:, None], (idx + 1) * self.h[a]) - np.maximum(lo[:, None], idx * self.h[a])
            bins.append(np.clip(idx, 0, self.shape[a] - 1))
            lens.append(np.maximum(overlap, 0.0))
        return bins, lens

    def forces(self, centres: np.ndarray) -> tuple[np.ndarray, float]:
        """Electrostatic force on each box and the overflow, for boxes given by their centres."""
        pos = centres - self.eff / 2
        (bx, by, bz), (lx, ly, lz) = self._overlaps(pos)
        weight = (lx[:, :, None, None] * ly[:, None, :, None] * lz[:, None, None, :]
                  * self.scale[:, None, None, None])
        index = (bx[:, :, None, None] * self.shape[1] * self.shape[2]
                 + by[:, None, :, None] * self.shape[2] + bz[:, None, None, :])
        volume = float(np.prod(self.h))
        rho = np.bincount(index.ravel(), weight.ravel(), int(np.prod(self.shape))).reshape(self.shape) / volume
        mean = self.charge.sum() / float(np.prod(self.dims))
        overflow = float(np.maximum(rho - max(mean, _TARGET_DENSITY), 0.0).sum() * volume / self.charge.sum())

        coeff = fft.dctn(rho - rho.mean(), type=2, norm="ortho") / self.laplace
        coeff[0, 0, 0] = 0.0
        phi = fft.idctn(coeff, type=2, norm="ortho")

        force = np.zeros_like(pos)
        for a in range(3):
            m = self.shape[a]
            up, down = np.minimum(np.arange(m) + 1, m - 1), np.maximum(np.arange(m) - 1, 0)
            field = (np.take(phi, down, axis=a) - np.take(phi, up, axis=a)) / (2 * self.h[a])
            force[:, a] = (field.ravel()[index.ravel()] * weight.ravel()).reshape(weight.shape).sum(axis=(1, 2, 3))
        return force, overflow


def _bit_index(comp_id: str) -> int:
    m = _BIT_INDEX_RE.search(comp_id)
    return int(m.group(1)) if m else 0


def _spread(targets: list[float], gap: int) -> list[int]:
    """Integers at least ``gap`` apart, in the given order, as close to ``targets`` as possible.

    Least squares: pool adjacent violators on the targets with the gaps taken out.
    """
    blocks: list[list[float]] = []  # [mean, count]
    for i, t in enumerate(targets):
        blocks.append([t - i * gap, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            mean, count = blocks.pop()
            blocks[-1][0] = (blocks[-1][0] * blocks[-1][1] + mean * count) / (blocks[-1][1] + count)
            blocks[-1][1] += count
    out: list[int] = []
    for mean, count in blocks:
        out += [int(round(mean)) + (len(out) + k) * gap for k in range(int(count))]
    return out


class _IoPlacer:
    """IO positions in domain coordinates, free but for the alignment rules.

    All bits of a signal share an X, signals keep at least a gutter between columns, and pins in a
    column rise with bit index, ``y_gap`` or more apart; each signal's X and each pin's Y otherwise
    follow the gate pins on its nets. Rows sit just outside the Z range here; _io_rows picks their final depth after placement.
    """

    def __init__(self, inputs: list[Component], outputs: list[Component], nets: list[NetConnection],
                 gutter: int, routing_gutter: int, y_gap: int, flat: bool,
                 output_x_order: dict[str, float] | None):
        self.inputs, self.outputs, self.nets = inputs, outputs, nets
        self.gutter, self.routing_gutter, self.y_gap = gutter, routing_gutter, y_gap
        self.flat, self.output_x_order = flat, output_x_order or {}
        self.neighbours: dict[str, list[tuple[str, str]]] = defaultdict(list)
        io_ids = {c.id for c in inputs + outputs}
        for net in nets:
            ends = [(e.component_id, e.pin_name) for e in (net.source, *net.sinks)]
            for cid, _ in ends:
                if cid in io_ids:
                    self.neighbours[cid].extend(e for e in ends if e[0] not in io_ids)

    def _side(self, comps: list[Component], pins: dict[tuple[str, str], tuple[float, float]],
              centre: float, x_order: dict[str, float]) -> dict[str, tuple[int, int]]:
        def wanted(cid: str, axis: int) -> float | None:
            vals = [pins[e][axis] for e in self.neighbours[cid] if e in pins]
            return float(np.median(vals)) if vals else None

        groups: dict[str, list[str]] = defaultdict(list)
        for c in comps:
            groups[_io_base_name(c.id)].append(c.id)
        keyed = []
        for base, ids in groups.items():
            xs = [v for v in (wanted(i, 0) for i in ids) if v is not None]
            target = float(np.median(xs)) if xs else centre
            fixed = [x_order[i] for i in ids if i in x_order]
            keyed.append((float(np.median(fixed)) if fixed else target, base, target))
        keyed.sort()
        columns = _spread([k[2] for k in keyed], self.gutter + 1)

        out = {}
        for (_, base, _), x in zip(keyed, columns):
            ys = {i: wanted(i, 1) or 0.0 for i in groups[base]}
            order = sorted(ys, key=lambda i: (_bit_index(i), i))  # Y rises with bit index
            spread = _spread([ys[i] for i in order], self.y_gap)
            lift = max(0, -spread[0])
            out.update({i: (x, y + lift) for i, y in zip(order, spread)})
        return out

    def place(self, gates: list[Component], pos: np.ndarray, width: float, z_max: float
              ) -> dict[str, tuple[int, int, int]]:
        output_z, input_z = -self.routing_gutter - 1, int(z_max) + self.routing_gutter
        if self.flat:  # one Y level, so every pin needs its own column
            placed = [_Placed(component=c, origin=(int(p[0]), int(p[1]), int(p[2]))) for c, p in zip(gates, pos)]
            xs = _assign_io_x_positions(self.inputs, placed, self.nets, 0, self.gutter, int(width), True)
            xs.update(_assign_io_x_positions(self.outputs, placed, self.nets, 0, self.gutter, int(width), True,
                                             x_order=self.output_x_order))
            return {**{c.id: (xs[c.id], 0, input_z) for c in self.inputs},
                    **{c.id: (xs[c.id], 0, output_z) for c in self.outputs}}
        pins = {(c.id, pin.name): (p[0] + pin.offset[0], p[1] + pin.offset[1])
                for c, p in zip(gates, pos) for pin in c.pins}
        centre = width / 2
        out = {}
        for comps, z, order in ((self.inputs, input_z, {}), (self.outputs, output_z, self.output_x_order)):
            out.update({i: (x, y, z) for i, (x, y) in self._side(comps, pins, centre, order).items()})
        return out


def _quadratic_start(netlist: _Netlist, pos: np.ndarray) -> None:
    """Star-model quadratic wirelength with IO fixed: the initial solution (ePlace-3D, step 1)."""
    n, m = netlist.n_gates, len(netlist.comps)
    if n == 0 or netlist.n_nets == 0:
        return
    sizes = np.bincount(netlist.pin_net, minlength=netlist.n_nets)
    weight = 1.0 / sizes[netlist.pin_net]
    node = netlist.pin_comp
    star = m + netlist.pin_net
    total = m + netlist.n_nets
    lap = sparse.coo_matrix(
        (np.concatenate([weight, weight, -weight, -weight]),
         (np.concatenate([node, star, node, star]), np.concatenate([node, star, star, node]))),
        shape=(total, total)).tocsr()
    free = np.concatenate([np.arange(n), np.arange(m, total)])
    fixed = np.arange(n, m)
    a = (lap[free][:, free] + 1e-6 * sparse.identity(len(free))).tocsr()
    b_part = lap[free][:, fixed]
    for axis in range(3):
        b = -b_part @ pos[fixed, axis]
        x, _ = cg(a, b, x0=np.full(len(free), pos[:n, axis].mean()), maxiter=200)
        pos[:n, axis] = x[:n]


def _global_place(netlist: _Netlist, sizes: np.ndarray, dims: np.ndarray, io: _IoPlacer, gates: list[Component],
                  seed: int) -> np.ndarray:
    """Electrostatic global placement; returns the origins of the padded boxes, inside the domain."""
    n, m = netlist.n_gates, len(netlist.comps)
    field = _Field(sizes, dims)
    pad = sizes / 2
    lo, hi = pad, np.maximum(dims - pad, pad)

    def refresh_io(pos: np.ndarray) -> None:
        coords = io.place(gates, np.maximum(pos[:n] - pad, 0), dims[0], dims[2])
        for i in range(n, m):
            pos[i] = coords[netlist.comps[i].id]

    # ``pos`` holds gate box centres but IO origins, and pins hang off the origin: wirelength needs origins.
    def origins(pos: np.ndarray) -> np.ndarray:
        out = pos.copy()
        out[:n] -= pad
        return out

    rng = np.random.default_rng(seed)
    pos = np.zeros((m, 3))
    pos[:n] = dims / 2 + rng.normal(0.0, 0.01, (n, 3)) * dims
    refresh_io(pos)
    start = origins(pos)
    _quadratic_start(netlist, start)
    pos[:n] = np.clip(start[:n] + pad + rng.normal(0.0, 0.01, (n, 3)) * dims, lo, hi)
    refresh_io(pos)

    v = pos.copy()
    u = pos.copy()
    a = 1.0
    lam = None
    prev_v = prev_g = None
    prev_hpwl = None
    best_overflow, best_at = math.inf, 0
    step = 0.1
    overflow = 1.0
    it = 0
    max_move = _MAX_STEP_BINS * field.h
    t0 = time.monotonic()
    for it in range(_MAX_ITERS):
        force, overflow = field.forces(v[:n])
        gamma = 8 * field.h * 10 ** (np.clip(overflow, 0.1, 1.0) * 20 / 9 - 11 / 9)
        wire = netlist.wa_gradient(origins(v), gamma)[:n]
        if lam is None:
            lam = np.abs(wire).sum() / max(np.abs(force).sum(), 1e-9)
        g = (wire - lam * force) / np.maximum(1.0, netlist.degree[:n, None] + lam * field.charge[:, None])
        if prev_g is not None:
            dv, dg = v[:n] - prev_v, g - prev_g
            step = np.linalg.norm(dv) / max(np.linalg.norm(dg), 1e-12)
        prev_v, prev_g = v[:n].copy(), g
        move = step * g
        worst = (np.abs(move) / max_move).max()
        if worst > 1:
            move = move / worst
        new = np.clip(v[:n] - move, lo, hi)
        a_next = (1 + math.sqrt(1 + 4 * a * a)) / 2
        v[:n] = np.clip(new + (a - 1) / a_next * (new - u[:n]), lo, hi)
        u[:n] = new
        a = a_next

        hpwl = netlist.hpwl(origins(u))
        if prev_hpwl is not None:
            ratio = (hpwl - prev_hpwl) / (_MU_REF * max(prev_hpwl, 1.0))
            lam *= _MU_MAX if ratio < 0 else max(_MU_MIN, _MU_MAX ** (1 - ratio))
        prev_hpwl = hpwl
        if it % _IO_REFRESH == _IO_REFRESH - 1:
            refresh_io(u)
            v[n:] = u[n:]
        if overflow < best_overflow - 1e-3:
            best_overflow, best_at = overflow, it
        if it >= _MIN_ITERS and (overflow < _STOP_OVERFLOW or it - best_at > _STALL_ITERS):
            break
    print(f"[place] poisson: grid {field.shape}, {it + 1} iterations, overflow {overflow:.2f}, "
          f"hpwl {netlist.hpwl(origins(u)):.0f} ({time.monotonic() - t0:.1f}s)")
    return np.maximum(u[:n] - pad, 0.0)


class _Occupancy:
    """Non-overlapping integer boxes, hashed on a coarse grid."""

    CELL = 16

    def __init__(self, sizes: np.ndarray):
        self.sizes = sizes
        self.grid: dict[tuple[int, int, int], set[int]] = defaultdict(set)
        self.origin: dict[int, tuple[int, int, int]] = {}

    def _cells(self, p: tuple[int, int, int], s: tuple[int, int, int]):
        c = self.CELL
        for x in range(p[0] // c, (p[0] + s[0] - 1) // c + 1):
            for y in range(p[1] // c, (p[1] + s[1] - 1) // c + 1):
                for z in range(p[2] // c, (p[2] + s[2] - 1) // c + 1):
                    yield x, y, z

    def conflicts(self, p: tuple[int, int, int], s: tuple[int, int, int], skip: int = -1) -> list[int]:
        seen: set[int] = set()
        for cell in self._cells(p, s):
            seen |= self.grid.get(cell, set())
        seen.discard(skip)
        out = []
        for j in sorted(seen):
            o = self.origin[j]
            t = self.sizes[j]
            if all(p[a] < o[a] + t[a] and o[a] < p[a] + s[a] for a in range(3)):
                out.append(j)
        return out

    def add(self, i: int, p: tuple[int, int, int]) -> None:
        self.origin[i] = p
        for cell in self._cells(p, tuple(int(v) for v in self.sizes[i])):
            self.grid[cell].add(i)

    def remove(self, i: int) -> None:
        for cell in self._cells(self.origin[i], tuple(int(v) for v in self.sizes[i])):
            self.grid[cell].discard(i)
        del self.origin[i]

    def nearest_free(self, i: int, target: tuple[int, int, int], flat: bool, max_pops: int,
                     ) -> tuple[int, int, int] | None:
        """Free origin closest to ``target``: best-first search, stepping out of each blocking box."""
        s = tuple(int(v) for v in self.sizes[i])

        def cost(p: tuple[int, int, int]) -> float:
            return sum(_WEIGHTS[a] * abs(p[a] - target[a]) for a in range(3))

        heap = [(0.0, target)]
        seen = {target}
        for _ in range(max_pops):
            if not heap:
                return None
            _, p = heapq.heappop(heap)
            blockers = self.conflicts(p, s, skip=i)
            if not blockers:
                return p
            for j in blockers:
                o = self.origin[j]
                t = self.sizes[j]
                for a in range(3):
                    if a == 1 and flat:
                        continue
                    for v in (int(o[a] + t[a]), int(o[a] - s[a])):
                        if a != 0 and v < 0:  # X may go negative; the final shift absorbs it
                            continue
                        q = tuple(v if b == a else p[b] for b in range(3))
                        if q not in seen:
                            seen.add(q)
                            heapq.heappush(heap, (cost(q), q))
        return None


def _legalize(pos: np.ndarray, sizes: np.ndarray, flat: bool) -> np.ndarray:
    """Remove overlap while keeping relative order: each pair of boxes is separated along one axis.

    That axis is the one needing the least movement (Y counts extra), the pair keeps its order on it,
    and a longest-path sweep along each axis then gives the smallest coordinates meeting all pairs.
    Sorting by centre keeps each axis's constraints acyclic, and every pair ends up apart on its axis.
    """
    centre = pos + sizes / 2
    # Movement needed to separate each pair along each axis; negative means already apart.
    need = np.stack([(sizes[:, a, None] + sizes[None, :, a]) / 2 - np.abs(centre[:, a, None] - centre[None, :, a])
                     for a in range(3)], axis=2)
    need = need * _LEGAL_AXIS_COST
    if flat:
        need[:, :, 1] = np.inf
    choice = need.argmin(axis=2)
    out = np.zeros_like(pos)
    for a in range(3):
        if a == 1 and flat:
            out[:, a] = pos[:, a]
            continue
        order = np.argsort(centre[:, a], kind="stable")
        for k, i in enumerate(order):
            before = order[:k]
            before = before[choice[i, before] == a]
            floor = (out[before, a] + sizes[before, a]).max() if len(before) else 0.0
            # Keep the global placement's spread where nothing pushes, so far apart stays far apart.
            out[i, a] = max(floor, np.rint(pos[i, a]))
    return out


def _compact(pos: np.ndarray, sizes: np.ndarray, axis: int) -> None:
    """Slide every box towards lower ``axis`` until it touches another box or zero."""
    others = [a for a in range(3) if a != axis]
    order = np.argsort(pos[:, axis], kind="stable")
    done: list[int] = []
    for i in order:
        if done:
            d = np.array(done)
            hit = np.ones(len(d), dtype=bool)
            for a in others:
                hit &= (pos[d, a] < pos[i, a] + sizes[i, a]) & (pos[i, a] < pos[d, a] + sizes[d, a])
            floor = (pos[d[hit], axis] + sizes[d[hit], axis]).max() if hit.any() else 0
        else:
            floor = 0
        pos[i, axis] = min(pos[i, axis], max(floor, 0))
        done.append(int(i))


class _NetCost:
    """Exact weighted HPWL with per-component recomputation, for refinement."""

    def __init__(self, netlist: _Netlist, pos: np.ndarray):
        self.pos = pos
        self.net_pins: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(netlist.n_nets)]
        self.comp_nets: list[list[int]] = [[] for _ in netlist.comps]
        for c, ni, off in zip(netlist.pin_comp.tolist(), netlist.pin_net.tolist(), netlist.pin_offset):
            self.net_pins[ni].append((c, off))
            if not self.comp_nets[c] or self.comp_nets[c][-1] != ni:
                self.comp_nets[c].append(ni)

    def net(self, ni: int) -> float:
        coords = [self.pos[c] + off for c, off in self.net_pins[ni]]
        return sum(_WEIGHTS[a] * (max(p[a] for p in coords) - min(p[a] for p in coords)) for a in range(3))

    def of(self, i: int) -> float:
        return sum(self.net(ni) for ni in self.comp_nets[i])

    def median_target(self, i: int) -> tuple[int, int, int]:
        """Origin at the median of the other pins' net bounding boxes: the HPWL-optimal region."""
        out = []
        for a in range(3):
            ends: list[float] = []
            own = 0.0
            for ni in self.comp_nets[i]:
                others = [self.pos[c][a] + off[a] for c, off in self.net_pins[ni] if c != i]
                mine = [off[a] for c, off in self.net_pins[ni] if c == i]
                own = mine[0]
                if others:
                    ends += [min(others) - own, max(others) - own]
            out.append(int(round(float(np.median(ends)))) if ends else int(self.pos[i][a]))
        return out[0], out[1], out[2]


def _refine(netlist: _Netlist, pos_all: np.ndarray, sizes: np.ndarray, flat: bool, deadline: float) -> None:
    """Greedy detailed placement: move each gate towards its optimal region if that lowers HPWL."""
    n = netlist.n_gates
    occ = _Occupancy(sizes)
    for i in range(n):
        occ.add(i, tuple(int(v) for v in pos_all[i]))
    cost = _NetCost(netlist, pos_all)
    for _ in range(_REFINE_PASSES):
        moved = 0
        for i in range(n):
            if time.monotonic() >= deadline:
                return
            target = cost.median_target(i)
            target = (target[0], max(0, target[1]) if not flat else int(pos_all[i][1]), max(0, target[2]))
            now = tuple(int(v) for v in pos_all[i])
            if target == now:
                continue
            spot = occ.nearest_free(i, target, flat, _REFINE_POPS)
            if spot is None or spot == now:
                continue
            before = cost.of(i)
            pos_all[i] = spot
            if cost.of(i) < before - 1e-9:
                occ.remove(i)
                occ.add(i, spot)
                moved += 1
            else:
                pos_all[i] = now
        if not moved:
            break


def _mirror_pass(netlist: _Netlist, pos_all: np.ndarray, gates: list[Component]) -> list[Component]:
    """Mirror each gate in X if that lowers HPWL (one pass)."""
    cost = _NetCost(netlist, pos_all)
    out = list(gates)
    for i, c in enumerate(gates):
        w = c.footprint.width
        if w < 2:
            continue
        before = cost.of(i)
        saved = {ni: cost.net_pins[ni] for ni in cost.comp_nets[i]}
        flipped = {ni: [(o, np.array([w - 1 - off[0], off[1], off[2]]) if o == i else off) for o, off in pins]
                   for ni, pins in saved.items()}
        for ni, pins in flipped.items():
            cost.net_pins[ni] = pins
        if cost.of(i) < before - 1e-9:
            out[i] = _mirror_component_x(c)
        else:
            for ni, pins in saved.items():
                cost.net_pins[ni] = pins
    return out


def _io_rows(netlist: _Netlist, pos_all: np.ndarray, gates: list[Component], routing_gutter: int,
             keep_output_row: bool, flat: bool = False) -> None:
    """Move the input row and the output row along Z to just outside their gates, in place.

    Nothing may be behind an input or ahead of an output (the builder keeps wires out, see
    builder._io_envelope_blocked), so a row stops short of any gate in one of its columns.
    keep_output_row: outputs stay in the front row, because extensions run ahead of them.
    flat: a row clears every gate, since it's all one plane and the clock region runs behind the inputs.
    """
    n = len(gates)
    x0 = pos_all[:n, 0]
    x1 = x0 + np.array([c.footprint.width for c in gates], dtype=float)
    z0 = pos_all[:n, 2]
    z1 = z0 + np.array([c.footprint.depth for c in gates], dtype=float)

    pin_z = pos_all[netlist.pin_comp, 2] + netlist.pin_offset[:, 2]
    net_gate_z: dict[int, list[float]] = defaultdict(list)
    net_io: dict[int, list[int]] = defaultdict(list)
    for c, ni, z in zip(netlist.pin_comp.tolist(), netlist.pin_net.tolist(), pin_z.tolist()):
        (net_gate_z[ni].append(z) if c < n else net_io[ni].append(c))
    connected: dict[int, list[float]] = defaultdict(list)
    for ni, ios in net_io.items():
        for i in ios:
            connected[i] += net_gate_z[ni]

    def in_way(x: float) -> np.ndarray:
        return flat | ((x0 - _IO_COLUMN_CLEAR <= x) & (x <= x1 - 1 + _IO_COLUMN_CLEAR))

    is_input = [c.type == ComponentType.INPUT_PIN for c in netlist.comps[n:]]
    inputs = [i + n for i, flag in enumerate(is_input) if flag]
    outputs = [i + n for i, flag in enumerate(is_input) if not flag]

    limits = [0.0]
    for i in inputs:
        limits += [*z1[in_way(pos_all[i, 0])], *(z + 1 for z in connected[i])]
    input_z = max(limits) + routing_gutter
    pos_all[inputs, 2] = input_z
    output_z = -routing_gutter - 1.0
    if not keep_output_row:
        limits = [input_z - 1]  # outputs stay in front of the input row
        for i in outputs:
            limits += [*z0[in_way(pos_all[i, 0])], *connected[i]]
        output_z = min(limits) - routing_gutter - 1
    pos_all[outputs, 2] = output_z


def _poisson_placement(
        components: list[Component],
        nets: list[NetConnection],
        *,
        base_y: int,
        gutter: int | None,
        io_margin: int,
        routing_gutter: int | None,
        y_spacing: int,
        seed: int = 0,
        flat: bool = False,
        output_x_order: dict[str, float] | None = None,
) -> list[_Placed]:
    """Place gates electrostatically; see the module docstring."""
    t0 = time.monotonic()
    gates = [c for c in components if c.type not in _IO_TYPES]
    inputs = [c for c in components if c.type == ComponentType.INPUT_PIN]
    outputs = [c for c in components if c.type == ComponentType.OUTPUT_PIN]
    loose = _DENSE if len(gates) >= _DENSE_GATES else _LOOSE
    gutter = loose[0] if gutter is None else gutter
    routing_gutter = loose[1] if routing_gutter is None else routing_gutter

    netlist = _Netlist(gates, inputs + outputs, nets)
    io = _IoPlacer(inputs, outputs, nets, gutter, routing_gutter, 1 + y_spacing, flat, output_x_order)
    if gates:
        sizes = np.array([[c.footprint.width + gutter,
                           1 if flat else c.footprint.height + y_spacing,
                           c.footprint.depth + routing_gutter] for c in gates], dtype=float)
        volume = float(np.prod(sizes, axis=1).sum()) / _TARGET_DENSITY
        height = 1.0 if flat else float(max(sizes[:, 1].max(), volume ** (1 / 3)))
        side = max(math.sqrt(volume / height), float(sizes[:, [0, 2]].max()))
        dims = np.array([side, height, side])

        placed_pos = _global_place(netlist, sizes, dims, io, gates, seed)
        pos = np.rint(placed_pos)
        pos = _legalize(pos, sizes, flat)
        pos[:, 0] -= pos[:, 0].min()
        for axis in (0, 2, 1):
            _compact(pos, sizes, axis)
        print(f"[place] poisson: legal in {tuple(int(v) for v in (pos + sizes).max(axis=0))}, "
              f"density {np.prod(sizes, axis=1).sum() / np.prod((pos + sizes).max(axis=0)):.2f}")
    else:
        pos = np.zeros((0, 3))
        sizes = np.zeros((0, 3))

    def io_rows() -> np.ndarray:
        depth_end = float((pos[:, 2] + sizes[:, 2]).max()) if len(pos) else 0.0
        width = float((pos[:, 0] + sizes[:, 0]).max()) if len(pos) else 0.0
        coords = io.place(gates, pos, width, depth_end)
        return np.array([coords[c.id] for c in netlist.comps[netlist.n_gates:]], dtype=float).reshape(-1, 3)

    pos_all = np.vstack([pos, io_rows()])
    _refine(netlist, pos_all, sizes, flat, time.monotonic() + 10.0)
    pos = pos_all[:len(gates)]
    pos_all[len(gates):] = io_rows()
    gates_final = _mirror_pass(netlist, pos_all, gates)
    pos_all[len(gates):] = io_rows()
    _io_rows(netlist, pos_all, gates_final, routing_gutter, bool(output_x_order), flat)
    print(f"[place] poisson: final hpwl {netlist.hpwl(pos_all):.0f} ({time.monotonic() - t0:.1f}s)")

    # World frame: margin in X, frontmost output at z=1 (as in layout._pack_layout).
    front = [p[2] for c, p in zip(netlist.comps, pos_all) if c.type == ComponentType.OUTPUT_PIN]
    z_shift = 1 - min(front) if front else routing_gutter + 2 - float(pos_all[:, 2].min(initial=0.0))
    shift = np.array([io_margin - min(0.0, float(pos_all[:, 0].min())), base_y, z_shift])
    placed = []
    for c, p in zip(gates_final + inputs + outputs, pos_all + shift):
        placed.append(_Placed(component=c, origin=(int(p[0]), int(p[1]), int(p[2]))))
    return placed
