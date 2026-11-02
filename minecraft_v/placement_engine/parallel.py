"""Parallel negotiation passes: route a batch of nets at once in forked worker processes.

A* is pure Python under the GIL, so parallelism has to come from routing several nets at
the same time. During negotiation every net is routed virtually against the static world
plus a `Congestion` table, so nets in one batch are independent if that table is frozen
for the batch (stale congestion, as in parallel FPGA PathFinder routers); history cost
cleans up the extra conflicts over later passes.

Workers are forked once, after the static world is built, so they inherit it (and the
route function closure) copy-on-write. Each keeps its own `Congestion` replica, kept in
sync by broadcasting deltas (claims added/removed, history bumps) before every batch —
much smaller than re-sending the whole table. Within a batch, nets are handed out one at
a time to whichever worker is idle; results come back in batch order, so the outcome
depends only on batch composition, not on completion order.
"""
from __future__ import annotations

import dataclasses
import multiprocessing
import time
from multiprocessing.connection import Connection, wait
from typing import Callable, NamedTuple

from litemapy import BlockState

from .negotiate import Congestion, NetRoute
from .pathfinding import RouteTimeout

Pos = tuple[int, int, int]
RouteFn = Callable[[str, float, Congestion, float | None], NetRoute]


class Claims(NamedTuple):
    """The part of a NetRoute that Congestion reads (duck-types NetRoute for add/remove)."""
    signal: frozenset[Pos]
    support: frozenset[Pos]
    halo: frozenset[Pos]


def claims_of(route: NetRoute) -> Claims:
    return Claims(route.signal, route.support, route.halo)


# Delta ops, applied in order: ("add" | "rm", net_id, Claims) or ("hist", None, [pos, ...]).
Delta = tuple[str, str | None, object]


def apply_deltas(cong: Congestion, deltas: list[Delta]) -> None:
    for op, net_id, arg in deltas:
        if op == "add":
            cong.add(net_id, arg)
        elif op == "rm":
            cong.remove(net_id, arg)
        else:
            for pos in arg:
                cong.hist[pos] += 1


# litemapy BlockState doesn't pickle; ship (id, sorted props) and rebuild on the main side.
_BlockKey = tuple[str, tuple[tuple[str, str], ...]]


def _encode_route(route: NetRoute) -> NetRoute:
    overlay = {pos: (b.id, tuple(sorted(b.properties()))) for pos, b in route.overlay.items()}
    return dataclasses.replace(route, overlay=overlay)


def _decode_route(route: NetRoute, cache: dict[_BlockKey, BlockState]) -> NetRoute:
    overlay = {}
    for pos, key in route.overlay.items():
        block = cache.get(key)
        if block is None:
            block = cache[key] = BlockState(key[0], **dict(key[1]))
        overlay[pos] = block
    return dataclasses.replace(route, overlay=overlay)


def _worker(conn: Connection, route_fn: RouteFn) -> None:
    cong = Congestion()
    while True:
        msg = conn.recv()
        if msg is None:
            return
        kind, payload = msg
        if kind == "sync":
            apply_deltas(cong, payload)
            continue
        net_id, pres, time_limit = payload
        t0 = time.monotonic()
        try:
            route = route_fn(net_id, pres, cong, time_limit)
        except Exception as e:
            conn.send((net_id, None, (isinstance(e, RouteTimeout), type(e).__name__, str(e)),
                       time.monotonic() - t0))
        else:
            conn.send((net_id, _encode_route(route), None, time.monotonic() - t0))


def _decode_error(err: tuple[bool, str, str]) -> Exception:
    is_timeout, type_name, msg = err
    return RouteTimeout(msg) if is_timeout else RuntimeError(f"{type_name}: {msg}")


class RouteWorkers:
    """N forked processes that each route one net at a time against a congestion replica."""

    def __init__(self, n: int, route_fn: RouteFn):
        ctx = multiprocessing.get_context("fork")
        self._conns: list[Connection] = []
        self._procs = []
        for _ in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, route_fn), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self._blocks: dict[_BlockKey, BlockState] = {}

    def __enter__(self) -> RouteWorkers:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._conns.clear()
        self._procs.clear()

    def run_batch(
            self,
            deltas: list[Delta],
            tasks: list[tuple[str, float, float | None]],
            on_done: Callable[[str], None] | None = None,
    ) -> dict[str, tuple[NetRoute | None, Exception | None, float]]:
        """Sync every replica with `deltas`, then route `tasks` (net_id, pres, time_limit).

        Tasks are dispatched in list order to idle workers. Returns net_id → (route, error,
        seconds); `on_done(net_id)` is called as each one finishes (progress output).
        """
        for conn in self._conns:
            conn.send(("sync", deltas))
        pending = list(reversed(tasks))
        busy: set[Connection] = set()
        results: dict[str, tuple[NetRoute | None, Exception | None, float]] = {}
        for conn in self._conns:
            if not pending:
                break
            conn.send(("route", pending.pop()))
            busy.add(conn)
        while busy:
            for conn in wait(list(busy)):
                try:
                    net_id, route, err, dt = conn.recv()
                except EOFError:
                    raise RuntimeError("route worker died") from None
                results[net_id] = (None if route is None else _decode_route(route, self._blocks),
                                   None if err is None else _decode_error(err), dt)
                if on_done is not None:
                    on_done(net_id)
                if pending:
                    conn.send(("route", pending.pop()))
                else:
                    busy.discard(conn)
        return results
