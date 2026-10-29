# distutils: language = c++
"""Compiled twin of pathfinding._search_py: the A* loop of _find_wire_path.

Same arguments, same results, same pop order. Cells' static moves still come from the Python
expand() callback (once per cell); this replaces the per-pop work: heap, node table, snapshot
re-centring, gating, budgets and stop conditions.

Floating point: g, f and costs are doubles combined in the same order as the Python loop, so
every comparison (heap order, f-cap, h-radius) matches it exactly.
"""
from libc.stdint cimport int64_t, uint32_t
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference
import time

cdef extern from *:
    """
    #include <algorithm>
    #include <vector>
    struct HeapEntry {
        double f; long long neg_h; double g; long long counter;
        double g_wire; long long node; long long parent; int group;
    };
    // heapq is a min-heap on the tuple (f, -h, g, counter, ...); counter is unique.
    struct HeapAfter {
        bool operator()(const HeapEntry& a, const HeapEntry& b) const {
            if (a.f != b.f) return a.f > b.f;
            if (a.neg_h != b.neg_h) return a.neg_h > b.neg_h;
            if (a.g != b.g) return a.g > b.g;
            return a.counter > b.counter;
        }
    };
    static inline void heap_push(std::vector<HeapEntry>& h, const HeapEntry& e) {
        h.push_back(e);
        std::push_heap(h.begin(), h.end(), HeapAfter());
    }
    static inline HeapEntry heap_pop(std::vector<HeapEntry>& h) {
        std::pop_heap(h.begin(), h.end(), HeapAfter());
        HeapEntry e = h.back();
        h.pop_back();
        return e;
    }
    """
    cdef struct HeapEntry:
        double f
        long long neg_h
        double g
        long long counter
        double g_wire
        long long node
        long long parent
        int group
    void heap_push(vector[HeapEntry]& h, const HeapEntry& e)
    HeapEntry heap_pop(vector[HeapEntry]& h)


cdef struct NodeInfo:
    double g          # g_score
    long long parent  # came_from (valid once expanded)
    bint expanded     # in came_from / explored


cdef struct Move:
    int dest
    int cost
    uint32_t forbid
    int gate          # 0 always, 1 not right after a tower move, 2 only right after one
    bint has_extra
    int ex, ey, ez    # 2x-slope intermediate, relative to the expanded cell
    int dx, dy, dz
    int shift         # snapshot shift table index for (dx, dy, dz)
    bint has_flat
    int fdx, fdz
    long long h
    bint near
    long long arrive
    double cong       # congestion cost of dest (0 without congestion)


cdef struct M4Side:
    uint32_t forbid
    int start, count  # into m4_cells (offsets relative to the expanded cell)


cdef struct CellMoves:
    int start, count  # into moves
    int m4_move       # index into moves, or -1
    int m4_start, m4_count  # into m4_sides


# Set by init() from pathfinding (single source of truth for the encoding).
cdef int SNAP_BIT_INDEX[27]      # (ox+1)*9 + (oy+1)*3 + (oz+1) -> bit index, or -1
cdef long long TOWER_FLAG = 0
cdef int NODE_SHIFT = 0
cdef long long SNAP_MASK = 0
cdef int STAGNATION = 0
cdef int MAX_NODES = 0
cdef double H_RADIUS_FACTOR = 0
cdef int N_SNAP_BITS = 0
cdef object _init_done = False

# Snapshot re-centring tables per move delta (dx in -2..2, dy in -4..4, dz in -2..2):
# three 256-entry byte tables plus the bit of the expanded cell itself.
cdef enum:
    N_SHIFTS = 225
cdef uint32_t SHIFT_TABLES[N_SHIFTS][3][256]
cdef uint32_t SHIFT_SELF[N_SHIFTS]
cdef bint SHIFT_READY[N_SHIFTS]
cdef int SNAP_OFF[32][3]


def init(snap_offsets, tower_flag, node_shift, stagnation, max_nodes, h_radius_factor):
    global TOWER_FLAG, NODE_SHIFT, SNAP_MASK, STAGNATION, MAX_NODES, H_RADIUS_FACTOR, N_SNAP_BITS, _init_done
    cdef int i
    for i in range(27):
        SNAP_BIT_INDEX[i] = -1
    N_SNAP_BITS = len(snap_offsets)
    assert N_SNAP_BITS <= 32
    for i, (ox, oy, oz) in enumerate(snap_offsets):
        SNAP_BIT_INDEX[(ox + 1) * 9 + (oy + 1) * 3 + (oz + 1)] = i
        SNAP_OFF[i][0] = ox
        SNAP_OFF[i][1] = oy
        SNAP_OFF[i][2] = oz
    for i in range(N_SHIFTS):
        SHIFT_READY[i] = False
    TOWER_FLAG = tower_flag
    NODE_SHIFT = node_shift
    SNAP_MASK = (1 << N_SNAP_BITS) - 1
    STAGNATION = stagnation
    MAX_NODES = max_nodes
    H_RADIUS_FACTOR = h_radius_factor
    _init_done = True


cdef inline uint32_t snap_bit(int ox, int oy, int oz) noexcept nogil:
    if ox < -1 or ox > 1 or oy < -1 or oy > 1 or oz < -1 or oz > 1:
        return 0
    cdef int i = SNAP_BIT_INDEX[(ox + 1) * 9 + (oy + 1) * 3 + (oz + 1)]
    return 0 if i < 0 else (<uint32_t>1) << i


cdef int shift_index(int dx, int dy, int dz) except -1:
    if dx < -2 or dx > 2 or dy < -4 or dy > 4 or dz < -2 or dz > 2:
        raise ValueError(f"A* move delta {(dx, dy, dz)} outside the kernel's shift table range")
    cdef int idx = (dx + 2) * 45 + (dy + 4) * 5 + (dz + 2)
    cdef int k, b, i
    cdef uint32_t moved[32]
    cdef uint32_t acc
    if not SHIFT_READY[idx]:
        for i in range(N_SNAP_BITS):
            moved[i] = snap_bit(SNAP_OFF[i][0] - dx, SNAP_OFF[i][1] - dy, SNAP_OFF[i][2] - dz)
        for k in range(3):
            for b in range(256):
                acc = 0
                for i in range(8):
                    if (b >> i) & 1 and 8 * k + i < N_SNAP_BITS:
                        acc |= moved[8 * k + i]
                SHIFT_TABLES[idx][k][b] = acc
        SHIFT_SELF[idx] = snap_bit(-dx, -dy, -dz)
        SHIFT_READY[idx] = True
    return idx


cdef class CameFrom:
    """came_from as a read-only mapping (node -> parent node), like the Python loop's dict."""
    cdef unordered_map[long long, NodeInfo] nodes

    def get(self, long long node, default=None):
        cdef unordered_map[long long, NodeInfo].iterator it = self.nodes.find(node)
        if it == self.nodes.end() or not dereference(it).second.expanded:
            return default
        return dereference(it).second.parent

    def __getitem__(self, long long node):
        cdef unordered_map[long long, NodeInfo].iterator it = self.nodes.find(node)
        if it == self.nodes.end() or not dereference(it).second.expanded:
            raise KeyError(node)
        return dereference(it).second.parent

    def __contains__(self, long long node):
        cdef unordered_map[long long, NodeInfo].iterator it = self.nodes.find(node)
        return it != self.nodes.end() and dereference(it).second.expanded


cdef class _Cells:
    """Coordinates, heuristic and converted static moves per cell id, mirrored from Python."""
    cdef vector[int] x, y, z
    cdef vector[long long] h
    cdef vector[CellMoves] cm
    cdef vector[Move] moves
    cdef vector[M4Side] m4_sides
    cdef vector[int] m4_cells  # flattened (ox, oy, oz) triples

    cdef void sync(self, list pos_list, list h_list):
        cdef Py_ssize_t i, n = len(pos_list)
        cdef CellMoves empty
        empty.start = -1
        empty.count = 0
        empty.m4_move = -1
        empty.m4_start = 0
        empty.m4_count = 0
        for i in range(<Py_ssize_t>self.x.size(), n):
            p = pos_list[i]
            self.x.push_back(p[0])
            self.y.push_back(p[1])
            self.z.push_back(p[2])
            self.h.push_back(h_list[i])
            self.cm.push_back(empty)

    cdef int convert(self, int cid, object entry, list pos_list, list h_list, object congestion_cost) except -1:
        """Append one Python move entry (see _find_wire_path) as a Move."""
        cdef Move m
        dest = entry[0]
        self.sync(pos_list, h_list)
        m.dest = dest
        m.cost = entry[1]
        m.forbid = entry[2]
        m.gate = entry[3]
        extra = entry[4]
        m.has_extra = extra is not None
        if m.has_extra:
            m.ex = extra[0] - self.x[cid]
            m.ey = extra[1] - self.y[cid]
            m.ez = extra[2] - self.z[cid]
        else:
            m.ex = m.ey = m.ez = 0
        m.dx = self.x[m.dest] - self.x[cid]
        m.dy = self.y[m.dest] - self.y[cid]
        m.dz = self.z[m.dest] - self.z[cid]
        m.shift = shift_index(m.dx, m.dy, m.dz)
        flat_dir = entry[6]
        m.has_flat = flat_dir is not None
        if m.has_flat:
            m.fdx = flat_dir[0]
            m.fdz = flat_dir[1]
        else:
            m.fdx = m.fdz = 0
        m.h = entry[7]
        m.near = entry[8]
        m.arrive = entry[9]
        m.cong = 0.0
        if congestion_cost is not None:
            m.cong = congestion_cost(pos_list[m.dest])
        self.moves.push_back(m)
        return 0

    cdef int load(self, int cid, object expand, list pos_list, list h_list, object congestion_cost) except -1:
        moves, m4 = expand(cid)
        self.sync(pos_list, h_list)
        cdef CellMoves c
        cdef M4Side side
        c.start = self.moves.size()
        for entry in moves:
            self.convert(cid, entry, pos_list, h_list, congestion_cost)
        c.count = <int>self.moves.size() - c.start
        c.m4_move = -1
        c.m4_start = self.m4_sides.size()
        c.m4_count = 0
        if m4 is not None:
            c.m4_move = self.moves.size()
            self.convert(cid, m4[0], pos_list, h_list, congestion_cost)
            for forbid, cells in m4[1]:
                side.forbid = forbid
                side.start = self.m4_cells.size()
                side.count = len(cells)
                for cell in cells:
                    self.m4_cells.push_back(cell[0] - self.x[cid])
                    self.m4_cells.push_back(cell[1] - self.y[cid])
                    self.m4_cells.push_back(cell[2] - self.z[cid])
                self.m4_sides.push_back(side)
            c.m4_count = <int>self.m4_sides.size() - c.m4_start
        self.cm[cid] = c
        return 0


def search(list seed_nodes, list seed_groups, goal_ids, list group_best_h, list group_max_f,
           long long best_node, long long best_h, double weight, time_limit, list pos_list,
           list h_list, expand, congestion_cost):
    """See pathfinding._search_py."""
    if not _init_done:
        raise RuntimeError("_astar.init() was not called")
    cdef _Cells cells = _Cells()
    cdef CameFrom came_from = CameFrom()
    cdef unordered_map[long long, NodeInfo]* nodes = &came_from.nodes
    cdef vector[HeapEntry] heap
    cdef vector[long long] gbest = group_best_h
    cdef vector[double] gmax = group_max_f
    cdef vector[char] explored_xyz
    cdef vector[char] is_goal
    cdef vector[int] taken
    cdef vector[int] extras  # (ex, ey, ez) triples of this expansion's taken 2x slopes
    cdef HeapEntry e, ne
    cdef unordered_map[long long, NodeInfo].iterator it
    cdef NodeInfo info
    cdef NodeInfo* ni
    cdef Move* m
    cdef CellMoves* cmv
    cdef M4Side* side
    cdef long long counter = 0, h, node, nkey, cur_snap, h_radius_pruned = 0
    cdef long long n_explored_xyz = 0, pops = 0
    cdef int stagnation = 0, cid, i, k, last, m4_side = -1
    cdef int in_dx = 0, in_dz = 0, nx, ny, nz
    cdef bint has_in_dir, came_from_tower, has_reached = False
    cdef long long reached = 0
    cdef double max_f, f_wire, h_limit, new_g, new_g_wire, cost
    cdef uint32_t snap, cb0, cb1, cb2
    cdef bint has_deadline = time_limit is not None
    cdef double deadline = 0
    early_stop_reason = None
    py_max_f = max(group_max_f)
    max_f = py_max_f

    cells.sync(pos_list, h_list)
    for g in goal_ids:
        if <Py_ssize_t>is_goal.size() <= g:
            is_goal.resize(g + 1, 0)
        is_goal[g] = 1

    for i in range(len(seed_nodes)):
        node = seed_nodes[i]
        info.g = 0
        info.parent = -1
        info.expanded = False
        nodes[0][node] = info
        h = cells.h[node >> NODE_SHIFT]
        e.f = weight * h
        e.neg_h = -h
        e.g = 0
        e.counter = counter
        e.g_wire = 0
        e.node = node
        e.parent = -1
        e.group = seed_groups[i]
        heap_push(heap, e)
        counter += 1

    if has_deadline:
        deadline = time.monotonic() + time_limit
    while heap.size():
        pops += 1
        if has_deadline and pops & 255 == 0 and time.monotonic() > deadline:
            early_stop_reason = f"timeout ({time_limit}s)"
            break
        e = heap_pop(heap)
        # f-cap per seed group; stop once past every group's.
        f_wire = e.f - e.g + e.g_wire
        if f_wire > gmax[e.group]:
            if f_wire > max_f:
                early_stop_reason = f"f-cap exceeded (f={f_wire} > max_f={py_max_f})"
                break
            continue
        ni = &dereference(nodes.find(e.node)).second
        if e.g > ni.g:
            continue
        cid = <int>(e.node >> NODE_SHIFT)
        h = cells.h[cid]
        if h < best_h:
            best_h = h
            best_node = e.node
        if h < gbest[e.group]:
            gbest[e.group] = h
        if ni.expanded:
            stagnation += 1
            if stagnation >= STAGNATION:
                early_stop_reason = f"stagnated ({stagnation} re-expansions without new territory)"
                break
        else:
            ni.expanded = True
            ni.parent = e.parent
            stagnation = 0
            if <Py_ssize_t>explored_xyz.size() <= cid:
                explored_xyz.resize(cid + 1, 0)
            if not explored_xyz[cid]:
                explored_xyz[cid] = 1
                n_explored_xyz += 1
            if n_explored_xyz >= MAX_NODES:
                early_stop_reason = f"node cap ({MAX_NODES} unique positions)"
                break
        if cid < <int>is_goal.size() and is_goal[cid]:
            reached = e.node
            has_reached = True
            break

        cur_snap = e.node & SNAP_MASK
        came_from_tower = (e.node & TOWER_FLAG) != 0
        if cells.cm[cid].start < 0:
            cells.load(cid, expand, pos_list, h_list, congestion_cost)
        cmv = &cells.cm[cid]
        taken.clear()
        extras.clear()
        for i in range(cmv.start, cmv.start + cmv.count):
            m = &cells.moves[i]
            if cur_snap & m.forbid:
                continue
            if m.gate and (m.gate == 1) == came_from_tower:
                continue
            taken.push_back(i)
            if m.has_extra:
                extras.push_back(m.ex)
                extras.push_back(m.ey)
                extras.push_back(m.ez)
        m4_side = -1
        for i in range(cmv.m4_start, cmv.m4_start + cmv.m4_count):
            if not cur_snap & cells.m4_sides[i].forbid:
                m4_side = i
                break
        # Re-centre the snapshot (plus current itself and the cells this expansion's moves
        # fill) onto each neighbour's window.
        cb0 = cur_snap & 255
        cb1 = (cur_snap >> 8) & 255
        cb2 = cur_snap >> 16
        has_in_dir = e.parent != -1
        if has_in_dir:
            k = <int>(e.parent >> NODE_SHIFT)
            in_dx = cells.x[cid] - cells.x[k]
            in_dz = cells.z[cid] - cells.z[k]
        h_limit = gbest[e.group] * H_RADIUS_FACTOR
        if m4_side >= 0:
            taken.push_back(cmv.m4_move)
        last = <int>taken.size() - 1
        for k in range(<int>taken.size()):
            m = &cells.moves[taken[k]]
            cost = m.cost
            # Penalise XZ direction change on flat moves only (avoids disrupting slope/tower costs).
            if has_in_dir and m.has_flat and (m.fdx != in_dx or m.fdz != in_dz):
                cost += 1
            new_g_wire = e.g_wire + cost
            if congestion_cost is not None:
                cost = cost + m.cong
            new_g = e.g + cost
            snap = (SHIFT_TABLES[m.shift][0][cb0] | SHIFT_TABLES[m.shift][1][cb1]
                    | SHIFT_TABLES[m.shift][2][cb2] | SHIFT_SELF[m.shift])
            for i in range(0, <int>extras.size(), 3):
                snap |= snap_bit(extras[i] - m.dx, extras[i + 1] - m.dy, extras[i + 2] - m.dz)
            if k == last and m4_side >= 0:
                side = &cells.m4_sides[m4_side]
                for i in range(side.start, side.start + 3 * side.count, 3):
                    snap |= snap_bit(cells.m4_cells[i] - m.dx, cells.m4_cells[i + 1] - m.dy,
                                     cells.m4_cells[i + 2] - m.dz)
            nkey = (<long long>m.dest << NODE_SHIFT) | m.arrive | snap
            it = nodes.find(nkey)
            if it != nodes.end():
                if not new_g < dereference(it).second.g:
                    continue
                dereference(it).second.g = new_g
            else:
                if not new_g < 1e9:
                    continue
                info.g = new_g
                info.parent = -1
                info.expanded = False
                nodes[0][nkey] = info
            if m.h <= h_limit or m.near:
                ne.f = new_g + weight * m.h
                ne.neg_h = -m.h
                ne.g = new_g
                ne.counter = counter
                ne.g_wire = new_g_wire
                ne.node = nkey
                ne.parent = e.node
                ne.group = e.group
                heap_push(heap, ne)
                counter += 1
            else:
                h_radius_pruned += 1

    for i in range(<int>gbest.size()):
        group_best_h[i] = gbest[i]
    return (reached if has_reached else None), best_node, h_radius_pruned, early_stop_reason, came_from
