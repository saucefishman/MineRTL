# distutils: language = c++
"""The A* search of _find_wire_path: the loop (search) and cells' static moves (World).

This is the maintained implementation. pathfinding._search_py and _static_moves are deprecated
Python versions (MINECRAFT_V_ASTAR=py/check, and the fallback when this isn't built); they
were exact twins when this was written but are no longer kept in sync. Without a World,
search() falls back to the Python expand() callback for moves.

Floating point: g, f and costs are doubles combined in the same order as the Python loop, so
every comparison (heap order, f-cap, h-radius) matches it exactly.
"""
from libc.stdint cimport int64_t, uint32_t
from libcpp.vector cimport vector
from cython.operator cimport dereference
from cpython.set cimport PySet_Contains
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
    // Open-addressing (linear probing) hash map from 64-bit keys; values value-initialised.
    // Key and value share a slot (one cache line per probe); ~0 marks an empty slot, which no
    // node or cell key uses. Pointers it returns are invalidated by the next get() that inserts.
    template <class V> struct FlatMap {
        struct Slot { unsigned long long key; V val; };
        static constexpr unsigned long long EMPTY = ~0ULL;
        std::vector<Slot> slots;
        size_t mask = 0, count = 0;
        FlatMap() { reset(1 << 12); }
        void reset(size_t cap) {
            slots.assign(cap, Slot{EMPTY, V()});
            mask = cap - 1; count = 0;
        }
        static inline size_t mix(unsigned long long k) {
            k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
            k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33;
            return (size_t)k;
        }
        V* find(unsigned long long k) {
            size_t i = mix(k) & mask;
            while (slots[i].key != EMPTY) {
                if (slots[i].key == k) return &slots[i].val;
                i = (i + 1) & mask;
            }
            return nullptr;
        }
        V* get(unsigned long long k, int* inserted) {
            if ((count + 1) * 2 > mask + 1) grow();
            size_t i = mix(k) & mask;
            while (slots[i].key != EMPTY) {
                if (slots[i].key == k) { *inserted = 0; return &slots[i].val; }
                i = (i + 1) & mask;
            }
            slots[i].key = k; slots[i].val = V(); count++;
            *inserted = 1;
            return &slots[i].val;
        }
        void grow() {
            std::vector<Slot> old;
            old.swap(slots);
            reset(old.size() * 2);
            int ins;
            for (const Slot& s : old) if (s.key != EMPTY) *get(s.key, &ins) = s.val;
        }
    };
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
    cdef cppclass FlatMap[V]:
        V* find(unsigned long long k)
        V* get(unsigned long long k, int* inserted)


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
cdef int HORIZ[4][2]             # _HORIZ_DIRS
cdef int SNAP_BIT_INDEX[27]      # (ox+1)*9 + (oy+1)*3 + (oz+1) -> bit index, or -1
cdef long long TOWER_FLAG = 0
cdef int NODE_SHIFT = 0
cdef int SIG_SHIFT = 0           # signal depth (4 bits) then repeater-candidate dir (3 bits)
cdef int SIG_LIMIT = 0           # _REPEATER_INTERVAL
cdef long long SNAP_MASK = 0
cdef int STAGNATION = 0
cdef int MAX_NODES = 0
cdef double H_RADIUS_FACTOR = 0
cdef int N_SNAP_BITS = 0
cdef int HORIZ_RADIUS_MIN = 0
cdef int Y_RADIUS_MIN = 0
cdef object _init_done = False

# Totals across searches in this process (see stats()).
cdef long long STAT_SEARCHES = 0, STAT_POPS = 0, STAT_PUSHES = 0, STAT_CELLS = 0
cdef double STAT_LOAD_TIME = 0


def stats():
    """{searches, pops, pushes, cells (move tables built), load_s (time building them)}."""
    return {"searches": STAT_SEARCHES, "pops": STAT_POPS, "pushes": STAT_PUSHES,
            "cells": STAT_CELLS, "load_s": STAT_LOAD_TIME}

# Snapshot re-centring tables per move delta (dx in -2..2, dy in -4..4, dz in -2..2):
# three 256-entry byte tables plus the bit of the expanded cell itself.
cdef enum:
    N_SHIFTS = 225
cdef uint32_t SHIFT_TABLES[N_SHIFTS][3][256]
cdef uint32_t SHIFT_SELF[N_SHIFTS]
cdef bint SHIFT_READY[N_SHIFTS]
cdef int SNAP_OFF[32][3]


def init(snap_offsets, tower_flag, node_shift, stagnation, max_nodes, h_radius_factor,
         horiz_dirs, horiz_radius_min, y_radius_min, sig_shift, sig_limit):
    global TOWER_FLAG, NODE_SHIFT, SNAP_MASK, STAGNATION, MAX_NODES, H_RADIUS_FACTOR, N_SNAP_BITS, _init_done
    global HORIZ_RADIUS_MIN, Y_RADIUS_MIN, SIG_SHIFT, SIG_LIMIT
    cdef int i
    assert len(horiz_dirs) == 4
    for i, (dx, dz) in enumerate(horiz_dirs):
        HORIZ[i][0] = dx
        HORIZ[i][1] = dz
    HORIZ_RADIUS_MIN = horiz_radius_min
    Y_RADIUS_MIN = y_radius_min
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
    SIG_SHIFT = sig_shift
    SIG_LIMIT = sig_limit
    assert SIG_SHIFT + 7 <= NODE_SHIFT and SIG_LIMIT <= 16
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
    cdef FlatMap[NodeInfo] nodes

    def get(self, long long node, default=None):
        cdef NodeInfo* ni = self.nodes.find(node)
        if ni == NULL or not ni.expanded:
            return default
        return ni.parent

    def __getitem__(self, long long node):
        cdef NodeInfo* ni = self.nodes.find(node)
        if ni == NULL or not ni.expanded:
            raise KeyError(node)
        return ni.parent

    def __contains__(self, long long node):
        cdef NodeInfo* ni = self.nodes.find(node)
        return ni != NULL and ni.expanded


# ---------------------------------------------------------------------------------------------
# Static move generation (compiled twin of _find_wire_path's _static_moves and its helpers).
#
# Reads the same Python containers the Python version does (sets/dicts via the C API, the
# keepout view through its parts), caching each cell's answers for the search. Produces the
# same move entries, in the same order, with the same cell-id assignment (shared pos_ids).

cdef enum:
    F_SOLID = 1 << 0
    F_NONAIR = 1 << 1
    F_WIRE = 1 << 2
    F_TORCH = 1 << 3
    F_REP = 1 << 4
    F_EPROT = 1 << 5     # effective protected (goal-relieved)
    F_PROT = 1 << 6      # raw protected
    F_EFP = 1 << 7       # effective footprint_blocked (footprint_blocked minus relief)
    F_FP = 1 << 8        # raw footprint_blocked
    F_INV = 1 << 9
    F_TERM = 1 << 10
    F_GLASS = 1 << 11    # solid cell holding GLASS
    F_POWERED = 1 << 12  # _is_powered_block
    F_OWNER = 1 << 13    # owner field known
    BASE_FLAGS = (F_SOLID | F_NONAIR | F_WIRE | F_TORCH | F_REP | F_EPROT | F_PROT | F_EFP | F_FP
                  | F_INV | F_TERM | F_OWNER)

cdef struct CellInfo:
    unsigned int known
    unsigned int val
    int owner            # 0 none, 1 this net, >= 2 another net

cdef extern from *:
    """
    static inline unsigned long long cell_key(long long x, long long y, long long z) {
        return ((unsigned long long)(x + (1 << 17)) << 36) | ((unsigned long long)(y + (1 << 17)) << 18)
             | (unsigned long long)(z + (1 << 17));
    }
    """
    unsigned long long cell_key(long long x, long long y, long long z) nogil



cdef inline void set3(int* c, int x, int y, int z) noexcept nogil:
    c[0] = x
    c[1] = y
    c[2] = z


cdef inline int exempt_code(int ox, int oy, int oz) noexcept nogil:
    """Single exempt cell as an offset from the queried cell (0 = no exempt)."""
    return 1 + (ox + 2) * 25 + (oy + 2) * 5 + (oz + 2)


cdef class World:
    cdef object non_air, wire, torch, repeater, solid, inverted, terminal, fp_relief, fp
    cdef dict dust_owner
    cdef object net_id
    cdef int min_x, min_y, min_z, max_x, max_y, max_z, max_bridge_y
    cdef int gx, gy, gz, x_h_factor
    cdef object eprot, prot  # ("set", s) or ("keepout", count, own, excluded, extra, halo, own_halo)
    cdef object workspace, glass, facing  # facing: per HORIZ direction, the input-facing name
    cdef dict pos_ids
    cdef list pos_list, h_list, move_cache
    cdef dict owner_ids
    cdef FlatMap[CellInfo] info
    cdef FlatMap[char] walk_cache, mid_cache  # 0 unknown, 1 false, 2 true
    cdef object no_exempt

    def __init__(self, non_air, wire, torch, repeater, solid, dust_owner, net_id, bounds,
                 int max_bridge_y, goal, int x_h_factor, eprot, prot, fp_relief, fp, inverted, terminal,
                 workspace, glass, facing, dict pos_ids, list pos_list, list h_list, list move_cache):
        for s in (non_air, wire, torch, repeater, solid, fp_relief, fp, inverted, terminal):
            if not isinstance(s, (set, frozenset)):
                raise TypeError(f"World needs sets, got {type(s).__name__}")
        self.non_air, self.wire, self.torch, self.repeater = non_air, wire, torch, repeater
        self.solid, self.inverted, self.terminal, self.fp_relief, self.fp = solid, inverted, terminal, fp_relief, fp
        if not isinstance(dust_owner, dict):
            raise TypeError(f"World needs a dict dust_owner, got {type(dust_owner).__name__}")
        self.dust_owner = <dict>dust_owner  # subclasses (JournaledDict, OverlayDict) keep dict.get
        self.net_id = net_id
        self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z = bounds
        self.max_bridge_y = max_bridge_y
        self.gx, self.gy, self.gz = goal
        self.x_h_factor = x_h_factor
        self.eprot, self.prot = eprot, prot
        self.workspace, self.glass, self.facing = workspace, glass, facing
        self.pos_ids, self.pos_list, self.h_list, self.move_cache = pos_ids, pos_list, h_list, move_cache
        self.owner_ids = {}

    # --- per-cell facts -----------------------------------------------------------------------

    cdef inline bint inb(self, int x, int y, int z) noexcept:
        return (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y
                and self.min_z <= z <= self.max_z)

    cdef bint keepout_has(self, object spec, object key) except -1:
        if spec[0] == "set":
            return PySet_Contains(spec[1], key)
        # _ForeignKeepout.__contains__
        if PySet_Contains(spec[4], key):
            return True
        n = (<dict>spec[1]).get(key, 0)
        if n == 0 or (n == 1 and PySet_Contains(spec[2], key)):
            return False
        if not PySet_Contains(spec[3], key):
            return True
        n = (<dict>spec[5]).get(key, 0)
        return n > 1 or (n == 1 and not PySet_Contains(spec[6], key))

    cdef unsigned int probe(self, int x, int y, int z, unsigned int bit) except? 0xFFFFFFFF:
        key = (x, y, z)
        cdef bint v
        if bit == F_SOLID:
            v = PySet_Contains(self.solid, key)
        elif bit == F_NONAIR:
            v = PySet_Contains(self.non_air, key)
        elif bit == F_WIRE:
            v = PySet_Contains(self.wire, key)
        elif bit == F_TORCH:
            v = PySet_Contains(self.torch, key)
        elif bit == F_REP:
            v = PySet_Contains(self.repeater, key)
        elif bit == F_EPROT:
            v = self.keepout_has(self.eprot, key)
        elif bit == F_PROT:
            v = self.keepout_has(self.prot, key)
        elif bit == F_EFP:
            v = PySet_Contains(self.fp, key) and not PySet_Contains(self.fp_relief, key)
        elif bit == F_FP:
            v = PySet_Contains(self.fp, key)
        elif bit == F_INV:
            v = PySet_Contains(self.inverted, key)
        elif bit == F_TERM:
            v = PySet_Contains(self.terminal, key)
        elif bit == F_GLASS:
            v = self.workspace[key] == self.glass
        elif bit == F_POWERED:
            v = self.powered(x, y, z)
        else:
            raise ValueError(bit)
        return bit if v else 0

    cdef CellInfo* cell(self, int x, int y, int z) except NULL:
        """The cell's cache entry; a new one gets every set membership and its owner at once
        (one key tuple), leaving only GLASS / POWERED to probe lazily."""
        cdef int ins
        cdef CellInfo* c = self.info.get(cell_key(x, y, z), &ins)
        if ins:
            self.fill(c, x, y, z)
        return c

    cdef int fill(self, CellInfo* c, int x, int y, int z) except -1:
        key = (x, y, z)
        cdef unsigned int v = 0
        if PySet_Contains(self.solid, key): v |= F_SOLID
        if PySet_Contains(self.non_air, key): v |= F_NONAIR
        if PySet_Contains(self.wire, key): v |= F_WIRE
        if PySet_Contains(self.torch, key): v |= F_TORCH
        if PySet_Contains(self.repeater, key): v |= F_REP
        if self.keepout_has(self.eprot, key): v |= F_EPROT
        if self.keepout_has(self.prot, key): v |= F_PROT
        if PySet_Contains(self.fp, key):
            v |= F_FP
            if not PySet_Contains(self.fp_relief, key): v |= F_EFP
        if PySet_Contains(self.inverted, key): v |= F_INV
        if PySet_Contains(self.terminal, key): v |= F_TERM
        o = self.dust_owner.get(key)
        if o is None:
            c.owner = 0
        elif o == self.net_id:
            c.owner = 1
        else:
            oid = self.owner_ids.get(o)
            if oid is None:
                oid = self.owner_ids[o] = 2 + len(self.owner_ids)
            c.owner = oid
        c.val = v
        c.known = BASE_FLAGS
        return 0

    cdef inline unsigned int flag(self, int x, int y, int z, unsigned int bit) except? 0xFFFFFFFF:
        cdef CellInfo* c = self.cell(x, y, z)
        cdef unsigned int v
        if not c.known & bit:
            v = self.probe(x, y, z, bit)
            c = self.cell(x, y, z)  # probing POWERED touches other cells (may move entries)
            c.val |= v
            c.known |= bit
        return c.val & bit

    cdef int owner(self, int x, int y, int z) except -1:
        return self.cell(x, y, z).owner

    cdef bint powered(self, int x, int y, int z) except -1:
        """_is_powered_block: a solid cell with a repeater facing into it."""
        cdef int d, bx, bz
        if not self.inb(x, y, z) or not (self.cell(x, y, z).val & F_SOLID):
            return False
        for d in range(4):
            bx, bz = x + HORIZ[d][0], z + HORIZ[d][1]
            if not self.inb(bx, y, bz):
                continue
            if not (self.cell(bx, y, bz).val & F_REP):
                continue
            if self.workspace[bx, y, bz]['facing'] == self.facing[d]:
                return True
        return False

    cdef inline bint can_support(self, int x, int y, int z) except -1:
        return self.inb(x, y, z) and ((self.cell(x, y, z).val & F_SOLID) != 0 or not (self.cell(x, y, z).val & F_NONAIR))

    cdef inline bint torch_below(self, int x, int y, int z) except -1:
        if not self.inb(x, y - 1, z):
            return False
        return (self.cell(x, y - 1, z).val & F_TORCH) != 0

    cdef bint wire_walkable(self, int x, int y, int z, int ex, int ey, int ez, bint has_exempt,
                            bint with_inv_fp) except -1:
        """_wire_walkable; with_inv_fp False = no inverted cells / footprint (mid_walkable).
        Only the boolean matters, so each neighbour cell is fetched once and all its checks run
        together (the Python version's check order differs; its result doesn't)."""
        cdef int d, dy, nx, ny, nz, vdy, vy
        cdef CellInfo* c
        cdef unsigned int v
        if not self.inb(x, y, z):
            return False
        c = self.cell(x, y, z)
        if c.val & F_SOLID:
            return False
        if c.val & F_NONAIR:
            if c.val & F_WIRE:
                if c.owner >= 2:
                    return False
            else:
                return False
        for d in range(4):
            for dy in range(-1, 2):
                nx, ny, nz = x + HORIZ[d][0], y + dy, z + HORIZ[d][1]
                c = self.cell(nx, ny, nz)
                if with_inv_fp and c.val & F_INV:
                    return False
                if has_exempt and nx == ex and ny == ey and nz == ez:
                    continue
                if not self.inb(nx, ny, nz):
                    continue
                if c.owner >= 2:
                    return False
                if c.owner == 0 and not (with_inv_fp and c.val & F_EFP):
                    if c.val & (F_TORCH | F_WIRE):
                        return False
        for vdy in (1, -1):
            vy = y + vdy
            if has_exempt and x == ex and vy == ey and z == ez:
                continue
            if not self.min_y <= vy <= self.max_y:
                continue
            if not (self.cell(x + 1, vy, z).val & F_REP or self.cell(x - 1, vy, z).val & F_REP
                    or self.cell(x, vy, z + 1).val & F_REP or self.cell(x, vy, z - 1).val & F_REP
                    or self.cell(x, vy - 1, z).val & F_TORCH):
                continue
            if self.cell(x, vy, z).owner >= 2:
                return False
        return True

    cdef bint walkable(self, int x, int y, int z, int ex=0, int ey=0, int ez=0, bint has_exempt=False) except -1:
        if (self.cell(x, y, z).val & F_EPROT) or (self.cell(x, y, z).val & F_EFP):
            return False
        cdef unsigned long long key = cell_key(x, y, z) << 7
        if has_exempt:
            key |= exempt_code(ex - x, ey - y, ez - z)
        cdef char* hit = self.walk_cache.find(key)
        if hit != NULL:
            return hit[0] == 2
        cdef bint v = self.wire_walkable(x, y, z, ex, ey, ez, has_exempt, True)
        cdef int ins
        self.walk_cache.get(key, &ins)[0] = 2 if v else 1
        return v

    cdef bint mid_walkable(self, int x, int y, int z, int ex, int ey, int ez) except -1:
        cdef unsigned long long key = cell_key(x, y, z) << 7 | exempt_code(ex - x, ey - y, ez - z)
        cdef char* hit = self.mid_cache.find(key)
        if hit != NULL:
            return hit[0] == 2
        cdef bint v = self.wire_walkable(x, y, z, ex, ey, ez, True, False)
        cdef int ins
        self.mid_cache.get(key, &ins)[0] = 2 if v else 1
        return v

    # --- move entries -------------------------------------------------------------------------

    cdef int pid(self, int x, int y, int z) except -1:
        key = (x, y, z)
        i = self.pos_ids.get(key)
        if i is None:
            i = self.pos_ids[key] = len(self.pos_list)
            self.pos_list.append(key)
            self.h_list.append(abs(x - self.gx) * self.x_h_factor + abs(y - self.gy) * 4 + abs(z - self.gz))
            self.move_cache.append(None)
        return i

    cdef object entry(self, int x, int y, int z, int tx, int ty, int tz, int cost, unsigned int forbid,
                      int gate=0, object extra=None):
        """_find_wire_path's _entry, with the snapshot tables slot left None."""
        cdef int dx = tx - x, dy = ty - y, dz = tz - z
        flat_dir = (dx, dz) if dy == 0 and abs(dx) <= 1 and abs(dz) <= 1 else None
        dest = self.pid(tx, ty, tz)
        h = self.h_list[dest]
        near = ((abs(self.gx - tx) + abs(self.gz - tz)) < HORIZ_RADIUS_MIN
                and abs(self.gy - ty) < Y_RADIUS_MIN)
        arrive = TOWER_FLAG if dy == 4 and ((abs(dx) == 2 and dz == 0) or (dx == 0 and abs(dz) == 2)) else 0
        return dest, cost, forbid, gate, extra, None, flat_dir, h, near, arrive

    cpdef tuple static_moves(self, int cid):
        x, y, z = self.pos_list[cid]
        return self._static_moves(x, y, z)

    cdef tuple _static_moves(self, int x, int y, int z):
        cdef list moves = []
        cdef int d, dx, dz, nx, nz, nx2, nz2, mx, my, mz, k, below_owner
        cdef unsigned int b_lo, b_mid, b_hi
        cdef bint has_ex, above_clear, breaks_slope, mid_ok, mid_dn_ok
        if (self.cell(x, y, z).val & F_INV):
            return [], None
        cdef int min_y = self.min_y, mby = self.max_bridge_y

        for d in range(4):
            dx, dz = HORIZ[d][0], HORIZ[d][1]
            nx, nz = x + dx, z + dz
            b_lo, b_mid, b_hi = snap_bit(dx, -1, dz), snap_bit(dx, 0, dz), snap_bit(dx, 1, dz)

            # Flat move
            if self.walkable(nx, y, nz) and not self.flag(nx, y + 1, nz, F_POWERED):
                if y <= min_y:
                    moves.append(self.entry(x, y, z, nx, y, nz, 1, b_hi | b_mid))
                elif self.can_support(nx, y - 1, nz) and not self.torch_below(nx, y - 1, nz):
                    moves.append(self.entry(x, y, z, nx, y, nz, 1, b_hi | b_mid | b_lo))

            # Slope-up
            if y < mby:
                if (not (self.cell(x, y + 1, z).val & F_NONAIR) and self.walkable(nx, y + 1, nz)
                        and not self.flag(nx, y + 2, nz, F_POWERED)):
                    if (((self.cell(nx, y, nz).val & F_SOLID) or (self.inb(nx, y, nz) and not (self.cell(nx, y, nz).val & F_NONAIR)))
                            and not self.torch_below(nx, y, nz)):
                        moves.append(self.entry(x, y, z, nx, y + 1, nz, 3, b_hi | b_mid, 1))

            # Slope-down
            if y > min_y:
                above_clear = (
                    not (self.cell(nx, y, nz).val & F_SOLID)
                    and self.inb(nx, y, nz)
                    and not (self.cell(nx, y, nz).val & F_NONAIR)
                    and not self.flag(nx, y, nz, F_POWERED)
                )
                has_ex = self.can_support(x, y - 1, z)
                if above_clear and self.walkable(nx, y - 1, nz, x, y - 2, z, has_ex):
                    if y - 1 <= min_y or (self.can_support(nx, y - 2, nz) and not self.torch_below(nx, y - 2, nz)):
                        below_owner = self.owner(nx, y - 3, nz)
                        breaks_slope = False
                        if below_owner >= 2:
                            for k in range(4):
                                if self.owner(nx + HORIZ[k][0], y - 2, nz + HORIZ[k][1]) == below_owner:
                                    breaks_slope = True
                                    break
                        if not breaks_slope:
                            moves.append(self.entry(x, y, z, nx, y - 1, nz, 2, b_lo | b_mid | b_hi))

        for d in range(4):
            dx, dz = HORIZ[d][0], HORIZ[d][1]
            nx2, nz2 = x + 2 * dx, z + 2 * dz
            b_lo, b_mid, b_hi = snap_bit(dx, -1, dz), snap_bit(dx, 0, dz), snap_bit(dx, 1, dz)

            # 2x slope-up
            if y + 2 <= mby:
                mx, my, mz = x + dx, y + 1, z + dz
                mid_ok = (
                    not (self.cell(x, y + 1, z).val & F_NONAIR)
                    and not (self.cell(mx, my, mz).val & F_PROT)
                    and not (self.cell(mx, my, mz).val & F_FP)
                    and self.inb(mx, my, mz)
                    and not (self.cell(mx, y + 2, mz).val & F_SOLID)
                    and not (self.cell(mx, my, mz).val & F_SOLID)
                    and not (self.cell(mx, my, mz).val & F_NONAIR)
                    and self.mid_walkable(mx, my, mz, nx2, y, nz2)
                )
                if mid_ok and self.walkable(nx2, y + 2, nz2):
                    if self.can_support(mx, y, mz) and not self.torch_below(mx, y, mz):
                        moves.append(self.entry(x, y, z, nx2, y + 2, nz2, 7, b_hi | b_mid, 1,
                                                (mx, my, mz)))

            # 2x slope-down
            if y - 2 >= min_y:
                mx, my, mz = x + dx, y - 1, z + dz
                above_clear = (
                    not (self.cell(mx, y, mz).val & F_SOLID)
                    and self.inb(mx, y, mz)
                    and not (self.cell(mx, y, mz).val & F_NONAIR)
                )
                # The mid -> dest dust step also needs the cell above dest open.
                mid_dn_ok = (
                    above_clear
                    and self.inb(nx2, y - 1, nz2)
                    and not (self.cell(nx2, y - 1, nz2).val & F_NONAIR)
                    and not (self.cell(mx, my, mz).val & F_PROT)
                    and not (self.cell(mx, my, mz).val & F_FP)
                    and self.inb(mx, my, mz)
                    and not (self.cell(mx, my, mz).val & F_SOLID)
                    and not (self.cell(mx, my, mz).val & F_NONAIR)
                    and self.mid_walkable(mx, my, mz, x, y - 2, z)
                )
                if mid_dn_ok and self.walkable(nx2, y - 2, nz2):
                    if y - 2 <= min_y or (self.can_support(nx2, y - 3, nz2) and not self.torch_below(nx2, y - 3, nz2)):
                        moves.append(self.entry(x, y, z, nx2, y - 2, nz2, 7, b_lo | b_mid | b_hi, 0,
                                                (mx, my, mz)))

        # +1Y from tower top (only right after a tower move)
        if y + 1 <= mby and self.walkable(x, y + 1, z):
            moves.append(self.entry(x, y, z, x, y + 1, z, 1, 0, 2))

        # Tower-up
        self.tower_moves(x, y, z, moves)

        return moves, self.powered_minus4(x, y, z)

    cdef int tower_moves(self, int x, int y, int z, list moves) except -1:
        """_tower_candidates, appending entries."""
        cdef int d, tdx, tdz, rx, rz, cx, cz, cy, cdy, k
        cdef bint cuts, col_clear
        if y + 4 > self.max_bridge_y:
            return 0
        for d in range(4):
            tdx, tdz = HORIZ[d][0], HORIZ[d][1]
            cx, cz = x + 2 * tdx, z + 2 * tdz
            if not self.walkable(cx, y + 4, cz):
                continue
            rx, rz = x + tdx, z + tdz
            if not self.inb(rx, y, rz) or (self.cell(rx, y, rz).val & F_FP) or (self.cell(rx, y, rz).val & F_PROT):
                continue
            if (self.cell(rx, y, rz).val & F_NONAIR):
                continue
            if not self.can_support(rx, y - 1, rz):
                continue
            cuts = False
            if self.inb(rx, y - 2, rz) and (self.cell(rx, y - 2, rz).val & F_WIRE):
                for k in range(4):
                    if (self.inb(rx + HORIZ[k][0], y - 1, rz + HORIZ[k][1])
                            and (self.cell(rx + HORIZ[k][0], y - 1, rz + HORIZ[k][1]).val & F_WIRE)):
                        cuts = True
                        break
            if cuts:
                continue
            col_clear = True
            for cdy in range(5):
                cy = y + cdy
                if not self.inb(cx, cy, cz) or (self.cell(cx, cy, cz).val & F_FP) or (self.cell(cx, cy, cz).val & F_PROT):
                    col_clear = False
                    break
                if (self.cell(cx, cy, cz).val & F_NONAIR):
                    col_clear = False
                    break
                for k in range(4):
                    if (self.inb(cx + HORIZ[k][0], cy, cz + HORIZ[k][1])
                            and (self.cell(cx + HORIZ[k][0], cy, cz + HORIZ[k][1]).val & F_WIRE)):
                        col_clear = False
                        break
                if not col_clear:
                    break
            if not col_clear:
                continue
            if self.inb(cx, y - 1, cz):
                if self.owner(cx, y - 1, cz) >= 2:
                    continue
                if (self.cell(cx, y - 1, cz).val & F_WIRE):
                    continue
            moves.append(self.entry(x, y, z, cx, y + 4, cz, 8,
                                    snap_bit(tdx, 0, tdz) | snap_bit(tdx, -1, tdz) | snap_bit(tdx, 1, tdz)))
        return 0

    cdef bint m4_cell_clear(self, int x, int y, int z) except -1:
        cdef int k, ddy, sx, sy, sz
        for k in range(4):
            for ddy in range(-1, 2):
                sx, sy, sz = x + HORIZ[k][0], y + ddy, z + HORIZ[k][1]
                if (self.cell(sx, sy, sz).val & F_PROT) or self.owner(sx, sy, sz) >= 2:
                    return False
        return self.owner(x, y, z) < 2

    cdef object powered_minus4(self, int x, int y, int z):
        """_powered_minus4_move."""
        cdef int d, dx, dz, nx, nz, k, ddy, i, bx, by, bz, cx, cy, cz
        cdef int cells[6][3]
        cdef unsigned int b_lo, b_mid, b_hi
        cdef bint ok
        if y - 4 < self.min_y:
            return None
        if not self.walkable(x, y - 4, z):
            return None
        bx = x
        by = y - 1
        bz = z
        if ((bx == self.gx and by == self.gy and bz == self.gz) or (self.cell(bx, by, bz).val & F_TERM)
                or (self.cell(bx, by, bz).val & F_FP) or (self.cell(bx, by, bz).val & F_PROT)):
            return None
        if (self.cell(bx, by, bz).val & F_SOLID):
            if self.flag(bx, by, bz, F_GLASS):
                return None
        elif not (self.inb(bx, by, bz) and not (self.cell(bx, by, bz).val & F_NONAIR)):
            return None
        sides = []
        for d in range(4):
            dx, dz = HORIZ[d][0], HORIZ[d][1]
            nx, nz = x + dx, z + dz
            b_lo, b_mid, b_hi = snap_bit(dx, -1, dz), snap_bit(dx, 0, dz), snap_bit(dx, 1, dz)
            set3(cells[0], nx, y - 1, nz)  # first wall torch (inverted)
            set3(cells[1], nx, y - 2, nz)  # inverted dust
            set3(cells[2], nx, y - 3, nz)  # second support block
            set3(cells[3], x, y - 3, z)    # second wall torch (active)
            set3(cells[4], nx, y, nz)      # clearance above first torch
            set3(cells[5], x, y - 2, z)    # clearance above second torch
            ok = True
            for i in range(6):
                cx, cy, cz = cells[i][0], cells[i][1], cells[i][2]
                if (not self.inb(cx, cy, cz) or (cx == self.gx and cy == self.gy and cz == self.gz)
                        or (self.cell(cx, cy, cz).val & F_TERM) or (self.cell(cx, cy, cz).val & F_FP)
                        or (self.cell(cx, cy, cz).val & F_PROT) or (self.cell(cx, cy, cz).val & F_SOLID)
                        or (self.cell(cx, cy, cz).val & F_NONAIR)):
                    ok = False
                    break
            if ok:
                for i in range(6):
                    if not self.m4_cell_clear(cells[i][0], cells[i][1], cells[i][2]):
                        ok = False
                        break
            if not ok:
                continue
            for k in range(4):
                for ddy in (-1, 1):
                    if self.owner(nx + HORIZ[k][0], y - 2 + ddy, nz + HORIZ[k][1]) != 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sides.append((b_lo | b_mid | b_hi, ((bx, by, bz),) + tuple(
                    (cells[i][0], cells[i][1], cells[i][2]) for i in range(6))))
        return (self.entry(x, y, z, x, y - 4, z, 8, 0), sides) if sides else None


def _strip_tables(static):
    """A static-moves result with each entry's snapshot-table slot blanked, for comparison."""
    moves, m4 = static
    strip = lambda e: e[:5] + (None,) + e[6:]
    return [strip(e) for e in moves], (None if m4 is None else (strip(m4[0]), m4[1]))


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

    cdef int load(self, int cid, object expand, World world, bint verify, list pos_list, list h_list,
                  object congestion_cost) except -1:
        if world is None:
            static = expand(cid)
        else:
            static = world.static_moves(cid)
            if verify:
                expected = _strip_tables(expand(cid))
                if static != expected:
                    raise AssertionError(f"A* kernel move generation differs at {pos_list[cid]}:\n"
                                         f"  kernel {static}\n  python {expected}")
        moves, m4 = static
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


cdef inline bint dominated(FlatMap[NodeInfo]* nodes, long long node, double g):
    cdef int d = (node >> SIG_SHIFT) & 15, k
    cdef long long base = node & ~(<long long>15 << SIG_SHIFT)
    cdef NodeInfo* other
    for k in range(d):
        other = nodes.find(base | (<long long>k << SIG_SHIFT))
        if other != NULL and other.g <= g:
            return True
    return False


def search(list seed_nodes, list seed_groups, goal_ids, list group_best_h, list group_max_f,
           long long best_node, long long best_h, double weight, time_limit, list pos_list,
           list h_list, expand, congestion_cost, World world=None, bint verify=False):
    """See pathfinding._search_py. With world, cells' static moves are generated here instead of
    by expand(); verify additionally checks each against expand()."""
    global STAT_SEARCHES, STAT_POPS, STAT_PUSHES, STAT_CELLS, STAT_LOAD_TIME
    if not _init_done:
        raise RuntimeError("_astar.init() was not called")
    cdef _Cells cells = _Cells()
    cdef CameFrom came_from = CameFrom()
    cdef FlatMap[NodeInfo]* nodes = &came_from.nodes
    cdef vector[HeapEntry] heap
    cdef vector[long long] gbest = group_best_h
    cdef vector[double] gmax = group_max_f
    cdef vector[char] explored_xyz
    cdef vector[char] is_goal
    cdef vector[int] taken
    cdef vector[int] extras  # (ex, ey, ez) triples of this expansion's taken 2x slopes
    cdef HeapEntry e, ne
    cdef NodeInfo* np
    cdef int ins
    cdef NodeInfo info
    cdef NodeInfo* ni
    cdef Move* m
    cdef CellMoves* cmv
    cdef M4Side* side
    cdef long long counter = 0, h, node, nkey, cur_snap, h_radius_pruned = 0
    cdef long long n_explored_xyz = 0, pops = 0
    cdef int stagnation = 0, cid, i, k, last, m4_side = -1
    cdef int in_dx = 0, in_dz = 0, nx, ny, nz
    cdef int cur_depth, cur_cand, ndepth, ncand, step
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
        np = nodes.get(node, &ins)
        np.g = 0
        np.parent = -1
        np.expanded = False
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
        ni = nodes.find(e.node)
        if e.g > ni.g:
            continue
        # Dominance: the same state with less signal depth at no more cost reaches everything
        # this one can (it is expanded first, or was dominated in turn).
        if not ni.expanded and dominated(nodes, e.node, e.g):
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
        cur_depth = (e.node >> SIG_SHIFT) & 15
        cur_cand = (e.node >> (SIG_SHIFT + 4)) & 7
        if cells.cm[cid].start < 0:
            t_load = time.perf_counter()
            cells.load(cid, expand, world, verify, pos_list, h_list, congestion_cost)
            STAT_LOAD_TIME += time.perf_counter() - t_load
            STAT_CELLS += 1
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
            # Signal state (pathfinding._signal_step): prune moves that would leave a stretch
            # with no cell _place_repeaters_for_net could put a repeater on.
            step = abs(m.dx) + abs(m.dz)
            if m.arrive or (m.dy == -4 and step == 0):
                ndepth = 1
                ncand = 0
            else:
                ndepth = cur_depth
                if (cur_cand and m.dy == 0 and step == 1 and HORIZ[cur_cand - 1][0] == m.dx
                        and HORIZ[cur_cand - 1][1] == m.dz):
                    ndepth = 0
                ndepth += step
                if ndepth >= SIG_LIMIT:
                    continue
                ncand = 0
                if step == 1 and (m.dy == 0 or m.dy == -1):
                    for i in range(4):
                        if HORIZ[i][0] == m.dx and HORIZ[i][1] == m.dz:
                            ncand = i + 1
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
            nkey = ((<long long>m.dest << NODE_SHIFT) | m.arrive | snap
                    | (<long long>(ndepth | (ncand << 4)) << SIG_SHIFT))
            if dominated(nodes, nkey, new_g):
                continue
            np = nodes.find(nkey)
            if np != NULL:
                if not new_g < np.g:
                    continue
                np.g = new_g
            else:
                if not new_g < 1e9:
                    continue
                np = nodes.get(nkey, &ins)
                np.g = new_g
                np.parent = -1
                np.expanded = False
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
                STAT_PUSHES += 1
            else:
                h_radius_pruned += 1

    STAT_SEARCHES += 1
    STAT_POPS += pops
    for i in range(<int>gbest.size()):
        group_best_h[i] = gbest[i]
    return (reached if has_reached else None), best_node, h_radius_pruned, early_stop_reason, came_from
