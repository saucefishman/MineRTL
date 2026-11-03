# distutils: language = c++
"""The A* search of _find_wire_path: the loop (search) and cells' static moves (World).

This is the maintained implementation. pathfinding._search_py and _static_moves are deprecated
Python versions (MINECRAFT_V_ASTAR=py/check, and the fallback when this isn't built); they
were exact twins when this was written but are no longer kept in sync. Without a World,
search() falls back to the Python expand() callback for moves.

Floating point: g, f and costs are doubles combined in the same order as the Python loop, so
every comparison (heap order, f-cap, h-radius) matches it exactly.
"""
from libc.stdint cimport int64_t, uint16_t, uint32_t
from libc.stdlib cimport calloc, free
from libcpp.vector cimport vector
from cython.operator cimport dereference
from cpython.set cimport PySet_Contains
from posix.time cimport clock_gettime, timespec, CLOCK_MONOTONIC
cimport cython
import time
import numpy as np
from minecraft_v.progress import pulse

cdef extern from *:
    """
    #include <algorithm>
    #include <functional>
    #include <vector>
    struct HeapEntry {
        double f; long long neg_h; double g; long long counter;
        double g_wire; long long node; long long parent; int group;
    };
    // heapq's order on the tuple (f, -h, g, counter, ...). counter is unique, so the order is total and
    // the pops don't depend on how the heap is laid out: 4-ary over 16 byte keys, with the rest of an
    // entry in an array by counter. Sifting is most of a big search's cache misses.
    struct HeapKey { double f; int neg_h; unsigned int counter; };
    struct HeapData { double g; double g_wire; long long node; long long parent; int group; int next; };
    // One queue per (f, -h), first in first out, which is the counter's order: a search has few of them.
    struct Bucket { unsigned long long key; int head, tail; };
    struct Heap {
        std::vector<HeapKey> keys;
        std::vector<HeapData> data;
        // With g and weight * h whole quarters, f and h fix g exactly, so it needs no looking at, and entries
        // go in buckets by (f, -h). Otherwise the 4-ary heap on keys.
        bool g_can_differ = true;
        std::vector<Bucket> buckets;     // open addressing; head < 0 is an empty queue
        size_t bucket_count = 0;
        std::vector<unsigned long long> active;  // min-heap of the keys of non-empty queues
        static constexpr unsigned long long NO_BUCKET = ~0ULL;
        inline bool before(const HeapKey& a, const HeapKey& b) const {
            if (a.f != b.f) return a.f < b.f;
            if (a.neg_h != b.neg_h) return a.neg_h < b.neg_h;
            if (g_can_differ) {
                double ga = data[a.counter].g, gb = data[b.counter].g;
                if (ga != gb) return ga < gb;
            }
            return a.counter < b.counter;
        }
        bool empty() const { return keys.empty() && active.empty(); }
        void clear() {
            keys.clear();
            data.clear();
            active.clear();
            if (data.capacity() > (1u << 22)) { std::vector<HeapData>().swap(data); std::vector<HeapKey>().swap(keys); }
            size_t cap = buckets.size() > (1u << 12) || buckets.empty() ? (1u << 10) : buckets.size();
            buckets.assign(cap, Bucket{NO_BUCKET, -1, -1});
            bucket_count = 0;
        }
        static inline size_t bucket_mix(unsigned long long k) {
            k *= 0x9e3779b97f4a7c15ULL;
            return (size_t)(k >> 40);
        }
        inline Bucket* bucket(unsigned long long key) {
            size_t mask = buckets.size() - 1, i = bucket_mix(key) & mask;
            while (buckets[i].key != key) {
                if (buckets[i].key == NO_BUCKET) {
                    if ((bucket_count + 1) * 2 > mask + 1) {
                        std::vector<Bucket> old;
                        old.swap(buckets);
                        buckets.assign(old.size() * 2, Bucket{NO_BUCKET, -1, -1});
                        size_t m = buckets.size() - 1;
                        for (const Bucket& b : old) {
                            if (b.key == NO_BUCKET) continue;
                            size_t j = bucket_mix(b.key) & m;
                            while (buckets[j].key != NO_BUCKET) j = (j + 1) & m;
                            buckets[j] = b;
                        }
                        return bucket(key);
                    }
                    buckets[i].key = key;
                    bucket_count++;
                    return &buckets[i];
                }
                i = (i + 1) & mask;
            }
            return &buckets[i];
        }
        // e.counter is its place in data: the number of pushes so far.
        inline void push(const HeapEntry& e) {
            data.push_back(HeapData{e.g, e.g_wire, e.node, e.parent, e.group, -1});
            if (!g_can_differ) {
                // f * 4 < 2^32: g is under 1e9 by the time it gets here, and h is a few thousand.
                unsigned long long key = ((unsigned long long)(e.f * 4) << 32) | (unsigned int)((int)e.neg_h + 0x40000000);
                Bucket* b = bucket(key);
                int at = (int)e.counter;
                if (b->head < 0) {
                    b->head = b->tail = at;
                    active.push_back(key);
                    std::push_heap(active.begin(), active.end(), std::greater<unsigned long long>());
                } else {
                    data[b->tail].next = at;
                    b->tail = at;
                }
                return;
            }
            HeapKey k{e.f, (int)e.neg_h, (unsigned int)e.counter};
            size_t i = keys.size();
            keys.push_back(k);
            HeapKey* h = keys.data();
            while (i) {
                size_t up = (i - 1) >> 2;
                if (!before(k, h[up])) break;
                h[i] = h[up];
                i = up;
            }
            h[i] = k;
        }
        inline HeapEntry pop() {
            if (!g_can_differ) {
                unsigned long long key = active[0];
                Bucket* b = bucket(key);
                int at = b->head;
                const HeapData& d = data[at];
                b->head = d.next;
                if (d.next < 0) {
                    std::pop_heap(active.begin(), active.end(), std::greater<unsigned long long>());
                    active.pop_back();
                } else {
                    __builtin_prefetch(&data[d.next]);
                }
                return HeapEntry{(double)(key >> 32) / 4, (long long)(int)((unsigned int)key - 0x40000000), d.g, at,
                                 d.g_wire, d.node, d.parent, d.group};
            }
            HeapKey* h = keys.data();
            HeapKey top = h[0];
            size_t n = keys.size() - 1;
            HeapKey k = h[n];
            keys.pop_back();
            if (n) {
                size_t i = 0;
                for (;;) {
                    size_t c = (i << 2) + 1;
                    if (c >= n) break;
                    size_t end = c + 4 < n ? c + 4 : n, best = c;
                    for (size_t j = c + 1; j < end; j++) if (before(h[j], h[best])) best = j;
                    if (!before(h[best], k)) break;
                    h[i] = h[best];
                    i = best;
                }
                h[i] = k;
            }
            const HeapData& d = data[top.counter];
            return HeapEntry{top.f, top.neg_h, d.g, top.counter, d.g_wire, d.node, d.parent, d.group};
        }
    };
    // Open-addressing (linear probing) hash map from 64-bit keys; values value-initialised.
    // Key and value share a slot (one cache line per probe); ~0 marks an empty slot, which no
    // node or cell key uses. Pointers it returns are invalidated by the next get() that inserts.
    template <class V> struct FlatMap {
        struct Slot { unsigned long long key; V val; };
        static constexpr unsigned long long EMPTY = ~0ULL;
        std::vector<Slot> slots;
        size_t mask = 0, count = 0;
        // A search makes several of these: storage goes round a pool rather than through malloc, which
        // hands big blocks back to the system. Single-threaded, like the rest of this module.
        static std::vector<std::vector<Slot>>& pool() {
            static std::vector<std::vector<Slot>> p;
            return p;
        }
        FlatMap() {
            std::vector<std::vector<Slot>>& p = pool();
            if (!p.empty()) { slots.swap(p.back()); p.pop_back(); }
            reset(1 << 12);
        }
        ~FlatMap() {
            std::vector<std::vector<Slot>>& p = pool();
            if (p.size() < 32 && slots.capacity() <= (1u << 20)) p.emplace_back(std::move(slots));
        }
        void reset(size_t cap) {
            slots.assign(cap, Slot{EMPTY, V()});
            mask = cap - 1; count = 0;
        }
        static inline size_t mix(unsigned long long k) {
            k ^= k >> 29;  // the high bits (a node's cell) have to reach the low bits of the product's top half
            k *= 0x9e3779b97f4a7c15ULL;
            return (size_t)(k >> 32);
        }
        inline void prefetch(unsigned long long k) const { __builtin_prefetch(&slots[mix(k) & mask]); }
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
            std::vector<std::vector<Slot>>& p = pool();
            if (!p.empty()) { slots.swap(p.back()); p.pop_back(); }
            reset(old.size() * 2);
            int ins;
            for (const Slot& s : old) if (s.key != EMPTY) *get(s.key, &ins) = s.val;
            if (p.size() < 32 && old.capacity() <= (1u << 20)) p.emplace_back(std::move(old));
        }
    };
    typedef int bint_;
    // A search's states. Hashed without the signal depth, so the same state at other depths sits next to it and
    // one walk answers both "known at no more cost" and "dominated by less depth". 16 byte slots: g is whole
    // without congestion, else an index into gs; parent is NOT_EXPANDED, UNREACHED or an index into parents.
    struct NodeSlot { unsigned long long key; int g; int parent; };
    struct NodeTable {
        static constexpr unsigned long long EMPTY = ~0ULL;
        static constexpr int NOT_EXPANDED_ = -2, UNREACHED_ = -3;
        std::vector<NodeSlot> slots;
        std::vector<long long> parents;
        std::vector<double> gs;
        size_t mask = 0, count = 0;
        unsigned long long depth_mask = 0;
        bool side_g = false;
        static std::vector<std::vector<NodeSlot>>& pool() {
            static std::vector<std::vector<NodeSlot>> p;
            return p;
        }
        NodeTable() {
            std::vector<std::vector<NodeSlot>>& p = pool();
            if (!p.empty()) { slots.swap(p.back()); p.pop_back(); }
            reset(1 << 12, 0, false);
        }
        ~NodeTable() {
            std::vector<std::vector<NodeSlot>>& p = pool();
            if (p.size() < 32 && slots.capacity() <= (1u << 20)) p.emplace_back(std::move(slots));
        }
        void reset(size_t cap, unsigned long long depths, bool side) {
            slots.assign(cap, NodeSlot{EMPTY, 0, 0});
            mask = cap - 1; count = 0;
            depth_mask = depths; side_g = side;
            parents.clear(); gs.clear();
        }
        inline size_t start(unsigned long long k) const { return FlatMap<char>::mix(k & ~depth_mask) & mask; }
        inline void prefetch(unsigned long long k) const { __builtin_prefetch(&slots[start(k)]); }
        inline double g_of(const NodeSlot* s) const { return side_g ? gs[s->g] : (double)s->g; }
        inline NodeSlot* find(unsigned long long k) {
            size_t i = start(k);
            while (slots[i].key != EMPTY) {
                if (slots[i].key == k) return &slots[i];
                i = (i + 1) & mask;
            }
            return nullptr;
        }
        // Whether the same state with less signal depth is known at no more cost.
        inline bool dominated(unsigned long long k, double g) const {
            unsigned long long d = k & depth_mask;
            if (!d) return false;
            for (size_t i = start(k); slots[i].key != EMPTY; i = (i + 1) & mask) {
                const NodeSlot& s = slots[i];
                if (!((s.key ^ k) & ~depth_mask) && (s.key & depth_mask) < d && s.parent != UNREACHED_ && g_of(&s) <= g)
                    return true;
            }
            return false;
        }
        // A move's arrival at k with cost g: *skip if k is known at no more cost or dominated, else k's slot if it has one.
        inline NodeSlot* probe(unsigned long long k, double g, bint_* skip) {
            unsigned long long d = k & depth_mask;
            NodeSlot* found = nullptr;
            *skip = 0;
            for (size_t i = start(k); slots[i].key != EMPTY; i = (i + 1) & mask) {
                NodeSlot* s = &slots[i];
                if (s->key == k) {
                    found = s;
                    if (s->parent != UNREACHED_ && !(g < g_of(s))) { *skip = 1; return s; }
                } else if (!((s->key ^ k) & ~depth_mask) && (s->key & depth_mask) < d && s->parent != UNREACHED_
                           && g_of(s) <= g) {
                    *skip = 1;
                    return s;
                }
            }
            return found;
        }
        // Invalidates slot pointers. k isn't in the table.
        inline NodeSlot* insert(unsigned long long k) {
            if ((count + 1) * 2 > mask + 1) grow();
            size_t i = start(k);
            while (slots[i].key != EMPTY) i = (i + 1) & mask;
            slots[i] = NodeSlot{k, 0, NOT_EXPANDED_};
            count++;
            return &slots[i];
        }
        // First cost of a slot that has none yet.
        inline void set_first_g(NodeSlot* s, double g) {
            if (side_g) { s->g = (int)gs.size(); gs.push_back(g); }
            else s->g = (int)g;
            s->parent = NOT_EXPANDED_;
        }
        inline void set_g(NodeSlot* s, double g) {
            if (side_g) gs[s->g] = g;
            else s->g = (int)g;
        }
        inline void set_unreached(NodeSlot* s) { s->parent = UNREACHED_; }
        inline void set_parent(NodeSlot* s, long long parent) { s->parent = (int)parents.size(); parents.push_back(parent); }
        inline long long parent_of(const NodeSlot* s) const { return parents[s->parent]; }
        void grow() {
            std::vector<NodeSlot> old;
            old.swap(slots);
            std::vector<std::vector<NodeSlot>>& p = pool();
            if (!p.empty()) { slots.swap(p.back()); p.pop_back(); }
            slots.assign(old.size() * 2, NodeSlot{EMPTY, 0, 0});
            mask = slots.size() - 1;
            for (const NodeSlot& s : old) {
                if (s.key == EMPTY) continue;
                size_t i = start(s.key);
                while (slots[i].key != EMPTY) i = (i + 1) & mask;
                slots[i] = s;
            }
            if (p.size() < 32 && old.capacity() <= (1u << 20)) p.emplace_back(std::move(old));
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
    cdef cppclass Heap:
        bint g_can_differ
        void clear()
        void push(const HeapEntry& e)
        HeapEntry pop()
        bint empty()
    cdef cppclass FlatMap[V]:
        void prefetch(unsigned long long k)
        V* find(unsigned long long k)
        V* get(unsigned long long k, int* inserted)
        void reset(size_t cap)
    cdef struct NodeSlot:
        int parent
    cdef cppclass NodeTable:
        void reset(size_t cap, unsigned long long depths, bint side)
        void prefetch(unsigned long long k)
        double g_of(const NodeSlot* s)
        NodeSlot* find(unsigned long long k)
        bint dominated(unsigned long long k, double g)
        NodeSlot* probe(unsigned long long k, double g, int* skip)
        NodeSlot* insert(unsigned long long k)
        void set_first_g(NodeSlot* s, double g)
        void set_g(NodeSlot* s, double g)
        void set_unreached(NodeSlot* s)
        void set_parent(NodeSlot* s, long long parent)
        long long parent_of(const NodeSlot* s)


cdef enum:
    MAX_CELL_MOVES = 64  # more than _static_moves' (flat, slopes, 2x slopes, +1, towers, -4)
    NOT_EXPANDED = -2   # NodeSlot.parent
    UNREACHED = -3      # a slot made for a cost past the limit


# A search reads a cell's moves some thirty times over: small, so that's few cache lines.
cdef struct Move:
    uint32_t forbid
    int dest
    int h
    short shift       # snapshot shift table index for (dx, dy, dz)
    signed char cost
    signed char gate  # 0 always, 1 not right after a tower move, 2 only right after one
    signed char dx, dy, dz
    signed char fdx, fdz
    signed char ex, ey, ez  # 2x-slope intermediate, relative to the expanded cell
    char has_extra, has_flat, near, arrive  # arrive: by a tower move
    double cong       # congestion cost of dest (0 without congestion)


cdef struct M4Side:
    uint32_t forbid
    int start, count  # into m4_cells (offsets relative to the expanded cell)


cdef struct CellMoves:
    int start, count  # into moves
    int m4_move       # index into moves, or -1
    int m4_start, m4_count  # into m4_sides


cdef CellMoves _NO_MOVES                # a cell whose moves aren't worked out yet
_NO_MOVES.start = -1
_NO_MOVES.count = 0
_NO_MOVES.m4_move = -1
_NO_MOVES.m4_start = 0
_NO_MOVES.m4_count = 0


# Set by init() from pathfinding (single source of truth for the encoding).
cdef int HORIZ[4][2]             # _HORIZ_DIRS
cdef int SNAP_BIT_INDEX[27]      # (ox+1)*9 + (oy+1)*3 + (oz+1) -> bit index, or -1
cdef long long TOWER_FLAG = 0
cdef int NODE_SHIFT = 0
cdef int SIG_SHIFT = 0           # signal depth (4 bits) then repeater-candidate dir (3 bits)
cdef int SIG_LIMIT = 0           # _REPEATER_INTERVAL
cdef bint FOOT_REPEATERS = False  # see pathfinding.set_foot_repeaters
cdef long long SNAP_MASK = 0
cdef int STAGNATION = 0
cdef int MAX_NODES = 0
cdef long long MAX_EXPANSIONS = 0
cdef double H_RADIUS_FACTOR = 0
cdef int N_SNAP_BITS = 0
cdef int HORIZ_RADIUS_MIN = 0
cdef int Y_RADIUS_MIN = 0
cdef object _init_done = False

# Totals across searches in this process (see stats()).
cdef long long STAT_SEARCHES = 0, STAT_POPS = 0, STAT_PUSHES = 0, STAT_CELLS = 0
cdef double STAT_LOAD_TIME = 0


cdef inline double now_seconds() noexcept nogil:
    cdef timespec t
    clock_gettime(CLOCK_MONOTONIC, &t)
    return t.tv_sec + 1e-9 * t.tv_nsec


LAST = {}  # counters of the most recent search (debugging / tuning)


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


def init(snap_offsets, tower_flag, node_shift, stagnation, max_nodes, max_expansions, h_radius_factor,
         horiz_dirs, horiz_radius_min, y_radius_min, sig_shift, sig_limit):
    global TOWER_FLAG, NODE_SHIFT, SNAP_MASK, STAGNATION, MAX_NODES, MAX_EXPANSIONS, H_RADIUS_FACTOR, N_SNAP_BITS, _init_done
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
    MAX_EXPANSIONS = max_expansions
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


@cython.final
cdef class CameFrom:
    """came_from as a read-only mapping (node -> parent node), like the Python loop's dict."""
    cdef NodeTable nodes

    def get(self, long long node, default=None):
        cdef NodeSlot* ni = self.nodes.find(node)
        if ni == NULL or ni.parent < 0:
            return default
        return self.nodes.parent_of(ni)

    def __getitem__(self, long long node):
        cdef NodeSlot* ni = self.nodes.find(node)
        if ni == NULL or ni.parent < 0:
            raise KeyError(node)
        return self.nodes.parent_of(ni)

    def __contains__(self, long long node):
        cdef NodeSlot* ni = self.nodes.find(node)
        return ni != NULL and ni.parent >= 0


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



# Grid bits (see CellGrid). Those shared with F_* have the same value, so a grid word masked by
# G_DIRECT reads as flags; the three that don't overlay F_EPROT, F_PROT and F_EFP.
cdef enum:
    G_OWNED = 1 << 5     # in dust_owner (CellGrid.owner holds who)
    G_KEEP1 = 1 << 6     # keepout count >= 1
    G_KEEP2 = 1 << 7     # keepout count >= 2
    G_FP_READ = 1 << 11  # F_FP has been read from the footprint set
    G_DIRECT = F_SOLID | F_NONAIR | F_WIRE | F_TORCH | F_REP | F_FP | F_INV | F_TERM
    GRID_MARGIN = 8      # cells either side of the region; queries beyond read the containers

# Per-search small sets, one map (World.small); bit << 0 for the effective keepout, << 1 for raw.
cdef enum:
    S_EXTRA = 1 << 0     # bits 0-1
    S_EXCL = 1 << 2      # bits 2-3
    S_RELIEF = 1 << 4
    S_PENDING = 1 << 5


@cython.final
cdef class CellGrid:
    """One word per cell of what the searches read, mirrored from the router's containers.

    A search asks about each cell a dozen times over a dozen Python containers; this answers from
    one read. It lives as long as a router, kept current by the containers' writers
    (JournaledSet, JournaledDict, _BlockIndex.update call put / put_owner), so nothing is rebuilt
    per search. Filled up front, region blocks from an array since walking millions of tuples is
    slow, except the footprint set: 13M tuples on the full CPU, three minutes to walk on a loaded
    machine, so each cell asks it once, when first read. Keepouts come from a count dict that
    doesn't change once built (load_keepout). Only cells with blocks are written, leaving other
    pages untouched. Cells outside the grid aren't mirrored; World reads those from the containers."""
    cdef uint16_t* g
    cdef int nx, ny, nz
    cdef long long sx, sy
    cdef FlatMap[int] owner_map     # cell_key -> interned owner, 0 for none (no erase in FlatMap)
    cdef dict owner_ids
    cdef FlatMap[char] own_map      # the cells of own_obj, last net's _own set
    cdef object own_obj, keep_count
    cdef object non_air, wire, torch, repeater, solid, dust_owner, inverted, fp, terminal

    def __cinit__(self):
        self.g = NULL

    def __init__(self, width, height, length, kinds, index, solid, dust_owner, inverted, fp, terminal):
        """kinds: the region's cells as an array of _BlockIndex's kind bits (index's sets, as they
        cost minutes to walk). The containers must call put / put_owner from here on."""
        self.nx = width + 2 * GRID_MARGIN
        self.ny = height + 2 * GRID_MARGIN
        self.nz = length + 2 * GRID_MARGIN
        self.sy = self.nz
        self.sx = <long long>self.ny * self.nz
        self.g = <uint16_t*>calloc(<size_t>(self.sx * self.nx), sizeof(uint16_t))
        if self.g == NULL:
            raise MemoryError("CellGrid")
        self.owner_ids = {}
        self.non_air, self.wire, self.torch, self.repeater = index.non_air, index.wire, index.torch, index.repeater
        self.solid, self.dust_owner, self.inverted, self.fp, self.terminal = solid, dust_owner, inverted, fp, terminal
        cdef size_t cells = <size_t>(self.sx * self.nx)
        flat = np.asarray(<uint16_t[:cells]>self.g)
        xs, ys, zs = np.nonzero(kinds)
        flat[(xs + GRID_MARGIN) * self.sx + (ys + GRID_MARGIN) * self.sy + (zs + GRID_MARGIN)] = (
            kinds[xs, ys, zs].astype(np.uint16) << 1)  # kind bits are F_NONAIR.. >> 1
        for cells_set, bit in ((solid, F_SOLID), (inverted, F_INV), (terminal, F_TERM)):
            self.put_all(cells_set, bit)
        for pos, value in dust_owner.items():
            self.put_owner(pos, value)
        index.grid = self
        index.grid_bits = (F_NONAIR, F_WIRE, F_TORCH, F_REP)
        solid._grid, solid._grid_bit = self, F_SOLID
        inverted._grid, inverted._grid_bit = self, F_INV
        dust_owner._grid = self

    def __dealloc__(self):
        free(self.g)

    def fp_set(self):
        return self.fp

    def terminal_set(self):
        return self.terminal

    cdef int read_fp(self, uint16_t* w, int x, int y, int z) except -1:
        w[0] |= G_FP_READ | (F_FP if PySet_Contains(self.fp, (x, y, z)) else 0)
        return 0

    cdef inline uint16_t* word(self, int x, int y, int z) noexcept:
        cdef unsigned int ux = <unsigned int>(x + GRID_MARGIN), uy = <unsigned int>(y + GRID_MARGIN)
        cdef unsigned int uz = <unsigned int>(z + GRID_MARGIN)
        if ux >= <unsigned int>self.nx or uy >= <unsigned int>self.ny or uz >= <unsigned int>self.nz:
            return NULL
        return &self.g[ux * self.sx + uy * self.sy + uz]

    cpdef put(self, tuple pos, unsigned int bit, bint on):
        cdef uint16_t* w = self.word(pos[0], pos[1], pos[2])
        if w != NULL:
            w[0] = (w[0] | bit) if on else (w[0] & ~bit)

    cdef int put_all(self, cells, unsigned int bit) except -1:
        cdef uint16_t* w
        cdef tuple pos
        for pos in cells:
            w = self.word(pos[0], pos[1], pos[2])
            if w != NULL:
                w[0] |= bit
        return 0

    cpdef put_owner(self, tuple pos, value):
        cdef int x = pos[0], y = pos[1], z = pos[2], ins
        cdef uint16_t* w = self.word(x, y, z)
        if w == NULL:
            return
        if value is None:
            w[0] &= ~G_OWNED
            return
        self.owner_map.get(cell_key(x, y, z), &ins)[0] = self.owner_id(value)
        w[0] |= G_OWNED

    cdef int owner_id(self, value) except -1:
        oid = self.owner_ids.get(value)
        if oid is None:
            oid = self.owner_ids[value] = 1 + len(self.owner_ids)
        return oid

    cdef int load_keepout(self, dict count) except -1:
        """Mirrors a keepout count dict; the bits of a previous one go."""
        if count is self.keep_count:
            return 0
        cdef uint16_t* w
        if self.keep_count is not None:
            for pos in self.keep_count:
                w = self.word(pos[0], pos[1], pos[2])
                if w != NULL:
                    w[0] &= ~(G_KEEP1 | G_KEEP2)
        self.keep_count = count
        for pos, n in count.items():
            w = self.word(pos[0], pos[1], pos[2])
            if w != NULL and n:
                w[0] |= G_KEEP1 | (G_KEEP2 if n >= 2 else 0)
        return 0

    cdef int load_own(self, own) except -1:
        cdef int ins
        if own is self.own_obj:
            return 0
        self.own_obj = own
        self.own_map.reset(1 << 12)
        for pos in own:
            self.own_map.get(cell_key(pos[0], pos[1], pos[2]), &ins)[0] = 1
        return 0


cdef inline void set3(int* c, int x, int y, int z) noexcept nogil:
    c[0] = x
    c[1] = y
    c[2] = z


cdef struct _CellVectors:
    vector[int]* x
    vector[int]* y
    vector[int]* z
    vector[long long]* h
    vector[CellMoves]* cm


cdef inline int exempt_code(int ox, int oy, int oz) noexcept nogil:
    """Single exempt cell as an offset from the queried cell (0 = no exempt)."""
    return 1 + (ox + 2) * 25 + (oy + 2) * 5 + (oz + 2)


@cython.final
cdef class World:
    cdef object non_air, wire, torch, repeater, solid, inverted, terminal, fp_relief, fp, pending
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
    # With a grid: reads go through it (see base(), owner(), eprot_at(), prot_at(), efp()).
    cdef CellGrid grid
    cdef uint16_t* gp
    cdef int gnx, gny, gnz, own_id
    cdef long long gsx, gsy
    cdef bint keep_fast, have_small
    cdef FlatMap[char] small         # S_* of the cells in this search's small sets
    cdef FlatMap[int] pid_map        # cell_key -> cell id, ahead of pos_ids
    cdef bint pids_seeded
    cdef bint grid_check
    # Set while _Cells.load runs: moves, -4 sides and new cells go straight into its vectors.
    cdef vector[Move]* out
    cdef vector[M4Side]* out_sides
    cdef vector[int]* out_m4_cells
    cdef _CellVectors* out_cells
    cdef bint have_extra             # a keepout has extra cells, so a keepout-free cell still needs small
    cdef int small_lo[3]             # bounding box of small's cells
    cdef int small_hi[3]

    def __init__(self, non_air, wire, torch, repeater, solid, dust_owner, net_id, bounds,
                 int max_bridge_y, goal, int x_h_factor, eprot, prot, fp_relief, fp, inverted, terminal,
                 workspace, glass, facing, dict pos_ids, list pos_list, list h_list, list move_cache,
                 pending=frozenset(), CellGrid grid=None, bint grid_check=False):
        for s in (non_air, wire, torch, repeater, solid, fp_relief, fp, inverted, terminal, pending):
            if not isinstance(s, (set, frozenset)):
                raise TypeError(f"World needs sets, got {type(s).__name__}")
        self.non_air, self.wire, self.torch, self.repeater = non_air, wire, torch, repeater
        self.solid, self.inverted, self.terminal, self.fp_relief, self.fp = solid, inverted, terminal, fp_relief, fp
        self.pending = pending
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
        self.gp = NULL
        self.small_lo = [1 << 30, 1 << 30, 1 << 30]
        self.small_hi = [-(1 << 30), -(1 << 30), -(1 << 30)]
        if grid is not None and (grid.non_air is non_air and grid.wire is wire and grid.torch is torch
                                 and grid.repeater is repeater and grid.solid is solid
                                 and grid.dust_owner is dust_owner and grid.inverted is inverted
                                 and grid.fp is fp and grid.terminal is terminal
                                 # A later router over the same containers took them over.
                                 and solid._grid is grid and dust_owner._grid is grid and inverted._grid is grid):
            self.use_grid(grid)
        self.grid_check = grid_check and self.gp != NULL

    cdef int use_grid(self, CellGrid grid) except -1:
        cdef int ins
        cdef char* b
        self.grid = grid
        self.gp = grid.g
        self.gnx, self.gny, self.gnz = grid.nx, grid.ny, grid.nz
        self.gsx, self.gsy = grid.sx, grid.sy
        self.own_id = grid.owner_id(self.net_id)
        # Keepouts: one count dict, or each cell reads the containers (keepout_has).
        self.keep_fast = (self.eprot[0] == "keepout" and self.prot[0] == "keepout"
                          and self.eprot[1] is self.prot[1] and self.eprot[2] is self.prot[2]
                          and self.eprot[5] is self.prot[5] and self.eprot[6] is self.prot[6])
        if self.keep_fast:
            grid.load_keepout(self.eprot[1])
            grid.load_own(self.eprot[2])
            for spec, shift in ((self.eprot, 0), (self.prot, 1)):
                self.add_small(spec[4], S_EXTRA << shift)
                if spec[4]:
                    self.have_extra = True
                self.add_small(spec[3], S_EXCL << shift)
        self.add_small(self.fp_relief, S_RELIEF)
        self.add_small(self.pending, S_PENDING)
        return 0

    cdef int add_small(self, cells, int bit) except -1:
        cdef int ins
        cdef char* b
        cdef int x, y, z
        for pos in cells:
            x, y, z = pos
            b = self.small.get(cell_key(x, y, z), &ins)
            b[0] |= bit
            self.have_small = True
            if x < self.small_lo[0]: self.small_lo[0] = x
            if y < self.small_lo[1]: self.small_lo[1] = y
            if z < self.small_lo[2]: self.small_lo[2] = z
            if x > self.small_hi[0]: self.small_hi[0] = x
            if y > self.small_hi[1]: self.small_hi[1] = y
            if z > self.small_hi[2]: self.small_hi[2] = z
        return 0

    # Reads. With a grid, a cell inside it needs no Python; everything else takes cell()'s path.

    cdef inline uint16_t* gw(self, int x, int y, int z) except? NULL:
        cdef unsigned int ux = <unsigned int>(x + GRID_MARGIN), uy = <unsigned int>(y + GRID_MARGIN)
        cdef unsigned int uz = <unsigned int>(z + GRID_MARGIN)
        if self.gp == NULL or ux >= <unsigned int>self.gnx or uy >= <unsigned int>self.gny or uz >= <unsigned int>self.gnz:
            return NULL
        cdef uint16_t* w = &self.gp[ux * self.gsx + uy * self.gsy + uz]
        if not w[0] & G_FP_READ:
            self.grid.read_fp(w, x, y, z)
        return w

    cdef inline unsigned int base(self, int x, int y, int z) except? 0xFFFFFFFF:
        """Flags of G_DIRECT only are meaningful."""
        cdef uint16_t* w = self.gw(x, y, z)
        if w != NULL:
            return w[0] & G_DIRECT
        return self.cell(x, y, z).val

    cdef inline int small_at(self, int x, int y, int z) noexcept:
        if (not self.have_small or x < self.small_lo[0] or x > self.small_hi[0] or y < self.small_lo[1]
                or y > self.small_hi[1] or z < self.small_lo[2] or z > self.small_hi[2]):
            return 0
        cdef char* b = self.small.find(cell_key(x, y, z))
        return 0 if b == NULL else b[0]

    cdef inline bint efp(self, int x, int y, int z) except -1:
        cdef uint16_t* w = self.gw(x, y, z)
        if w == NULL:
            return (self.cell(x, y, z).val & F_EFP) != 0
        return (w[0] & F_FP) != 0 and not self.small_at(x, y, z) & S_RELIEF

    cdef bint keep_slow(self, int x, int y, int z, int shift) except -1:
        """A keepout's answer for a cell its count dict has, after excluded: the halo."""
        cdef object spec = self.prot if shift else self.eprot
        key = (x, y, z)
        cdef object n = (<dict>spec[5]).get(key, 0)
        return n > 1 or (n == 1 and not PySet_Contains(spec[6], key))

    cdef inline bint keep_at(self, uint16_t g, int x, int y, int z, int shift) except -1:
        """_ForeignKeepout.__contains__, for a cell in the grid."""
        if not self.have_extra and not g & G_KEEP1:
            return False
        cdef int s = self.small_at(x, y, z)
        if s & (S_EXTRA << shift):
            return True
        if not g & G_KEEP1:
            return False
        if not g & G_KEEP2 and self.grid.own_map.find(cell_key(x, y, z)) != NULL:
            return False
        if not s & (S_EXCL << shift):
            return True
        return self.keep_slow(x, y, z, shift)

    cdef inline bint eprot_at(self, int x, int y, int z) except -1:
        cdef uint16_t* w = self.gw(x, y, z)
        if w != NULL and self.keep_fast:
            return self.keep_at(w[0], x, y, z, 0)
        return (self.cell(x, y, z).val & F_EPROT) != 0

    cdef inline bint prot_at(self, int x, int y, int z) except -1:
        cdef uint16_t* w = self.gw(x, y, z)
        if w != NULL and self.keep_fast:
            return self.keep_at(w[0], x, y, z, 1)
        return (self.cell(x, y, z).val & F_PROT) != 0

    cdef inline int owner_code(self, uint16_t g, int x, int y, int z) noexcept:
        if not g & G_OWNED:
            return 0
        cdef int o = self.grid.owner_map.find(cell_key(x, y, z))[0]
        return 1 if o == self.own_id else o + 1

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
        cdef uint16_t* w = self.gw(x, y, z)
        cdef unsigned int v = 0
        if w != NULL and self.keep_fast:
            # Only flag() memos here; everything else reads the grid.
            v = w[0] & G_DIRECT
            if self.eprot_at(x, y, z): v |= F_EPROT
            if self.prot_at(x, y, z): v |= F_PROT
            if self.efp(x, y, z): v |= F_EFP
            c.owner = self.owner_code(w[0], x, y, z)
            c.val = v
            c.known = BASE_FLAGS
            return 0
        key = (x, y, z)
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
        cdef CellInfo* c
        cdef unsigned int v
        if bit == F_TERM:
            return self.base(x, y, z) & F_TERM
        if bit == F_POWERED and not self.base(x, y, z) & F_SOLID:  # what powered() checks first
            return 0
        c = self.cell(x, y, z)
        if not c.known & bit:
            v = self.probe(x, y, z, bit)
            c = self.cell(x, y, z)  # probing POWERED touches other cells (may move entries)
            c.val |= v
            c.known |= bit
        return c.val & bit

    cdef inline int owner(self, int x, int y, int z) except -1:
        cdef uint16_t* w = self.gw(x, y, z)
        if w != NULL:
            return self.owner_code(w[0], x, y, z)
        return self.cell(x, y, z).owner

    cdef bint powered(self, int x, int y, int z) except -1:
        """_is_powered_block: a solid cell with a repeater facing into it."""
        cdef int d, bx, bz
        if not self.inb(x, y, z) or not (self.base(x, y, z) & F_SOLID):
            return False
        for d in range(4):
            bx, bz = x + HORIZ[d][0], z + HORIZ[d][1]
            if not self.inb(bx, y, bz):
                continue
            if not (self.base(bx, y, bz) & F_REP):
                continue
            if self.workspace[bx, y, bz]['facing'] == self.facing[d]:
                return True
        return False

    cdef inline bint can_support(self, int x, int y, int z) except -1:
        return self.inb(x, y, z) and ((self.base(x, y, z) & F_SOLID) != 0 or not (self.base(x, y, z) & F_NONAIR))

    cdef inline bint torch_below(self, int x, int y, int z) except -1:
        if not self.inb(x, y - 1, z):
            return False
        return (self.base(x, y - 1, z) & F_TORCH) != 0

    cdef bint wire_walkable(self, int x, int y, int z, int ex, int ey, int ez, bint has_exempt,
                            bint with_inv_fp) except -1:
        """_wire_walkable; with_inv_fp False = no footprint checks (mid_walkable).
        Only the boolean matters, so each neighbour cell is fetched once and all its checks run
        together (the Python version's check order differs; its result doesn't)."""
        cdef int d, dy, nx, ny, nz, vdy, vy
        cdef int o
        cdef unsigned int v
        if not self.inb(x, y, z):
            return False
        v = self.base(x, y, z)
        if v & F_SOLID:
            return False
        if v & F_NONAIR:
            if v & F_WIRE:
                if self.owner(x, y, z) >= 2:
                    return False
            else:
                return False
        for d in range(4):
            for dy in range(-1, 2):
                nx, ny, nz = x + HORIZ[d][0], y + dy, z + HORIZ[d][1]
                v = self.base(nx, ny, nz)
                if v & F_INV:
                    return False
                if has_exempt and nx == ex and ny == ey and nz == ez:
                    continue
                if not self.inb(nx, ny, nz):
                    continue
                o = self.owner(nx, ny, nz)
                if o >= 2:
                    return False
                if o == 0 and v & (F_TORCH | F_WIRE) and not (with_inv_fp and self.efp(nx, ny, nz)):
                    return False
        for vdy in (1, -1):
            vy = y + vdy
            if has_exempt and x == ex and vy == ey and z == ez:
                continue
            if not self.min_y <= vy <= self.max_y:
                continue
            if not (self.base(x + 1, vy, z) & F_REP or self.base(x - 1, vy, z) & F_REP
                    or self.base(x, vy, z + 1) & F_REP or self.base(x, vy, z - 1) & F_REP
                    or self.base(x, vy - 1, z) & F_TORCH):
                continue
            if self.owner(x, vy, z) >= 2:
                return False
        return True

    cdef inline bint walkable(self, int x, int y, int z) except -1:
        return self.walkable_ex(x, y, z, 0, 0, 0, False)

    cdef bint walkable_ex(self, int x, int y, int z, int ex, int ey, int ez, bint has_exempt) except -1:
        if self.eprot_at(x, y, z) or self.efp(x, y, z):
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
        cdef int ins
        cdef int* slot = self.pid_map.get(cell_key(x, y, z), &ins)
        if not ins:
            return slot[0]
        if self.out_cells == NULL:
            slot[0] = self.pid_slow(x, y, z)
            return slot[0]
        # Filling the search's vectors: the cell stays out of the Python containers unless flush_cells asks.
        slot[0] = <int>self.out_cells.x.size()
        self.out_cells.x.push_back(x)
        self.out_cells.y.push_back(y)
        self.out_cells.z.push_back(z)
        self.out_cells.h.push_back(abs(x - self.gx) * self.x_h_factor + abs(y - self.gy) * 4 + abs(z - self.gz))
        self.out_cells.cm.push_back(_NO_MOVES)
        return slot[0]

    cdef int pid_slow(self, int x, int y, int z) except -1:
        key = (x, y, z)
        i = self.pos_ids.get(key)
        if i is None:
            i = self.pos_ids[key] = len(self.pos_list)
            self.pos_list.append(key)
            self.h_list.append(abs(x - self.gx) * self.x_h_factor + abs(y - self.gy) * 4 + abs(z - self.gz))
            self.move_cache.append(None)
        return i

    cdef int seed_pids(self) except -1:
        """The cells Python has ids for, once, before the first cell's moves go in the vectors."""
        cdef int ins, x, y, z
        if self.pids_seeded:
            return 0
        self.pids_seeded = True
        for key, i in self.pos_ids.items():
            x, y, z = key
            self.pid_map.get(cell_key(x, y, z), &ins)[0] = i
        return 0

    cdef int emit(self, list moves, int x, int y, int z, int tx, int ty, int tz, int cost, unsigned int forbid,
                  int gate, bint has_extra, int ex, int ey, int ez) except -1:
        """_find_wire_path's _entry. Into the vector being filled, or moves as a tuple with its snapshot tables slot None."""
        cdef int dx = tx - x, dy = ty - y, dz = tz - z
        cdef Move m
        if self.out == NULL:
            flat_dir = (dx, dz) if dy == 0 and abs(dx) <= 1 and abs(dz) <= 1 else None
            dest = self.pid(tx, ty, tz)
            near = ((abs(self.gx - tx) + abs(self.gz - tz)) < HORIZ_RADIUS_MIN
                    and abs(self.gy - ty) < Y_RADIUS_MIN)
            arrive = TOWER_FLAG if dy == 4 and ((abs(dx) == 2 and dz == 0) or (dx == 0 and abs(dz) == 2)) else 0
            moves.append((dest, cost, forbid, gate, (ex, ey, ez) if has_extra else None, None, flat_dir, self.h_list[dest],
                          near, arrive))
            return 0
        m.dest = self.pid(tx, ty, tz)
        m.cost = cost
        m.forbid = forbid
        m.gate = gate
        m.has_extra = has_extra
        if has_extra:
            m.ex, m.ey, m.ez = ex - x, ey - y, ez - z
        else:
            m.ex = m.ey = m.ez = 0
        m.dx, m.dy, m.dz = dx, dy, dz
        m.shift = shift_index(dx, dy, dz)
        m.has_flat = dy == 0 and abs(dx) <= 1 and abs(dz) <= 1
        m.fdx = dx if m.has_flat else 0
        m.fdz = dz if m.has_flat else 0
        m.h = abs(tx - self.gx) * self.x_h_factor + abs(ty - self.gy) * 4 + abs(tz - self.gz)
        m.near = ((abs(self.gx - tx) + abs(self.gz - tz)) < HORIZ_RADIUS_MIN
                  and abs(self.gy - ty) < Y_RADIUS_MIN)
        m.arrive = dy == 4 and ((abs(dx) == 2 and dz == 0) or (dx == 0 and abs(dz) == 2))
        m.cong = 0.0
        self.out.push_back(m)
        return 0

    cpdef tuple static_moves(self, int cid):
        x, y, z = self.pos_list[cid]
        if not self.grid_check:
            return self._static_moves(x, y, z)
        got = self._static_moves(x, y, z)
        # The same cell from the containers alone (caches off for it: they were filled from the grid).
        cdef uint16_t* gp = self.gp
        self.gp = NULL
        self.walk_cache.reset(1 << 12)
        self.mid_cache.reset(1 << 12)
        self.info.reset(1 << 12)
        want = self._static_moves(x, y, z)
        self.gp = gp
        self.walk_cache.reset(1 << 12)
        self.mid_cache.reset(1 << 12)
        self.info.reset(1 << 12)
        if got != want:
            raise AssertionError(f"CellGrid moves differ at {(x, y, z)} for net {self.net_id}:\n"
                                 f"  grid {got}\n  containers {want}")
        return got

    cdef tuple _static_moves(self, int x, int y, int z):
        cdef list moves = []
        if self.base(x, y, z) & F_INV:
            return [], None
        self.plain_moves(x, y, z, moves)
        return moves, self.powered_minus4(x, y, z)

    cdef int plain_moves(self, int x, int y, int z, list moves) except -1:
        """Every move but powered -4, in moves or the vector being filled. The cell isn't inverted."""
        cdef int d, dx, dz, nx, nz, nx2, nz2, mx, my, mz, k, below_owner
        cdef unsigned int b_lo, b_mid, b_hi
        cdef bint has_ex, above_clear, breaks_slope, mid_ok, mid_dn_ok
        cdef int min_y = self.min_y, mby = self.max_bridge_y

        for d in range(4):
            dx, dz = HORIZ[d][0], HORIZ[d][1]
            nx, nz = x + dx, z + dz
            b_lo, b_mid, b_hi = snap_bit(dx, -1, dz), snap_bit(dx, 0, dz), snap_bit(dx, 1, dz)

            # Flat move
            if self.walkable(nx, y, nz) and not self.flag(nx, y + 1, nz, F_POWERED):
                if y <= min_y:
                    self.emit(moves, x, y, z, nx, y, nz, 1, b_hi | b_mid, 0, False, 0, 0, 0)
                elif self.can_support(nx, y - 1, nz) and not self.torch_below(nx, y - 1, nz):
                    self.emit(moves, x, y, z, nx, y, nz, 1, b_hi | b_mid | b_lo, 0, False, 0, 0, 0)

            # Slope-up
            if y < mby:
                if (not (self.base(x, y + 1, z) & F_NONAIR) and self.walkable(nx, y + 1, nz)
                        and not self.flag(nx, y + 2, nz, F_POWERED)):
                    if (((self.base(nx, y, nz) & F_SOLID) or (self.inb(nx, y, nz) and not (self.base(nx, y, nz) & F_NONAIR)))
                            and not self.torch_below(nx, y, nz)):
                        self.emit(moves, x, y, z, nx, y + 1, nz, 3, b_hi | b_mid, 1, False, 0, 0, 0)

            # Slope-down
            if y > min_y:
                above_clear = (
                    not (self.base(nx, y, nz) & F_SOLID)
                    and self.inb(nx, y, nz)
                    and not (self.base(nx, y, nz) & F_NONAIR)
                    and not self.flag(nx, y, nz, F_POWERED)
                )
                has_ex = self.can_support(x, y - 1, z)
                if above_clear and self.walkable_ex(nx, y - 1, nz, x, y - 2, z, has_ex):
                    if y - 1 <= min_y or (self.can_support(nx, y - 2, nz) and not self.torch_below(nx, y - 2, nz)):
                        below_owner = self.owner(nx, y - 3, nz)
                        breaks_slope = False
                        if below_owner >= 2:
                            for k in range(4):
                                if self.owner(nx + HORIZ[k][0], y - 2, nz + HORIZ[k][1]) == below_owner:
                                    breaks_slope = True
                                    break
                        if not breaks_slope:
                            self.emit(moves, x, y, z, nx, y - 1, nz, 2, b_lo | b_mid | b_hi, 0, False, 0, 0, 0)

        for d in range(4):
            dx, dz = HORIZ[d][0], HORIZ[d][1]
            nx2, nz2 = x + 2 * dx, z + 2 * dz
            b_lo, b_mid, b_hi = snap_bit(dx, -1, dz), snap_bit(dx, 0, dz), snap_bit(dx, 1, dz)

            # 2x slope-up
            if y + 2 <= mby:
                mx, my, mz = x + dx, y + 1, z + dz
                mid_ok = (
                    not (self.base(x, y + 1, z) & F_NONAIR)
                    and not self.prot_at(mx, my, mz)
                    and not (self.base(mx, my, mz) & F_FP)
                    and self.inb(mx, my, mz)
                    and not (self.base(mx, y + 2, mz) & F_SOLID)
                    and not (self.base(mx, my, mz) & F_SOLID)
                    and not (self.base(mx, my, mz) & F_NONAIR)
                    and self.mid_walkable(mx, my, mz, nx2, y, nz2)
                )
                if mid_ok and self.walkable(nx2, y + 2, nz2):
                    if self.can_support(mx, y, mz) and not self.torch_below(mx, y, mz):
                        self.emit(moves, x, y, z, nx2, y + 2, nz2, 7, b_hi | b_mid, 1, True, mx, my, mz)

            # 2x slope-down
            if y - 2 >= min_y:
                mx, my, mz = x + dx, y - 1, z + dz
                above_clear = (
                    not (self.base(mx, y, mz) & F_SOLID)
                    and self.inb(mx, y, mz)
                    and not (self.base(mx, y, mz) & F_NONAIR)
                )
                # The mid -> dest dust step also needs the cell above dest open.
                mid_dn_ok = (
                    above_clear
                    and self.inb(nx2, y - 1, nz2)
                    and not (self.base(nx2, y - 1, nz2) & F_NONAIR)
                    and not self.prot_at(mx, my, mz)
                    and not (self.base(mx, my, mz) & F_FP)
                    and self.inb(mx, my, mz)
                    and not (self.base(mx, my, mz) & F_SOLID)
                    and not (self.base(mx, my, mz) & F_NONAIR)
                    and self.mid_walkable(mx, my, mz, x, y - 2, z)
                )
                if mid_dn_ok and self.walkable(nx2, y - 2, nz2):
                    if y - 2 <= min_y or (self.can_support(nx2, y - 3, nz2) and not self.torch_below(nx2, y - 3, nz2)):
                        self.emit(moves, x, y, z, nx2, y - 2, nz2, 7, b_lo | b_mid | b_hi, 0, True, mx, my, mz)

        # +1Y from tower top (only right after a tower move)
        if y + 1 <= mby and self.walkable(x, y + 1, z):
            self.emit(moves, x, y, z, x, y + 1, z, 1, 0, 2, False, 0, 0, 0)

        # Tower-up
        self.tower_moves(x, y, z, moves)
        return 0

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
            if not self.inb(rx, y, rz) or (self.base(rx, y, rz) & F_FP) or self.prot_at(rx, y, rz):
                continue
            if (self.base(rx, y, rz) & F_NONAIR):
                continue
            if not self.can_support(rx, y - 1, rz):
                continue
            cuts = False
            if self.inb(rx, y - 2, rz) and (self.base(rx, y - 2, rz) & F_WIRE):
                for k in range(4):
                    if (self.inb(rx + HORIZ[k][0], y - 1, rz + HORIZ[k][1])
                            and (self.base(rx + HORIZ[k][0], y - 1, rz + HORIZ[k][1]) & F_WIRE)):
                        cuts = True
                        break
            if cuts:
                continue
            col_clear = True
            for cdy in range(5):
                cy = y + cdy
                if not self.inb(cx, cy, cz) or (self.base(cx, cy, cz) & F_FP) or self.prot_at(cx, cy, cz):
                    col_clear = False
                    break
                if (self.base(cx, cy, cz) & F_NONAIR):
                    col_clear = False
                    break
                for k in range(4):
                    if (self.inb(cx + HORIZ[k][0], cy, cz + HORIZ[k][1])
                            and (self.base(cx + HORIZ[k][0], cy, cz + HORIZ[k][1]) & F_WIRE)):
                        col_clear = False
                        break
                if not col_clear:
                    break
            if not col_clear:
                continue
            if self.next_to_terminal(cx, y + 1, cz) or self.next_to_terminal(cx, y + 2, cz):
                continue
            if self.inb(cx, y - 1, cz):
                if self.owner(cx, y - 1, cz) >= 2:
                    continue
                if (self.base(cx, y - 1, cz) & F_WIRE):
                    continue
            self.emit(moves, x, y, z, cx, y + 4, cz, 8, snap_bit(tdx, 0, tdz) | snap_bit(tdx, -1, tdz) | snap_bit(tdx, 1, tdz),
                      0, False, 0, 0, 0)
        return 0

    cdef bint next_to_terminal(self, int x, int y, int z) except -1:
        """Whether an inverted cell here would make a pin terminal unwalkable (wire_walkable
        rejects cells beside inverted ones); own-net terminals of later sinks aren't keepout.
        Sinks this net has yet to reach also need their approach cells walkable, and those are
        one step further out, so an inverted cell can wall a pin off from a diagonal."""
        cdef int k, ddy
        for k in range(4):
            for ddy in range(-1, 2):
                if self.flag(x + HORIZ[k][0], y + ddy, z + HORIZ[k][1], F_TERM):
                    return True
        if not self.pending:
            return False
        if self.gp != NULL:
            for ddy in range(-2, 3):
                if self.small_at(x, y + ddy, z) & S_PENDING:
                    return True
                for k in range(4):
                    if (self.small_at(x + 2 * HORIZ[k][0], y + ddy, z + 2 * HORIZ[k][1]) & S_PENDING
                            or self.small_at(x + HORIZ[k][0] + HORIZ[k][1], y + ddy,
                                             z + HORIZ[k][1] - HORIZ[k][0]) & S_PENDING):
                        return True
            return False
        for ddy in range(-2, 3):
            if PySet_Contains(self.pending, (x, y + ddy, z)):
                return True
            for k in range(4):
                if (PySet_Contains(self.pending, (x + 2 * HORIZ[k][0], y + ddy, z + 2 * HORIZ[k][1]))
                        or PySet_Contains(self.pending, (x + HORIZ[k][0] + HORIZ[k][1], y + ddy,
                                                         z + HORIZ[k][1] - HORIZ[k][0]))):
                    return True
        return False

    cdef bint m4_cell_clear(self, int x, int y, int z) except -1:
        cdef int k, ddy, sx, sy, sz
        for k in range(4):
            for ddy in range(-1, 2):
                sx, sy, sz = x + HORIZ[k][0], y + ddy, z + HORIZ[k][1]
                if self.prot_at(sx, sy, sz) or self.owner(sx, sy, sz) >= 2:
                    return False
        return self.owner(x, y, z) < 2

    cdef object powered_minus4(self, int x, int y, int z):
        """_powered_minus4_move: its entry and sides. Filling vectors: the sides, then the entry, returning None."""
        cdef int d, dx, dz, nx, nz, k, ddy, i, bx, by, bz, cx, cy, cz
        cdef int cells[6][3]
        cdef unsigned int b_lo, b_mid, b_hi
        cdef bint ok
        cdef int n_sides = 0
        cdef M4Side m4
        cdef list moves = None
        if y - 4 < self.min_y:
            return None
        if not self.walkable(x, y - 4, z):
            return None
        bx = x
        by = y - 1
        bz = z
        if ((bx == self.gx and by == self.gy and bz == self.gz) or (self.base(bx, by, bz) & F_TERM)
                or (self.base(bx, by, bz) & F_FP) or self.prot_at(bx, by, bz)):
            return None
        if (self.base(bx, by, bz) & F_SOLID):
            if self.flag(bx, by, bz, F_GLASS):
                return None
        elif not (self.inb(bx, by, bz) and not (self.base(bx, by, bz) & F_NONAIR)):
            return None
        sides = [] if self.out == NULL else None
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
                        or (self.base(cx, cy, cz) & F_TERM) or (self.base(cx, cy, cz) & F_FP)
                        or self.prot_at(cx, cy, cz) or (self.base(cx, cy, cz) & F_SOLID)
                        or (self.base(cx, cy, cz) & F_NONAIR)):
                    ok = False
                    break
            if ok:
                for i in range(6):
                    if not self.m4_cell_clear(cells[i][0], cells[i][1], cells[i][2]):
                        ok = False
                        break
            if ok and (self.next_to_terminal(cells[0][0], cells[0][1], cells[0][2])
                       or self.next_to_terminal(cells[1][0], cells[1][1], cells[1][2])):
                ok = False
            if not ok:
                continue
            for k in range(4):
                for ddy in (-1, 1):
                    if self.owner(nx + HORIZ[k][0], y - 2 + ddy, nz + HORIZ[k][1]) != 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            n_sides += 1
            if self.out != NULL:
                m4.forbid = b_lo | b_mid | b_hi
                m4.start = self.out_m4_cells.size()
                m4.count = 7
                self.out_sides.push_back(m4)
                self.out_m4_cells.push_back(bx - x)
                self.out_m4_cells.push_back(by - y)
                self.out_m4_cells.push_back(bz - z)
                for i in range(6):
                    self.out_m4_cells.push_back(cells[i][0] - x)
                    self.out_m4_cells.push_back(cells[i][1] - y)
                    self.out_m4_cells.push_back(cells[i][2] - z)
            else:
                sides.append((b_lo | b_mid | b_hi, ((bx, by, bz),) + tuple(
                    [(cells[i][0], cells[i][1], cells[i][2]) for i in range(6)])))
        if not n_sides:
            return None
        if self.out == NULL:
            moves = []
        self.emit(moves, x, y, z, x, y - 4, z, 8, 0, 0, False, 0, 0, 0)
        return (moves[0], sides) if self.out == NULL else None


def _strip_tables(static):
    """A static-moves result with each entry's snapshot-table slot blanked, for comparison."""
    moves, m4 = static
    strip = lambda e: e[:5] + (None,) + e[6:]
    return [strip(e) for e in moves], (None if m4 is None else (strip(m4[0]), m4[1]))


@cython.final
cdef class _Cells:
    """Coordinates, heuristic and converted static moves per cell id, mirrored from Python."""
    cdef vector[int] x, y, z
    cdef vector[long long] h
    cdef vector[CellMoves] cm
    cdef vector[Move] moves
    cdef vector[M4Side] m4_sides
    cdef vector[int] m4_cells  # flattened (ox, oy, oz) triples

    cdef void clear(self):
        self.x.clear()
        self.y.clear()
        self.z.clear()
        self.h.clear()
        self.cm.clear()
        self.moves.clear()
        self.m4_sides.clear()
        self.m4_cells.clear()
        if self.moves.capacity() > (1u << 22):  # a huge search: not kept for the next ones
            vector[Move]().swap(self.moves)

    cdef int flush_cells(self, dict pos_ids, list pos_list, list h_list, list move_cache) except -1:
        """Cells only the vectors have, into the Python containers: a failed search's report and its retry read them."""
        cdef Py_ssize_t i
        cdef int x, y, z
        for i in range(len(pos_list), <Py_ssize_t>self.x.size()):
            x, y, z = self.x[i], self.y[i], self.z[i]
            key = (x, y, z)
            pos_ids[key] = i
            pos_list.append(key)
            h_list.append(self.h[i])
            move_cache.append(None)
        return 0

    cdef void sync(self, list pos_list, list h_list):
        cdef Py_ssize_t i, n = len(pos_list)
        for i in range(<Py_ssize_t>self.x.size(), n):
            p = pos_list[i]
            self.x.push_back(p[0])
            self.y.push_back(p[1])
            self.z.push_back(p[2])
            self.h.push_back(h_list[i])
            self.cm.push_back(_NO_MOVES)

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
        m.arrive = entry[9] != 0
        m.cong = 0.0
        if congestion_cost is not None:
            m.cong = congestion_cost(pos_list[m.dest])
        self.moves.push_back(m)
        return 0

    cdef int add_m4_sides(self, int cid, list sides) except -1:
        cdef M4Side side
        for forbid, cells in sides:
            side.forbid = forbid
            side.start = self.m4_cells.size()
            side.count = len(cells)
            for cell in cells:
                self.m4_cells.push_back(cell[0] - self.x[cid])
                self.m4_cells.push_back(cell[1] - self.y[cid])
                self.m4_cells.push_back(cell[2] - self.z[cid])
            self.m4_sides.push_back(side)
        return 0

    cdef int load(self, int cid, object expand, World world, bint verify, list pos_list, list h_list,
                  object congestion_cost) except -1:
        cdef CellMoves c
        cdef _CellVectors vectors
        cdef int cx, cy, cz
        if world is not None and not verify and not world.grid_check and congestion_cost is None:
            # The kernel's move generation without going through Python tuples.
            world.seed_pids()
            c.start = self.moves.size()
            c.m4_start = self.m4_sides.size()
            c.m4_move = -1
            c.m4_count = 0
            cx, cy, cz = self.x[cid], self.y[cid], self.z[cid]
            world.out = &self.moves
            world.out_sides = &self.m4_sides
            world.out_m4_cells = &self.m4_cells
            vectors.x, vectors.y, vectors.z, vectors.h, vectors.cm = &self.x, &self.y, &self.z, &self.h, &self.cm
            world.out_cells = &vectors
            try:
                if not world.base(cx, cy, cz) & F_INV:
                    world.plain_moves(cx, cy, cz, None)
                    c.count = <int>self.moves.size() - c.start
                    world.powered_minus4(cx, cy, cz)
                    c.m4_count = <int>self.m4_sides.size() - c.m4_start
                    if c.m4_count:
                        c.m4_move = <int>self.moves.size() - 1
                else:
                    c.count = 0
            finally:
                world.out = NULL
                world.out_sides = NULL
                world.out_m4_cells = NULL
                world.out_cells = NULL
            self.cm[cid] = c
            return 0
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
            self.add_m4_sides(cid, m4[1])
            c.m4_count = <int>self.m4_sides.size() - c.m4_start
        self.cm[cid] = c
        return 0


cdef _Cells SEARCH_CELLS = _Cells()  # a search's storage, kept for the next one
cdef Heap SEARCH_HEAP


cdef extern from *:
    "#define count_trailing_zeros(x) __builtin_ctz(x)"
    int count_trailing_zeros(unsigned int x) nogil


def cell_pos(int cid):
    """A cell of the last search, which may be past pos_list: good until the next search starts."""
    cdef int x = SEARCH_CELLS.x[cid], y = SEARCH_CELLS.y[cid], z = SEARCH_CELLS.z[cid]
    return (x, y, z)


def set_foot_repeaters(bint on) -> None:
    global FOOT_REPEATERS
    FOOT_REPEATERS = on


def search(list seed_nodes, list seed_groups, goal_ids, list group_best_h, list group_max_f,
           long long best_node, long long best_h, double weight, time_limit, list pos_list,
           list h_list, expand, congestion_cost, World world=None, bint verify=False,
           double h_radius_factor=-1, int stagnation=-1, int budget=1):
    """See pathfinding._search_py. With world, cells' static moves are generated here instead of
    by expand(); verify additionally checks each against expand(). stagnation: this search's limit
    (negative: _ROUTE_STAGNATION). h_radius_factor: the h-radius
    cull for this search (0: off; negative: _H_RADIUS_FACTOR). budget: multiplies the stagnation,
    position and expansion limits."""
    global STAT_SEARCHES, STAT_POPS, STAT_PUSHES, STAT_CELLS, STAT_LOAD_TIME
    if not _init_done:
        raise RuntimeError("_astar.init() was not called")
    cdef _Cells cells = SEARCH_CELLS
    cells.clear()
    cdef Heap* heap = &SEARCH_HEAP
    heap.clear()
    heap.g_can_differ = congestion_cost is not None or (weight * 4) % 1 != 0
    cdef CameFrom came_from = CameFrom()
    cdef NodeTable* nodes = &came_from.nodes
    # Most searches outgrow the default, and growing rehashes.
    nodes.reset(1 << 14, <unsigned long long>15 << SIG_SHIFT, congestion_cost is not None)
    cdef vector[long long] gbest = group_best_h
    cdef vector[double] gmax = group_max_f
    cdef vector[char] explored_xyz
    cdef vector[char] is_goal
    cdef vector[int] taken
    cdef vector[int] extras  # (ex, ey, ez) triples of this expansion's taken 2x slopes
    cdef HeapEntry e, ne
    cdef NodeSlot* np
    cdef int skip
    cdef NodeSlot* ni
    cdef Move* m
    cdef CellMoves* cmv
    cdef M4Side* side
    cdef long long counter = 0, h, node, nkey, cur_snap, h_radius_pruned = 0
    cdef long long n_explored_xyz = 0, pops = 0, n_exp = 0, n_reexp = 0, last_new_pos = 0, max_gap = 0
    cdef int cid, i, k, last, m4_side = -1
    cdef int in_dx = 0, in_dz = 0, nx, ny, nz
    cdef int cur_depth, cur_cand, ndepth, ncand, step
    cdef bint has_in_dir, came_from_tower, has_reached = False
    cdef long long reached = 0
    cdef double max_f, f_wire, h_limit, new_g, new_g_wire, cost, t_load
    cdef int next_move[MAX_CELL_MOVES]
    cdef long long next_key[MAX_CELL_MOVES]
    cdef double next_g[MAX_CELL_MOVES]
    cdef double next_g_wire[MAX_CELL_MOVES]
    cdef int n_next
    cdef uint32_t snap, cb0, cb1, cb2
    cdef bint has_deadline = time_limit is not None
    cdef double deadline = 0
    early_stop_reason = None
    cdef double radius = H_RADIUS_FACTOR if h_radius_factor < 0 else h_radius_factor
    cdef long long stagnation_limit = budget * (STAGNATION if stagnation < 0 else stagnation)
    cdef long long max_nodes = budget * MAX_NODES, max_expansions = budget * MAX_EXPANSIONS
    py_max_f = max(group_max_f)
    max_f = py_max_f

    cells.sync(pos_list, h_list)
    for g in goal_ids:
        if <Py_ssize_t>is_goal.size() <= g:
            is_goal.resize(g + 1, 0)
        is_goal[g] = 1

    for i in range(len(seed_nodes)):
        node = seed_nodes[i]
        np = nodes.find(node)
        if np == NULL:
            np = nodes.insert(node)
        nodes.set_first_g(np, 0)
        h = cells.h[node >> NODE_SHIFT]
        e.f = weight * h
        e.neg_h = -h
        e.g = 0
        e.counter = counter
        e.g_wire = 0
        e.node = node
        e.parent = -1
        e.group = seed_groups[i]
        heap.push(e)
        counter += 1

    if has_deadline:
        deadline = time.monotonic() + time_limit
    while not heap.empty():
        pops += 1
        if has_deadline and pops & 255 == 0 and time.monotonic() > deadline:
            early_stop_reason = f"timeout ({time_limit}s)"
            break
        if pops & 16383 == 0:
            pulse(pops)
        e = heap.pop()
        # f-cap per seed group; stop once past every group's.
        f_wire = e.f - e.g + e.g_wire
        if f_wire > gmax[e.group]:
            if f_wire > max_f:
                early_stop_reason = f"f-cap exceeded (f={f_wire} > max_f={py_max_f})"
                break
            continue
        ni = nodes.find(e.node)
        if e.g > nodes.g_of(ni):
            continue
        # Dominance: the same state with less signal depth at no more cost reaches everything
        # this one can (it is expanded first, or was dominated in turn).
        if ni.parent == NOT_EXPANDED and nodes.dominated(e.node, e.g):
            continue
        cid = <int>(e.node >> NODE_SHIFT)
        h = cells.h[cid]
        if h < best_h:
            best_h = h
            best_node = e.node
        if h < gbest[e.group]:
            gbest[e.group] = h
        n_exp += 1
        if ni.parent != NOT_EXPANDED:
            n_reexp += 1
        else:
            nodes.set_parent(ni, e.parent)
            if <Py_ssize_t>explored_xyz.size() <= cid:
                explored_xyz.resize(cid + 1, 0)
            if not explored_xyz[cid]:
                explored_xyz[cid] = 1
                n_explored_xyz += 1
                if n_exp - last_new_pos > max_gap:
                    max_gap = n_exp - last_new_pos
                last_new_pos = n_exp
            if n_explored_xyz >= max_nodes:
                early_stop_reason = f"node cap ({max_nodes} unique positions)"
                break
        if cid < <int>is_goal.size() and is_goal[cid]:
            reached = e.node
            has_reached = True
            break
        # Stagnation: the frontier keeps churning states (snapshots, signal depths, cheaper
        # re-expansions) over cells it has already reached.
        if n_exp - last_new_pos >= stagnation_limit:
            early_stop_reason = f"stagnated ({stagnation_limit} expansions without a new position)"
            break
        if n_exp >= max_expansions:
            early_stop_reason = f"expansion cap ({max_expansions})"
            break

        cur_snap = e.node & SNAP_MASK
        came_from_tower = (e.node & TOWER_FLAG) != 0
        cur_depth = (e.node >> SIG_SHIFT) & 15
        cur_cand = (e.node >> (SIG_SHIFT + 4)) & 7
        if cells.cm[cid].start < 0:
            t_load = now_seconds()
            cells.load(cid, expand, world, verify, pos_list, h_list, congestion_cost)
            STAT_LOAD_TIME += now_seconds() - t_load
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
        h_limit = gbest[e.group] * radius if radius > 0 else 1e18
        if m4_side >= 0:
            taken.push_back(cmv.m4_move)
        last = <int>taken.size() - 1
        n_next = 0
        for k in range(<int>taken.size()):
            m = &cells.moves[taken[k]]
            # Signal state (pathfinding._signal_step): prune moves that would leave a stretch
            # with no cell _place_repeaters_for_net could put a repeater on.
            step = abs(<int>m.dx) + abs(<int>m.dz)  # ints: abs of a char goes through Python
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
                if step == 1 and (m.dy == 0 or (FOOT_REPEATERS and m.dy == -1)):
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
            nkey = ((<long long>m.dest << NODE_SHIFT) | (TOWER_FLAG if m.arrive else 0) | snap
                    | (<long long>(ndepth | (ncand << 4)) << SIG_SHIFT))
            next_move[n_next] = taken[k]
            next_key[n_next] = nkey
            next_g[n_next] = new_g
            next_g_wire[n_next] = new_g_wire
            n_next += 1
            nodes.prefetch(nkey)
        # The table is big and these are scattered: asking for them all first saves waiting for each in turn.
        for k in range(n_next):
            m = &cells.moves[next_move[k]]
            nkey, new_g, new_g_wire = next_key[k], next_g[k], next_g_wire[k]
            np = nodes.probe(nkey, new_g, &skip)
            if skip:
                continue
            if np == NULL or np.parent == UNREACHED:
                if np == NULL:
                    np = nodes.insert(nkey)
                if not new_g < 1e9:  # stays looking like nothing was ever here
                    nodes.set_unreached(np)
                    continue
                nodes.set_first_g(np, new_g)
            else:
                nodes.set_g(np, new_g)
            if m.h <= h_limit or m.near:
                ne.f = new_g + weight * m.h
                ne.neg_h = -m.h
                ne.g = new_g
                ne.counter = counter
                ne.g_wire = new_g_wire
                ne.node = nkey
                ne.parent = e.node
                ne.group = e.group
                heap.push(ne)
                counter += 1
                STAT_PUSHES += 1
            else:
                h_radius_pruned += 1

    if not has_reached and world is not None:
        cells.flush_cells(world.pos_ids, pos_list, h_list, world.move_cache)
    STAT_SEARCHES += 1
    STAT_POPS += pops
    LAST.update(pops=pops, exp=n_exp, reexp=n_reexp, pos=n_explored_xyz, max_gap=max_gap, ok=has_reached,
                pruned=h_radius_pruned, seeds=len(seed_nodes))
    for i in range(<int>gbest.size()):
        group_best_h[i] = gbest[i]
    return (reached if has_reached else None), best_node, h_radius_pruned, early_stop_reason, came_from


# ---------------------------------------------------------------------------------------------
# The checks route_job runs on each finished net, compiled: verify._latch_cycle,
# verify._inverted_contacts and verify.find_gate_contacts over the same powers relation as
# verify._World, read from the region's array. All they say is whether a check has something to
# report. Next to no net does, and for those verify works out what, so what they report stays
# verify's: change the rules there and here together (MINECRAFT_V_NET_CHECKS=check compares them).

cdef enum:  # verify's _AIR, _DUST, ...
    K_AIR, K_DUST, K_REPEATER, K_TORCH, K_WALL_TORCH, K_OPAQUE, K_OTHER

cdef enum:
    MODE_PLAIN, MODE_STRONG, MODE_WEAK
    MAX_SUCCESSORS = 32

FAULT_LATCH, FAULT_INVERTED, FAULT_GATE = 1, 2, 4


cdef struct BlockKind:
    int kind, dx, dz      # dx, dz: _FACING_TO_DELTA of a repeater or wall torch
    bint redstone_block


cdef struct CheckNode:
    int x, y, z, mode


cdef struct TarjanFrame:
    int node, edge


@cython.final
cdef class NetChecks:
    cdef CellGrid grid
    cdef object region, palette, blocks_obj, classify
    cdef const uint32_t* blocks
    cdef int nx, ny, nz
    cdef long long sx, sy
    cdef vector[BlockKind] kinds
    cdef BlockKind air
    cdef bint hiding               # footprint cells read as air, unless terminals or in own
    cdef FlatMap[char] own         # this net's written cells
    cdef FlatMap[char] stop_own    # terminals the signal is followed through
    cdef FlatMap[char] gate
    cdef bint has_gate
    cdef FlatMap[int] ids
    cdef FlatMap[char] seen, polarity
    cdef vector[CheckNode] nodes, feeds
    cdef vector[char] flips
    cdef vector[int] offsets, heads, index, low, scc_stack
    cdef vector[char] on_stack
    cdef vector[TarjanFrame] frames
    cdef vector[int] own_cells
    cdef CheckNode out[MAX_SUCCESSORS]
    cdef int n_out

    def __init__(self, region, CellGrid grid, gate_cells, classify):
        """classify(block): verify's (kind, dx, dz, is redstone block) of a palette entry."""
        cdef int ins
        self.region, self.grid, self.classify = region, grid, classify
        self.palette = region._Region__palette
        self.air.kind = K_AIR
        self.air.dx = self.air.dz = 0
        self.air.redstone_block = False
        self.has_gate = bool(gate_cells)
        for pos in gate_cells:
            self.gate.get(cell_key(pos[0], pos[1], pos[2]), &ins)[0] = 1

    @staticmethod
    def usable(region):
        """Whether the region is the array this reads directly; views aren't."""
        blocks = getattr(region, "_Region__blocks", None)
        return blocks is not None and blocks.dtype == np.uint32 and blocks.flags.c_contiguous

    cdef int bind(self) except -1:
        blocks = self.region._Region__blocks
        if blocks is not self.blocks_obj:
            self.blocks_obj = blocks
            self.nx, self.ny, self.nz = blocks.shape
            self.sx, self.sy = <long long>self.ny * self.nz, self.nz
            self.blocks = <const uint32_t*><size_t>blocks.ctypes.data
        cdef BlockKind k
        for i in range(<Py_ssize_t>self.kinds.size(), len(self.palette)):
            k.kind, k.dx, k.dz, k.redstone_block = self.classify(self.palette[i])
            self.kinds.push_back(k)
        return 0

    cdef inline const BlockKind* at(self, int x, int y, int z) except NULL:
        if <unsigned int>x >= <unsigned int>self.nx or <unsigned int>y >= <unsigned int>self.ny or <unsigned int>z >= <unsigned int>self.nz:
            return &self.air
        cdef const BlockKind* k = &self.kinds[self.blocks[x * self.sx + y * self.sy + z]]
        cdef uint16_t* w
        if self.hiding and k.kind != K_AIR:
            w = self.grid.word(x, y, z)
            if not w[0] & G_FP_READ:
                self.grid.read_fp(w, x, y, z)
            if w[0] & F_FP and not w[0] & F_TERM and self.own.find(cell_key(x, y, z)) == NULL:
                return &self.air
        return k

    cdef inline int kind(self, int x, int y, int z) except -1:
        return self.at(x, y, z).kind

    cdef inline void add(self, int x, int y, int z, int mode) noexcept:
        cdef CheckNode* n = &self.out[self.n_out]
        n.x, n.y, n.z, n.mode = x, y, z, mode
        self.n_out += 1

    cdef int dust_links(self, int x, int y, int z) except -1:
        """Adds the dust x, y, z is wired to; the directions they are in, as bits of HORIZ."""
        cdef int d, sx, sz, side, connected = 0
        cdef bint above_open = self.kind(x, y + 1, z) != K_OPAQUE
        for d in range(4):
            sx, sz = x + HORIZ[d][0], z + HORIZ[d][1]
            side = self.kind(sx, y, sz)
            if side == K_DUST:
                self.add(sx, y, sz, MODE_PLAIN)
                connected |= 1 << d
                continue
            if above_open and self.kind(sx, y + 1, sz) == K_DUST:
                self.add(sx, y + 1, sz, MODE_PLAIN)
                connected |= 1 << d
            if side != K_OPAQUE and self.kind(sx, y - 1, sz) == K_DUST:
                self.add(sx, y - 1, sz, MODE_PLAIN)
                connected |= 1 << d
        return connected

    cdef int dust_directions(self, int x, int y, int z, int connected) except -1:
        """_World.dust_directions, from dust_links' answer."""
        cdef int d, e, sx, sz
        cdef const BlockKind* k
        for d in range(4):
            sx, sz = x + HORIZ[d][0], z + HORIZ[d][1]
            k = self.at(sx, y, sz)
            if k.kind == K_REPEATER:
                if (sx + k.dx == x and sz + k.dz == z) or (sx - k.dx == x and sz - k.dz == z):
                    connected |= 1 << d
            elif k.kind == K_TORCH or k.kind == K_WALL_TORCH or (k.kind == K_OPAQUE and k.redstone_block):
                connected |= 1 << d
        if not connected:
            return 15
        if connected & (connected - 1) == 0:
            d = count_trailing_zeros(connected)
            for e in range(4):
                if HORIZ[e][0] == -HORIZ[d][0] and HORIZ[e][1] == -HORIZ[d][1]:
                    connected |= 1 << e
        return connected

    cdef int successors(self, int x, int y, int z, int mode) except -1:
        """_World._find_successors, into out."""
        self.n_out = 0
        cdef const BlockKind* here = self.at(x, y, z)
        cdef const BlockKind* other
        cdef int k = here.kind, d, qx, qy, qz, kq, ax, ay, az, connected
        if k == K_DUST:
            connected = self.dust_links(x, y, z)
            for d in range(4):
                qx, qz = x + HORIZ[d][0], z + HORIZ[d][1]
                other = self.at(qx, y, qz)
                if other.kind == K_REPEATER and qx + other.dx == x and qz + other.dz == z:
                    self.add(qx, y, qz, MODE_PLAIN)
            if self.kind(x, y - 1, z) == K_OPAQUE:
                self.add(x, y - 1, z, MODE_WEAK)
            connected = self.dust_directions(x, y, z, connected)
            for d in range(4):
                if connected >> d & 1 and self.kind(x + HORIZ[d][0], y, z + HORIZ[d][1]) == K_OPAQUE:
                    self.add(x + HORIZ[d][0], y, z + HORIZ[d][1], MODE_WEAK)
        elif k == K_REPEATER:
            qx, qz = x - here.dx, z - here.dz
            other = self.at(qx, y, qz)
            if other.kind == K_DUST:
                self.add(qx, y, qz, MODE_PLAIN)
            elif other.kind == K_REPEATER and qx + other.dx == x and qz + other.dz == z:
                self.add(qx, y, qz, MODE_PLAIN)
            elif other.kind == K_OPAQUE:
                self.add(qx, y, qz, MODE_STRONG)
        elif k == K_TORCH or k == K_WALL_TORCH:
            if k == K_TORCH:
                ax, ay, az = x, y - 1, z
            else:
                ax, ay, az = x - here.dx, y, z - here.dz
            if self.kind(x, y + 1, z) == K_OPAQUE:
                self.add(x, y + 1, z, MODE_STRONG)
            for d in range(5):
                if d < 4:
                    qx, qy, qz = x + HORIZ[d][0], y, z + HORIZ[d][1]
                else:
                    qx, qy, qz = x, y - 1, z
                if qx == ax and qy == ay and qz == az:
                    continue
                other = self.at(qx, qy, qz)
                if other.kind == K_DUST:
                    self.add(qx, qy, qz, MODE_PLAIN)
                elif other.kind == K_REPEATER and qx + other.dx == x and qy == y and qz + other.dz == z:
                    self.add(qx, qy, qz, MODE_PLAIN)
        elif k == K_OPAQUE:
            for d in range(6):
                if d < 4:
                    qx, qy, qz = x + HORIZ[d][0], y, z + HORIZ[d][1]
                else:
                    qx, qy, qz = x, y + (1 if d == 4 else -1), z
                other = self.at(qx, qy, qz)
                kq = other.kind
                if kq == K_TORCH:
                    if qx == x and qy - 1 == y and qz == z:
                        self.add(qx, qy, qz, MODE_PLAIN)
                elif kq == K_WALL_TORCH:
                    if qx - other.dx == x and qy == y and qz - other.dz == z:
                        self.add(qx, qy, qz, MODE_PLAIN)
                elif kq == K_REPEATER:
                    if qx + other.dx == x and qy == y and qz + other.dz == z:
                        self.add(qx, qy, qz, MODE_PLAIN)
                elif kq == K_DUST and mode == MODE_STRONG:
                    self.add(qx, qy, qz, MODE_PLAIN)
        return 0

    cdef inline bint stops(self, int x, int y, int z) except -1:
        """In verify's stop: a terminal the signal isn't followed through."""
        return (self.grid.word(x, y, z)[0] & F_TERM) != 0 and self.stop_own.find(cell_key(x, y, z)) == NULL

    cdef bint has_latch(self, int sx, int sy, int sz) except -1:
        """_latch_cycle(...) != []: a repeater or torch on a cycle of more than itself."""
        self.hiding = False
        self.nodes.clear()
        self.offsets.clear()
        self.heads.clear()
        self.ids.reset(1 << 10)
        cdef CheckNode start
        start.x, start.y, start.z = sx, sy, sz
        start.mode = MODE_STRONG if self.kind(sx, sy, sz) == K_OPAQUE else MODE_PLAIN
        self.nodes.push_back(start)
        cdef int ins, i = 0, j, head
        cdef int* slot
        cdef CheckNode node
        cdef bint repeaters = False
        self.ids.get(cell_key(sx, sy, sz) << 2 | start.mode, &ins)[0] = 0
        while i < <int>self.nodes.size():
            node = self.nodes[i]
            self.offsets.push_back(self.heads.size())
            i += 1
            if self.kind(node.x, node.y, node.z) in (K_REPEATER, K_TORCH, K_WALL_TORCH):
                repeaters = True
            if self.stops(node.x, node.y, node.z):
                continue
            self.successors(node.x, node.y, node.z, node.mode)
            for j in range(self.n_out):
                slot = self.ids.get(cell_key(self.out[j].x, self.out[j].y, self.out[j].z) << 2 | self.out[j].mode, &ins)
                if ins:
                    slot[0] = self.nodes.size()
                    self.nodes.push_back(self.out[j])
                self.heads.push_back(slot[0])
        if not repeaters:
            return False
        self.offsets.push_back(self.heads.size())
        return self.repeater_on_cycle()

    cdef bint repeater_on_cycle(self) except -1:
        """Tarjan over offsets and heads, without recursion: wire runs long."""
        cdef int n = self.nodes.size(), counter = 1, root, v, w, popped, size
        cdef bint repeater
        cdef TarjanFrame* top
        cdef TarjanFrame frame
        self.index.assign(n, 0)
        self.low.assign(n, 0)
        self.on_stack.assign(n, 0)
        self.scc_stack.clear()
        for root in range(n):
            if self.index[root]:
                continue
            self.index[root] = self.low[root] = counter
            counter += 1
            self.scc_stack.push_back(root)
            self.on_stack[root] = 1
            frame.node, frame.edge = root, self.offsets[root]
            self.frames.push_back(frame)
            while self.frames.size():
                top = &self.frames.back()
                v = top.node
                if top.edge < self.offsets[v + 1]:
                    w = self.heads[top.edge]
                    top.edge += 1
                    if not self.index[w]:
                        self.index[w] = self.low[w] = counter
                        counter += 1
                        self.scc_stack.push_back(w)
                        self.on_stack[w] = 1
                        frame.node, frame.edge = w, self.offsets[w]
                        self.frames.push_back(frame)
                    elif self.on_stack[w] and self.index[w] < self.low[v]:
                        self.low[v] = self.index[w]
                    continue
                self.frames.pop_back()
                if self.frames.size() and self.low[v] < self.low[self.frames.back().node]:
                    self.low[self.frames.back().node] = self.low[v]
                if self.low[v] != self.index[v]:
                    continue
                size = 0
                repeater = False
                while True:
                    popped = self.scc_stack.back()
                    self.scc_stack.pop_back()
                    self.on_stack[popped] = 0
                    size += 1
                    if self.kind(self.nodes[popped].x, self.nodes[popped].y, self.nodes[popped].z) in (K_REPEATER, K_TORCH, K_WALL_TORCH):
                        repeater = True
                    if popped == v:
                        break
                if size > 1 and repeater:
                    self.frames.clear()
                    return True
        return False

    cdef inline void note_polarity(self, int x, int y, int z, bint flipped) noexcept:
        cdef int ins
        self.polarity.get(cell_key(x, y, z), &ins)[0] |= 2 if flipped else 1

    cdef inline int polarity_at(self, int x, int y, int z) noexcept:
        cdef char* p = self.polarity.find(cell_key(x, y, z))
        return 0 if p == NULL else p[0]

    cdef bint has_inverted(self, int sx, int sy, int sz, sinks) except -1:
        """_inverted_contacts(...) != [], over the world find_inverted_contacts makes."""
        self.hiding = True
        self.nodes.clear()
        self.flips.clear()
        self.feeds.clear()
        self.seen.reset(1 << 10)
        self.polarity.reset(1 << 10)
        cdef CheckNode start, node, q
        cdef int ins, i = 0, j, kp, kq
        cdef bint flipped, out
        start.x, start.y, start.z = sx, sy, sz
        start.mode = MODE_STRONG if self.kind(sx, sy, sz) == K_OPAQUE else MODE_PLAIN
        self.nodes.push_back(start)
        self.flips.push_back(False)
        self.seen.get(cell_key(sx, sy, sz) << 3 | start.mode << 1, &ins)
        self.note_polarity(sx, sy, sz, False)
        while i < <int>self.nodes.size():
            node = self.nodes[i]
            flipped = self.flips[i]
            i += 1
            if self.stops(node.x, node.y, node.z):
                continue
            kp = self.kind(node.x, node.y, node.z)
            out = flipped != ((kp == K_TORCH or kp == K_WALL_TORCH) and not (node.x == sx and node.y == sy and node.z == sz))
            self.successors(node.x, node.y, node.z, node.mode)
            for j in range(self.n_out):
                q = self.out[j]
                if out:
                    self.feeds.push_back(node)
                    self.feeds.push_back(q)
                self.note_polarity(q.x, q.y, q.z, out)
                self.seen.get(cell_key(q.x, q.y, q.z) << 3 | q.mode << 1 | out, &ins)
                if ins:
                    self.nodes.push_back(q)
                    self.flips.push_back(out)
        for i in range(0, <int>self.feeds.size(), 2):
            node, q = self.feeds[i], self.feeds[i + 1]
            if self.polarity_at(q.x, q.y, q.z) != 3:
                continue
            kq = self.kind(q.x, q.y, q.z)
            if kq != K_DUST and kq != K_REPEATER:
                continue
            kp = self.kind(node.x, node.y, node.z)
            if (kp != K_DUST and kp != K_REPEATER) or self.polarity_at(node.x, node.y, node.z) == 2:
                return True
        for sink in sinks:
            if self.polarity_at(sink[0], sink[1], sink[2]) == 2:
                return True
        return False

    cdef bint has_gate_contact(self) except -1:
        """find_gate_contacts(...) != [] for own_cells."""
        if not self.has_gate:
            return False
        self.hiding = False
        cdef int i, d, x, y, z, qx, qz, connected
        for i in range(0, <int>self.own_cells.size(), 3):
            x, y, z = self.own_cells[i], self.own_cells[i + 1], self.own_cells[i + 2]
            if self.kind(x, y, z) != K_DUST:
                continue
            if self.kind(x, y - 1, z) == K_OPAQUE and self.gate.find(cell_key(x, y - 1, z)) != NULL:
                return True
            self.n_out = 0
            connected = self.dust_directions(x, y, z, self.dust_links(x, y, z))
            for d in range(4):
                if connected >> d & 1:
                    qx, qz = x + HORIZ[d][0], z + HORIZ[d][1]
                    if self.kind(qx, y, qz) == K_OPAQUE and self.gate.find(cell_key(qx, y, qz)) != NULL:
                        return True
        return False

    cpdef int faults(self, tuple source, sinks, stop_own, list journal) except -1:
        """Bits of FAULT_*: those of a finished net's checks with something to report.
        stop_own: the terminals its signal is followed through. journal: the net's journal."""
        cdef int ins, result = 0, x, y, z
        cdef int sx = source[0], sy = source[1], sz = source[2]
        self.bind()
        self.stop_own.reset(1 << 10)
        for pos in stop_own:
            self.stop_own.get(cell_key(pos[0], pos[1], pos[2]), &ins)
        self.own.reset(1 << 12)
        self.own_cells.clear()
        for entry in journal:
            if (<tuple>entry)[0] == "block":
                pos = (<tuple>entry)[2]
                x, y, z = pos
                self.own.get(cell_key(x, y, z), &ins)
                if ins:
                    self.own_cells.push_back(x)
                    self.own_cells.push_back(y)
                    self.own_cells.push_back(z)
        if self.has_latch(sx, sy, sz):
            result |= FAULT_LATCH
        if self.has_inverted(sx, sy, sz, sinks):
            result |= FAULT_INVERTED
        if self.has_gate_contact():
            result |= FAULT_GATE
        return result
