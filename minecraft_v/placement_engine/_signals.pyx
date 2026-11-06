# distutils: language = c++
"""verify._World's powers relation and the searches over it, on one byte per cell.

The Python in verify.py and hold.py stays the definition of what these compute (tests run both and compare). Results match
it exactly, ties and all: a node's key sorts as its (pos, mode) tuple does, successors come in the same order, and
dust_directions' Python set order is read from a table of real sets, since it depends on insertion order.
"""
from libc.stdint cimport int32_t, int64_t, uint8_t, uint32_t, uint64_t
from libcpp.vector cimport vector
import itertools
import numpy as np

cdef extern from *:
    """
    #include <algorithm>
    #include <cstdint>
    #include <functional>
    #include <vector>
    // Cell byte: bits 0-2 kind, 3-4 facing (index into HX/HZ), 5-6 repeater delay - 1, 7 a stop cell.
    enum { K_AIR, K_DUST, K_REPEATER, K_TORCH, K_WALL_TORCH, K_OPAQUE, K_OTHER, K_POWER_BLOCK };  // first 7 as in verify.py
    enum { M_NONE, M_STRONG, M_WEAK };
    static const int HX[4] = {1, -1, 0, 0}, HZ[4] = {0, 0, 1, -1};

    struct IntMap {  // key -> int, open addressing; -1 when absent
        std::vector<int64_t> keys; std::vector<int> vals; std::vector<size_t> used; size_t mask = 0, count = 0;
        IntMap() { reset(1 << 10); }
        void reset(size_t cap) { keys.assign(cap, -1); vals.assign(cap, -1); used.clear(); mask = cap - 1; count = 0; }
        void clear() { for (size_t i : used) keys[i] = -1; used.clear(); count = 0; }
        size_t slot(int64_t k) const {
            size_t i = (size_t)((uint64_t)k * 0x9E3779B97F4A7C15ULL >> 20) & mask;
            while (keys[i] != -1 && keys[i] != k) i = (i + 1) & mask;
            return i;
        }
        int get(int64_t k) const { size_t i = slot(k); return keys[i] == k ? vals[i] : -1; }
        void put(int64_t k, int v) {
            if ((count + 1) * 2 > keys.size()) {
                std::vector<int64_t> ok; ok.swap(keys); std::vector<int> ov; ov.swap(vals);
                size_t cap = ok.size() * 2; keys.assign(cap, -1); vals.assign(cap, -1); mask = cap - 1; used.clear();
                for (size_t j = 0; j < ok.size(); j++) if (ok[j] != -1) { size_t s = slot(ok[j]); keys[s] = ok[j]; vals[s] = ov[j]; used.push_back(s); }
            }
            size_t i = slot(k);
            if (keys[i] != k) { keys[i] = k; used.push_back(i); count++; }
            vals[i] = v;
        }
    };

    struct SignalGrid {
        uint8_t* g = nullptr; int w = 0, h = 0, l = 0;
        const signed char* dir_table = nullptr;  // insertion sequence (base 5, digit = dir + 1) -> order of the set, 4 per entry, -1 ends
        inline int64_t cell(int x, int y, int z) const { return ((int64_t)x * h + y) * l + z; }
        inline void decode(int64_t c, int& x, int& y, int& z) const { z = (int)(c % l); c /= l; y = (int)(c % h); x = (int)(c / h); }
        inline int raw(int x, int y, int z) const {
            return (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < l) ? g[cell(x, y, z)] : 0;
        }
        inline int kind(int x, int y, int z) const { return raw(x, y, z) & 7; }
        static inline bool opaque(int k) { return k == K_OPAQUE || k == K_POWER_BLOCK; }
        static inline bool torch(int k) { return k == K_TORCH || k == K_WALL_TORCH; }
        inline bool repeater_input_is(int qx, int qy, int qz, int px, int py, int pz) const {
            int f = (raw(qx, qy, qz) >> 3) & 3;
            return qx + HX[f] == px && qy == py && qz + HZ[f] == pz;
        }
        inline void attachment(int x, int y, int z, int k, int& ax, int& ay, int& az) const {
            if (k == K_TORCH) { ax = x; ay = y - 1; az = z; return; }
            int f = (raw(x, y, z) >> 3) & 3;
            ax = x - HX[f]; ay = y; az = z - HZ[f];
        }
        // Dust links as cells; dirs gets each one's direction index.
        inline int dust_links(int x, int y, int z, int* lx, int* ly, int* lz, int* dirs) const {
            bool above_open = !opaque(kind(x, y + 1, z));
            int n = 0;
            for (int d = 0; d < 4; d++) {
                int sx = x + HX[d], sz = z + HZ[d];
                int ks = kind(sx, y, sz);
                if (ks == K_DUST) { lx[n] = sx; ly[n] = y; lz[n] = sz; dirs[n++] = d; continue; }
                if (above_open && kind(sx, y + 1, sz) == K_DUST) { lx[n] = sx; ly[n] = y + 1; lz[n] = sz; dirs[n++] = d; }
                if (!opaque(ks) && kind(sx, y - 1, sz) == K_DUST) { lx[n] = sx; ly[n] = y - 1; lz[n] = sz; dirs[n++] = d; }
            }
            return n;
        }
        // dust_directions in its set's iteration order; returns the count.
        inline int dust_directions(int x, int y, int z, int nlinks, const int* link_dirs, int* out) const {
            int seen = 0, seq = 0, count = 0, first = -1;
            auto add = [&](int d) { if (!(seen >> d & 1)) { seen |= 1 << d; seq = seq * 5 + d + 1; if (!count) first = d; count++; } };
            for (int i = 0; i < nlinks; i++) add(link_dirs[i]);
            for (int d = 0; d < 4; d++) {
                int sx = x + HX[d], sz = z + HZ[d];
                int k = kind(sx, y, sz);
                if (k == K_REPEATER) {
                    int f = (raw(sx, y, sz) >> 3) & 3;
                    // p is its input or its output: it lies along the repeater's axis
                    if (HX[f] == HX[d] * -1 && HZ[f] == HZ[d] * -1) add(d);
                    else if (HX[f] == HX[d] && HZ[f] == HZ[d]) add(d);
                } else if (torch(k) || k == K_POWER_BLOCK) add(d);
            }
            if (!count) { seq = 1 * 125 + 2 * 25 + 3 * 5 + 4; }            // set(_HORIZ_DIRS)
            else if (count == 1) { int o = first ^ 1; seq = (first + 1) * 5 + o + 1; }  // {d, -d}
            const signed char* e = dir_table + 4 * seq;
            int n = 0;
            while (n < 4 && e[n] >= 0) { out[n] = e[n]; n++; }
            return n;
        }
        // _World._find_successors, as keys cell * 3 + mode.
        void successors(int64_t node, std::vector<int64_t>& out) const {
            int mode = (int)(node % 3);
            int x, y, z; decode(node / 3, x, y, z);
            int k = kind(x, y, z);
            auto push = [&](int a, int b, int c, int m) { out.push_back(cell(a, b, c) * 3 + m); };
            if (k == K_DUST) {
                int lx[12], ly[12], lz[12], ld[12], dirs[4];
                int n = dust_links(x, y, z, lx, ly, lz, ld);
                for (int i = 0; i < n; i++) push(lx[i], ly[i], lz[i], M_NONE);
                for (int d = 0; d < 4; d++) {
                    int qx = x + HX[d], qz = z + HZ[d];
                    if (kind(qx, y, qz) == K_REPEATER && repeater_input_is(qx, y, qz, x, y, z)) push(qx, y, qz, M_NONE);
                }
                if (opaque(kind(x, y - 1, z))) push(x, y - 1, z, M_WEAK);
                int m = dust_directions(x, y, z, n, ld, dirs);
                for (int i = 0; i < m; i++) {
                    int qx = x + HX[dirs[i]], qz = z + HZ[dirs[i]];
                    if (opaque(kind(qx, y, qz))) push(qx, y, qz, M_WEAK);
                }
            } else if (k == K_REPEATER) {
                int f = (raw(x, y, z) >> 3) & 3;
                int ox = x - HX[f], oz = z - HZ[f];
                int ko = kind(ox, y, oz);
                if (ko == K_DUST) push(ox, y, oz, M_NONE);
                else if (ko == K_REPEATER) { if (repeater_input_is(ox, y, oz, x, y, z)) push(ox, y, oz, M_NONE); }
                else if (opaque(ko)) push(ox, y, oz, M_STRONG);
            } else if (torch(k)) {
                int ax, ay, az; attachment(x, y, z, k, ax, ay, az);
                if (opaque(kind(x, y + 1, z))) push(x, y + 1, z, M_STRONG);
                for (int d = 0; d < 5; d++) {
                    int qx = d < 4 ? x + HX[d] : x, qy = d < 4 ? y : y - 1, qz = d < 4 ? z + HZ[d] : z;
                    if (qx == ax && qy == ay && qz == az) continue;
                    int kq = kind(qx, qy, qz);
                    if (kq == K_DUST) push(qx, qy, qz, M_NONE);
                    else if (kq == K_REPEATER && repeater_input_is(qx, qy, qz, x, y, z)) push(qx, qy, qz, M_NONE);
                }
            } else if (opaque(k)) {
                for (int d = 0; d < 6; d++) {
                    int qx = d < 4 ? x + HX[d] : x, qy = d < 4 ? y : (d == 4 ? y + 1 : y - 1), qz = d < 4 ? z + HZ[d] : z;
                    int kq = kind(qx, qy, qz);
                    if (torch(kq)) {
                        int ax, ay, az; attachment(qx, qy, qz, kq, ax, ay, az);
                        if (ax == x && ay == y && az == z) { push(qx, qy, qz, M_NONE); continue; }
                    }
                    if (kq == K_REPEATER && repeater_input_is(qx, qy, qz, x, y, z)) push(qx, qy, qz, M_NONE);
                    else if (kq == K_DUST && mode == M_STRONG) push(qx, qy, qz, M_NONE);
                }
            }
        }
    };

    // Everything a source's signal can visit: the graph the searches walk. A stop cell other than the source has no edges.
    struct NetGraph {
        std::vector<int64_t> node;       // id -> key, in the order found
        std::vector<int> off, edges;     // off[id]..off[id + 1] in edges, in successor order
        IntMap ids;
        std::vector<int64_t> scratch;
        void build(const SignalGrid& grid, int64_t source_cell) {
            node.clear(); off.clear(); edges.clear(); ids.clear();
            int sx, sy, sz; grid.decode(source_cell, sx, sy, sz);
            int64_t start = source_cell * 3 + (SignalGrid::opaque(grid.kind(sx, sy, sz)) ? M_STRONG : M_NONE);
            node.push_back(start); ids.put(start, 0); off.push_back(0);
            for (size_t i = 0; i < node.size(); i++) {
                int64_t c = node[i] / 3;
                if (c == source_cell || !(grid.g[c] & 128)) {
                    scratch.clear();
                    grid.successors(node[i], scratch);
                    for (int64_t nxt : scratch) {
                        int id = ids.get(nxt);
                        if (id < 0) { id = (int)node.size(); ids.put(nxt, id); node.push_back(nxt); }
                        edges.push_back(id);
                    }
                }
                off.push_back((int)edges.size());
            }
        }
    };

    // _reach: cells in the order first reached, source first.
    static void reach_cells(const NetGraph& G, std::vector<int64_t>& out, std::vector<uint8_t>& seen, std::vector<int>& queue) {
        size_t n = G.node.size();
        seen.assign(n, 0); queue.clear(); out.clear();
        seen[0] = 1; queue.push_back(0); out.push_back(G.node[0] / 3);
        for (size_t qi = 0; qi < queue.size(); qi++) {
            int a = queue[qi];
            for (int e = G.off[a]; e < G.off[a + 1]; e++) {
                int b = G.edges[e];
                if (seen[b]) continue;
                int64_t key = G.node[b];
                if (key % 3 == M_WEAK) {  // a strong visit subsumes a weak one
                    int s = G.ids.get(key - M_WEAK + M_STRONG);
                    if (s >= 0 && seen[s]) continue;
                }
                seen[b] = 1; out.push_back(key / 3); queue.push_back(b);
            }
        }
    }

    // _strengths: best[id] > 0 where signal gets with strength left.
    static void strengths(const SignalGrid& grid, const NetGraph& G, std::vector<int>& best, std::vector<int>& queue) {
        size_t n = G.node.size();
        best.assign(n, 0); queue.clear();
        best[0] = 15; queue.push_back(0);
        for (size_t qi = 0; qi < queue.size(); qi++) {
            int a = queue[qi];
            int level = best[a];
            int64_t ca = G.node[a] / 3;
            bool dust = (grid.g[ca] & 7) == K_DUST;
            for (int e = G.off[a]; e < G.off[a + 1]; e++) {
                int b = G.edges[e];
                int next = dust && (grid.g[G.node[b] / 3] & 7) == K_DUST ? level - 1 : 15;
                if (next > best[b]) { best[b] = next; queue.push_back(b); }
            }
        }
    }

    // Whether _latch_cycle would find anything: a repeater or torch on a cycle of more than one node.
    static bool driver(int k) { return k == K_REPEATER || SignalGrid::torch(k); }
    static bool has_latch(const SignalGrid& grid, const NetGraph& G) {
        int n = (int)G.node.size();
        bool any = false;
        for (int i = 0; i < n && !any; i++) any = driver(grid.g[G.node[i] / 3] & 7);
        if (!any) return false;
        std::vector<int> index(n, -1), low(n, 0), stack, frames, next_edge(n, 0);
        std::vector<uint8_t> on(n, 0);
        int counter = 0;
        for (int root = 0; root < n; root++) {
            if (index[root] >= 0) continue;
            frames.push_back(root);
            index[root] = low[root] = counter++; stack.push_back(root); on[root] = 1; next_edge[root] = G.off[root];
            while (!frames.empty()) {
                int a = frames.back();
                if (next_edge[a] < G.off[a + 1]) {
                    int b = G.edges[next_edge[a]++];
                    if (index[b] < 0) {
                        index[b] = low[b] = counter++; stack.push_back(b); on[b] = 1; next_edge[b] = G.off[b];
                        frames.push_back(b);
                    } else if (on[b]) low[a] = std::min(low[a], index[b]);
                } else {
                    frames.pop_back();
                    if (!frames.empty()) low[frames.back()] = std::min(low[frames.back()], low[a]);
                    if (low[a] == index[a]) {
                        int size = 0; bool has_driver = false;
                        for (;;) {
                            int b = stack.back(); stack.pop_back(); on[b] = 0; size++;
                            if (driver(grid.g[G.node[b] / 3] & 7)) has_driver = true;
                            if (b == a) break;
                        }
                        if (size > 1 && has_driver) return true;
                    }
                }
            }
        }
        return false;
    }

    // _inverted_contacts' first (lowest) cell, or -1. sinks: their cells.
    static int64_t first_inverted(const SignalGrid& grid, const NetGraph& G, const std::vector<int64_t>& sinks) {
        int n = (int)G.node.size();
        int64_t source = G.node[0] / 3;
        IntMap cells;
        std::vector<int64_t> cell_key;
        std::vector<int> cell_of(n);
        for (int i = 0; i < n; i++) {
            int64_t c = G.node[i] / 3;
            int id = cells.get(c);
            if (id < 0) { id = (int)cell_key.size(); cells.put(c, id); cell_key.push_back(c); }
            cell_of[i] = id;
        }
        std::vector<uint8_t> pol(cell_key.size(), 0), seen(2 * (size_t)n, 0);
        std::vector<std::pair<int, int>> feeds;
        std::vector<int> queue;
        pol[cell_of[0]] = 1;          // bit 0: reached as itself, bit 1: as its opposite
        seen[0] = 1; queue.push_back(0);
        for (size_t qi = 0; qi < queue.size(); qi++) {
            int state = queue[qi], a = state / 2, flipped = state & 1;
            int64_t c = G.node[a] / 3;
            int out = flipped ^ (SignalGrid::torch(grid.g[c] & 7) && c != source ? 1 : 0);
            for (int e = G.off[a]; e < G.off[a + 1]; e++) {
                int b = G.edges[e];
                if (out) feeds.push_back({cell_of[a], cell_of[b]});
                pol[cell_of[b]] |= out ? 2 : 1;
                int next = b * 2 + out;
                if (!seen[next]) { seen[next] = 1; queue.push_back(next); }
            }
        }
        auto wired = [&](int cid) { int k = grid.g[cell_key[cid]] & 7; return k == K_DUST || k == K_REPEATER; };
        int64_t best = -1;
        auto take = [&](int64_t c) { if (best < 0 || c < best) best = c; };
        for (auto& f : feeds)
            if (pol[f.second] == 3 && wired(f.second) && (!wired(f.first) || pol[f.first] == 2)) take(cell_key[f.second]);
        for (int64_t s : sinks) {
            int id = cells.get(s);
            if (id >= 0 && pol[id] == 2) take(s);
        }
        return best;
    }

    // wire_arrivals. parent[id] is the search's parent, -1 at the source; best[id] the least ticks.
    static void arrivals(const SignalGrid& grid, const NetGraph& G, std::vector<int>& best, std::vector<int>& parent) {
        int n = (int)G.node.size();
        const int unreached = 1 << 30;
        best.assign(n, unreached); parent.assign(n, -1);
        // heapq on (t, node): the node's key sorts as its tuple does
        std::vector<std::pair<int, int64_t>> heap;
        auto after = [](const std::pair<int, int64_t>& a, const std::pair<int, int64_t>& b) { return a > b; };
        best[0] = 0; heap.push_back({0, G.node[0]});
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), after);
            auto top = heap.back(); heap.pop_back();
            int a = G.ids.get(top.second);
            if (top.first > best[a]) continue;
            for (int e = G.off[a]; e < G.off[a + 1]; e++) {
                int b = G.edges[e];
                int bk = grid.g[G.node[b] / 3];
                int k = bk & 7;
                int step = k == K_REPEATER ? ((bk >> 5) & 3) + 1 : SignalGrid::torch(k) ? 1 : 0;
                int at = top.first + step;
                if (at < best[b]) {
                    best[b] = at; parent[b] = a;
                    heap.push_back({at, G.node[b]});
                    std::push_heap(heap.begin(), heap.end(), after);
                }
            }
        }
    }
    """
    cdef cppclass SignalGrid:
        uint8_t* g
        int w, h, l
        const signed char* dir_table
        int64_t cell(int, int, int)
        void successors(int64_t, vector[int64_t]&)
        int dust_links(int, int, int, int*, int*, int*, int*)
        int dust_directions(int, int, int, int, const int*, int*)
        int kind(int, int, int)
    cdef cppclass IntMap:
        int get(int64_t)
        void put(int64_t, int)
        void reset(size_t)
    cdef cppclass NetGraph:
        vector[int64_t] node
        IntMap ids
        void build(const SignalGrid&, int64_t)
    void reach_cells(const NetGraph&, vector[int64_t]&, vector[uint8_t]&, vector[int]&)
    void strengths(const SignalGrid&, const NetGraph&, vector[int]&, vector[int]&)
    bint has_latch(const SignalGrid&, const NetGraph&)
    int64_t first_inverted(const SignalGrid&, const NetGraph&, const vector[int64_t]&)
    void arrivals(const SignalGrid&, const NetGraph&, vector[int]&, vector[int]&)


cdef extern from *:
    """
    static inline void heap_make(std::vector<int64_t>& h) { std::make_heap(h.begin(), h.end(), std::greater<int64_t>()); }
    static inline int64_t heap_take(std::vector<int64_t>& h) {
        std::pop_heap(h.begin(), h.end(), std::greater<int64_t>());
        int64_t v = h.back(); h.pop_back(); return v;
    }
    static inline void heap_put(std::vector<int64_t>& h, int64_t v) {
        h.push_back(v); std::push_heap(h.begin(), h.end(), std::greater<int64_t>());
    }
    """
    void _heapify "heap_make"(vector[int64_t]&)
    int64_t _heappop "heap_take"(vector[int64_t]&)
    void _heappush "heap_put"(vector[int64_t]&, int64_t)


_HORIZ = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _direction_table() -> bytes:
    """For each insertion sequence of directions, the order Python's set gives them."""
    table = bytearray(b"\xff" * (4 * 5 ** 4))
    for size in range(1, 5):
        for seq in itertools.permutations(range(4), size):
            code = 0
            found = set()
            for d in seq:
                code = code * 5 + d + 1
                found.add(_HORIZ[d])
            for i, t in enumerate(found):
                table[4 * code + i] = _HORIZ.index(t)
    return bytes(table)


cdef bytes _DIR_TABLE = _direction_table()
_FACE = {"east": 0, "west": 1, "south": 2, "north": 3}


cdef class Grid:
    """Cell bytes of a region, as a _World with `blocked` and `terminals` sees them. grid: from cell_bytes in verify.py."""
    cdef SignalGrid c
    cdef object array
    cdef uint8_t[:, :, :] view
    cdef NetGraph graph
    cdef vector[int64_t] cells_out
    cdef vector[uint8_t] seen
    cdef vector[int] queue, best, parent
    cdef vector[int64_t] sink_cells

    def __init__(self, array):
        self.array = array
        self.view = array
        self.c.g = &self.view[0, 0, 0]
        self.c.w, self.c.h, self.c.l = array.shape
        self.c.dir_table = <const signed char*><const char*>_DIR_TABLE

    def set_byte(self, p, int value):
        cdef int x, y, z
        x, y, z = p
        self.view[x, y, z] = (self.view[x, y, z] & 128) | value

    cdef inline tuple _pos(self, int64_t c):
        cdef int64_t l = self.c.l, h = self.c.h
        return (int(c // (h * l)), int((c // l) % h), int(c % l))

    cdef inline void _set_stop(self, object cells, bint on):
        cdef int x, y, z
        for x, y, z in cells:
            if on:
                self.view[x, y, z] |= 128
            else:
                self.view[x, y, z] &= 127

    def check_net(self, source, sinks, own):
        """(cells reached in order, whether each sink has strength left, whether there's a latch, lowest inverted contact)."""
        return self._check_net(source, sinks, own, False)

    def check_net_keys(self, source, sinks, own):
        """check_net with the cells reached as an int64 array of flat indices into the grid, x * h * l + y * l + z, and
        after the rest whether each sink was reached."""
        return self._check_net(source, sinks, own, True)

    cdef tuple _check_net(self, source, sinks, own, bint as_keys):
        cdef int x, y, z, m, node
        cdef size_t i, n
        cdef int64_t c, inverted
        cdef int64_t[:] kv
        self._set_stop(own, False)
        try:
            x, y, z = source
            self.graph.build(self.c, self.c.cell(x, y, z))
            reach_cells(self.graph, self.cells_out, self.seen, self.queue)
            if as_keys:
                n = self.cells_out.size()
                reached = np.empty(n, dtype=np.int64)
                kv = reached
                for i in range(n):
                    kv[i] = self.cells_out[i]
            else:
                reached = [self._pos(self.cells_out[i]) for i in range(self.cells_out.size())]
            strengths(self.c, self.graph, self.best, self.queue)
            strong = []
            hit = []
            self.sink_cells.clear()
            for x, y, z in sinks:
                c = self.c.cell(x, y, z)
                self.sink_cells.push_back(c)
                ok = False
                seen = False
                for m in range(3):
                    node = self.graph.ids.get(c * 3 + m)
                    if node >= 0 and self.best[node] > 0:
                        ok = True
                    if node >= 0 and self.seen[node]:
                        seen = True
                strong.append(ok)
                hit.append(seen)
            latch = has_latch(self.c, self.graph)
            inverted = first_inverted(self.c, self.graph, self.sink_cells)
        finally:
            self._set_stop(own, True)
        if as_keys:
            return reached, strong, latch, (self._pos(inverted) if inverted >= 0 else None), hit
        return reached, strong, latch, (self._pos(inverted) if inverted >= 0 else None)

    def search(self, source, own, sinks):
        """wire_arrivals from source: (node keys, parent ids, each sink's least ticks or -1)."""
        cdef int x, y, z, m, node, least
        cdef int64_t c
        cdef size_t i, n
        self._set_stop(own, False)
        try:
            x, y, z = source
            self.graph.build(self.c, self.c.cell(x, y, z))
            arrivals(self.c, self.graph, self.best, self.parent)
        finally:
            self._set_stop(own, True)
        n = self.graph.node.size()
        keys = np.empty(n, dtype=np.int64)
        parents = np.empty(n, dtype=np.int32)
        cdef int64_t[:] kv = keys
        cdef int32_t[:] pv = parents
        for i in range(n):
            kv[i] = self.graph.node[i]
            pv[i] = self.parent[i]
        ticks = []
        for x, y, z in sinks:
            c = self.c.cell(x, y, z)
            least = -1
            for m in range(3):
                node = self.graph.ids.get(c * 3 + m)
                if node >= 0 and (least < 0 or self.best[node] < least):
                    least = self.best[node]
            ticks.append(least)
        return keys, parents, ticks

    def node_key(self, p, int mode) -> int:
        cdef int x, y, z
        x, y, z = p
        return self.c.cell(x, y, z) * 3 + mode

    def pos_of_key(self, int64_t key) -> tuple:
        return self._pos(key // 3)

    def dust_shapes(self, xs, ys, zs):
        """For each dust: bits per direction of _DIR_NAME's order, 2 each: 0 none, 1 side, 2 up."""
        cdef int32_t[:] xv = xs, yv = ys, zv = zs
        out = np.empty(len(xv), dtype=np.int32)
        cdef int32_t[:] ov = out
        cdef Py_ssize_t i
        cdef int n, m, j, x, y, z, code, ups, dir_bits
        cdef int lx[12], ly[12], lz[12], ld[12], dirs[4]
        for i in range(xv.shape[0]):
            x, y, z = xv[i], yv[i], zv[i]
            n = self.c.dust_links(x, y, z, lx, ly, lz, ld)
            m = self.c.dust_directions(x, y, z, n, ld, dirs)
            ups = 0
            for j in range(n):
                if ly[j] == y + 1:
                    ups |= 1 << ld[j]
            dir_bits = 0
            for j in range(m):
                dir_bits |= 1 << dirs[j]
            code = 0
            for j in range(4):
                if dir_bits >> j & 1:
                    code |= (2 if ups >> j & 1 else 1) << (2 * j)
            ov[i] = code
        return out


cdef class HoldKernel:
    """_HoldTiming.propagate over numbered gates and pins. A gate's number sorts as its id does, so the heap pops the same."""
    cdef vector[int] net_off, sink_gate, sink_pin, sink_class, sink_arrival
    cdef vector[int] from_off, from_net
    cdef vector[uint8_t] searched
    cdef vector[int] into, via_pin_net, via_pin_driver, via_pin_sink, via_net, via_driver, via_sink
    cdef vector[int] best
    cdef object search_net
    cdef int unreached

    def __init__(self, net_sinks, gates_from, int n_gates, int n_pins, search_net):
        """net_sinks: per net, (gate, pin, class) per sink: class 0 flip-flop, 1 output pin, 2 other.
        gates_from: per gate, its nets. search_net(net) returns each sink's ticks, -1 for one not reached."""
        self.unreached = 1 << 30
        self.search_net = search_net
        self.net_off.push_back(0)
        for net, sinks in enumerate(net_sinks):
            for g, p, k in sinks:
                self.sink_gate.push_back(g)
                self.sink_pin.push_back(p)
                self.sink_class.push_back(k)
                self.sink_arrival.push_back(-1)
            self.net_off.push_back(self.sink_gate.size())
        self.searched.assign(len(net_sinks), 0)
        self.from_off.push_back(0)
        for nets in gates_from:
            for n in nets:
                self.from_net.push_back(n)
            self.from_off.push_back(self.from_net.size())
        self.into.assign(n_pins, self.unreached)
        self.via_pin_net.assign(n_pins, -1)
        self.via_pin_driver.assign(n_pins, -1)
        self.via_pin_sink.assign(n_pins, -1)
        self.best.assign(n_gates, self.unreached)
        self.via_net.assign(n_gates, -1)
        self.via_driver.assign(n_gates, -1)
        self.via_sink.assign(n_gates, -1)

    def forget(self, int net):
        self.searched[net] = 0

    cdef _ensure(self, int net):
        cdef int i, off = self.net_off[net]
        if not self.searched[net]:
            ticks = self.search_net(net)
            for i in range(self.net_off[net + 1] - off):
                self.sink_arrival[off + i] = ticks[i]
            self.searched[net] = 1

    def ensure(self, int net):
        self._ensure(net)

    def arrival(self, int net, int sink_index) -> int:
        self._ensure(net)
        return self.sink_arrival[self.net_off[net] + sink_index]

    def propagate(self, starts, gate_delays):
        """starts: gate -> time. Returns [(pin, time)] in the order the pins were first reached."""
        cdef int unreached = self.unreached
        cdef int n_gates = self.best.size(), n_pins = self.into.size()
        cdef int[:] delays = gate_delays
        cdef vector[int] pin_order
        cdef vector[int64_t] heap
        cdef int g, t, i, at, s, net, pin, out, kind
        cdef int64_t top
        for i in range(n_gates):
            self.best[i] = unreached
            self.via_net[i] = -1
        for i in range(n_pins):
            self.into[i] = unreached
            self.via_pin_net[i] = -1
        for g, t in starts:
            self.best[g] = t
            heap.push_back((<int64_t>t << 32) | g)
        _heapify(heap)
        while heap.size():
            top = _heappop(heap)
            t = <int>(top >> 32)
            g = <int>(top & 0xFFFFFFFF)
            if t > self.best[g]:
                continue
            for i in range(self.from_off[g], self.from_off[g + 1]):
                net = self.from_net[i]
                if not self.searched[net]:
                    self._ensure(net)
                for s in range(self.net_off[net], self.net_off[net + 1]):
                    if self.sink_arrival[s] < 0:
                        continue
                    at = t + self.sink_arrival[s]
                    kind = self.sink_class[s]
                    if kind == 0:
                        pin = self.sink_pin[s]
                        if at < self.into[pin]:
                            if self.into[pin] == unreached:
                                pin_order.push_back(pin)
                            self.into[pin] = at
                            self.via_pin_net[pin] = net
                            self.via_pin_driver[pin] = g
                            self.via_pin_sink[pin] = s - self.net_off[net]
                    elif kind != 1:
                        out = at + delays[self.sink_gate[s]]
                        if out < self.best[self.sink_gate[s]]:
                            self.best[self.sink_gate[s]] = out
                            self.via_net[self.sink_gate[s]] = net
                            self.via_driver[self.sink_gate[s]] = g
                            self.via_sink[self.sink_gate[s]] = s - self.net_off[net]
                            _heappush(heap, (<int64_t>out << 32) | self.sink_gate[s])
        return [(pin_order[i], self.into[pin_order[i]]) for i in range(pin_order.size())]

    def via_of_pin(self, int pin):
        """(net, driver gate, sink index), or None."""
        if self.via_pin_net[pin] < 0:
            return None
        return self.via_pin_net[pin], self.via_pin_driver[pin], self.via_pin_sink[pin]

    def via_of_gate(self, int gate):
        if self.via_net[gate] < 0:
            return None
        return self.via_net[gate], self.via_driver[gate], self.via_sink[gate]


cdef extern from *:
    """
    static const int FX[6] = {0, 0, 0, 0, -1, 1}, FY[6] = {1, -1, 0, 0, 0, 0}, FZ[6] = {0, 0, -1, 1, 0, 0};
    """
    const int FX[6]
    const int FY[6]
    const int FZ[6]


cdef struct SettleCell:
    int x, y, z
    int kind, pid, facing, power, level, saved_power, run
    int links_off, links_len, takers_off, takers_len, inputs_off, inputs_len
    int sides_off, sides_len, sources_off, sources_len, readers_off, readers_len
    int side_mask          # directions dust points, -1 until asked
    bint locked, saved_locked, inside, subtract, queued


cdef enum:  # as propagate.py numbers them
    S_DUST, S_REPEATER, S_COMPARATOR, S_TORCH, S_WALL_TORCH, S_LAMP, S_LEVER, S_BLOCK
cdef enum:  # indexes of propagate._FACES; a direction's opposite is its index ^ 1
    D_TOP, D_BOTTOM


cdef inline int _horizontal(int dx, int dz) noexcept:
    if dx == 0:
        return 2 if dz < 0 else 3
    return 4 if dx < 0 else 5


cdef class Settler:
    """propagate._Propagation, step for step: cells get numbered in the same lazy order and a block's readers come out of a
    real Python set of those numbers, so the queue runs in the same order and flip-flops land the same way."""
    cdef uint32_t[:, :, :] blocks
    cdef int w, h, l
    cdef vector[uint8_t] solid, transparent
    cdef vector[int] kind_id, pal_facing, pal_level, pal_attach
    cdef vector[uint8_t] pal_locked, pal_subtract, pal_dot
    cdef vector[int] box
    cdef vector[int] reach_x, reach_y, reach_z
    cdef IntMap index, signals
    cdef vector[SettleCell] cells
    cdef vector[int] arena
    cdef vector[vector[int]] runs
    cdef vector[uint8_t] run_queued
    cdef vector[vector[int]] buckets
    cdef public long long steps
    cdef public bint oscillated

    def __init__(self, blocks, tables, boxes, reach, comparator_signal):
        self.blocks = blocks
        self.w, self.h, self.l = blocks.shape
        for v in tables["solid"]:
            self.solid.push_back(v)
        for v in tables["transparent"]:
            self.transparent.push_back(v)
        for v in tables["kind"]:
            self.kind_id.push_back(v)
        for v in tables["facing"]:
            self.pal_facing.push_back(v)
        for v in tables["level"]:
            self.pal_level.push_back(v)
        for v in tables["attach"]:
            self.pal_attach.push_back(v)
        for v in tables["locked"]:
            self.pal_locked.push_back(v)
        for v in tables["subtract"]:
            self.pal_subtract.push_back(v)
        for v in tables["dot"]:
            self.pal_dot.push_back(v)
        for lo, hi in boxes:
            for v in (*lo, *hi):
                self.box.push_back(v)
        for x, y, z in reach:
            self.reach_x.push_back(x)
            self.reach_y.push_back(y)
            self.reach_z.push_back(z)
        for (x, y, z), level in comparator_signal.items():
            if 0 <= x < self.w and 0 <= y < self.h and 0 <= z < self.l:
                self.signals.put(self._key(x, y, z), level)
        self.buckets.resize(16)
        self.steps = 0
        self.oscillated = False

    cdef inline int64_t _key(self, int x, int y, int z) noexcept:
        return (<int64_t>x * self.h + y) * self.l + z

    cdef inline int _id(self, int x, int y, int z) noexcept:
        if 0 <= x < self.w and 0 <= y < self.h and 0 <= z < self.l:
            return self.blocks[x, y, z]
        return 0

    cdef inline int _kind_id_at(self, int x, int y, int z) noexcept:
        return self.kind_id[self._id(x, y, z)]

    cdef inline bint _solid_at(self, int x, int y, int z) noexcept:
        return self.solid[self._id(x, y, z)]

    cdef int _at(self, int x, int y, int z) noexcept:
        if not (0 <= x < self.w and 0 <= y < self.h and 0 <= z < self.l):
            return -1
        if self.kind_id[self.blocks[x, y, z]] < 0:  # blocks don't change here: only redstone is ever numbered
            return -1
        cdef int i = self.index.get(self._key(x, y, z))
        if i < 0:
            i = self._load(x, y, z)
        return i

    cdef int _load(self, int x, int y, int z) noexcept:
        cdef int pid = self.blocks[x, y, z]
        cdef int i = self.cells.size(), j, level, k = self.kind_id[pid]
        cdef SettleCell c
        c.x, c.y, c.z = x, y, z
        c.kind, c.pid = k, pid
        c.facing = self.pal_facing[pid]
        level = self.pal_level[pid]
        if k == S_COMPARATOR:
            level = self.signals.get(self._key(x, y, z))
            if level < 0:
                level = 0
        c.power, c.level, c.saved_power = level, 0, level
        c.locked = c.saved_locked = self.pal_locked[pid]
        c.subtract = self.pal_subtract[pid]
        c.inside = False
        for j in range(0, self.box.size(), 6):
            if (self.box[j] <= x < self.box[j + 3] and self.box[j + 1] <= y < self.box[j + 4]
                    and self.box[j + 2] <= z < self.box[j + 5]):
                c.inside = True
                break
        c.run = -1
        c.side_mask = -1
        c.queued = False
        c.links_off = c.takers_off = c.inputs_off = c.sides_off = c.sources_off = c.readers_off = -1
        c.links_len = c.takers_len = c.inputs_len = c.sides_len = c.sources_len = c.readers_len = 0
        self.index.put(self._key(x, y, z), i)
        self.cells.push_back(c)
        return i

    # ---- wire shape and connections

    cdef bint _connects(self, int x, int y, int z, int d) noexcept:
        cdef int pid = self._id(x, y, z), k = self.kind_id[pid]
        if k < 0 or k == S_LAMP:
            return False
        return k != S_REPEATER or (self.pal_facing[pid] >> 1) == (d >> 1)

    cdef int _dust_sides(self, int i) noexcept:
        cdef int found = self.cells[i].side_mask
        cdef int x, y, z, d, sx, sz, count = 0, last = 0
        cdef bint above_solid
        if found >= 0:
            return found
        x, y, z = self.cells[i].x, self.cells[i].y, self.cells[i].z
        above_solid = self._solid_at(x, y + 1, z)
        found = 0
        for d in range(2, 6):
            sx, sz = x + FX[d], z + FZ[d]
            if (self._connects(sx, y, sz, d)
                    or not above_solid and self._kind_id_at(sx, y + 1, sz) == S_DUST
                    or not self._solid_at(sx, y, sz) and self._kind_id_at(sx, y - 1, sz) == S_DUST):
                found |= 1 << d
                count += 1
                last = d
        if count == 1:
            found |= 1 << (last ^ 1)
        elif count == 0:
            found = 0 if self.pal_dot[self.cells[i].pid] else 0b111100
        self.cells[i].side_mask = found
        return found

    cdef inline void _keep_links(self, int i, vector[int]& found) noexcept:
        self.cells[i].links_off = self.arena.size()
        self.cells[i].links_len = found.size()
        self.arena.insert(self.arena.end(), found.begin(), found.end())

    cdef void _dust_links(self, int i) noexcept:
        if self.cells[i].links_off >= 0:
            return
        cdef vector[int] found
        cdef int x = self.cells[i].x, y = self.cells[i].y, z = self.cells[i].z, d, qx, qy, qz, q_id
        cdef bint above_solid = self._solid_at(x, y + 1, z)
        for d in range(6):
            qx, qy, qz = x + FX[d], y + FY[d], z + FZ[d]
            if self._kind_id_at(qx, qy, qz) == S_DUST:
                found.push_back(self._at(qx, qy, qz))
            if d >= 2:
                q_id = self._id(qx, qy, qz)
                if not above_solid and not self.transparent[q_id] and self._kind_id_at(qx, qy + 1, qz) == S_DUST:
                    found.push_back(self._at(qx, qy + 1, qz))
                if not self.solid[q_id] and self._kind_id_at(qx, qy - 1, qz) == S_DUST:
                    found.push_back(self._at(qx, qy - 1, qz))
        self._keep_links(i, found)

    cdef inline bint _in_arena(self, int off, int n, int e) noexcept:
        cdef int j
        for j in range(off, off + n):
            if self.arena[j] == e:
                return True
        return False

    cdef void _take_from(self, int i, int qx, int qy, int qz, vector[int]& found) noexcept:
        cdef int j
        if self._kind_id_at(qx, qy, qz) == S_DUST:
            j = self._at(qx, qy, qz)
            self._dust_links(j)
            if self._in_arena(self.cells[j].links_off, self.cells[j].links_len, i):
                found.push_back(j)

    cdef void _dust_takers(self, int i) noexcept:
        if self.cells[i].takers_off >= 0:
            return
        cdef vector[int] found
        cdef int x = self.cells[i].x, y = self.cells[i].y, z = self.cells[i].z, d
        for d in range(6):
            self._take_from(i, x + FX[d], y + FY[d], z + FZ[d], found)
        for d in range(2, 6):
            self._take_from(i, x + FX[d], y + 1, z + FZ[d], found)
            self._take_from(i, x + FX[d], y - 1, z + FZ[d], found)
        self.cells[i].takers_off = self.arena.size()
        self.cells[i].takers_len = found.size()
        self.arena.insert(self.arena.end(), found.begin(), found.end())

    cdef int _run(self, int start) noexcept:
        cdef int number = self.cells[start].run, a, j, q
        if number >= 0:
            return number
        number = self.runs.size()
        self.cells[start].run = number
        cdef vector[int] run
        run.push_back(start)
        cdef size_t at = 0
        while at < run.size():
            a = run[at]
            at += 1
            self._dust_links(a)
            self._dust_takers(a)
            for j in range(self.cells[a].links_off, self.cells[a].links_off + self.cells[a].links_len):
                q = self.arena[j]
                if self.cells[q].run < 0:
                    self.cells[q].run = number
                    run.push_back(q)
            for j in range(self.cells[a].takers_off, self.cells[a].takers_off + self.cells[a].takers_len):
                q = self.arena[j]
                if self.cells[q].run < 0:
                    self.cells[q].run = number
                    run.push_back(q)
        self.runs.push_back(run)
        self.run_queued.push_back(False)
        return number

    # ---- what emits into a cell: `side` is the direction from the reader to p

    cdef int _weak(self, int x, int y, int z, int side, bint dust_power) noexcept:
        cdef int i = self._at(x, y, z), k
        cdef bint emits
        if i < 0:
            return -1
        k = self.cells[i].kind
        if k == S_TORCH:
            emits = side != D_TOP
        elif k == S_WALL_TORCH:
            emits = self.cells[i].facing != side
        elif k == S_BLOCK or k == S_LEVER:
            emits = True
        elif k == S_REPEATER or k == S_COMPARATOR:
            emits = self.cells[i].facing == side
        elif k == S_DUST:
            emits = dust_power and side != D_BOTTOM and (side == D_TOP or (self._dust_sides(i) >> (side ^ 1)) & 1)
        else:
            emits = False
        return i if emits else -1

    cdef int _strong(self, int x, int y, int z, int side, bint dust_power) noexcept:
        cdef int i = self._at(x, y, z), k
        if i < 0:
            return -1
        k = self.cells[i].kind
        if k == S_TORCH or k == S_WALL_TORCH:
            return i if side == D_BOTTOM else -1
        if k == S_LEVER:
            return i if self.pal_attach[self.cells[i].pid] == side else -1
        if k == S_DUST or k == S_REPEATER or k == S_COMPARATOR:
            return self._weak(x, y, z, side, dust_power)
        return -1

    cdef void _signal(self, int x, int y, int z, int side, bint dust_power, vector[int]& out) noexcept:
        cdef int d, e
        if self._solid_at(x, y, z):
            for d in range(6):
                e = self._strong(x + FX[d], y + FY[d], z + FZ[d], d, dust_power)
                if e >= 0:
                    out.push_back(e)
        else:
            e = self._weak(x, y, z, side, dust_power)
            if e >= 0:
                out.push_back(e)

    cdef void _inputs(self, int i) noexcept:
        if self.cells[i].inputs_off >= 0:
            return
        cdef vector[int] found
        cdef int k = self.cells[i].kind, x = self.cells[i].x, y = self.cells[i].y, z = self.cells[i].z, d, j
        if k == S_TORCH:
            self._signal(x, y - 1, z, D_TOP, True, found)
        elif k == S_WALL_TORCH:
            d = self.cells[i].facing ^ 1
            self._signal(x + FX[d], y + FY[d], z + FZ[d], d, True, found)
        elif k == S_LAMP:
            for d in range(6):
                self._signal(x + FX[d], y + FY[d], z + FZ[d], d, True, found)
        else:
            d = self.cells[i].facing
            self._signal(x + FX[d], y + FY[d], z + FZ[d], d, True, found)
            if found.empty():
                j = self._at(x + FX[d], y + FY[d], z + FZ[d])
                if j >= 0 and self.cells[j].kind == S_DUST:
                    found.push_back(j)
        self.cells[i].inputs_off = self.arena.size()
        self.cells[i].inputs_len = found.size()
        self.arena.insert(self.arena.end(), found.begin(), found.end())

    cdef void _side_inputs(self, int i) noexcept:
        cdef int k = self.cells[i].kind
        if self.cells[i].sides_off >= 0 or (k != S_REPEATER and k != S_COMPARATOR):
            return
        cdef vector[int] found
        cdef int x = self.cells[i].x, y = self.cells[i].y, z = self.cells[i].z
        cdef int f = self.cells[i].facing, dx = FX[f], dz = FZ[f], n, ax, az, d, j, kq, e
        for n in range(2):
            ax, az = (dz, -dx) if n == 0 else (-dz, dx)
            d = _horizontal(ax, az)
            j = self._at(x + ax, y, z + az)
            kq = self.cells[j].kind if j >= 0 else -1
            if kq == S_REPEATER or kq == S_COMPARATOR:
                e = self._weak(x + ax, y, z + az, d, False)
                if e >= 0:
                    found.push_back(e)
            elif k == S_COMPARATOR and (kq == S_DUST or kq == S_BLOCK):
                found.push_back(j)
        self.cells[i].sides_off = self.arena.size()
        self.cells[i].sides_len = found.size()
        self.arena.insert(self.arena.end(), found.begin(), found.end())

    cdef void _dust_sources(self, int i) noexcept:
        if self.cells[i].sources_off >= 0:
            return
        cdef vector[int] found
        cdef int x = self.cells[i].x, y = self.cells[i].y, z = self.cells[i].z, d
        for d in range(6):
            self._signal(x + FX[d], y + FY[d], z + FZ[d], d, False, found)
        self.cells[i].sources_off = self.arena.size()
        self.cells[i].sources_len = found.size()
        self.arena.insert(self.arena.end(), found.begin(), found.end())

    cdef void _readers(self, int e):
        if self.cells[e].readers_off >= 0:
            return
        cdef vector[int] found
        cdef int x = self.cells[e].x, y = self.cells[e].y, z = self.cells[e].z, n, qx, qy, qz, k, c, r
        cdef size_t j
        cdef bint fresh
        for n in range(self.reach_x.size()):
            qx, qy, qz = x + self.reach_x[n], y + self.reach_y[n], z + self.reach_z[n]
            k = self._kind_id_at(qx, qy, qz)
            r = 0
            fresh = False
            if k == S_DUST:
                c = self._at(qx, qy, qz)
                self._dust_sources(c)
                if self._in_arena(self.cells[c].sources_off, self.cells[c].sources_len, e):
                    r = ~self._run(c)
                    fresh = True
            elif k == S_REPEATER or k == S_COMPARATOR or k == S_TORCH or k == S_WALL_TORCH or k == S_LAMP:
                c = self._at(qx, qy, qz)
                self._inputs(c)
                if self._in_arena(self.cells[c].inputs_off, self.cells[c].inputs_len, e):
                    r = c
                    fresh = True
                else:
                    self._side_inputs(c)
                    if self._in_arena(self.cells[c].sides_off, self.cells[c].sides_len, e):
                        r = c
                        fresh = True
            if fresh:
                for j in range(found.size()):
                    if found[j] == r:
                        fresh = False
                        break
                if fresh:
                    found.push_back(r)
        if found.size() > 1:  # the order a set of them iterates in
            ordered = set()
            for j in range(found.size()):
                ordered.add(found[j])
            j = 0
            for v in ordered:
                found[j] = v
                j += 1
        self.cells[e].readers_off = self.arena.size()
        self.cells[e].readers_len = found.size()
        self.arena.insert(self.arena.end(), found.begin(), found.end())

    # ---- evaluation

    cdef inline bint _any_on(self, int off, int n) noexcept:
        cdef int j
        for j in range(off, off + n):
            if self.cells[self.arena[j]].power:
                return True
        return False

    cdef inline int _max_on(self, int off, int n) noexcept:
        cdef int j, best = 0, p
        for j in range(off, off + n):
            p = self.cells[self.arena[j]].power
            if p > best:
                best = p
        return best

    cdef int _output(self, int i) noexcept:
        cdef int k = self.cells[i].kind, level, sides
        self._inputs(i)
        cdef int io = self.cells[i].inputs_off, il = self.cells[i].inputs_len
        if k == S_REPEATER:
            self._side_inputs(i)
            self.cells[i].locked = self._any_on(self.cells[i].sides_off, self.cells[i].sides_len)
            if self.cells[i].locked:
                return self.cells[i].power
            return 15 if self._any_on(io, il) else 0
        if k == S_COMPARATOR:
            level = self._max_on(io, il)
            self._side_inputs(i)
            sides = self._max_on(self.cells[i].sides_off, self.cells[i].sides_len)
            if self.cells[i].subtract:
                return max(level - sides, 0)
            return level if level >= sides else 0
        if k == S_LAMP:
            return 15 if self._any_on(io, il) else 0
        return 0 if self._any_on(io, il) else 15

    cdef void _dust_levels(self, int number, vector[int]& changed) noexcept:
        cdef size_t a
        cdef int i, level, q, j, n = self.runs[number].size()
        for a in range(n):
            self.cells[self.runs[number][a]].level = 0
        for level in range(16):
            self.buckets[level].clear()
        for a in range(n):
            i = self.runs[number][a]
            self._dust_sources(i)
            level = self._max_on(self.cells[i].sources_off, self.cells[i].sources_len)
            if level:
                self.cells[i].level = level
                self.buckets[level].push_back(i)
        for level in range(15, 1, -1):
            a = 0
            while a < self.buckets[level].size():
                i = self.buckets[level][a]
                a += 1
                if self.cells[i].level != level:
                    continue
                self._dust_takers(i)
                for j in range(self.cells[i].takers_off, self.cells[i].takers_off + self.cells[i].takers_len):
                    q = self.arena[j]
                    if self.cells[q].level < level - 1:
                        self.cells[q].level = level - 1
                        self.buckets[level - 1].push_back(q)
        changed.clear()
        for a in range(n):
            i = self.runs[number][a]
            if self.cells[i].power != self.cells[i].level:
                changed.push_back(i)

    cdef inline void _mark(self, int item, bint on) noexcept:
        if item >= 0:
            self.cells[item].queued = on
        else:
            self.run_queued[~item] = on

    cdef inline bint _is_queued(self, int item) noexcept:
        return self.cells[item].queued if item >= 0 else self.run_queued[~item]

    def run(self, starting, int steps_per_item, int steps_per_cell):
        """starting: the cells to begin from, an n x 3 int32 array."""
        cdef int[:, :] start = starting
        cdef vector[int] queue, in_box, waiting, steppers, changed
        cdef size_t qh = 0, bh = 0, wh = 0
        cdef Py_ssize_t n
        cdef int i, k, number, item, level, j, reader, a
        cdef long long queued_count = 0, limit, steps = 0
        for n in range(start.shape[0]):
            i = self._at(start[n, 0], start[n, 1], start[n, 2])
            k = self.cells[i].kind
            if k == S_DUST:
                number = self._run(i)
                if self.runs[number][0] == i:
                    queue.push_back(~number)
            elif k == S_REPEATER or k == S_COMPARATOR or k == S_TORCH or k == S_WALL_TORCH or k == S_LAMP:
                steppers.push_back(i)
        for a in range(steppers.size()):  # latches first, as the set of them is built before the queues
            self._side_inputs(steppers[a])
        for a in range(steppers.size()):
            i = steppers[a]
            if self.cells[i].sides_len == 0:
                queue.push_back(i)
            else:
                waiting.push_back(i)
        for a in range(queue.size()):
            self._mark(queue[a], True)
        for a in range(waiting.size()):
            self._mark(waiting[a], True)
        queued_count = queue.size() + waiting.size()
        limit = <long long>steps_per_item * max(queued_count, 1)
        while qh < queue.size() or bh < in_box.size() or wh < waiting.size():
            steps += 1
            if steps > limit:
                limit = <long long>steps_per_cell * max(<long long>self.cells.size(), 1)
                if steps > limit:
                    self.oscillated = True
                    break
            if qh < queue.size():
                item = queue[qh]
                qh += 1
            elif bh < in_box.size():
                item = in_box[bh]
                bh += 1
            else:
                item = waiting[wh]
                wh += 1
            self._mark(item, False)
            if item < 0:
                self._dust_levels(~item, changed)
                for a in range(changed.size()):
                    i = changed[a]
                    self.cells[i].power = self.cells[i].level
            else:
                level = self._output(item)
                changed.clear()
                if level != self.cells[item].power:
                    self.cells[item].power = level
                    changed.push_back(item)
            for a in range(changed.size()):
                i = changed[a]
                self._readers(i)
                for j in range(self.cells[i].readers_off, self.cells[i].readers_off + self.cells[i].readers_len):
                    reader = self.arena[j]
                    if self._is_queued(reader):
                        continue
                    self._mark(reader, True)
                    if reader >= 0:
                        self._side_inputs(reader)
                    if reader >= 0 and self.cells[reader].sides_len:
                        waiting.push_back(reader)
                    elif self.cells[reader if reader >= 0 else self.runs[~reader][0]].inside:
                        in_box.push_back(reader)
                    else:
                        queue.push_back(reader)
        self.steps = steps

    @property
    def cell_count(self):
        return self.cells.size()

    def results(self):
        """Per block that has a state to write, sorted by position: positions n x 3, what kind of state, power, locked.
        Kind: 0 repeater, 1 dust, 2 torch or lamp, 3 comparator. Inside a box, only the ones that moved."""
        cdef size_t i, row, n
        cdef int k
        cdef vector[int] keep
        for i in range(self.cells.size()):
            k = self.cells[i].kind
            if k == S_LEVER or k == S_BLOCK:
                continue
            if self.cells[i].inside and self.cells[i].power == self.cells[i].saved_power \
                    and self.cells[i].locked == self.cells[i].saved_locked:
                continue
            keep.push_back(i)
        n = keep.size()
        pos = np.empty((n, 3), dtype=np.int32)
        state = np.empty(n, dtype=np.int8)
        power = np.empty(n, dtype=np.int32)
        locked = np.empty(n, dtype=bool)
        cdef int32_t[:, :] pv = pos
        cdef signed char[:] sv = state
        cdef int32_t[:] wv = power
        cdef uint8_t[:] lv = locked.view(np.uint8)
        for row in range(n):
            i = keep[row]
            pv[row, 0], pv[row, 1], pv[row, 2] = self.cells[i].x, self.cells[i].y, self.cells[i].z
            k = self.cells[i].kind
            sv[row] = 0 if k == S_REPEATER else 1 if k == S_DUST else 3 if k == S_COMPARATOR else 2
            wv[row] = self.cells[i].power
            lv[row] = 1 if self.cells[i].locked else 0
        # By x, then y, then z: one key per cell sorts as lexsort would, and several times faster.
        order = np.argsort((pos[:, 0].astype(np.int64) * self.h + pos[:, 1]) * self.l + pos[:, 2])
        return pos[order], state[order], power[order], locked[order]


def short_suspects(int64_t[::1] keys, int64_t[::1] offsets, owner, dict ids, int w, int h, int l):
    """verify._verify_on_grid's bulk short check, with hash maps instead of sorting. keys: every net's reached cells, net
    after net, offsets[k]:offsets[k + 1] net k's; owner: cell -> owning net's name; ids: net name -> k.
    Returns (per net, whether it has a cell owned by another net or reached by an earlier one; per key, the first net
    that reached it)."""
    cdef Py_ssize_t n_nets = offsets.shape[0] - 1, k, i
    suspect = np.zeros(n_nets, dtype=bool)
    first = np.empty(keys.shape[0], dtype=np.int64)
    cdef uint8_t[:] sv = suspect.view(np.uint8)
    cdef int64_t[:] fv = first
    cdef IntMap owners, firsts
    cdef size_t cap = 1 << 10
    cdef int x, y, z, f, o
    cdef int64_t key
    while cap < 2 * (<size_t>len(owner) + 1):
        cap <<= 1
    owners.reset(cap)
    others = {}
    for pos, name in owner.items():
        x, y, z = pos
        if not (0 <= x < w and 0 <= y < h and 0 <= z < l):
            continue
        oid = ids.get(name)
        if oid is None:
            oid = others.get(name)
            if oid is None:
                oid = others[name] = len(ids) + len(others)
        owners.put((<int64_t>x * h + y) * l + z, oid)
    cap = 1 << 10
    while cap < 2 * (<size_t>keys.shape[0] + 1):
        cap <<= 1
    firsts.reset(cap)
    for k in range(n_nets):
        for i in range(offsets[k], offsets[k + 1]):
            key = keys[i]
            f = firsts.get(key)
            if f < 0:
                firsts.put(key, <int>k)
                f = <int>k
            fv[i] = f
            if f != k:
                sv[k] = 1
            o = owners.get(key)
            if o >= 0 and o != k:
                sv[k] = 1
    return suspect, first


def paint_boxes(mask, int64_t[:, ::1] lo, int64_t[:, ::1] hi):
    """mask[lo[i]:hi[i]] = True for each box, every box inside mask (a C-ordered bool array)."""
    cdef uint8_t[:, :, ::1] m = mask.view(np.uint8)
    cdef Py_ssize_t i
    cdef int64_t x, y, z
    for i in range(lo.shape[0]):
        for x in range(lo[i, 0], hi[i, 0]):
            for y in range(lo[i, 1], hi[i, 1]):
                for z in range(lo[i, 2], hi[i, 2]):
                    m[x, y, z] = 1


def table_gather(uint32_t[:, :, ::1] blocks, uint8_t[::1] table):
    """table[blocks], a uint8 array the shape of blocks: numpy's fancy indexing takes several times as long."""
    out = np.empty((blocks.shape[0], blocks.shape[1], blocks.shape[2]), dtype=np.uint8)
    if out.size == 0:
        return out
    cdef uint8_t[:, :, ::1] o = out
    cdef uint32_t* p = &blocks[0, 0, 0]
    cdef uint8_t* q = &o[0, 0, 0]
    cdef Py_ssize_t i, size = blocks.shape[0] * blocks.shape[1] * blocks.shape[2]
    for i in range(size):
        q[i] = table[p[i]]
    return out


def table_where(uint32_t[:, :, ::1] blocks, uint8_t[::1] table, boxes=()):
    """np.argwhere(table[blocks]) with the cells in boxes (lowest cell, one past the highest) left out: an n x 3 int64
    array in the same order, without a mask the size of blocks."""
    cdef int nx = blocks.shape[0], ny = blocks.shape[1], nz = blocks.shape[2], x, y, z, z0, z1, j, nb
    cdef vector[int] box
    cdef vector[int] cut  # per row, the z ranges boxes cover there
    cdef vector[int64_t] found
    for lo, hi in boxes:
        for v in (*lo, *hi):
            box.push_back(v)
    nb = box.size() // 6
    cdef uint32_t* row
    for x in range(nx):
        for y in range(ny):
            row = &blocks[x, y, 0]
            cut.clear()
            for j in range(nb):
                if box[6 * j] <= x < box[6 * j + 3] and box[6 * j + 1] <= y < box[6 * j + 4]:
                    z0, z1 = max(box[6 * j + 2], 0), min(box[6 * j + 5], nz)
                    if z0 < z1:
                        cut.push_back(z0)
                        cut.push_back(z1)
            if cut.empty():
                for z in range(nz):
                    if table[row[z]]:
                        found.push_back((<int64_t>x * ny + y) * nz + z)
            else:
                for z in range(nz):
                    if table[row[z]]:
                        for j in range(0, cut.size(), 2):
                            if cut[j] <= z < cut[j + 1]:
                                break
                        else:
                            found.push_back((<int64_t>x * ny + y) * nz + z)
    out = np.empty((found.size(), 3), dtype=np.int64)
    cdef int64_t[:, ::1] o = out
    cdef int64_t k
    cdef size_t n
    for n in range(found.size()):
        k = found[n]
        o[n, 2] = k % nz
        k //= nz
        o[n, 1] = k % ny
        o[n, 0] = k // ny
    return out


def used_ids(uint32_t[:, :, :] blocks, int n):
    """Which of n palette entries blocks use, as a bool array."""
    used = np.zeros(n, dtype=np.uint8)
    cdef uint8_t[:] u = used
    cdef uint32_t* p = &blocks[0, 0, 0]
    cdef Py_ssize_t i, size = blocks.shape[0] * blocks.shape[1] * blocks.shape[2]
    for i in range(size):
        u[p[i]] = 1
    return used.astype(bool)


def remap_ids(uint32_t[:, :, :] blocks, uint32_t[:] table):
    """blocks[...] = table[blocks], without the copy."""
    cdef uint32_t* p = &blocks[0, 0, 0]
    cdef Py_ssize_t i, size = blocks.shape[0] * blocks.shape[1] * blocks.shape[2]
    for i in range(size):
        p[i] = table[p[i]]


def pack_states(uint32_t[:, :, :] blocks, int nbits, out):
    """block_utils._pack_states' bits, written to out, a uint64 array: y, then z, then x, lowest bits first."""
    cdef uint64_t[:] o = out
    cdef int n_x = blocks.shape[0], n_y = blocks.shape[1], n_z = blocks.shape[2], x, y, z, pos = 0
    cdef uint64_t held = 0, v, mask = (<uint64_t>1 << nbits) - 1
    cdef Py_ssize_t done = 0, step = n_y * n_z, at
    cdef uint32_t* p = &blocks[0, 0, 0]
    for y in range(n_y):
        for z in range(n_z):
            at = y * n_z + z
            for x in range(n_x):
                v = p[at] & mask
                at += step
                held |= v << pos
                pos += nbits
                if pos >= 64:
                    o[done] = held
                    done += 1
                    pos -= 64
                    held = v >> (nbits - pos) if pos else 0
    if pos:
        o[done] = held
