# distutils: language = c++
"""verify._World's powers relation and the searches over it, on one byte per cell.

The Python in verify.py and hold.py stays the definition of what these compute (tests run both and compare). Results match
it exactly, ties and all: a node's key sorts as its (pos, mode) tuple does, successors come in the same order, and
dust_directions' Python set order is read from a table of real sets, since it depends on insertion order.
"""
from libc.stdint cimport int32_t, int64_t, uint8_t
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

    // Whether _latch_cycle would find anything: a repeater on a cycle of more than one node.
    static bool has_latch(const SignalGrid& grid, const NetGraph& G) {
        int n = (int)G.node.size();
        bool any = false;
        for (int i = 0; i < n && !any; i++) any = (grid.g[G.node[i] / 3] & 7) == K_REPEATER;
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
                        int size = 0; bool repeater = false;
                        for (;;) {
                            int b = stack.back(); stack.pop_back(); on[b] = 0; size++;
                            if ((grid.g[G.node[b] / 3] & 7) == K_REPEATER) repeater = true;
                            if (b == a) break;
                        }
                        if (size > 1 && repeater) return true;
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
        cdef int x, y, z, m, node
        cdef size_t i
        cdef int64_t c, inverted
        self._set_stop(own, False)
        try:
            x, y, z = source
            self.graph.build(self.c, self.c.cell(x, y, z))
            reach_cells(self.graph, self.cells_out, self.seen, self.queue)
            reached = [self._pos(self.cells_out[i]) for i in range(self.cells_out.size())]
            strengths(self.c, self.graph, self.best, self.queue)
            strong = []
            self.sink_cells.clear()
            for x, y, z in sinks:
                c = self.c.cell(x, y, z)
                self.sink_cells.push_back(c)
                ok = False
                for m in range(3):
                    node = self.graph.ids.get(c * 3 + m)
                    if node >= 0 and self.best[node] > 0:
                        ok = True
                strong.append(ok)
            latch = has_latch(self.c, self.graph)
            inverted = first_inverted(self.c, self.graph, self.sink_cells)
        finally:
            self._set_stop(own, True)
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
