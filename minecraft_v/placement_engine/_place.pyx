# distutils: language = c++
"""poisson._Occupancy and poisson._refine's greedy moves, and _Field's charge deposit and force gather, in C++.

The Python in poisson.py stays the definition of what these compute (tests run both and compare). Results match it
exactly: the best-first search pops (cost, x, y, z) in the same order as heapq does with (cost, (x, y, z)) entries, and
every coordinate and cost is a whole number, so the float comparisons come out the same.
"""
from libc.stdint cimport int64_t
from libcpp cimport bool as cbool
from libcpp.vector cimport vector
import numpy as np

cdef extern from *:
    """
    #include <algorithm>
    #include <cmath>
    #include <cstdint>
    #include <queue>
    #include <thread>
    #include <tuple>
    #include <unordered_map>
    #include <unordered_set>
    #include <vector>

    static inline int64_t floordiv(int64_t a, int64_t b) { int64_t q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
    static inline int64_t pack3(int64_t x, int64_t y, int64_t z) {
        const int64_t o = 1 << 20;
        return ((x + o) << 42) | ((y + o) << 21) | (z + o);
    }

    struct Occ {
        static const int64_t CELL = 16;
        std::vector<int64_t> size;    // 3 per id
        std::vector<int64_t> origin;  // 3 per id
        std::vector<char> present;
        std::vector<int> mark;
        int epoch = 0;
        std::unordered_map<int64_t, std::vector<int>> grid;
        std::vector<int> found;

        void init(const std::vector<int64_t>& sizes) {
            size = sizes;
            size_t n = sizes.size() / 3;
            origin.assign(3 * n, 0); present.assign(n, 0); mark.assign(n, 0);
        }
        template <class F> void cells(const int64_t* p, const int64_t* s, F f) const {
            for (int64_t x = floordiv(p[0], CELL); x <= floordiv(p[0] + s[0] - 1, CELL); ++x)
                for (int64_t y = floordiv(p[1], CELL); y <= floordiv(p[1] + s[1] - 1, CELL); ++y)
                    for (int64_t z = floordiv(p[2], CELL); z <= floordiv(p[2] + s[2] - 1, CELL); ++z)
                        f(pack3(x, y, z));
        }
        void add(int i, const int64_t* p) {
            origin[3 * i] = p[0]; origin[3 * i + 1] = p[1]; origin[3 * i + 2] = p[2]; present[i] = 1;
            cells(p, &size[3 * i], [&](int64_t k) { grid[k].push_back(i); });
        }
        void remove(int i) {
            cells(&origin[3 * i], &size[3 * i], [&](int64_t k) {
                auto it = grid.find(k);
                if (it == grid.end()) return;
                auto& v = it->second;
                auto f = std::find(v.begin(), v.end(), i);
                if (f != v.end()) { *f = v.back(); v.pop_back(); }
            });
            present[i] = 0;
        }
        // Boxes overlapping a box of size s at p, other than skip, into `found` (unsorted).
        void conflicts(const int64_t* p, const int64_t* s, int skip) {
            found.clear();
            if (++epoch == 0) { std::fill(mark.begin(), mark.end(), 0); epoch = 1; }
            const int64_t e0 = p[0] + s[0], e1 = p[1] + s[1], e2 = p[2] + s[2];
            cells(p, s, [&](int64_t k) {
                auto it = grid.find(k);
                if (it == grid.end()) return;
                for (int j : it->second) {
                    if (j == skip || mark[j] == epoch) continue;
                    mark[j] = epoch;
                    const int64_t* o = &origin[3 * j];
                    const int64_t* t = &size[3 * j];
                    if (p[0] < o[0] + t[0] && o[0] < e0 && p[1] < o[1] + t[1] && o[1] < e1 && p[2] < o[2] + t[2] && o[2] < e2)
                        found.push_back(j);
                }
            });
        }
        // Free origin closest to target (weights 1, 2, 1), or false after max_pops.
        bool nearest_free(int i, const int64_t* target, bool flat, int64_t max_pops, int64_t* out) {
            typedef std::tuple<int64_t, int64_t, int64_t, int64_t> Entry;
            std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
            std::unordered_set<int64_t> seen;
            const int64_t* s = &size[3 * i];
            heap.emplace(0, target[0], target[1], target[2]);
            seen.insert(pack3(target[0], target[1], target[2]));
            const int axes3[3] = {0, 1, 2}, axes2[2] = {0, 2};
            const int* axes = flat ? axes2 : axes3;
            int n_axes = flat ? 2 : 3;
            std::vector<int> blockers;
            for (int64_t pop = 0; pop < max_pops; ++pop) {
                if (heap.empty()) return false;
                Entry e = heap.top(); heap.pop();
                int64_t p[3] = {std::get<1>(e), std::get<2>(e), std::get<3>(e)};
                conflicts(p, s, i);
                if (found.empty()) { out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; return true; }
                blockers = found;
                for (int j : blockers) {
                    const int64_t* o = &origin[3 * j];
                    const int64_t* t = &size[3 * j];
                    for (int k = 0; k < n_axes; ++k) {
                        int a = axes[k];
                        int64_t vs[2] = {o[a] + t[a], o[a] - s[a]};
                        for (int64_t v : vs) {
                            if (a != 0 && v < 0) continue;
                            int64_t q[3] = {p[0], p[1], p[2]};
                            q[a] = v;
                            if (seen.insert(pack3(q[0], q[1], q[2])).second) {
                                int64_t c = std::llabs(q[0] - target[0]) + 2 * std::llabs(q[1] - target[1]) + std::llabs(q[2] - target[2]);
                                heap.emplace(c, q[0], q[1], q[2]);
                            }
                        }
                    }
                }
            }
            return false;
        }
    };


    // Charge deposit and force gather for _Field.forces. Boxes come in deposit order; every box touches span[a] bins
    // along axis a from floor(lo / h), clipped to the grid, with its overlap length computed before clipping (as numpy).
    struct FieldKernel {
        int nx = 1, ny = 1, nz = 1;
        double h[3] = {1, 1, 1};
        int threads = 1;

        template <class F> void parallel(int n, F f) const {
            int t = std::max(1, std::min(threads, n));
            if (t == 1) { f(0, n); return; }
            std::vector<std::thread> pool;
            for (int k = 0; k < t; ++k) {
                int a = (int)((int64_t)n * k / t), b = (int)((int64_t)n * (k + 1) / t);
                pool.emplace_back([=, &f]() { f(a, b); });
            }
            for (auto& th : pool) th.join();
        }
        // Bins and overlaps of box k along axis a: up to span entries.
        inline int axis_bins(double lo, double eff, int span, int a, int* bins, double* lens) const {
            const int m = a == 0 ? nx : a == 1 ? ny : nz;
            const double hh = h[a], hi = lo + eff;
            const int64_t first = (int64_t)std::floor(lo / hh);
            for (int j = 0; j < span; ++j) {
                int64_t idx = first + j;
                double ov = std::min(hi, (double)(idx + 1) * hh) - std::max(lo, (double)idx * hh);
                lens[j] = std::max(ov, 0.0);
                bins[j] = (int)std::min<int64_t>(std::max<int64_t>(idx, 0), m - 1);
            }
            return span;
        }
        void deposit(int n, const double* lo, const double* eff, const double* scale, const int* span, double* held) const {
            std::fill(held, held + (size_t)nx * ny * nz, 0.0);
            parallel(nx, [&](int x0, int x1) {
                int bx[64], by[64], bz[64];
                double lx[64], ly[64], lz[64];
                for (int k = 0; k < n; ++k) {
                    const double* l = lo + 3 * k; const double* e = eff + 3 * k; const int* s = span + 3 * k;
                    axis_bins(l[0], e[0], s[0], 0, bx, lx);
                    bool mine = false;
                    for (int i = 0; i < s[0]; ++i) if (bx[i] >= x0 && bx[i] < x1) { mine = true; break; }
                    if (!mine) continue;
                    axis_bins(l[1], e[1], s[1], 1, by, ly);
                    axis_bins(l[2], e[2], s[2], 2, bz, lz);
                    for (int i = 0; i < s[0]; ++i) {
                        if (bx[i] < x0 || bx[i] >= x1) continue;
                        double* plane = held + (size_t)bx[i] * ny * nz;
                        for (int j = 0; j < s[1]; ++j) {
                            double* row = plane + (size_t)by[j] * nz;
                            double lxy = lx[i] * ly[j];
                            for (int q = 0; q < s[2]; ++q) row[bz[q]] += lxy * lz[q] * scale[k];
                        }
                    }
                }
            });
        }
        // -dphi/dx per bin: central differences, one-sided at the walls, all over 2h (as _Field.forces).
        void fields(const double* phi, double* f0, double* f1, double* f2) const {
            const size_t sx = (size_t)ny * nz, sy = nz;
            const double c0 = 1 / (2 * h[0]), c1 = 1 / (2 * h[1]), c2 = 1 / (2 * h[2]);
            parallel(nx, [&](int x0, int x1) {
                for (int x = x0; x < x1; ++x) for (int y = 0; y < ny; ++y) for (int z = 0; z < nz; ++z) {
                    size_t c = x * sx + y * sy + z;
                    if (nx > 1) {
                        size_t a = x == 0 ? c : c - sx, b = x == nx - 1 ? c : c + sx;
                        if (x == 0) b = c + sx;
                        if (x == nx - 1) a = c - sx;
                        f0[c] = (phi[a] - phi[b]) * c0;
                    } else f0[c] = 0;
                    if (ny > 1) {
                        size_t a = y == 0 ? c : c - sy, b = y == ny - 1 ? c : c + sy;
                        if (y == 0) b = c + sy;
                        if (y == ny - 1) a = c - sy;
                        f1[c] = (phi[a] - phi[b]) * c1;
                    } else f1[c] = 0;
                    if (nz > 1) {
                        size_t a = z == 0 ? c : c - 1, b = z == nz - 1 ? c : c + 1;
                        if (z == 0) b = c + 1;
                        if (z == nz - 1) a = c - 1;
                        f2[c] = (phi[a] - phi[b]) * c2;
                    } else f2[c] = 0;
                }
            });
        }
        // Sum of max(held - capacity, 0), in fixed chunks so the total doesn't depend on the thread count.
        double overflow(const double* held, const double* capacity, size_t cells) const {
            const int chunks = 64;
            double part[chunks];
            parallel(chunks, [&](int c0, int c1) {
                for (int c = c0; c < c1; ++c) {
                    size_t a = cells * c / chunks, b = cells * (c + 1) / chunks;
                    double sum = 0;
                    for (size_t i = a; i < b; ++i) sum += std::max(held[i] - capacity[i], 0.0);
                    part[c] = sum;
                }
            });
            double total = 0;
            for (int c = 0; c < chunks; ++c) total += part[c];
            return total;
        }
        void axpb(double* x, const double* a, const double* b, size_t cells) const {
            parallel(64, [&](int c0, int c1) {
                size_t lo = cells * c0 / 64, hi = cells * c1 / 64;
                for (size_t i = lo; i < hi; ++i) x[i] = x[i] * a[i] + b[i];
            });
        }
        void gather(int n, const double* lo, const double* eff, const double* scale, const int* span,
                    const double* f0, const double* f1, const double* f2, double* force) const {
            parallel(n, [&](int k0, int k1) {
                int bx[64], by[64], bz[64];
                double lx[64], ly[64], lz[64];
                for (int k = k0; k < k1; ++k) {
                    const double* l = lo + 3 * k; const double* e = eff + 3 * k; const int* s = span + 3 * k;
                    axis_bins(l[0], e[0], s[0], 0, bx, lx);
                    axis_bins(l[1], e[1], s[1], 1, by, ly);
                    axis_bins(l[2], e[2], s[2], 2, bz, lz);
                    double g0 = 0, g1 = 0, g2 = 0;
                    for (int i = 0; i < s[0]; ++i) for (int j = 0; j < s[1]; ++j) {
                        double lxy = lx[i] * ly[j];
                        size_t base = ((size_t)bx[i] * ny + by[j]) * nz;
                        for (int q = 0; q < s[2]; ++q) {
                            double w = lxy * lz[q] * scale[k];
                            size_t c = base + bz[q];
                            g0 += w * f0[c]; g1 += w * f1[c]; g2 += w * f2[c];
                        }
                    }
                    force[3 * k] = g0; force[3 * k + 1] = g1; force[3 * k + 2] = g2;
                }
            });
        }
    };


    // Weighted-average wirelength gradient and HPWL (poisson._Netlist), nets in parallel.
    struct WireKernel {
        std::vector<int64_t> pin_comp, net_start;  // net_start: n_nets + 1
        std::vector<double> pin_off, weights;      // 3 per pin; 3
        std::vector<double> pin_grad;              // 3 per pin
        std::vector<double> part;
        int threads = 1;
        template <class F> void parallel(int n, F f) const {
            int t = std::max(1, std::min(threads, n));
            if (t == 1) { f(0, n); return; }
            std::vector<std::thread> pool;
            for (int k = 0; k < t; ++k) {
                int a = (int)((int64_t)n * k / t), b = (int)((int64_t)n * (k + 1) / t);
                pool.emplace_back([=, &f]() { f(a, b); });
            }
            for (auto& th : pool) th.join();
        }
        int n_nets() const { return (int)net_start.size() - 1; }
        void gradient(const double* pos, int m, const double* gamma, double* grad) {
            pin_grad.assign(pin_comp.size() * 3, 0.0);
            parallel(n_nets(), [&](int a0, int a1) {
                std::vector<double> x, e;
                for (int net = a0; net < a1; ++net) {
                    int64_t p0 = net_start[net], p1 = net_start[net + 1];
                    size_t k = (size_t)(p1 - p0);
                    x.resize(k); e.resize(k);
                    for (int a = 0; a < 3; ++a) {
                        for (int64_t q = p0; q < p1; ++q) x[q - p0] = pos[3 * pin_comp[q] + a] + pin_off[3 * q + a];
                        for (double sign : {1.0, -1.0}) {
                            double top = -1e300;
                            for (size_t i = 0; i < k; ++i) top = std::max(top, sign * x[i]);
                            double s0 = 0, s1 = 0;
                            for (size_t i = 0; i < k; ++i) {
                                e[i] = std::exp((sign * x[i] - top) / gamma[a]);
                                s0 += e[i]; s1 += x[i] * e[i];
                            }
                            double mean = s1 / s0;
                            for (size_t i = 0; i < k; ++i)
                                pin_grad[3 * (p0 + i) + a] += sign * weights[a] * (1 + sign * (x[i] - mean) / gamma[a]) * e[i] / s0;
                        }
                    }
                }
            });
            std::fill(grad, grad + 3 * (size_t)m, 0.0);
            for (size_t q = 0; q < pin_comp.size(); ++q)
                for (int a = 0; a < 3; ++a) grad[3 * pin_comp[q] + a] += pin_grad[3 * q + a];
        }
        double hpwl(const double* pos) {
            const int chunks = 64;
            part.assign(chunks, 0.0);
            int nn = n_nets();
            parallel(chunks, [&](int c0, int c1) {
                for (int c = c0; c < c1; ++c) {
                    double sum = 0;
                    for (int net = (int)((int64_t)nn * c / chunks); net < (int)((int64_t)nn * (c + 1) / chunks); ++net) {
                        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
                        for (int64_t q = net_start[net]; q < net_start[net + 1]; ++q)
                            for (int a = 0; a < 3; ++a) {
                                double v = pos[3 * pin_comp[q] + a] + pin_off[3 * q + a];
                                lo[a] = std::min(lo[a], v); hi[a] = std::max(hi[a], v);
                            }
                        for (int a = 0; a < 3; ++a) sum += (hi[a] - lo[a]) * weights[a];
                    }
                    part[c] = sum;
                }
            });
            double total = 0;
            for (double v : part) total += v;
            return total;
        }
    };

    // Exact weighted HPWL and median targets over a netlist's pins (poisson._NetCost).
    struct Nets {
        std::vector<int64_t> pin_comp, net_start;  // net_start has n_nets + 1 entries
        std::vector<int64_t> pin_off;              // 3 per pin
        std::vector<int64_t> comp_net_start, comp_nets;
        double* pos = nullptr;                     // 3 per component
        std::vector<double> ends;

        double of(int i) const {
            double total = 0;
            for (int64_t k = comp_net_start[i]; k < comp_net_start[i + 1]; ++k) {
                int64_t net = comp_nets[k];
                double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
                for (int64_t q = net_start[net]; q < net_start[net + 1]; ++q)
                    for (int a = 0; a < 3; ++a) {
                        double v = pos[3 * pin_comp[q] + a] + pin_off[3 * q + a];
                        lo[a] = std::min(lo[a], v); hi[a] = std::max(hi[a], v);
                    }
                total += (hi[0] - lo[0]) + 2 * (hi[1] - lo[1]) + (hi[2] - lo[2]);
            }
            return total;
        }
        void median_target(int i, int64_t* out) {
            int64_t first = comp_net_start[i], last = comp_net_start[i + 1];
            for (int a = 0; a < 3; ++a) {
                ends.clear();
                for (int64_t k = first; k < last; ++k) {
                    int64_t net = comp_nets[k];
                    double lo = 1e300, hi = -1e300;
                    int64_t own = -1;
                    bool other = false;
                    for (int64_t q = net_start[net]; q < net_start[net + 1]; ++q) {
                        if (pin_comp[q] == i) { if (own < 0) own = q; continue; }
                        other = true;
                        double v = pos[3 * pin_comp[q] + a] + pin_off[3 * q + a];
                        lo = std::min(lo, v); hi = std::max(hi, v);
                    }
                    if (!other) continue;
                    double o = (double)pin_off[3 * own + a];
                    ends.push_back(lo - o); ends.push_back(hi - o);
                }
                if (ends.empty()) { out[a] = (int64_t)pos[3 * i + a]; continue; }
                size_t m = ends.size();
                std::sort(ends.begin(), ends.end());
                double med = (m % 2) ? ends[m / 2] : (ends[m / 2 - 1] + ends[m / 2]) / 2.0;
                out[a] = (int64_t)std::nearbyint(med);  // round half to even, as Python's round
            }
        }
    };
    """
    cdef cppclass Occ:
        void init(const vector[int64_t]&)
        void add(int, const int64_t*)
        void remove(int)
        void conflicts(const int64_t*, const int64_t*, int)
        cbool nearest_free(int, const int64_t*, cbool, int64_t, int64_t*)
        vector[int] found
        vector[char] present
        vector[int64_t] origin
    cdef cppclass FieldKernel:
        int nx, ny, nz, threads
        double h[3]
        void deposit(int, const double*, const double*, const double*, const int*, double*) nogil
        void fields(const double*, double*, double*, double*) nogil
        double overflow(const double*, const double*, size_t) nogil
        void axpb(double*, const double*, const double*, size_t) nogil
        void gather(int, const double*, const double*, const double*, const int*,
                    const double*, const double*, const double*, double*) nogil
    cdef cppclass WireKernel:
        vector[int64_t] pin_comp, net_start
        vector[double] pin_off, weights
        int threads
        void gradient(const double*, int, const double*, double*) nogil
        double hpwl(const double*) nogil
    cdef cppclass Nets:
        vector[int64_t] pin_comp, net_start, pin_off, comp_net_start, comp_nets
        double* pos
        double of(int)
        void median_target(int, int64_t*)


cdef vector[int64_t] _whole(sizes) except *:
    arr = np.asarray(sizes, dtype=float).reshape(-1, 3)
    whole = np.floor(arr)
    if not np.array_equal(whole, arr):
        raise ValueError("box sizes must be whole numbers")
    return whole.astype(np.int64).ravel().tolist()


cdef class Occupancy:
    """poisson._Occupancy: non-overlapping integer boxes; same methods, same answers."""
    cdef Occ c

    def __init__(self, sizes):
        self.c.init(_whole(sizes))

    def add(self, int i, p):
        cdef int64_t q[3]
        q[0], q[1], q[2] = int(p[0]), int(p[1]), int(p[2])
        self.c.add(i, q)

    def remove(self, int i):
        self.c.remove(i)

    @property
    def origin(self):
        return {i: (self.c.origin[3 * i], self.c.origin[3 * i + 1], self.c.origin[3 * i + 2])
                for i in range(self.c.present.size()) if self.c.present[i]}

    def conflicts(self, p, s, int skip=-1):
        cdef int64_t q[3]
        cdef int64_t t[3]
        q[0], q[1], q[2] = int(p[0]), int(p[1]), int(p[2])
        t[0], t[1], t[2] = int(s[0]), int(s[1]), int(s[2])
        self.c.conflicts(q, t, skip)
        return sorted(self.c.found)

    def nearest_free(self, int i, target, bint flat, int64_t max_pops):
        cdef int64_t g[3]
        cdef int64_t out[3]
        g[0], g[1], g[2] = int(target[0]), int(target[1]), int(target[2])
        if self.c.nearest_free(i, g, flat, max_pops, out):
            return (out[0], out[1], out[2])
        return None


def refine(Occupancy occ, double[:, ::1] pos, pin_comp, pin_offset, net_start, comp_nets, comp_net_start,
           skip, bint flat, int passes, int64_t pops, status=None):
    """poisson._refine's passes over gates 0..len(skip)-1, moving pos in place; occ holds every unskipped gate."""
    cdef Nets nets
    cdef int n = len(skip)
    cdef int i, p, moved
    cdef int64_t target[3]
    cdef int64_t now[3]
    cdef int64_t spot[3]
    cdef double before
    cdef vector[char] skipped = np.asarray(skip, dtype=np.int8).tolist()
    nets.pin_comp = np.asarray(pin_comp, dtype=np.int64).tolist()
    nets.net_start = np.asarray(net_start, dtype=np.int64).tolist()
    nets.pin_off = np.asarray(pin_offset, dtype=np.int64).ravel().tolist()
    nets.comp_nets = np.asarray(comp_nets, dtype=np.int64).tolist()
    nets.comp_net_start = np.asarray(comp_net_start, dtype=np.int64).tolist()
    nets.pos = &pos[0, 0]
    for p in range(passes):
        if status is not None:
            status(f"refining, pass {p + 1}/{passes}")
        moved = 0
        for i in range(n):
            if skipped[i]:
                continue
            nets.median_target(i, target)
            now[0], now[1], now[2] = <int64_t>pos[i, 0], <int64_t>pos[i, 1], <int64_t>pos[i, 2]
            if flat:
                target[1] = now[1]
            elif target[1] < 0:
                target[1] = 0
            if target[2] < 0:
                target[2] = 0
            if target[0] == now[0] and target[1] == now[1] and target[2] == now[2]:
                continue
            if not occ.c.nearest_free(i, target, flat, pops, spot):
                continue
            if spot[0] == now[0] and spot[1] == now[1] and spot[2] == now[2]:
                continue
            before = nets.of(i)
            pos[i, 0], pos[i, 1], pos[i, 2] = spot[0], spot[1], spot[2]
            if nets.of(i) < before - 1e-9:
                occ.c.remove(i)
                occ.c.add(i, spot)
                moved += 1
            else:
                pos[i, 0], pos[i, 1], pos[i, 2] = now[0], now[1], now[2]
        if not moved:
            break


cdef class Field:
    """_Field.forces's deposit and gather over one bin grid; boxes in the order given."""
    cdef FieldKernel k
    cdef double[:, ::1] eff
    cdef double[::1] scale
    cdef int[:, ::1] span
    cdef double[:, :, ::1] f0, f1, f2

    def __init__(self, shape, h, eff, scale, span, int threads):
        self.k.nx, self.k.ny, self.k.nz = int(shape[0]), int(shape[1]), int(shape[2])
        self.k.h[0], self.k.h[1], self.k.h[2] = float(h[0]), float(h[1]), float(h[2])
        self.k.threads = threads
        self.eff = np.ascontiguousarray(eff, dtype=np.float64)
        self.scale = np.ascontiguousarray(scale, dtype=np.float64)
        self.span = np.ascontiguousarray(span, dtype=np.int32)
        if np.asarray(self.span).max(initial=0) > 64:
            raise ValueError("a box spans more than 64 bins")
        self.f0 = np.empty(tuple(shape))
        self.f1 = np.empty(tuple(shape))
        self.f2 = np.empty(tuple(shape))

    def deposit(self, lo):
        """Charge per bin for boxes with these lower corners."""
        cdef double[:, ::1] l = np.ascontiguousarray(lo, dtype=np.float64)
        out = np.empty((self.k.nx, self.k.ny, self.k.nz))
        cdef double[:, :, ::1] held = out
        with nogil:
            self.k.deposit(l.shape[0], &l[0, 0], &self.eff[0, 0], &self.scale[0], &self.span[0, 0], &held[0, 0, 0])
        return out

    def gather(self, lo, phi):
        """Force on each box from potential phi."""
        cdef double[:, ::1] l = np.ascontiguousarray(lo, dtype=np.float64)
        cdef double[:, :, ::1] p = np.ascontiguousarray(phi, dtype=np.float64)
        out = np.empty((l.shape[0], 3))
        cdef double[:, ::1] force = out
        with nogil:
            self.k.fields(&p[0, 0, 0], &self.f0[0, 0, 0], &self.f1[0, 0, 0], &self.f2[0, 0, 0])
            self.k.gather(l.shape[0], &l[0, 0], &self.eff[0, 0], &self.scale[0], &self.span[0, 0],
                          &self.f0[0, 0, 0], &self.f1[0, 0, 0], &self.f2[0, 0, 0], &force[0, 0])
        return out

    def overflow(self, held, capacity):
        """Charge over capacity, summed over bins."""
        cdef double[:, :, ::1] h = held
        cdef double[:, :, ::1] c = capacity
        cdef double out
        with nogil:
            out = self.k.overflow(&h[0, 0, 0], &c[0, 0, 0], <size_t>(self.k.nx * self.k.ny * self.k.nz))
        return out

    def axpb(self, x, a, b):
        """x = x * a + b, in place."""
        cdef double[:, :, ::1] xv = x
        cdef double[:, :, ::1] av = a
        cdef double[:, :, ::1] bv = b
        with nogil:
            self.k.axpb(&xv[0, 0, 0], &av[0, 0, 0], &bv[0, 0, 0], <size_t>(self.k.nx * self.k.ny * self.k.nz))


cdef class Wire:
    """_Netlist.wa_gradient and hpwl over fixed pins; pins come net after net."""
    cdef WireKernel k

    def __init__(self, pin_comp, pin_offset, net_start, weights, int threads):
        self.k.pin_comp = np.asarray(pin_comp, dtype=np.int64).tolist()
        self.k.pin_off = np.asarray(pin_offset, dtype=np.float64).ravel().tolist()
        self.k.net_start = np.asarray(net_start, dtype=np.int64).tolist()
        self.k.weights = np.asarray(weights, dtype=np.float64).tolist()
        self.k.threads = threads

    def gradient(self, pos, gamma):
        cdef double[:, ::1] p = np.ascontiguousarray(pos, dtype=np.float64)
        cdef double[::1] g = np.ascontiguousarray(gamma, dtype=np.float64)
        out = np.empty((p.shape[0], 3))
        cdef double[:, ::1] o = out
        with nogil:
            self.k.gradient(&p[0, 0], p.shape[0], &g[0], &o[0, 0])
        return out

    def hpwl(self, pos):
        cdef double[:, ::1] p = np.ascontiguousarray(pos, dtype=np.float64)
        cdef double out
        with nogil:
            out = self.k.hpwl(&p[0, 0])
        return out
