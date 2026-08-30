// _core.cpp -- C++ core for the Sparse eXtra Ensemble (SXE).
//
// Ported from the original sxe.cpp with three changes for packaging:
//   * OpenMP replaced by a persistent std::thread pool (no libomp to ship,
//     an explicit n_jobs, and no oversubscription fight with joblib)
//   * pybind11 boundary instead of a 27-positional-argument ctypes ABI:
//     array layout and dtype are enforced at the boundary, shape mismatches
//     raise Python exceptions rather than reading out of bounds
//   * serialization, so a fitted model survives pickling
//
// The growth / pruning / export logic is unchanged from the reference
// implementation, so results reproduce bit-for-bit.  Per-tree RNG seeds
// derive from splitmix64(seed + c*tree_index), which is why the output is
// independent of how many threads happen to run.
//
// Algorithm:
//   * trees grown on pre-binned (uint8, <=256 bins) column-major features
//   * extremely randomized splits: K random candidate features per node,
//     random bin cut-points (or the exact best cut) evaluated from per-node
//     256-bin histograms -- no sorting, no search over raw values
//   * per-tree random subsample without replacement; the out-of-subsample
//     rows are that tree's private holdout
//   * holdout reduced-error pruning with a complexity penalty:
//     collapse when  SSE_leaf <= SSE_subtree + lambda*varh*(leaves-1)
//   * batch boosting: small batches of trees grown in parallel against the
//     current residual, added with shrinkage, early-stopped on a val fold
//
// The convex non-negative weight fit stays in Python (negligible cost).

#include <atomic>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------
// thread pool
//
// Replaces `#pragma omp parallel for schedule(dynamic, 1)`.  Workers pull
// item indices off an atomic counter, which is the same dynamic-chunk-1
// schedule.  The pool is created once per fit rather than per batch: the
// California configuration runs ~1200 batches, and re-spawning threads that
// often is measurable.  The calling thread participates in the work.
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(int n_extra_workers) {
        workers_.reserve(n_extra_workers > 0 ? (size_t)n_extra_workers : 0);
        for (int i = 0; i < n_extra_workers; ++i)
            workers_.emplace_back([this] { loop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_start_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int n_threads() const { return (int)workers_.size() + 1; }

    // Run fn(i) for i in [0, n_items) and block until all have completed.
    void run(int64_t n_items, const std::function<void(int64_t)>& fn) {
        if (workers_.empty() || n_items <= 1) {
            for (int64_t i = 0; i < n_items; ++i) fn(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_ = &fn;
            n_ = n_items;
            next_.store(0, std::memory_order_relaxed);
            running_ = (int)workers_.size();
            err_ = nullptr;
            ++epoch_;
        }
        cv_start_.notify_all();
        drain();  // the caller is a worker too
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [this] { return running_ == 0; });
        fn_ = nullptr;
        if (err_) {
            std::exception_ptr e = err_;
            err_ = nullptr;
            std::rethrow_exception(e);
        }
    }

private:
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_start_.wait(lk, [&] { return stop_ || epoch_ != seen; });
                if (stop_) return;
                seen = epoch_;
            }
            drain();
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--running_ == 0) cv_done_.notify_all();
            }
        }
    }

    // Pull items until the queue is empty.  On an exception, record the first
    // one and poison the counter so the remaining workers stop promptly.
    void drain() {
        for (;;) {
            const int64_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_) break;
            try {
                (*fn_)(i);
            } catch (...) {
                std::lock_guard<std::mutex> lk(mu_);
                if (!err_) err_ = std::current_exception();
                next_.store(n_, std::memory_order_relaxed);
                break;
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_start_, cv_done_;
    const std::function<void(int64_t)>* fn_ = nullptr;
    int64_t n_ = 0;
    std::atomic<int64_t> next_{0};
    int running_ = 0;
    uint64_t epoch_ = 0;
    bool stop_ = false;
    std::exception_ptr err_;
};

int resolve_threads(int n_jobs) {
    if (n_jobs > 0) return n_jobs;
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    if (n_jobs == 0) return 1;
    // negative, numpy/joblib convention: -1 = all, -2 = all but one, ...
    const int want = (int)hc + 1 + n_jobs;
    return std::max(1, want);
}

// ---------------------------------------------------------------------------
// tree representation
// ---------------------------------------------------------------------------
struct Tree {
    // compact arrays; feat = -1 marks a leaf
    std::vector<int32_t> feat, left, right;
    std::vector<uint8_t> thr;
    std::vector<double>  val;    // nout values per node: val[id*nout + k]
    std::vector<double>  gain;   // split gain at internal nodes, 0 at leaves
    int32_t nout = 1;
    int32_t n_leaves = 0;
    int32_t base_id = -1;   // which grown tree this member came from
    int32_t variant = 0;    // 0 raw, 2 random lambda
    // Leaf values ship as 16-bit codes: val = leaf_a + leaf_b * q. Set by
    // quantize_leaves(), which also rounds `val` in place so the model in
    // memory is the model on disk.
    bool   quantized = false;
    double leaf_a = 0.0, leaf_b = 0.0;
};

struct GrownTree {
    // full grown tree + per-node training stats + holdout stats
    std::vector<int32_t> feat, left, right;
    std::vector<uint8_t> thr;
    std::vector<double>  val;                       // train means per node (K each)
    std::vector<double>  gain;
    std::vector<double>  h_cnt, h_sum, h_ssq;       // holdout stats per node:
                                                    // cnt, K sums, one total ssq
    std::vector<int64_t> lo, hi;                    // node's row range in ws.idx
};

struct Hist {
    // all-feature histogram: cnt laid out as [feature*256 + bin], sums as
    // [(feature*256 + bin)*K + k] for K target columns
    std::vector<int32_t> cnt;
    std::vector<double>  sum;    // exact residual sums, or
    std::vector<int32_t> qsum;   // sums of the stochastically rounded residual
    void reset(int p, int K, bool quant) {
        cnt.assign((size_t)p * 256, 0);
        if (quant) qsum.assign((size_t)p * 256 * (size_t)K, 0);
        else sum.assign((size_t)p * 256 * (size_t)K, 0.0);
    }
};

struct Workspace {
    std::vector<int64_t> idx;        // sample indices being partitioned
    std::vector<int>     featbuf;    // for sampling K distinct features
    std::vector<int32_t> hcnt;       // small-node histogram: counts
    std::vector<double>  hsum;       // small-node histogram: sums
    std::vector<int>     touched;    // bins touched (for cheap clearing)
    std::vector<Hist*>   pool;       // reusable all-feature histograms
    std::vector<double>  ybuf;       // node-local y cache
    std::vector<int16_t> qbuf;       // node-local q cache
    std::vector<double>  sumk, meank, sl;  // per-target scratch (K each)
    std::vector<double>  sumq;       // per-target sums of q over the node
    std::vector<int32_t> hqsum;      // small-node histogram: quantized sums
    std::vector<int16_t> q;          // quantized residual, n*K, when quant
    bool                 quant = false;
    int                  nout = 1;   // target columns K (1 = scalar target)
    int                  hybrid_min = 600;  // node size above which the
                                            // subtraction scheme is used
    std::mt19937_64      rng;

    Hist* acquire(int p) {
        Hist* h;
        if (pool.empty()) h = new Hist();
        else { h = pool.back(); pool.pop_back(); }
        h->reset(p, nout, quant);
        return h;
    }
    void release(Hist* h) { pool.push_back(h); }
    ~Workspace() { for (Hist* h : pool) delete h; }
};

struct TreeParams {
    int max_depth, min_leaf, K, n_thresh;
    int argmax_bins = 0;   // 1 = evaluate the BEST cut per candidate feature
    int max_leaves = 0;    // >0 = leaf-wise (best-first) growth with this budget
    double lam_lo, lam_hi;
};

// ---------------------------------------------------------------------------
// growth
// ---------------------------------------------------------------------------
// Variance-reduction gain of a cut with (cl, sl[k]) on the left, summed over
// the K targets. With K == 1 this is the scalar expression, operation for
// operation.
inline double split_gain(int K, int64_t cl, int64_t cr, const double* sl,
                         const double* sumk, double parent_term)
{
    double g = 0.0;
    for (int k = 0; k < K; ++k) {
        const double sr = sumk[k] - sl[k];
        g += sl[k] * sl[k] / (double)cl + sr * sr / (double)cr;
    }
    return g - parent_term;
}

// Node statistics over rows idx[lo, hi): per-target sums, total sum of
// squares, per-target means and the parent term sum_k S_k^2 / m.
inline double node_stats(const double* y, int K, const std::vector<int64_t>& idx,
                         int64_t lo, int64_t hi, double* sumk, double* meank,
                         double& parent_term)
{
    const int64_t m = hi - lo;
    for (int k = 0; k < K; ++k) sumk[k] = 0.0;
    double ssq = 0.0;
    if (K == 1) {
        double sum = 0.0;
        for (int64_t i = lo; i < hi; ++i) { const double v = y[idx[i]]; sum += v; ssq += v * v; }
        sumk[0] = sum;
    } else {
        for (int64_t i = lo; i < hi; ++i) {
            const double* yi = y + idx[i] * K;
            for (int k = 0; k < K; ++k) { const double v = yi[k]; sumk[k] += v; ssq += v * v; }
        }
    }
    parent_term = 0.0;
    for (int k = 0; k < K; ++k) {
        meank[k] = sumk[k] / (double)m;
        parent_term += sumk[k] * sumk[k] / (double)m;
    }
    return ssq;
}

// Sums of the quantized residual over a node, and the parent term they
// imply. Split gains are homogeneous of degree two in the sums, so with
// every term on the same integer scale the argmax is the argmax of the
// exact gain up to the rounding (Shi et al., NeurIPS 2022).
inline void node_qsums(const int16_t* q, int K, const std::vector<int64_t>& idx,
                       int64_t lo, int64_t hi, double* sumq, double& parent_term)
{
    const int64_t m = hi - lo;
    for (int k = 0; k < K; ++k) sumq[k] = 0.0;
    if (K == 1) {
        int64_t acc = 0;
        for (int64_t i = lo; i < hi; ++i) acc += q[idx[i]];
        sumq[0] = (double)acc;
    } else {
        for (int64_t i = lo; i < hi; ++i) {
            const int16_t* qi = q + idx[i] * K;
            for (int k = 0; k < K; ++k) sumq[k] += (double)qi[k];
        }
    }
    parent_term = 0.0;
    for (int k = 0; k < K; ++k) parent_term += sumq[k] * sumq[k] / (double)m;
}

inline bool node_pure(int K, double ssq, const double* sumk, const double* meank) {
    double sm = 0.0;
    for (int k = 0; k < K; ++k) sm += sumk[k] * meank[k];
    return (ssq - sm) <= 1e-12 * (std::fabs(ssq) + 1e-12);
}

inline int32_t push_node(GrownTree& T, int K, const double* meank, int64_t lo, int64_t hi) {
    const int32_t id = (int32_t)T.feat.size();
    T.feat.push_back(-1); T.thr.push_back(0);
    T.left.push_back(-1); T.right.push_back(-1);
    for (int k = 0; k < K; ++k) T.val.push_back(meank[k]);
    T.gain.push_back(0.0);
    T.lo.push_back(lo); T.hi.push_back(hi);
    return id;
}

int32_t grow_node_legacy(GrownTree& T, const uint8_t* X, const double* y,
                  int64_t n, int p, Workspace& ws,
                  int64_t lo, int64_t hi, int depth, const TreeParams& tp)
{
    const int64_t m = hi - lo;
    const int Kt = ws.nout;
    double* sumk = ws.sumk.data(); double* meank = ws.meank.data(); double* sl = ws.sl.data();
    double parent_term;
    const double ssq = node_stats(y, Kt, ws.idx, lo, hi, sumk, meank, parent_term);
    const int32_t id = push_node(T, Kt, meank, lo, hi);

    if (depth >= tp.max_depth || m < 2 * (int64_t)tp.min_leaf || node_pure(Kt, ssq, sumk, meank))
        return id;

    // sample K distinct candidate features (partial Fisher-Yates)
    for (int j = 0; j < p; ++j) ws.featbuf[j] = j;
    const int K = std::min(tp.K, p);

    int best_f = -1, best_t = -1;
    double best_gain = -1.0;
    // with quantized sums the gains are on the q scale: node sums to match
    if (ws.quant) {
        node_qsums(ws.q.data(), Kt, ws.idx, lo, hi, ws.sumq.data(), parent_term);
        sumk = ws.sumq.data();
    }

    for (int c = 0; c < K; ++c) {
        std::uniform_int_distribution<int> pick(c, p - 1);
        std::swap(ws.featbuf[c], ws.featbuf[pick(ws.rng)]);
        const int f = ws.featbuf[c];
        const uint8_t* col = X + (int64_t)f * n;

        // build per-node histogram for this feature
        for (int b : ws.touched) {
            ws.hcnt[b] = 0;
            for (int k = 0; k < Kt; ++k) {
                if (ws.quant) ws.hqsum[(size_t)b * Kt + k] = 0;
                else ws.hsum[(size_t)b * Kt + k] = 0.0;
            }
        }
        ws.touched.clear();
        int minb = 256, maxb = -1;
        if (ws.quant) {
            const int16_t* q = ws.q.data();
            for (int64_t i = lo; i < hi; ++i) {
                const int b = col[ws.idx[i]];
                if (!ws.hcnt[b]) ws.touched.push_back(b);
                ws.hcnt[b]++;
                const int16_t* qi = q + ws.idx[i] * Kt;
                for (int k = 0; k < Kt; ++k) ws.hqsum[(size_t)b * Kt + k] += qi[k];
                if (b < minb) minb = b;
                if (b > maxb) maxb = b;
            }
        } else if (Kt == 1) {
            for (int64_t i = lo; i < hi; ++i) {
                const int b = col[ws.idx[i]];
                if (!ws.hcnt[b]) ws.touched.push_back(b);
                ws.hcnt[b]++; ws.hsum[b] += y[ws.idx[i]];
                if (b < minb) minb = b;
                if (b > maxb) maxb = b;
            }
        } else {
            for (int64_t i = lo; i < hi; ++i) {
                const int b = col[ws.idx[i]];
                if (!ws.hcnt[b]) ws.touched.push_back(b);
                ws.hcnt[b]++;
                const double* yi = y + ws.idx[i] * Kt;
                for (int k = 0; k < Kt; ++k) ws.hsum[(size_t)b * Kt + k] += yi[k];
                if (b < minb) minb = b;
                if (b > maxb) maxb = b;
            }
        }
        if (maxb <= minb) continue;  // constant feature in node
        if (ws.quant) {
            // scan the quantized 256-bin slice through the shared template:
            // one feature, sums on the q scale
            const int32_t* hq = ws.hqsum.data();
            if (tp.argmax_bins) {
                int64_t cl = 0; double* sl = ws.sl.data();
                for (int k = 0; k < Kt; ++k) sl[k] = 0.0;
                for (int t = minb; t < maxb; ++t) {
                    cl += ws.hcnt[t];
                    for (int k = 0; k < Kt; ++k) sl[k] += (double)hq[(size_t)t * Kt + k];
                    const int64_t cr = m - cl;
                    if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                    const double gain = split_gain(Kt, cl, cr, sl, sumk, parent_term);
                    if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
                }
            } else {
                std::uniform_int_distribution<int> tdist(minb, maxb - 1);
                for (int r = 0; r < tp.n_thresh; ++r) {
                    const int t = tdist(ws.rng);
                    int64_t cl = 0; double* sl = ws.sl.data();
                    for (int k = 0; k < Kt; ++k) sl[k] = 0.0;
                    for (int b = minb; b <= t; ++b) {
                        cl += ws.hcnt[b];
                        for (int k = 0; k < Kt; ++k) sl[k] += (double)hq[(size_t)b * Kt + k];
                    }
                    const int64_t cr = m - cl;
                    if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                    const double gain = split_gain(Kt, cl, cr, sl, sumk, parent_term);
                    if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
                }
            }
            continue;
        }

        if (tp.argmax_bins && Kt == 1) {
            const double sum = sumk[0];
            int64_t cl = 0; double s1 = 0.0;
            for (int t = minb; t < maxb; ++t) {
                cl += ws.hcnt[t]; s1 += ws.hsum[t];
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double sr = sum - s1;
                const double gain = s1 * s1 / (double)cl + sr * sr / (double)cr - parent_term;
                if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
            }
        } else if (tp.argmax_bins) {
            int64_t cl = 0;
            for (int k = 0; k < Kt; ++k) sl[k] = 0.0;
            for (int t = minb; t < maxb; ++t) {
                cl += ws.hcnt[t];
                for (int k = 0; k < Kt; ++k) sl[k] += ws.hsum[(size_t)t * Kt + k];
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double gain = split_gain(Kt, cl, cr, sl, sumk, parent_term);
                if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
            }
        } else {
            std::uniform_int_distribution<int> tdist(minb, maxb - 1);
            for (int r = 0; r < tp.n_thresh; ++r) {
                const int t = tdist(ws.rng);
                int64_t cl = 0;
                for (int k = 0; k < Kt; ++k) sl[k] = 0.0;
                for (int b = minb; b <= t; ++b) {
                    cl += ws.hcnt[b];
                    for (int k = 0; k < Kt; ++k) sl[k] += ws.hsum[(size_t)b * Kt + k];
                }
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double gain = split_gain(Kt, cl, cr, sl, sumk, parent_term);
                if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
            }
        }
    }
    if (best_f < 0) return id;

    // partition indices in place
    const uint8_t* col = X + (int64_t)best_f * n;
    int64_t* base = ws.idx.data();
    int64_t* mid = std::stable_partition(base + lo, base + hi,
                                  [&](int64_t r) { return col[r] <= (uint8_t)best_t; });
    const int64_t split = mid - base;
    if (split == lo || split == hi) return id;  // safety

    T.feat[id] = best_f; T.thr[id] = (uint8_t)best_t; T.gain[id] = best_gain;

    // Smaller child first, matching grow_node_full. The two growers pick splits
    // identically but draw from one shared per-tree RNG stream, so recursing in
    // different orders would make the node a given draw lands on depend on
    // ws.hybrid_min -- a performance knob silently changing the fitted model.
    const bool left_small = (split - lo) <= (hi - split);
    const int64_t slo = left_small ? lo : split, shi = left_small ? split : hi;
    const int64_t llo = left_small ? split : lo, lhi = left_small ? hi : split;
    const int32_t id_small = grow_node_legacy(T, X, y, n, p, ws, slo, shi, depth + 1, tp);
    const int32_t id_large = grow_node_legacy(T, X, y, n, p, ws, llo, lhi, depth + 1, tp);
    T.left[id]  = left_small ? id_small : id_large;
    T.right[id] = left_small ? id_large : id_small;
    return id;
}

// ---------------------------------------------------------------------------
// full-histogram growth with the parent-minus-child subtraction trick
// (used for nodes >= ws.hybrid_min; smaller subtrees fall back to the
// per-candidate touched-bin scheme above, which has no memset overhead)
// ---------------------------------------------------------------------------
void fill_hist(Hist& H, const uint8_t* X, const double* y, int64_t n, int p,
               const std::vector<int64_t>& idx, int64_t lo, int64_t hi,
               std::vector<double>& ybuf, int K, const int16_t* q = nullptr,
               std::vector<int16_t>* qb = nullptr)
{
    const int64_t m = hi - lo;
    const int16_t* qbuf = nullptr;
    if (!H.qsum.empty()) {
        qb->resize((size_t)m * K);
        for (int64_t i = 0; i < m; ++i)
            for (int k = 0; k < K; ++k) (*qb)[(size_t)i * K + k] = q[idx[lo + i] * K + k];
        qbuf = qb->data();
    } else {
        ybuf.resize((size_t)m * K);
        for (int64_t i = 0; i < m; ++i)
            for (int k = 0; k < K; ++k) ybuf[(size_t)i * K + k] = y[idx[lo + i] * K + k];
    }
    for (int f = 0; f < p; ++f) {
        const uint8_t* col = X + (int64_t)f * n;
        int32_t* c = H.cnt.data() + (size_t)f * 256;
        if (!H.qsum.empty()) {
            // quantized sums: integer adds, order-independent by construction
            int32_t* qs = H.qsum.data() + (size_t)f * 256 * K;
            const int16_t* q = qbuf;
            for (int64_t i = 0; i < m; ++i) {
                const int b = col[idx[lo + i]];
                c[b]++;
                for (int k = 0; k < K; ++k) qs[(size_t)b * K + k] += q[(size_t)i * K + k];
            }
            continue;
        }
        double*  s = H.sum.data() + (size_t)f * 256 * K;
        if (K == 1) {
            for (int64_t i = 0; i < m; ++i) {
                const int b = col[idx[lo + i]];
                c[b]++; s[b] += ybuf[i];
            }
        } else {
            for (int64_t i = 0; i < m; ++i) {
                const int b = col[idx[lo + i]];
                c[b]++;
                for (int k = 0; k < K; ++k) s[(size_t)b * K + k] += ybuf[(size_t)i * K + k];
            }
        }
    }
}

struct SplitEval { double gain = -1.0; int f = -1, t = -1; };

// Best cut over K random candidate features from an all-feature histogram.
// S is the histogram sum type: double for exact sums, int32 for quantized.
template <class S>
SplitEval best_split_scan(const int32_t* cnt, const S* sums, int p, int64_t m,
                          const double* sumk, double parent_term, Workspace& ws,
                          const TreeParams& tp)
{
    SplitEval out;
    const int Kt = ws.nout;
    double* sl = ws.sl.data();
    for (int j = 0; j < p; ++j) ws.featbuf[j] = j;
    const int K = std::min(tp.K, p);
    for (int c = 0; c < K; ++c) {
        std::uniform_int_distribution<int> pick(c, p - 1);
        std::swap(ws.featbuf[c], ws.featbuf[pick(ws.rng)]);
        const int f = ws.featbuf[c];
        const int32_t* hc = cnt + (size_t)f * 256;
        const S*       hs = sums + (size_t)f * 256 * Kt;
        int minb = 0, maxb = 255;
        while (minb < 256 && hc[minb] == 0) ++minb;
        while (maxb >= 0 && hc[maxb] == 0) --maxb;
        if (maxb <= minb) continue;
        if (tp.argmax_bins && Kt == 1) {
            const double sum = sumk[0];
            int64_t cl = 0; double s1 = 0.0;
            for (int t = minb; t < maxb; ++t) {
                cl += hc[t]; s1 += (double)hs[t];
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double sr = sum - s1;
                const double g = s1 * s1 / (double)cl + sr * sr / (double)cr - parent_term;
                if (g > out.gain) { out.gain = g; out.f = f; out.t = t; }
            }
        } else if (tp.argmax_bins) {
            int64_t cl = 0;
            for (int k = 0; k < Kt; ++k) sl[k] = 0.0;
            for (int t = minb; t < maxb; ++t) {
                cl += hc[t];
                for (int k = 0; k < Kt; ++k) sl[k] += (double)hs[(size_t)t * Kt + k];
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double g = split_gain(Kt, cl, cr, sl, sumk, parent_term);
                if (g > out.gain) { out.gain = g; out.f = f; out.t = t; }
            }
        } else {
            std::uniform_int_distribution<int> tdist(minb, maxb - 1);
            for (int r = 0; r < tp.n_thresh; ++r) {
                const int t = tdist(ws.rng);
                int64_t cl = 0;
                for (int k = 0; k < Kt; ++k) sl[k] = 0.0;
                for (int b = minb; b <= t; ++b) {
                    cl += hc[b];
                    for (int k = 0; k < Kt; ++k) sl[k] += (double)hs[(size_t)b * Kt + k];
                }
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double g = split_gain(Kt, cl, cr, sl, sumk, parent_term);
                if (g > out.gain) { out.gain = g; out.f = f; out.t = t; }
            }
        }
    }
    return out;
}

// Dispatch on whether the histogram carries exact or quantized sums. With
// quantized sums the node's own sums must be on the same scale (sumq).
inline SplitEval best_split_from_hist(const Hist& H, int p, int64_t m, const double* sumk,
                                      double parent_term, Workspace& ws, const TreeParams& tp)
{
    if (ws.quant)
        return best_split_scan<int32_t>(H.cnt.data(), H.qsum.data(), p, m, sumk, parent_term, ws, tp);
    return best_split_scan<double>(H.cnt.data(), H.sum.data(), p, m, sumk, parent_term, ws, tp);
}

int32_t grow_node_full(GrownTree& T, const uint8_t* X, const double* y,
                       int64_t n, int p, Workspace& ws,
                       int64_t lo, int64_t hi, int depth, const TreeParams& tp,
                       Hist* H)  // H: this node's all-feature histogram (owned by caller)
{
    const int64_t m = hi - lo;
    const int Kt = ws.nout;
    double* sumk = ws.sumk.data(); double* meank = ws.meank.data();
    double parent_term;
    const double ssq = node_stats(y, Kt, ws.idx, lo, hi, sumk, meank, parent_term);
    const int32_t id = push_node(T, Kt, meank, lo, hi);

    if (depth >= tp.max_depth || m < 2 * (int64_t)tp.min_leaf || node_pure(Kt, ssq, sumk, meank))
        return id;

    if (ws.quant) node_qsums(ws.q.data(), Kt, ws.idx, lo, hi, ws.sumq.data(), parent_term);
    SplitEval e = best_split_from_hist(*H, p, m, ws.quant ? ws.sumq.data() : sumk,
                                       parent_term, ws, tp);
    const int best_f = e.f, best_t = e.t;
    const double best_gain = e.gain;
    if (best_f < 0) return id;

    const uint8_t* col = X + (int64_t)best_f * n;
    int64_t* base = ws.idx.data();
    int64_t* mid = std::stable_partition(base + lo, base + hi,
                                  [&](int64_t r) { return col[r] <= (uint8_t)best_t; });
    const int64_t split = mid - base;
    if (split == lo || split == hi) return id;

    T.feat[id] = best_f; T.thr[id] = (uint8_t)best_t; T.gain[id] = best_gain;

    const bool left_small = (split - lo) <= (hi - split);
    const int64_t slo = left_small ? lo : split, shi = left_small ? split : hi;   // smaller
    const int64_t llo = left_small ? split : lo, lhi = left_small ? hi : split;   // larger

    // build smaller child's histogram from data; larger child = parent - smaller
    Hist* HS = ws.acquire(p);
    fill_hist(*HS, X, y, n, p, ws.idx, slo, shi, ws.ybuf, Kt, ws.q.data(), &ws.qbuf);
    {
        const size_t sz = (size_t)p * 256;
        int32_t* pc = H->cnt.data(); const int32_t* sc = HS->cnt.data();
        for (size_t k = 0; k < sz; ++k) pc[k] -= sc[k];
        if (ws.quant) {
            int32_t* pq = H->qsum.data(); const int32_t* sq = HS->qsum.data();
            for (size_t k = 0; k < sz * (size_t)Kt; ++k) pq[k] -= sq[k];
        } else {
            double* ps = H->sum.data(); const double* ss = HS->sum.data();
            for (size_t k = 0; k < sz * (size_t)Kt; ++k) ps[k] -= ss[k];
        }
    }

    int32_t id_small, id_large;
    if (shi - slo >= ws.hybrid_min)
        id_small = grow_node_full(T, X, y, n, p, ws, slo, shi, depth + 1, tp, HS);
    else
        id_small = grow_node_legacy(T, X, y, n, p, ws, slo, shi, depth + 1, tp);
    ws.release(HS);
    if (lhi - llo >= ws.hybrid_min)
        id_large = grow_node_full(T, X, y, n, p, ws, llo, lhi, depth + 1, tp, H);
    else
        id_large = grow_node_legacy(T, X, y, n, p, ws, llo, lhi, depth + 1, tp);

    T.left[id]  = left_small ? id_small : id_large;
    T.right[id] = left_small ? id_large : id_small;
    return id;
}

// ---------------------------------------------------------------------------
// leaf-wise (best-first) growth: always split the pending leaf with the
// largest gain, LightGBM-style, until the leaf budget is exhausted.
// ---------------------------------------------------------------------------
// A pending leaf carries its best split, not the histogram that found it.
// Keeping the histogram would let the parent-minus-child subtraction serve the
// children, but it costs p*256*12 bytes per pending leaf -- on 500 features
// with a 127-leaf budget that is ~195 MB per tree, times the batch width.
// Rebuilding a child's histogram when it is popped bounds the workspace at one
// live histogram instead, for roughly one extra fill per node.
struct Pending {
    double gain; int32_t id; int64_t lo, hi; int depth; int f, t;
    bool operator<(const Pending& o) const { return gain < o.gain; }
};

void grow_tree_leafwise(GrownTree& T, const uint8_t* X, const double* y,
                        int64_t n, int p, Workspace& ws, int64_t n_sub,
                        const TreeParams& tp)
{
    const int Kt = ws.nout;
    auto make_node = [&](int64_t lo, int64_t hi) -> int32_t {
        double* sumk = ws.sumk.data(); double* meank = ws.meank.data();
        for (int k = 0; k < Kt; ++k) sumk[k] = 0.0;
        for (int64_t i = lo; i < hi; ++i) {
            const double* yi = y + ws.idx[i] * Kt;
            for (int k = 0; k < Kt; ++k) sumk[k] += yi[k];
        }
        for (int k = 0; k < Kt; ++k) meank[k] = sumk[k] / (double)(hi - lo);
        return push_node(T, Kt, meank, lo, hi);
    };
    // per-target sums and the parent term of a node, from its stored means
    auto node_sums = [&](int32_t id, int64_t lo, int64_t hi, double* sumk) {
        double pt = 0.0;
        for (int k = 0; k < Kt; ++k) {
            sumk[k] = T.val[(size_t)id * Kt + k] * (double)(hi - lo);
            pt += sumk[k] * sumk[k] / (double)(hi - lo);
        }
        return pt;
    };

    std::priority_queue<Pending> heap;

    // Evaluate a node's best split from a freshly built histogram and queue it
    // if it clears the depth, size and gain floors. The histogram is released
    // immediately, so at most one is live per workspace.
    auto consider = [&](int32_t cid, int64_t clo, int64_t chi, int depth) {
        const int64_t cm = chi - clo;
        if (depth >= tp.max_depth || cm < 2 * (int64_t)tp.min_leaf) return;
        Hist* H = ws.acquire(p);
        fill_hist(*H, X, y, n, p, ws.idx, clo, chi, ws.ybuf, Kt, ws.q.data(), &ws.qbuf);
        double* sumk = ws.sumk.data();
        double pt = node_sums(cid, clo, chi, sumk);
        if (ws.quant) node_qsums(ws.q.data(), Kt, ws.idx, clo, chi, sumk, pt);
        SplitEval e = best_split_from_hist(*H, p, cm, sumk, pt, ws, tp);
        ws.release(H);
        if (e.gain > 1e-12) heap.push({e.gain, cid, clo, chi, depth, e.f, e.t});
    };

    const int32_t root = make_node(0, n_sub);
    consider(root, 0, n_sub, 0);
    int leaves = 1;
    while (!heap.empty() && leaves < tp.max_leaves) {
        Pending nd = heap.top(); heap.pop();
        const uint8_t* col = X + (int64_t)nd.f * n;
        int64_t* base = ws.idx.data();
        int64_t* mid = std::stable_partition(base + nd.lo, base + nd.hi,
                                      [&](int64_t r) { return col[r] <= (uint8_t)nd.t; });
        const int64_t split = mid - base;
        if (split == nd.lo || split == nd.hi) continue;

        T.feat[nd.id] = nd.f; T.thr[nd.id] = (uint8_t)nd.t; T.gain[nd.id] = nd.gain;
        const int32_t idl = make_node(nd.lo, split);
        const int32_t idr = make_node(split, nd.hi);
        T.left[nd.id] = idl; T.right[nd.id] = idr;
        ++leaves;

        // smaller child first, as both depth-wise growers do, so the per-tree
        // RNG stream is consumed in one order throughout
        const bool left_small = (split - nd.lo) <= (nd.hi - split);
        if (left_small) {
            consider(idl, nd.lo, split, nd.depth + 1);
            consider(idr, split, nd.hi, nd.depth + 1);
        } else {
            consider(idr, split, nd.hi, nd.depth + 1);
            consider(idl, nd.lo, split, nd.depth + 1);
        }
    }
}

// route holdout rows, accumulating (cnt,sum,ssq) at every node on the path,
// and remembering the leaf each row reached so it is never walked again
void route_holdout(GrownTree& T, const uint8_t* X, const double* y, int K,
                   int64_t n, const std::vector<int64_t>& hold,
                   std::vector<int32_t>& leaf_of)
{
    const size_t nn = T.feat.size();
    T.h_cnt.assign(nn, 0.0); T.h_sum.assign(nn * (size_t)K, 0.0); T.h_ssq.assign(nn, 0.0);
    leaf_of.resize(hold.size());
    for (size_t j = 0; j < hold.size(); ++j) {
        const int64_t r = hold[j];
        const double* yr = y + r * K;
        int32_t id = 0;
        for (;;) {
            T.h_cnt[id] += 1.0;
            for (int k = 0; k < K; ++k) {
                const double v = yr[k];
                T.h_sum[(size_t)id * K + k] += v; T.h_ssq[id] += v * v;
            }
            const int32_t f = T.feat[id];
            if (f < 0) break;
            id = (X[(int64_t)f * n + r] <= T.thr[id]) ? T.left[id] : T.right[id];
        }
        leaf_of[j] = id;
    }
}

// holdout SSE of a node predicting its training means
inline double leaf_sse(const GrownTree& T, int K, int32_t id) {
    if (K == 1) {
        const double mu = T.val[id];
        return T.h_ssq[id] - 2.0 * mu * T.h_sum[id] + T.h_cnt[id] * mu * mu;
    }
    double cross = 0.0, sq = 0.0;
    for (int k = 0; k < K; ++k) {
        const double mu = T.val[(size_t)id * K + k];
        cross += mu * T.h_sum[(size_t)id * K + k];
        sq += mu * mu;
    }
    return T.h_ssq[id] - 2.0 * cross + T.h_cnt[id] * sq;
}

// rep[id] = the node that predicts for rows reaching id after pruning: the
// nearest collapsed ancestor, else id itself when it is a grown leaf, else -1.
// Children always have larger ids than their parent, so one ascending pass
// pushes the answer down the tree.
void exported_leaf_of(const GrownTree& T, const std::vector<uint8_t>* collapsed,
                      std::vector<int32_t>& rep)
{
    const int32_t nn = (int32_t)T.feat.size();
    rep.assign(nn, -1);
    for (int32_t id = 0; id < nn; ++id) {
        if (rep[id] < 0 && (T.feat[id] < 0 || (collapsed && (*collapsed)[id]))) rep[id] = id;
        if (T.feat[id] >= 0) { rep[T.left[id]] = rep[id]; rep[T.right[id]] = rep[id]; }
    }
}

// Training-fold prediction without a tree walk: every subsample row sits in
// its grown leaf's range of ws.idx, and every holdout row's leaf was recorded
// while routing. Both read the same leaf value the exported member carries,
// so this equals predict_row on the member, bit for bit.
void predict_train_from_ranges(const GrownTree& T, int K, const std::vector<int32_t>& rep,
                               const std::vector<int64_t>& idx,
                               const std::vector<int64_t>& hold,
                               const std::vector<int32_t>& leaf_of, double* out)
{
    const int32_t nn = (int32_t)T.feat.size();
    for (int32_t id = 0; id < nn; ++id) {
        if (T.feat[id] >= 0) continue;
        const double* v = T.val.data() + (size_t)rep[id] * K;
        for (int64_t i = T.lo[id]; i < T.hi[id]; ++i)
            for (int k = 0; k < K; ++k) out[idx[i] * K + k] = v[k];
    }
    for (size_t j = 0; j < hold.size(); ++j) {
        const double* v = T.val.data() + (size_t)rep[leaf_of[j]] * K;
        for (int k = 0; k < K; ++k) out[hold[j] * K + k] = v[k];
    }
}

// bottom-up collapse decision; pre-order ids => descending id is bottom-up.
// returns per-node "collapsed" flags for penalty lambda (in units of holdout
// variance per extra leaf).
void prune_flags(const GrownTree& T, int K, double lam, double varh,
                 std::vector<uint8_t>& collapsed, std::vector<int32_t>& leaves)
{
    const int32_t nn = (int32_t)T.feat.size();
    std::vector<double> sse_sub(nn);
    collapsed.assign(nn, 0); leaves.assign(nn, 1);
    for (int32_t id = nn - 1; id >= 0; --id) {
        const double sse_leaf = leaf_sse(T, K, id);
        if (T.feat[id] < 0) { sse_sub[id] = sse_leaf; continue; }
        const double child = sse_sub[T.left[id]] + sse_sub[T.right[id]];
        const int32_t lv = leaves[T.left[id]] + leaves[T.right[id]];
        if (sse_leaf <= child + lam * varh * (double)(lv - 1)) {
            collapsed[id] = 1; sse_sub[id] = sse_leaf; leaves[id] = 1;
        } else {
            sse_sub[id] = child; leaves[id] = lv;
        }
    }
}

// Holdout-fitted leaf step size (PaloBoost, Park & Ho 2018, eqs. 12-13).
// A leaf predicts its training mean mu; the multiplier c that minimises the
// holdout SSE of c*mu is sum_h(r)/(n_h*mu). With a prior of strength lam
// pulling c toward 1 (the unshrunk leaf) that is
//     c = (sum_h(r)/mu + lam) / (n_h + lam),
// clipped to [0, 1]: a leaf the holdout contradicts is damped, never
// amplified. Leaves the holdout never visits keep their mean.
void shrink_leaves(GrownTree& T, int K, const std::vector<uint8_t>* collapsed, double lam)
{
    const size_t nn = T.feat.size();
    for (size_t id = 0; id < nn; ++id) {
        const bool leaf = (T.feat[id] < 0) || (collapsed && (*collapsed)[id]);
        if (!leaf) continue;
        if (T.h_cnt[id] <= 0.0) continue;
        double fit;  // the unpenalised holdout-optimal multiplier
        if (K == 1) {
            const double mu = T.val[id];
            if (mu == 0.0) continue;
            fit = T.h_sum[id] / mu;
        } else {
            double cross = 0.0, sq = 0.0;
            for (int k = 0; k < K; ++k) {
                const double mu = T.val[id * K + k];
                cross += mu * T.h_sum[id * K + k]; sq += mu * mu;
            }
            if (sq == 0.0) continue;
            fit = cross / sq;
        }
        double c = (fit + lam) / (T.h_cnt[id] + lam);
        if (c < 0.0) c = 0.0;
        if (c > 1.0) c = 1.0;
        for (int k = 0; k < K; ++k) T.val[id * K + k] *= c;
    }
}

// safe two-pass export (avoids pointer-into-growing-vector issues)
void export_member_safe(const GrownTree& T, int K, const std::vector<uint8_t>* collapsed,
                        Tree& out, int32_t base_id, int32_t variant)
{
    out.base_id = base_id; out.variant = variant; out.nout = K;
    out.feat.clear(); out.thr.clear(); out.left.clear();
    out.right.clear(); out.val.clear(); out.gain.clear();
    // pass 1: pre-order list of kept nodes + mapping
    std::vector<int32_t> order; order.reserve(T.feat.size());
    std::vector<int32_t> map(T.feat.size(), -1);
    std::vector<int32_t> st; st.push_back(0);
    while (!st.empty()) {
        int32_t src = st.back(); st.pop_back();
        map[src] = (int32_t)order.size();
        order.push_back(src);
        const bool leaf = (T.feat[src] < 0) || (collapsed && (*collapsed)[src]);
        if (!leaf) { st.push_back(T.right[src]); st.push_back(T.left[src]); }
    }
    const int32_t nn = (int32_t)order.size();
    out.feat.resize(nn); out.thr.resize(nn); out.left.resize(nn);
    out.right.resize(nn); out.val.resize((size_t)nn * K); out.gain.resize(nn);
    int32_t leaves = 0;
    for (int32_t i = 0; i < nn; ++i) {
        const int32_t src = order[i];
        const bool leaf = (T.feat[src] < 0) || (collapsed && (*collapsed)[src]);
        for (int k = 0; k < K; ++k) out.val[(size_t)i * K + k] = T.val[(size_t)src * K + k];
        if (leaf) {
            out.feat[i] = -1; out.thr[i] = 0;
            out.left[i] = out.right[i] = -1; out.gain[i] = 0.0; ++leaves;
        } else {
            out.feat[i] = T.feat[src]; out.thr[i] = T.thr[src];
            out.left[i] = map[T.left[src]]; out.right[i] = map[T.right[src]];
            out.gain[i] = T.gain[src];
        }
    }
    out.n_leaves = leaves;
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// the K leaf values row r reaches
inline const double* predict_leaf(const Tree& T, const uint8_t* X, int64_t n, int64_t r) {
    int32_t id = 0;
    while (T.feat[id] >= 0)
        id = (X[(int64_t)T.feat[id] * n + r] <= T.thr[id]) ? T.left[id] : T.right[id];
    return T.val.data() + (size_t)id * T.nout;
}
inline double predict_row(const Tree& T, const uint8_t* X, int64_t n, int64_t r) {
    return predict_leaf(T, X, n, r)[0];
}

// ---------------------------------------------------------------------------
// serialization
// ---------------------------------------------------------------------------
// The blob stores what prediction needs and nothing else:
//
//   * `left` is implicit -- export_member_safe emits nodes in pre-order, so a
//     node's left child is always the next id.
//   * `val` on internal nodes and `gain` everywhere are not written (gains
//     are accumulated once at fit time into a p-vector).
//   * feature ids and right-child ids take 2 bytes when the model has at most
//     65535 features / the tree at most 65535 nodes, else 4.
//   * leaf values are 16-bit codes under a per-tree affine map (v3). Each
//     leaf enters a prediction with weight w_b/batch as one of hundreds of
//     summands, so a 1/65535-of-range rounding is far below the noise floor;
//     quantize_leaves() applies the same rounding in memory.
//
// That is 5 bytes per internal node and 2 per leaf on models of ordinary
// width, plus one bit per node saying which is which.
const char kMagic[4] = {'S', 'X', 'E', 'B'};
const uint32_t kFormatVersion = 3;
// Leaf code width: 16 bits for a scalar leaf, 8 for a K-vector leaf, whose
// K entries would otherwise dominate the node cost (K*2 bytes against 5 for
// the split above it). Each code is one of hundreds of shrunk summands, so
// even 1/255 of the tree's leaf range is below the ensemble's noise floor.
inline int leaf_levels(int n_out) { return n_out > 1 ? 255 : 65535; }

inline bool narrow_feat(int n_features) { return n_features >= 0 && n_features <= 65535; }
inline bool narrow_node(size_t nn) { return nn <= 65535; }

inline size_t bitmap_bytes(size_t n_nodes) { return (n_nodes + 7) / 8; }

template <class T>
void put(std::string& buf, const T& v) {
    buf.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
void put_vec(std::string& buf, const std::vector<T>& v) {
    const uint64_t sz = v.size();
    put(buf, sz);
    if (sz) buf.append(reinterpret_cast<const char*>(v.data()), sz * sizeof(T));
}

struct Reader {
    const char* p;
    const char* end;
    void need(size_t k) const {
        if ((size_t)(end - p) < k) throw std::runtime_error("SXE: truncated model blob");
    }
    template <class T> T get() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    template <class T> std::vector<T> get_vec() {
        const uint64_t sz = get<uint64_t>();
        need(sz * sizeof(T));
        std::vector<T> v(sz);
        if (sz) std::memcpy(v.data(), p, sz * sizeof(T));
        p += sz * sizeof(T);
        return v;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// the Python-visible ensemble
// ---------------------------------------------------------------------------
namespace {

using ArrU8F = py::array_t<uint8_t, py::array::f_style | py::array::forcecast>;
using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrI32 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

class Ensemble {
public:
    std::vector<Tree> members;
    int n_features = -1;   // set at fit; -1 until then
    int n_out = 1;         // target columns per leaf (K); 1 for scalar targets

    int n_members() const { return (int)members.size(); }

    // -- validation --------------------------------------------------------
    void check_matrix(const ArrU8F& X, int64_t& n, int& p) const {
        if (X.ndim() != 2)
            throw std::invalid_argument("X must be 2-dimensional, got " +
                                        std::to_string(X.ndim()) + " dimensions");
        n = X.shape(0);
        p = (int)X.shape(1);
        if (n_features >= 0 && p != n_features)
            throw std::invalid_argument(
                "X has " + std::to_string(p) + " features, but this model was fitted with " +
                std::to_string(n_features) + " features");
    }

    std::vector<int32_t> check_members(const ArrI32& mem) const {
        auto m = mem.unchecked<1>();
        std::vector<int32_t> out((size_t)m.shape(0));
        const int32_t total = (int32_t)members.size();
        for (py::ssize_t j = 0; j < m.shape(0); ++j) {
            const int32_t v = m(j);
            if (v < 0 || v >= total)
                throw std::out_of_range("member index " + std::to_string(v) +
                                        " out of range [0, " + std::to_string(total) + ")");
            out[(size_t)j] = v;
        }
        return out;
    }

    // -- batch-boosted dictionary generation --------------------------------
    //
    // Grow `batch` trees in parallel against the current residual, update the
    // residual with a shrunk batch average, early-stop on validation MSE.
    // Each tree emits ONE member: itself, pruned on its own holdout at a
    // per-tree random strength.  The sparse weight fit happens once, in
    // Python, over the whole dictionary afterwards.
    py::tuple boost(const ArrU8F& Xa, const ArrF64& ya,
                    const ArrU8F& Xva, const ArrF64& yva,
                    int batch, int max_batches, double shrink, int patience, double tol,
                    int n_thresh, int argmax_bins, int max_leaves,
                    double sub_lo, double sub_hi,
                    const std::vector<int>& depth_choices,
                    int leaf_lo, int leaf_hi,
                    double feat_lo, double feat_hi,
                    double lam_lo, double lam_hi,
                    double leaf_shrink, int hist_bits,
                    int hybrid_min, uint64_t seed, int n_jobs)
    {
        if (hist_bits < 0 || hist_bits > 15)
            throw std::invalid_argument("hist_bits must be 0 (exact) or in [1, 15]");
        if (Xa.ndim() != 2 || Xva.ndim() != 2)
            throw std::invalid_argument("X and X_val must be 2-dimensional");
        const int64_t n = Xa.shape(0), nv = Xva.shape(0);
        const int p = (int)Xa.shape(1);
        if ((int)Xva.shape(1) != p)
            throw std::invalid_argument("X and X_val must have the same number of features");
        // y is (n,) for a scalar target or (n, K) for K targets fitted jointly
        // (one tree, K-vector leaves, gain summed over the columns)
        if (!(ya.ndim() == 1 || ya.ndim() == 2) || ya.shape(0) != n)
            throw std::invalid_argument("y must have shape (n_samples,) or (n_samples, K)");
        const int K = ya.ndim() == 2 ? (int)ya.shape(1) : 1;
        if (K < 1) throw std::invalid_argument("y must have at least one column");
        if (yva.ndim() != ya.ndim() || yva.shape(0) != nv || (K > 1 && (int)yva.shape(1) != K))
            throw std::invalid_argument("y_val must have the same layout as y with len(y_val) == X_val.shape[0]");
        if (n < 1) throw std::invalid_argument("X must have at least one row");
        if (nv < 1) throw std::invalid_argument("X_val must have at least one row");
        if (p < 1) throw std::invalid_argument("X must have at least one feature");
        if (batch < 1) throw std::invalid_argument("batch_size must be >= 1");
        if (depth_choices.empty()) throw std::invalid_argument("max_depth_choices must be non-empty");
        if (leaf_lo < 1 || leaf_hi < leaf_lo)
            throw std::invalid_argument("min_samples_leaf_range must satisfy 1 <= lo <= hi");
        if (!(lam_lo > 0.0) || lam_hi < lam_lo)
            throw std::invalid_argument("prune_strength_range must satisfy 0 < lo <= hi");

        const uint8_t* X = Xa.data();
        const uint8_t* Xv = Xva.data();
        const double* yv = yva.data();
        const int n_depth = (int)depth_choices.size();

        std::vector<double> r(ya.data(), ya.data() + (size_t)n * K);  // residual (training target)
        std::vector<double> Fv((size_t)nv * K, 0.0);          // generation-ensemble val preds
        std::vector<double> bt((size_t)batch * (size_t)n * K);   // per-tree train predictions
        std::vector<double> bv((size_t)batch * (size_t)nv * K);  // per-tree val predictions

        double best = 1e300;
        int bad = 0, n_batches = 0;
        // Quantized training: each batch the residual is scaled so its largest
        // magnitude maps to 2^(bits-1) - 1, and each tree rounds every row
        // stochastically (unbiased) from its own RNG stream. Split gains are
        // then computed from integer histograms; leaf values stay exact.
        const bool quant = hist_bits > 0;
        const double qmax = quant ? (double)((1 << (hist_bits - 1)) - 1) : 1.0;
        double qscale = 1.0;

        {
            py::gil_scoped_release nogil;
            ThreadPool pool(resolve_threads(n_jobs) - 1);

            for (int b = 0; b < max_batches; ++b) {
                const size_t offset = members.size();
                members.resize(offset + (size_t)batch);
                if (quant) {
                    double amax = 0.0;
                    for (size_t i = 0; i < r.size(); ++i) amax = std::max(amax, std::fabs(r[i]));
                    qscale = amax > 0.0 ? qmax / amax : 1.0;
                }

                pool.run(batch, [&](int64_t t64) {
                    const int t = (int)t64;
                    Workspace ws;
                    ws.rng.seed(splitmix64(seed + 0x9E3779B9ULL * (uint64_t)(b * batch + t + 1)));
                    ws.featbuf.resize(p);
                    ws.nout = K;
                    ws.hcnt.assign(256, 0); ws.hsum.assign((size_t)256 * K, 0.0);
                    ws.sumk.assign(K, 0.0); ws.meank.assign(K, 0.0); ws.sl.assign(K, 0.0);
                    ws.touched.reserve(256);
                    ws.hybrid_min = hybrid_min;

                    std::uniform_real_distribution<double> U(0.0, 1.0);
                    TreeParams tp;
                    int d = depth_choices[(int)(U(ws.rng) * n_depth) % n_depth];
                    tp.max_depth = (d <= 0) ? 64 : d;
                    tp.min_leaf = (int)std::lround(std::exp(
                        std::log((double)leaf_lo) +
                        U(ws.rng) * (std::log((double)leaf_hi) - std::log((double)leaf_lo))));
                    tp.K = std::max(1, (int)std::lround((feat_lo + U(ws.rng) * (feat_hi - feat_lo)) * p));
                    tp.n_thresh = n_thresh;
                    tp.argmax_bins = argmax_bins;

                    const double rate = sub_lo + U(ws.rng) * (sub_hi - sub_lo);
                    // clamp to [1, n]: a zero-row subsample would divide by zero
                    // in the node mean. Only reachable on very small inputs.
                    const int64_t msub = std::min<int64_t>(
                        n, std::max<int64_t>(1, (int64_t)std::llround(rate * (double)n)));
                    std::vector<int64_t> all((size_t)n);
                    for (int64_t i = 0; i < n; ++i) all[(size_t)i] = i;
                    for (int64_t i = 0; i < msub; ++i) {
                        std::uniform_int_distribution<int64_t> pick(i, n - 1);
                        std::swap(all[(size_t)i], all[(size_t)pick(ws.rng)]);
                    }
                    // Sorted row ids, kept sorted by stable partitions below:
                    // every histogram fill and holdout route then gathers
                    // X[f*n + idx] in increasing address order, which the
                    // prefetcher can follow. The row set is the same; only
                    // the order the sums are folded in changes, and that
                    // order is fixed by the data, not by the thread count.
                    ws.idx.assign(all.begin(), all.begin() + msub);
                    std::sort(ws.idx.begin(), ws.idx.end());
                    std::vector<int64_t> hold(all.begin() + msub, all.end());
                    std::sort(hold.begin(), hold.end());

                    if (quant) {
                        // drawn after every hyperparameter so those stay put
                        ws.quant = true;
                        ws.sumq.assign(K, 0.0);
                        ws.hqsum.assign((size_t)256 * K, 0);
                        ws.q.resize((size_t)n * K);
                        for (size_t i = 0; i < (size_t)n * K; ++i) {
                            const double v = r[i] * qscale;
                            const double fl = std::floor(v);
                            const double frac = v - fl;
                            ws.q[i] = (int16_t)(fl + ((U(ws.rng) < frac) ? 1.0 : 0.0));
                        }
                    }

                    GrownTree G;
                    G.feat.reserve(512);
                    if (max_leaves > 0) {
                        tp.max_leaves = max_leaves;
                        grow_tree_leafwise(G, X, r.data(), n, p, ws, msub, tp);
                    } else if (msub >= (int64_t)hybrid_min) {
                        Hist* Hh = ws.acquire(p);
                        fill_hist(*Hh, X, r.data(), n, p, ws.idx, 0, msub, ws.ybuf, K,
                                  ws.q.data(), &ws.qbuf);
                        grow_node_full(G, X, r.data(), n, p, ws, 0, msub, 0, tp, Hh);
                        ws.release(Hh);
                    } else {
                        grow_node_legacy(G, X, r.data(), n, p, ws, 0, msub, 0, tp);
                    }
                    std::vector<int32_t> leaf_of;
                    route_holdout(G, X, r.data(), K, n, hold, leaf_of);
                    const double nh = std::max(1.0, G.h_cnt.empty() ? 1.0 : G.h_cnt[0]);
                    double root_sq = 0.0;
                    for (int k = 0; k < K; ++k) root_sq += G.h_sum[k] * G.h_sum[k] / nh;
                    const double varh = std::max(1e-12, (G.h_ssq[0] - root_sq) / nh);

                    std::vector<uint8_t> collapsed; std::vector<int32_t> leaves, rep;
                    Tree& M = members[offset + (size_t)t];
                    // Small-holdout guard: subtrees the holdout never visits satisfy the
                    // collapse inequality trivially and would prune the tree to a stump.
                    if ((int64_t)hold.size() >= std::max<int64_t>(50, n / 50)) {
                        const double lam = std::exp(std::log(lam_lo) +
                            U(ws.rng) * (std::log(lam_hi) - std::log(lam_lo)));
                        prune_flags(G, K, lam, varh, collapsed, leaves);
                        if (leaf_shrink >= 0.0) shrink_leaves(G, K, &collapsed, leaf_shrink);
                        export_member_safe(G, K, &collapsed, M, b * batch + t, 2);
                        exported_leaf_of(G, &collapsed, rep);
                    } else {
                        export_member_safe(G, K, nullptr, M, b * batch + t, 0);
                        exported_leaf_of(G, nullptr, rep);
                    }

                    // this tree's predictions on train rows come from the leaf
                    // ranges; only the val rows need a walk
                    double* pt = bt.data() + (size_t)t * (size_t)n * K;
                    predict_train_from_ranges(G, K, rep, ws.idx, hold, leaf_of, pt);
                    double* pv = bv.data() + (size_t)t * (size_t)nv * K;
                    for (int64_t i = 0; i < nv; ++i) {
                        const double* lv = predict_leaf(M, Xv, nv, i);
                        for (int k = 0; k < K; ++k) pv[i * K + k] = lv[k];
                    }
                });

                ++n_batches;
                const double step = shrink / (double)batch;
                // Each row folds the batch's trees in the same fixed order
                // whichever thread owns it, so the result does not depend on
                // the chunking.
                {
                    const int64_t chunk = 16384;
                    const int64_t nc_tr = (n + chunk - 1) / chunk, nc_v = (nv + chunk - 1) / chunk;
                    pool.run(nc_tr + nc_v, [&](int64_t c) {
                        if (c < nc_tr) {
                            const int64_t lo = c * chunk * K, hi = std::min(n * K, lo + chunk * K);
                            for (int t = 0; t < batch; ++t) {
                                const double* pt = bt.data() + (size_t)t * (size_t)n * K;
                                for (int64_t i = lo; i < hi; ++i) r[(size_t)i] -= step * pt[i];
                            }
                        } else {
                            const int64_t lo = (c - nc_tr) * chunk * K, hi = std::min(nv * K, lo + chunk * K);
                            for (int t = 0; t < batch; ++t) {
                                const double* pv = bv.data() + (size_t)t * (size_t)nv * K;
                                for (int64_t i = lo; i < hi; ++i) Fv[(size_t)i] += step * pv[i];
                            }
                        }
                    });
                }
                double mse = 0.0;
                for (int64_t i = 0; i < nv * K; ++i) {
                    const double e = yv[i] - Fv[(size_t)i];
                    mse += e * e;
                }
                mse /= (double)nv;
                if (mse < best - tol) { best = mse; bad = 0; }
                else if (++bad >= patience) break;
            }
        }
        n_features = p;
        n_out = K;
        return py::make_tuple((int)members.size(), n_batches);
    }

    // -- prediction ---------------------------------------------------------

    // out[i, j] = member mem[j] evaluated at row i
    ArrF64 predict_members(const ArrU8F& Xa, const ArrI32& mem, int n_jobs) {
        int64_t n; int p;
        check_matrix(Xa, n, p);
        const std::vector<int32_t> idx = check_members(mem);
        const int m = (int)idx.size();

        // one row per (sample, target): out[i*K + k, j]; for K == 1 that is (n, m)
        const int K = n_out;
        ArrF64 out({(py::ssize_t)(n * K), (py::ssize_t)m});
        double* o = out.mutable_data();
        const uint8_t* X = Xa.data();
        {
            py::gil_scoped_release nogil;
            ThreadPool pool(resolve_threads(n_jobs) - 1);
            pool.run(m, [&](int64_t j) {
                const Tree& T = members[(size_t)idx[(size_t)j]];
                for (int64_t r = 0; r < n; ++r) {
                    const double* lv = predict_leaf(T, X, n, r);
                    for (int k = 0; k < K; ++k) o[(r * K + k) * (int64_t)m + j] = lv[k];
                }
            });
        }
        return out;
    }

    // out[i, b] = mean over members [b*batch, (b+1)*batch) at row i.
    // Averaging in C++ keeps the design matrix for the weight fit at
    // n_val x n_batches instead of n_val x n_members.
    ArrF64 predict_batch_means(const ArrU8F& Xa, int batch, int n_jobs) {
        int64_t n; int p;
        check_matrix(Xa, n, p);
        if (batch < 1) throw std::invalid_argument("batch_size must be >= 1");
        const int nb = (int)(members.size() / (size_t)batch);
        if (nb == 0) throw std::invalid_argument("fewer members than batch_size");

        const int K = n_out;
        ArrF64 out({(py::ssize_t)(n * K), (py::ssize_t)nb});
        double* o = out.mutable_data();
        const uint8_t* X = Xa.data();
        {
            py::gil_scoped_release nogil;
            ThreadPool pool(resolve_threads(n_jobs) - 1);
            std::vector<double> acc((size_t)K);
            pool.run(nb, [&](int64_t b) {
                std::vector<double> acc((size_t)K);
                for (int64_t r = 0; r < n; ++r) {
                    for (int k = 0; k < K; ++k) acc[k] = 0.0;
                    for (int t = 0; t < batch; ++t) {
                        const double* lv = predict_leaf(members[(size_t)(b * batch + t)], X, n, r);
                        for (int k = 0; k < K; ++k) acc[k] += lv[k];
                    }
                    // divide (not multiply by a reciprocal) so this matches
                    // numpy's mean bit-for-bit on the reference path
                    for (int k = 0; k < K; ++k)
                        o[(r * K + k) * (int64_t)nb + b] = acc[k] / (double)batch;
                }
            });
        }
        return out;
    }

    ArrF64 predict_weighted(const ArrU8F& Xa, const ArrI32& mem,
                            const ArrF64& wa, double intercept, int n_jobs) {
        int64_t n; int p;
        check_matrix(Xa, n, p);
        const std::vector<int32_t> idx = check_members(mem);
        const int m = (int)idx.size();
        if (wa.ndim() != 1 || (int)wa.shape(0) != m)
            throw std::invalid_argument("weights must be 1-dimensional with len(weights) == len(members)");

        // (n,) for a scalar target, (n, K) otherwise
        const int K = n_out;
        ArrF64 out = K == 1 ? ArrF64((py::ssize_t)n) : ArrF64({(py::ssize_t)n, (py::ssize_t)K});
        double* o = out.mutable_data();
        const double* w = wa.data();
        const uint8_t* X = Xa.data();
        {
            py::gil_scoped_release nogil;
            ThreadPool pool(resolve_threads(n_jobs) - 1);
            // parallelise over rows: members are summed per row, so each
            // thread owns a disjoint slice of the output
            const int nt = pool.n_threads();
            const int64_t chunk = (n + nt - 1) / std::max(1, nt);
            const int64_t n_chunks = chunk > 0 ? (n + chunk - 1) / chunk : 0;
            pool.run(n_chunks, [&](int64_t c) {
                const int64_t lo = c * chunk, hi = std::min(n, lo + chunk);
                for (int64_t r = lo; r < hi; ++r) {
                    double* orow = o + r * K;
                    for (int k = 0; k < K; ++k) orow[k] = intercept;
                    for (int j = 0; j < m; ++j) {
                        const double* lv = predict_leaf(members[(size_t)idx[(size_t)j]], X, n, r);
                        for (int k = 0; k < K; ++k) orow[k] += w[j] * lv[k];
                    }
                }
            });
        }
        return out;
    }

    // -- introspection ------------------------------------------------------
    py::dict member_info(int i) const {
        if (i < 0 || i >= (int)members.size())
            throw std::out_of_range("member index out of range");
        const Tree& T = members[(size_t)i];
        py::dict d;
        d["n_nodes"] = (int)T.feat.size();
        d["n_leaves"] = (int)T.n_leaves;
        d["base_id"] = (int)T.base_id;
        d["variant"] = (int)T.variant;
        return d;
    }

    // (total nodes, total leaves) over the given members
    py::tuple structure(const ArrI32& mem) const {
        const std::vector<int32_t> idx = check_members(mem);
        int64_t nodes = 0, leaves = 0;
        for (int32_t i : idx) {
            nodes += (int64_t)members[(size_t)i].feat.size();
            leaves += (int64_t)members[(size_t)i].n_leaves;
        }
        return py::make_tuple(nodes, leaves);
    }

    // split-gain importance, accumulated over members and weighted by each
    // member's ensemble weight; not normalised here
    ArrF64 feature_gains(const ArrI32& mem, const ArrF64& wa, int n_features_out) const {
        const std::vector<int32_t> idx = check_members(mem);
        if (wa.ndim() != 1 || (size_t)wa.shape(0) != idx.size())
            throw std::invalid_argument("weights must be 1-dimensional with len(weights) == len(members)");
        if (n_features_out < 0) throw std::invalid_argument("n_features must be >= 0");

        ArrF64 out((py::ssize_t)n_features_out);
        double* o = out.mutable_data();
        std::fill(o, o + n_features_out, 0.0);
        const double* w = wa.data();
        for (size_t j = 0; j < idx.size(); ++j) {
            const Tree& T = members[(size_t)idx[j]];
            const double wj = std::fabs(w[j]);
            for (size_t k = 0; k < T.feat.size(); ++k) {
                const int32_t f = T.feat[k];
                if (f < 0) continue;
                if (f >= n_features_out)
                    throw std::runtime_error("model references a feature index beyond n_features");
                o[f] += wj * T.gain[k];
            }
        }
        return out;
    }

    // Which input columns any selected member actually splits on. The bin
    // table only has to ship edges for these -- on wide data most columns are
    // never indexed and their quantile edges are dead weight.
    py::array_t<bool> used_features(const ArrI32& mem, int n_features_out) const {
        const std::vector<int32_t> idx = check_members(mem);
        if (n_features_out < 0) throw std::invalid_argument("n_features must be >= 0");
        py::array_t<bool> out((py::ssize_t)n_features_out);
        bool* o = out.mutable_data();
        std::fill(o, o + n_features_out, false);
        for (int32_t i : idx) {
            const Tree& T = members[(size_t)i];
            for (size_t k = 0; k < T.feat.size(); ++k) {
                const int32_t f = T.feat[k];
                if (f < 0) continue;
                if (f >= n_features_out)
                    throw std::runtime_error("model references a feature index beyond n_features");
                o[f] = true;
            }
        }
        return out;
    }

    // -- compaction ---------------------------------------------------------
    // Keep only the given members, renumbered 0..len(mem)-1.  This is the
    // "physically delete the zero-weight members" step: after it, the model
    // holds exactly the trees that contribute, which is what makes the
    // serialized model small.
    Ensemble subset(const ArrI32& mem) const {
        const std::vector<int32_t> idx = check_members(mem);
        Ensemble out;
        out.n_features = n_features;
        out.n_out = n_out;
        out.members.reserve(idx.size());
        for (int32_t i : idx) out.members.push_back(members[(size_t)i]);
        return out;
    }

    // -- leaf quantization --------------------------------------------------
    // Round every leaf to the 16-bit affine code the blob stores, in place.
    // Idempotent: a tree carries its map once set, so a reload followed by a
    // second call changes nothing.
    void quantize_leaves() {
        for (Tree& T : members) {
            if (T.quantized) continue;
            const int K = T.nout;
            double lo = 1e300, hi = -1e300;
            for (size_t i = 0; i < T.feat.size(); ++i) {
                if (T.feat[i] >= 0) continue;
                for (int k = 0; k < K; ++k) {
                    lo = std::min(lo, T.val[i * K + k]); hi = std::max(hi, T.val[i * K + k]);
                }
            }
            if (!(lo <= hi)) { lo = 0.0; hi = 0.0; }  // no leaves (cannot happen)
            T.leaf_a = lo;
            T.leaf_b = (hi > lo) ? (hi - lo) / (double)leaf_levels(K) : 0.0;
            for (size_t i = 0; i < T.feat.size(); ++i) {
                if (T.feat[i] >= 0) continue;
                for (int k = 0; k < K; ++k)
                    T.val[i * K + k] = T.leaf_a + T.leaf_b * (double)leaf_code(T, T.val[i * K + k]);
            }
            T.quantized = true;
        }
    }

    static uint16_t leaf_code(const Tree& T, double v) {
        if (T.leaf_b <= 0.0) return 0;
        const double top = (double)leaf_levels(T.nout);
        double q = std::nearbyint((v - T.leaf_a) / T.leaf_b);
        if (q < 0.0) q = 0.0;
        if (q > top) q = top;
        return (uint16_t)q;
    }

    // -- threshold compaction -----------------------------------------------
    // out[f, t] is true if any selected member cuts feature f at bin t. The
    // bin table only has to ship the edges behind those cuts.
    py::array_t<bool> used_thresholds(const ArrI32& mem, int n_features_out) const {
        const std::vector<int32_t> idx = check_members(mem);
        if (n_features_out < 0) throw std::invalid_argument("n_features must be >= 0");
        py::array_t<bool> out({(py::ssize_t)n_features_out, (py::ssize_t)256});
        bool* o = out.mutable_data();
        std::fill(o, o + (size_t)n_features_out * 256, false);
        for (int32_t i : idx) {
            const Tree& T = members[(size_t)i];
            for (size_t k = 0; k < T.feat.size(); ++k) {
                const int32_t f = T.feat[k];
                if (f < 0) continue;
                if (f >= n_features_out)
                    throw std::runtime_error("model references a feature index beyond n_features");
                o[(size_t)f * 256 + T.thr[k]] = true;
            }
        }
        return out;
    }

    // Rewrite every split's bin threshold through table[f, t] -- the bin id
    // the same cut has after the bin table was compacted.
    void remap_thresholds(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& table) {
        if (table.ndim() != 2 || table.shape(1) != 256)
            throw std::invalid_argument("table must have shape (n_features, 256)");
        const int nf = (int)table.shape(0);
        const uint8_t* tb = table.data();
        for (Tree& T : members) {
            for (size_t k = 0; k < T.feat.size(); ++k) {
                const int32_t f = T.feat[k];
                if (f < 0) continue;
                if (f >= nf) throw std::runtime_error("model references a feature index beyond the table");
                T.thr[k] = tb[(size_t)f * 256 + T.thr[k]];
            }
        }
    }

    // -- serialization ------------------------------------------------------
    py::bytes serialize() const {
        std::string buf;
        buf.append(kMagic, 4);
        put(buf, kFormatVersion);
        put(buf, (int32_t)n_features);
        put(buf, (int32_t)n_out);
        put(buf, (uint64_t)members.size());
        const bool nf16 = narrow_feat(n_features);
        for (const Tree& T : members) {
            if (!T.quantized)
                throw std::runtime_error("SXE: quantize_leaves() must run before serialize()");
            const size_t nn = T.feat.size();
            const bool nn16 = narrow_node(nn);
            put(buf, T.n_leaves);
            put(buf, T.base_id);
            put(buf, T.variant);
            put(buf, (uint32_t)nn);
            put(buf, T.leaf_a);
            put(buf, T.leaf_b);

            std::vector<uint8_t> mask(bitmap_bytes(nn), 0);
            for (size_t i = 0; i < nn; ++i)
                if (T.feat[i] >= 0) mask[i >> 3] |= (uint8_t)(1u << (i & 7));
            buf.append(reinterpret_cast<const char*>(mask.data()), mask.size());

            for (size_t i = 0; i < nn; ++i) {
                if (T.feat[i] < 0) continue;
                if (T.left[i] != (int32_t)i + 1)
                    throw std::runtime_error("SXE: tree is not in pre-order; cannot serialize");
                if (nf16) put(buf, (uint16_t)T.feat[i]); else put(buf, T.feat[i]);
                put(buf, T.thr[i]);
                if (nn16) put(buf, (uint16_t)T.right[i]); else put(buf, T.right[i]);
            }
            for (size_t i = 0; i < nn; ++i) {
                if (T.feat[i] >= 0) continue;
                for (int k = 0; k < n_out; ++k) {
                    const uint16_t code = leaf_code(T, T.val[i * n_out + k]);
                    if (n_out > 1) put(buf, (uint8_t)code); else put(buf, code);
                }
            }
        }
        return py::bytes(buf);
    }

    // Bytes serialize() would produce, without building the buffer. Used at
    // fit time so summary() can report the size of the real artefact.
    int64_t serialized_size() const {
        int64_t total = 4 + (int64_t)sizeof(kFormatVersion) + 4 + 4 + 8;
        const int64_t fw = narrow_feat(n_features) ? 2 : 4;
        for (const Tree& T : members) {
            const size_t nn = T.feat.size();
            const int64_t rw = narrow_node(nn) ? 2 : 4;
            total += 4 + 4 + 4 + 4 + 8 + 8 + (int64_t)bitmap_bytes(nn);
            total += (int64_t)(nn - (size_t)T.n_leaves) * (fw + 1 + rw);
            total += (int64_t)T.n_leaves * (n_out > 1 ? 1 : 2) * n_out;
        }
        return total;
    }

    static Ensemble deserialize(const py::bytes& blob) {
        const std::string s = blob;  // copy; the buffer must outlive the read
        Reader rd{s.data(), s.data() + s.size()};
        rd.need(4);
        if (std::memcmp(rd.p, kMagic, 4) != 0)
            throw std::runtime_error("SXE: not a model blob (bad magic)");
        rd.p += 4;
        const uint32_t ver = rd.get<uint32_t>();
        if (ver != kFormatVersion)
            throw std::runtime_error("SXE: model blob version " + std::to_string(ver) +
                                     " is not supported by this build (expected " +
                                     std::to_string(kFormatVersion) + ")");
        Ensemble out;
        out.n_features = rd.get<int32_t>();
        out.n_out = rd.get<int32_t>();
        if (out.n_out < 1) throw std::runtime_error("SXE: implausible target count in model blob");
        const int K = out.n_out;
        const bool nf16 = narrow_feat(out.n_features);
        const uint64_t m = rd.get<uint64_t>();
        if (m > (uint64_t)INT32_MAX) throw std::runtime_error("SXE: implausible member count");
        out.members.resize((size_t)m);
        for (uint64_t i = 0; i < m; ++i) {
            Tree& T = out.members[(size_t)i];
            T.n_leaves = rd.get<int32_t>();
            T.base_id = rd.get<int32_t>();
            T.variant = rd.get<int32_t>();
            const uint32_t nn32 = rd.get<uint32_t>();
            if (nn32 == 0 || nn32 > (uint32_t)INT32_MAX)
                throw std::runtime_error("SXE: implausible node count in model blob");
            const size_t nn = (size_t)nn32;
            const bool nn16 = narrow_node(nn);
            T.leaf_a = rd.get<double>();
            T.leaf_b = rd.get<double>();
            T.quantized = true;
            T.nout = K;

            rd.need(bitmap_bytes(nn));
            const uint8_t* mask = reinterpret_cast<const uint8_t*>(rd.p);
            rd.p += bitmap_bytes(nn);

            T.feat.assign(nn, -1); T.thr.assign(nn, 0);
            T.left.assign(nn, -1); T.right.assign(nn, -1);
            T.val.assign(nn * (size_t)K, 0.0); T.gain.assign(nn, 0.0);

            int32_t leaves = 0;
            for (size_t k = 0; k < nn; ++k) {
                if (!(mask[k >> 3] & (1u << (k & 7)))) { ++leaves; continue; }
                T.feat[k] = nf16 ? (int32_t)rd.get<uint16_t>() : rd.get<int32_t>();
                T.thr[k] = rd.get<uint8_t>();
                T.right[k] = nn16 ? (int32_t)rd.get<uint16_t>() : rd.get<int32_t>();
                T.left[k] = (int32_t)k + 1;
                // a corrupt blob must not become a wild read
                if (out.n_features >= 0 && (T.feat[k] < 0 || T.feat[k] >= out.n_features))
                    throw std::runtime_error("SXE: feature index out of range in model blob");
                if (T.left[k] >= (int32_t)nn || T.right[k] <= (int32_t)k ||
                    T.right[k] >= (int32_t)nn)
                    throw std::runtime_error("SXE: child index out of range in model blob");
            }
            if (leaves != T.n_leaves)
                throw std::runtime_error("SXE: leaf count disagrees with the node map");
            for (size_t k = 0; k < nn; ++k)
                if (T.feat[k] < 0)
                    for (int c = 0; c < K; ++c) {
                        const double code = K > 1 ? (double)rd.get<uint8_t>() : (double)rd.get<uint16_t>();
                        T.val[k * K + c] = T.leaf_a + T.leaf_b * code;
                    }
        }
        return out;
    }
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "SXE C++ core: batch-boosted, holdout-pruned histogram trees";
    m.attr("__format_version__") = kFormatVersion;

    m.def("n_threads_available", [] { return resolve_threads(-1); },
          "Number of threads used when n_jobs=-1.");

    py::class_<Ensemble>(m, "Ensemble")
        .def(py::init<>())
        .def_readonly("n_features", &Ensemble::n_features)
        .def_readonly("n_out", &Ensemble::n_out)
        .def_property_readonly("n_members", &Ensemble::n_members)
        .def("boost", &Ensemble::boost,
             py::arg("X"), py::arg("y"), py::arg("X_val"), py::arg("y_val"),
             py::arg("batch"), py::arg("max_batches"), py::arg("shrink"),
             py::arg("patience"), py::arg("tol"),
             py::arg("n_thresh"), py::arg("argmax_bins"), py::arg("max_leaves"),
             py::arg("sub_lo"), py::arg("sub_hi"),
             py::arg("depth_choices"),
             py::arg("leaf_lo"), py::arg("leaf_hi"),
             py::arg("feat_lo"), py::arg("feat_hi"),
             py::arg("lam_lo"), py::arg("lam_hi"),
             py::arg("leaf_shrink"), py::arg("hist_bits"),
             py::arg("hybrid_min"), py::arg("seed"), py::arg("n_jobs"),
             "Grow the batch-boosted dictionary. Returns (n_members, n_batches).")
        .def("predict_members", &Ensemble::predict_members,
             py::arg("X"), py::arg("members"), py::arg("n_jobs") = -1)
        .def("predict_batch_means", &Ensemble::predict_batch_means,
             py::arg("X"), py::arg("batch"), py::arg("n_jobs") = -1)
        .def("predict_weighted", &Ensemble::predict_weighted,
             py::arg("X"), py::arg("members"), py::arg("weights"),
             py::arg("intercept"), py::arg("n_jobs") = -1)
        .def("used_features", &Ensemble::used_features,
             py::arg("members"), py::arg("n_features"),
             "Boolean mask of the input columns any selected member splits on.")
        .def("used_thresholds", &Ensemble::used_thresholds,
             py::arg("members"), py::arg("n_features"),
             "(n_features, 256) mask of the bin cuts any selected member uses.")
        .def("remap_thresholds", &Ensemble::remap_thresholds, py::arg("table"),
             "Rewrite split thresholds through a (n_features, 256) uint8 table.")
        .def("quantize_leaves", &Ensemble::quantize_leaves,
             "Round leaf values to the 16-bit codes the blob stores.")
        .def("serialized_size", &Ensemble::serialized_size,
             "Bytes serialize() would produce, without building the buffer.")
        .def("member_info", &Ensemble::member_info, py::arg("i"))
        .def("structure", &Ensemble::structure, py::arg("members"))
        .def("feature_gains", &Ensemble::feature_gains,
             py::arg("members"), py::arg("weights"), py::arg("n_features"))
        .def("subset", &Ensemble::subset, py::arg("members"))
        .def("serialize", &Ensemble::serialize)
        .def_static("deserialize", &Ensemble::deserialize, py::arg("blob"))
        .def(py::pickle(
            [](const Ensemble& e) { return e.serialize(); },
            [](const py::bytes& b) { return Ensemble::deserialize(b); }));
}
