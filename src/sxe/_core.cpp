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
    std::vector<double>  val;
    std::vector<double>  gain;   // split gain at internal nodes, 0 at leaves
    int32_t n_leaves = 0;
    int32_t base_id = -1;   // which grown tree this member came from
    int32_t variant = 0;    // 0 raw, 2 random lambda
};

struct GrownTree {
    // full grown tree + per-node training stats + holdout stats
    std::vector<int32_t> feat, left, right;
    std::vector<uint8_t> thr;
    std::vector<double>  val;                       // train mean per node
    std::vector<double>  gain;
    std::vector<double>  h_cnt, h_sum, h_ssq;       // holdout stats per node
};

struct Hist {
    // all-feature histogram: cnt/sum laid out as [feature*256 + bin]
    std::vector<int32_t> cnt;
    std::vector<double>  sum;
    void reset(int p) {
        cnt.assign((size_t)p * 256, 0);
        sum.assign((size_t)p * 256, 0.0);
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
    int                  hybrid_min = 600;  // node size above which the
                                            // subtraction scheme is used
    std::mt19937_64      rng;

    Hist* acquire(int p) {
        Hist* h;
        if (pool.empty()) h = new Hist();
        else { h = pool.back(); pool.pop_back(); }
        h->reset(p);
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
int32_t grow_node_legacy(GrownTree& T, const uint8_t* X, const double* y,
                  int64_t n, int p, Workspace& ws,
                  int64_t lo, int64_t hi, int depth, const TreeParams& tp)
{
    const int64_t m = hi - lo;
    double sum = 0.0, ssq = 0.0;
    for (int64_t i = lo; i < hi; ++i) { double v = y[ws.idx[i]]; sum += v; ssq += v * v; }
    const double mean = sum / (double)m;

    const int32_t id = (int32_t)T.feat.size();
    T.feat.push_back(-1); T.thr.push_back(0);
    T.left.push_back(-1); T.right.push_back(-1);
    T.val.push_back(mean); T.gain.push_back(0.0);

    const bool pure = (ssq - sum * mean) <= 1e-12 * (std::fabs(ssq) + 1e-12);
    if (depth >= tp.max_depth || m < 2 * (int64_t)tp.min_leaf || pure) return id;

    // sample K distinct candidate features (partial Fisher-Yates)
    for (int j = 0; j < p; ++j) ws.featbuf[j] = j;
    const int K = std::min(tp.K, p);

    int best_f = -1, best_t = -1;
    double best_gain = -1.0;
    const double parent_term = sum * sum / (double)m;

    for (int c = 0; c < K; ++c) {
        std::uniform_int_distribution<int> pick(c, p - 1);
        std::swap(ws.featbuf[c], ws.featbuf[pick(ws.rng)]);
        const int f = ws.featbuf[c];
        const uint8_t* col = X + (int64_t)f * n;

        // build per-node histogram for this feature
        for (int b : ws.touched) { ws.hcnt[b] = 0; ws.hsum[b] = 0.0; }
        ws.touched.clear();
        int minb = 256, maxb = -1;
        for (int64_t i = lo; i < hi; ++i) {
            const int b = col[ws.idx[i]];
            if (!ws.hcnt[b]) ws.touched.push_back(b);
            ws.hcnt[b]++; ws.hsum[b] += y[ws.idx[i]];
            if (b < minb) minb = b;
            if (b > maxb) maxb = b;
        }
        if (maxb <= minb) continue;  // constant feature in node

        if (tp.argmax_bins) {
            int64_t cl = 0; double sl = 0.0;
            for (int t = minb; t < maxb; ++t) {
                cl += ws.hcnt[t]; sl += ws.hsum[t];
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double sr = sum - sl;
                const double gain = sl * sl / (double)cl + sr * sr / (double)cr - parent_term;
                if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
            }
        } else {
            std::uniform_int_distribution<int> tdist(minb, maxb - 1);
            for (int r = 0; r < tp.n_thresh; ++r) {
                const int t = tdist(ws.rng);
                int64_t cl = 0; double sl = 0.0;
                for (int b = minb; b <= t; ++b) { cl += ws.hcnt[b]; sl += ws.hsum[b]; }
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double sr = sum - sl;
                const double gain = sl * sl / (double)cl + sr * sr / (double)cr - parent_term;
                if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
            }
        }
    }
    if (best_f < 0) return id;

    // partition indices in place
    const uint8_t* col = X + (int64_t)best_f * n;
    int64_t* base = ws.idx.data();
    int64_t* mid = std::partition(base + lo, base + hi,
                                  [&](int64_t r) { return col[r] <= (uint8_t)best_t; });
    const int64_t split = mid - base;
    if (split == lo || split == hi) return id;  // safety

    T.feat[id] = best_f; T.thr[id] = (uint8_t)best_t; T.gain[id] = best_gain;
    T.left[id]  = grow_node_legacy(T, X, y, n, p, ws, lo, split, depth + 1, tp);
    T.right[id] = grow_node_legacy(T, X, y, n, p, ws, split, hi, depth + 1, tp);
    return id;
}

// ---------------------------------------------------------------------------
// full-histogram growth with the parent-minus-child subtraction trick
// (used for nodes >= ws.hybrid_min; smaller subtrees fall back to the
// per-candidate touched-bin scheme above, which has no memset overhead)
// ---------------------------------------------------------------------------
void fill_hist(Hist& H, const uint8_t* X, const double* y, int64_t n, int p,
               const std::vector<int64_t>& idx, int64_t lo, int64_t hi,
               std::vector<double>& ybuf)
{
    const int64_t m = hi - lo;
    ybuf.resize(m);
    for (int64_t i = 0; i < m; ++i) ybuf[i] = y[idx[lo + i]];
    for (int f = 0; f < p; ++f) {
        const uint8_t* col = X + (int64_t)f * n;
        int32_t* c = H.cnt.data() + (size_t)f * 256;
        double*  s = H.sum.data() + (size_t)f * 256;
        for (int64_t i = 0; i < m; ++i) {
            const int b = col[idx[lo + i]];
            c[b]++; s[b] += ybuf[i];
        }
    }
}

int32_t grow_node_full(GrownTree& T, const uint8_t* X, const double* y,
                       int64_t n, int p, Workspace& ws,
                       int64_t lo, int64_t hi, int depth, const TreeParams& tp,
                       Hist* H)  // H: this node's all-feature histogram (owned by caller)
{
    const int64_t m = hi - lo;
    double sum = 0.0, ssq = 0.0;
    for (int64_t i = lo; i < hi; ++i) { double v = y[ws.idx[i]]; sum += v; ssq += v * v; }
    const double mean = sum / (double)m;

    const int32_t id = (int32_t)T.feat.size();
    T.feat.push_back(-1); T.thr.push_back(0);
    T.left.push_back(-1); T.right.push_back(-1);
    T.val.push_back(mean); T.gain.push_back(0.0);

    const bool pure = (ssq - sum * mean) <= 1e-12 * (std::fabs(ssq) + 1e-12);
    if (depth >= tp.max_depth || m < 2 * (int64_t)tp.min_leaf || pure) return id;

    for (int j = 0; j < p; ++j) ws.featbuf[j] = j;
    const int K = std::min(tp.K, p);
    int best_f = -1, best_t = -1;
    double best_gain = -1.0;
    const double parent_term = sum * sum / (double)m;

    for (int c = 0; c < K; ++c) {
        std::uniform_int_distribution<int> pick(c, p - 1);
        std::swap(ws.featbuf[c], ws.featbuf[pick(ws.rng)]);
        const int f = ws.featbuf[c];
        const int32_t* hc = H->cnt.data() + (size_t)f * 256;
        const double*  hs = H->sum.data() + (size_t)f * 256;
        int minb = 0, maxb = 255;
        while (minb < 256 && hc[minb] == 0) ++minb;
        while (maxb >= 0 && hc[maxb] == 0) --maxb;
        if (maxb <= minb) continue;
        if (tp.argmax_bins) {
            int64_t cl = 0; double sl = 0.0;
            for (int t = minb; t < maxb; ++t) {
                cl += hc[t]; sl += hs[t];
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double sr = sum - sl;
                const double gain = sl * sl / (double)cl + sr * sr / (double)cr - parent_term;
                if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
            }
        } else {
            std::uniform_int_distribution<int> tdist(minb, maxb - 1);
            for (int r = 0; r < tp.n_thresh; ++r) {
                const int t = tdist(ws.rng);
                int64_t cl = 0; double sl = 0.0;
                for (int b = minb; b <= t; ++b) { cl += hc[b]; sl += hs[b]; }
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double sr = sum - sl;
                const double gain = sl * sl / (double)cl + sr * sr / (double)cr - parent_term;
                if (gain > best_gain) { best_gain = gain; best_f = f; best_t = t; }
            }
        }
    }
    if (best_f < 0) return id;

    const uint8_t* col = X + (int64_t)best_f * n;
    int64_t* base = ws.idx.data();
    int64_t* mid = std::partition(base + lo, base + hi,
                                  [&](int64_t r) { return col[r] <= (uint8_t)best_t; });
    const int64_t split = mid - base;
    if (split == lo || split == hi) return id;

    T.feat[id] = best_f; T.thr[id] = (uint8_t)best_t; T.gain[id] = best_gain;

    const bool left_small = (split - lo) <= (hi - split);
    const int64_t slo = left_small ? lo : split, shi = left_small ? split : hi;   // smaller
    const int64_t llo = left_small ? split : lo, lhi = left_small ? hi : split;   // larger

    // build smaller child's histogram from data; larger child = parent - smaller
    Hist* HS = ws.acquire(p);
    fill_hist(*HS, X, y, n, p, ws.idx, slo, shi, ws.ybuf);
    {
        const size_t sz = (size_t)p * 256;
        int32_t* pc = H->cnt.data(); const int32_t* sc = HS->cnt.data();
        double*  ps = H->sum.data(); const double*  ss = HS->sum.data();
        for (size_t k = 0; k < sz; ++k) { pc[k] -= sc[k]; ps[k] -= ss[k]; }
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
struct SplitEval { double gain = -1.0; int f = -1, t = -1; };

SplitEval best_split_from_hist(const Hist& H, int p, int64_t m, double sum,
                               Workspace& ws, const TreeParams& tp)
{
    SplitEval out;
    for (int j = 0; j < p; ++j) ws.featbuf[j] = j;
    const int K = std::min(tp.K, p);
    const double parent_term = sum * sum / (double)m;
    for (int c = 0; c < K; ++c) {
        std::uniform_int_distribution<int> pick(c, p - 1);
        std::swap(ws.featbuf[c], ws.featbuf[pick(ws.rng)]);
        const int f = ws.featbuf[c];
        const int32_t* hc = H.cnt.data() + (size_t)f * 256;
        const double*  hs = H.sum.data() + (size_t)f * 256;
        int minb = 0, maxb = 255;
        while (minb < 256 && hc[minb] == 0) ++minb;
        while (maxb >= 0 && hc[maxb] == 0) --maxb;
        if (maxb <= minb) continue;
        if (tp.argmax_bins) {
            int64_t cl = 0; double sl = 0.0;
            for (int t = minb; t < maxb; ++t) {
                cl += hc[t]; sl += hs[t];
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double sr = sum - sl;
                const double g = sl * sl / (double)cl + sr * sr / (double)cr - parent_term;
                if (g > out.gain) { out.gain = g; out.f = f; out.t = t; }
            }
        } else {
            std::uniform_int_distribution<int> tdist(minb, maxb - 1);
            for (int r = 0; r < tp.n_thresh; ++r) {
                const int t = tdist(ws.rng);
                int64_t cl = 0; double sl = 0.0;
                for (int b = minb; b <= t; ++b) { cl += hc[b]; sl += hs[b]; }
                const int64_t cr = m - cl;
                if (cl < tp.min_leaf || cr < tp.min_leaf) continue;
                const double sr = sum - sl;
                const double g = sl * sl / (double)cl + sr * sr / (double)cr - parent_term;
                if (g > out.gain) { out.gain = g; out.f = f; out.t = t; }
            }
        }
    }
    return out;
}

struct Pending {
    double gain; int32_t id; int64_t lo, hi; int depth; Hist* H; int f, t;
    bool operator<(const Pending& o) const { return gain < o.gain; }
};

void grow_tree_leafwise(GrownTree& T, const uint8_t* X, const double* y,
                        int64_t n, int p, Workspace& ws, int64_t n_sub,
                        const TreeParams& tp)
{
    auto make_node = [&](int64_t lo, int64_t hi) -> int32_t {
        double sum = 0.0;
        for (int64_t i = lo; i < hi; ++i) sum += y[ws.idx[i]];
        const int32_t id = (int32_t)T.feat.size();
        T.feat.push_back(-1); T.thr.push_back(0);
        T.left.push_back(-1); T.right.push_back(-1);
        T.val.push_back(sum / (double)(hi - lo)); T.gain.push_back(0.0);
        return id;
    };
    auto node_sum = [&](int32_t id, int64_t lo, int64_t hi) {
        return T.val[id] * (double)(hi - lo);
    };

    std::priority_queue<Pending> heap;
    const int32_t root = make_node(0, n_sub);
    {
        Hist* H = ws.acquire(p);
        fill_hist(*H, X, y, n, p, ws.idx, 0, n_sub, ws.ybuf);
        SplitEval e = best_split_from_hist(*H, p, n_sub, node_sum(root, 0, n_sub), ws, tp);
        if (e.gain > 0) heap.push({e.gain, root, 0, n_sub, 0, H, e.f, e.t});
        else ws.release(H);
    }
    int leaves = 1;
    while (!heap.empty() && leaves < tp.max_leaves) {
        Pending nd = heap.top(); heap.pop();
        const uint8_t* col = X + (int64_t)nd.f * n;
        int64_t* base = ws.idx.data();
        int64_t* mid = std::partition(base + nd.lo, base + nd.hi,
                                      [&](int64_t r) { return col[r] <= (uint8_t)nd.t; });
        const int64_t split = mid - base;
        if (split == nd.lo || split == nd.hi) { ws.release(nd.H); continue; }

        T.feat[nd.id] = nd.f; T.thr[nd.id] = (uint8_t)nd.t; T.gain[nd.id] = nd.gain;
        const int32_t idl = make_node(nd.lo, split);
        const int32_t idr = make_node(split, nd.hi);
        T.left[nd.id] = idl; T.right[nd.id] = idr;
        ++leaves;

        const bool left_small = (split - nd.lo) <= (nd.hi - split);
        const int64_t slo = left_small ? nd.lo : split, shi = left_small ? split : nd.hi;
        const int64_t llo = left_small ? split : nd.lo, lhi = left_small ? nd.hi : split;
        const int32_t id_s = left_small ? idl : idr, id_l = left_small ? idr : idl;

        Hist* HS = ws.acquire(p);
        fill_hist(*HS, X, y, n, p, ws.idx, slo, shi, ws.ybuf);
        {
            const size_t sz = (size_t)p * 256;
            int32_t* pc = nd.H->cnt.data(); const int32_t* sc = HS->cnt.data();
            double*  ps = nd.H->sum.data(); const double*  ss = HS->sum.data();
            for (size_t k = 0; k < sz; ++k) { pc[k] -= sc[k]; ps[k] -= ss[k]; }
        }
        auto consider = [&](int32_t cid, int64_t clo, int64_t chi, Hist* CH) {
            const int64_t cm = chi - clo;
            if (nd.depth + 1 >= tp.max_depth || cm < 2 * (int64_t)tp.min_leaf) { ws.release(CH); return; }
            SplitEval e = best_split_from_hist(*CH, p, cm, node_sum(cid, clo, chi), ws, tp);
            if (e.gain > 1e-12) heap.push({e.gain, cid, clo, chi, nd.depth + 1, CH, e.f, e.t});
            else ws.release(CH);
        };
        consider(id_s, slo, shi, HS);
        consider(id_l, llo, lhi, nd.H);
    }
    while (!heap.empty()) { ws.release(heap.top().H); heap.pop(); }
}

// route holdout rows, accumulating (cnt,sum,ssq) at every node on the path
void route_holdout(GrownTree& T, const uint8_t* X, const double* y,
                   int64_t n, const std::vector<int64_t>& hold)
{
    const size_t nn = T.feat.size();
    T.h_cnt.assign(nn, 0.0); T.h_sum.assign(nn, 0.0); T.h_ssq.assign(nn, 0.0);
    for (int64_t r : hold) {
        const double v = y[r];
        int32_t id = 0;
        for (;;) {
            T.h_cnt[id] += 1.0; T.h_sum[id] += v; T.h_ssq[id] += v * v;
            const int32_t f = T.feat[id];
            if (f < 0) break;
            id = (X[(int64_t)f * n + r] <= T.thr[id]) ? T.left[id] : T.right[id];
        }
    }
}

// bottom-up collapse decision; pre-order ids => descending id is bottom-up.
// returns per-node "collapsed" flags for penalty lambda (in units of holdout
// variance per extra leaf).
void prune_flags(const GrownTree& T, double lam, double varh,
                 std::vector<uint8_t>& collapsed, std::vector<int32_t>& leaves)
{
    const int32_t nn = (int32_t)T.feat.size();
    std::vector<double> sse_sub(nn);
    collapsed.assign(nn, 0); leaves.assign(nn, 1);
    for (int32_t id = nn - 1; id >= 0; --id) {
        const double mu = T.val[id];
        const double sse_leaf = T.h_ssq[id] - 2.0 * mu * T.h_sum[id] + T.h_cnt[id] * mu * mu;
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

// safe two-pass export (avoids pointer-into-growing-vector issues)
void export_member_safe(const GrownTree& T, const std::vector<uint8_t>* collapsed,
                        Tree& out, int32_t base_id, int32_t variant)
{
    out.base_id = base_id; out.variant = variant;
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
    out.right.resize(nn); out.val.resize(nn); out.gain.resize(nn);
    int32_t leaves = 0;
    for (int32_t i = 0; i < nn; ++i) {
        const int32_t src = order[i];
        const bool leaf = (T.feat[src] < 0) || (collapsed && (*collapsed)[src]);
        out.val[i] = T.val[src];
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

inline double predict_row(const Tree& T, const uint8_t* X, int64_t n, int64_t r) {
    int32_t id = 0;
    while (T.feat[id] >= 0)
        id = (X[(int64_t)T.feat[id] * n + r] <= T.thr[id]) ? T.left[id] : T.right[id];
    return T.val[id];
}

// ---------------------------------------------------------------------------
// serialization
// ---------------------------------------------------------------------------
const char kMagic[4] = {'S', 'X', 'E', 'B'};
const uint32_t kFormatVersion = 1;

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
                    int hybrid_min, uint64_t seed, int n_jobs)
    {
        if (Xa.ndim() != 2 || Xva.ndim() != 2)
            throw std::invalid_argument("X and X_val must be 2-dimensional");
        const int64_t n = Xa.shape(0), nv = Xva.shape(0);
        const int p = (int)Xa.shape(1);
        if ((int)Xva.shape(1) != p)
            throw std::invalid_argument("X and X_val must have the same number of features");
        if (ya.ndim() != 1 || ya.shape(0) != n)
            throw std::invalid_argument("y must be 1-dimensional with len(y) == X.shape[0]");
        if (yva.ndim() != 1 || yva.shape(0) != nv)
            throw std::invalid_argument("y_val must be 1-dimensional with len(y_val) == X_val.shape[0]");
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

        std::vector<double> r(ya.data(), ya.data() + n);  // residual (training target)
        std::vector<double> Fv((size_t)nv, 0.0);          // generation-ensemble val preds
        std::vector<double> bt((size_t)batch * (size_t)n);   // per-tree train predictions
        std::vector<double> bv((size_t)batch * (size_t)nv);  // per-tree val predictions

        double best = 1e300;
        int bad = 0, n_batches = 0;

        {
            py::gil_scoped_release nogil;
            ThreadPool pool(resolve_threads(n_jobs) - 1);

            for (int b = 0; b < max_batches; ++b) {
                const size_t offset = members.size();
                members.resize(offset + (size_t)batch);

                pool.run(batch, [&](int64_t t64) {
                    const int t = (int)t64;
                    Workspace ws;
                    ws.rng.seed(splitmix64(seed + 0x9E3779B9ULL * (uint64_t)(b * batch + t + 1)));
                    ws.featbuf.resize(p);
                    ws.hcnt.assign(256, 0); ws.hsum.assign(256, 0.0);
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
                    ws.idx.assign(all.begin(), all.begin() + msub);
                    std::vector<int64_t> hold(all.begin() + msub, all.end());

                    GrownTree G;
                    G.feat.reserve(512);
                    if (max_leaves > 0) {
                        tp.max_leaves = max_leaves;
                        grow_tree_leafwise(G, X, r.data(), n, p, ws, msub, tp);
                    } else if (msub >= (int64_t)hybrid_min) {
                        Hist* Hh = ws.acquire(p);
                        fill_hist(*Hh, X, r.data(), n, p, ws.idx, 0, msub, ws.ybuf);
                        grow_node_full(G, X, r.data(), n, p, ws, 0, msub, 0, tp, Hh);
                        ws.release(Hh);
                    } else {
                        grow_node_legacy(G, X, r.data(), n, p, ws, 0, msub, 0, tp);
                    }
                    route_holdout(G, X, r.data(), n, hold);
                    const double nh = std::max(1.0, G.h_cnt.empty() ? 1.0 : G.h_cnt[0]);
                    const double varh = std::max(1e-12,
                        (G.h_ssq[0] - G.h_sum[0] * G.h_sum[0] / nh) / nh);

                    std::vector<uint8_t> collapsed; std::vector<int32_t> leaves;
                    Tree& M = members[offset + (size_t)t];
                    // Small-holdout guard: subtrees the holdout never visits satisfy the
                    // collapse inequality trivially and would prune the tree to a stump.
                    if ((int64_t)hold.size() >= std::max<int64_t>(50, n / 50)) {
                        const double lam = std::exp(std::log(lam_lo) +
                            U(ws.rng) * (std::log(lam_hi) - std::log(lam_lo)));
                        prune_flags(G, lam, varh, collapsed, leaves);
                        export_member_safe(G, &collapsed, M, b * batch + t, 2);
                    } else {
                        export_member_safe(G, nullptr, M, b * batch + t, 0);
                    }

                    // this tree's predictions on train and val rows
                    double* pt = bt.data() + (size_t)t * (size_t)n;
                    for (int64_t i = 0; i < n; ++i) pt[i] = predict_row(M, X, n, i);
                    double* pv = bv.data() + (size_t)t * (size_t)nv;
                    for (int64_t i = 0; i < nv; ++i) pv[i] = predict_row(M, Xv, nv, i);
                });

                ++n_batches;
                const double step = shrink / (double)batch;
                for (int t = 0; t < batch; ++t) {
                    const double* pt = bt.data() + (size_t)t * (size_t)n;
                    for (int64_t i = 0; i < n; ++i) r[(size_t)i] -= step * pt[i];
                    const double* pv = bv.data() + (size_t)t * (size_t)nv;
                    for (int64_t i = 0; i < nv; ++i) Fv[(size_t)i] += step * pv[i];
                }
                double mse = 0.0;
                for (int64_t i = 0; i < nv; ++i) {
                    const double e = yv[i] - Fv[(size_t)i];
                    mse += e * e;
                }
                mse /= (double)nv;
                if (mse < best - tol) { best = mse; bad = 0; }
                else if (++bad >= patience) break;
            }
        }
        n_features = p;
        return py::make_tuple((int)members.size(), n_batches);
    }

    // -- prediction ---------------------------------------------------------

    // out[i, j] = member mem[j] evaluated at row i
    ArrF64 predict_members(const ArrU8F& Xa, const ArrI32& mem, int n_jobs) {
        int64_t n; int p;
        check_matrix(Xa, n, p);
        const std::vector<int32_t> idx = check_members(mem);
        const int m = (int)idx.size();

        ArrF64 out({(py::ssize_t)n, (py::ssize_t)m});
        double* o = out.mutable_data();
        const uint8_t* X = Xa.data();
        {
            py::gil_scoped_release nogil;
            ThreadPool pool(resolve_threads(n_jobs) - 1);
            pool.run(m, [&](int64_t j) {
                const Tree& T = members[(size_t)idx[(size_t)j]];
                for (int64_t r = 0; r < n; ++r)
                    o[r * (int64_t)m + j] = predict_row(T, X, n, r);
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

        ArrF64 out({(py::ssize_t)n, (py::ssize_t)nb});
        double* o = out.mutable_data();
        const uint8_t* X = Xa.data();
        {
            py::gil_scoped_release nogil;
            ThreadPool pool(resolve_threads(n_jobs) - 1);
            pool.run(nb, [&](int64_t b) {
                for (int64_t r = 0; r < n; ++r) {
                    double acc = 0.0;
                    for (int t = 0; t < batch; ++t)
                        acc += predict_row(members[(size_t)(b * batch + t)], X, n, r);
                    // divide (not multiply by a reciprocal) so this matches
                    // numpy's mean bit-for-bit on the reference path
                    o[r * (int64_t)nb + b] = acc / (double)batch;
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

        ArrF64 out((py::ssize_t)n);
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
                    double acc = intercept;
                    for (int j = 0; j < m; ++j)
                        acc += w[j] * predict_row(members[(size_t)idx[(size_t)j]], X, n, r);
                    o[r] = acc;
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

    // -- compaction ---------------------------------------------------------
    // Keep only the given members, renumbered 0..len(mem)-1.  This is the
    // "physically delete the zero-weight members" step: after it, the model
    // holds exactly the trees that contribute, which is what makes the
    // serialized model small.
    Ensemble subset(const ArrI32& mem) const {
        const std::vector<int32_t> idx = check_members(mem);
        Ensemble out;
        out.n_features = n_features;
        out.members.reserve(idx.size());
        for (int32_t i : idx) out.members.push_back(members[(size_t)i]);
        return out;
    }

    // -- serialization ------------------------------------------------------
    py::bytes serialize() const {
        std::string buf;
        buf.append(kMagic, 4);
        put(buf, kFormatVersion);
        put(buf, (int32_t)n_features);
        put(buf, (uint64_t)members.size());
        for (const Tree& T : members) {
            put(buf, T.n_leaves);
            put(buf, T.base_id);
            put(buf, T.variant);
            put_vec(buf, T.feat);
            put_vec(buf, T.left);
            put_vec(buf, T.right);
            put_vec(buf, T.thr);
            put_vec(buf, T.val);
            put_vec(buf, T.gain);
        }
        return py::bytes(buf);
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
        const uint64_t m = rd.get<uint64_t>();
        out.members.resize((size_t)m);
        for (uint64_t i = 0; i < m; ++i) {
            Tree& T = out.members[(size_t)i];
            T.n_leaves = rd.get<int32_t>();
            T.base_id = rd.get<int32_t>();
            T.variant = rd.get<int32_t>();
            T.feat = rd.get_vec<int32_t>();
            T.left = rd.get_vec<int32_t>();
            T.right = rd.get_vec<int32_t>();
            T.thr = rd.get_vec<uint8_t>();
            T.val = rd.get_vec<double>();
            T.gain = rd.get_vec<double>();
            const size_t nn = T.feat.size();
            if (T.left.size() != nn || T.right.size() != nn || T.thr.size() != nn ||
                T.val.size() != nn || T.gain.size() != nn || nn == 0)
                throw std::runtime_error("SXE: inconsistent node arrays in model blob");
            // structural validation: a corrupt blob must not become a wild read
            for (size_t k = 0; k < nn; ++k) {
                if (T.feat[k] < 0) continue;
                if (out.n_features >= 0 && T.feat[k] >= out.n_features)
                    throw std::runtime_error("SXE: feature index out of range in model blob");
                if (T.left[k] <= (int32_t)k || T.right[k] <= (int32_t)k ||
                    T.left[k] >= (int32_t)nn || T.right[k] >= (int32_t)nn)
                    throw std::runtime_error("SXE: child index out of range in model blob");
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
             py::arg("hybrid_min"), py::arg("seed"), py::arg("n_jobs"),
             "Grow the batch-boosted dictionary. Returns (n_members, n_batches).")
        .def("predict_members", &Ensemble::predict_members,
             py::arg("X"), py::arg("members"), py::arg("n_jobs") = -1)
        .def("predict_batch_means", &Ensemble::predict_batch_means,
             py::arg("X"), py::arg("batch"), py::arg("n_jobs") = -1)
        .def("predict_weighted", &Ensemble::predict_weighted,
             py::arg("X"), py::arg("members"), py::arg("weights"),
             py::arg("intercept"), py::arg("n_jobs") = -1)
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
