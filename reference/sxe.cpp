// sxe.cpp -- C++ core for the Sparse eXtra Ensemble (SXE).
//
// Implements the hot path of the method designed in conversation:
//   * trees grown on pre-binned (uint8, <=256 bins) column-major features
//   * extremely randomized splits: K random candidate features per node,
//     a few random bin cut-points per feature, evaluated via per-node
//     256-bin histograms (count,sum) -- no sorting, no full split search
//   * per-tree random subsample (without replacement); the out-of-subsample
//     rows are the tree's holdout
//   * holdout reduced-error pruning with a complexity penalty:
//     collapse a subtree when   SSE_leaf <= SSE_subtree + lambda*varh*(leaves-1)
//     evaluated bottom-up after routing the holdout once through the tree.
//     lambda = 0 removes exactly the splits that do not help the holdout
//     ("dead wood"); larger lambda gives nested, more aggressive subtrees.
//     Each grown tree can emit several variants (raw / lambda=0 / random
//     lambda) as separate dictionary members.
//   * OpenMP parallelism over trees (trees within a round are independent).
//
// The convex non-negative weight fit stays in Python (negligible cost).
//
// Build:  g++ -O3 -march=native -fopenmp -shared -fPIC sxe.cpp -o libsxe.so

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Tree {
    // compact arrays; feat = -1 marks a leaf
    std::vector<int32_t> feat, left, right;
    std::vector<uint8_t> thr;
    std::vector<double>  val;
    int32_t n_leaves = 0;
    int32_t base_id = -1;   // which grown tree this member came from
    int32_t variant = 0;    // 0 raw, 1 lambda=0, 2 random lambda
};

struct GrownTree {
    // full grown tree + per-node training stats + holdout stats
    std::vector<int32_t> feat, left, right;
    std::vector<uint8_t> thr;
    std::vector<double>  val;                       // train mean per node
    std::vector<double>  h_cnt, h_sum, h_ssq;       // holdout stats per node
};

struct Params {
    int     n_thresh;
    double  sub_lo, sub_hi;
    std::vector<int> depth_choices;
    int     leaf_lo, leaf_hi;
    double  feat_lo, feat_hi;
    double  lam_lo, lam_hi;
    int     emit_raw, emit_zero, emit_rand;
};

struct Handle {
    std::vector<Tree> members;
};

struct Hist {
    // all-feature histogram: cnt/sum laid out as [feature*256 + bin]
    std::vector<int32_t> cnt;
    std::vector<double>  sum;
    void reset(int p) { cnt.assign((size_t)p * 256, 0); sum.assign((size_t)p * 256, 0.0); }
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
    int max_leaves = 0;    // >0 = leaf-wise (best-first) growth with this leaf budget
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
    T.val.push_back(mean);

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

    T.feat[id] = best_f; T.thr[id] = (uint8_t)best_t;
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
    T.val.push_back(mean);

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

    T.feat[id] = best_f; T.thr[id] = (uint8_t)best_t;

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
        T.val.push_back(sum / (double)(hi - lo));
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

        T.feat[nd.id] = nd.f; T.thr[nd.id] = (uint8_t)nd.t;
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
    out.feat.clear(); out.thr.clear(); out.left.clear(); out.right.clear(); out.val.clear();
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
    out.right.resize(nn); out.val.resize(nn);
    int32_t leaves = 0;
    for (int32_t i = 0; i < nn; ++i) {
        const int32_t src = order[i];
        const bool leaf = (T.feat[src] < 0) || (collapsed && (*collapsed)[src]);
        out.val[i] = T.val[src];
        if (leaf) { out.feat[i] = -1; out.thr[i] = 0; out.left[i] = out.right[i] = -1; ++leaves; }
        else {
            out.feat[i] = T.feat[src]; out.thr[i] = T.thr[src];
            out.left[i] = map[T.left[src]]; out.right[i] = map[T.right[src]];
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

} // namespace

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------
extern "C" {

void* sxe_create() { return new Handle(); }
void  sxe_free(void* h) { delete (Handle*)h; }
int   sxe_total_members(void* h) { return (int)((Handle*)h)->members.size(); }

// X: uint8 column-major (n rows x p cols), y: double[n]
int sxe_grow(void* hv, const uint8_t* X, const double* y, int64_t n, int p,
             int n_trees, int n_thresh,
             double sub_lo, double sub_hi,
             const int* depth_choices, int n_depth,
             int leaf_lo, int leaf_hi,
             double feat_lo, double feat_hi,
             double lam_lo, double lam_hi,
             int emit_raw, int emit_zero, int emit_rand,
             int hybrid_min, uint64_t seed)
{
    Handle* H = (Handle*)hv;
    const int emit = emit_raw + emit_zero + emit_rand;
    const size_t offset = H->members.size();
    H->members.resize(offset + (size_t)n_trees * emit);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int t = 0; t < n_trees; ++t) {
        Workspace ws;
        ws.rng.seed(splitmix64(seed + 0x1000003ULL * (uint64_t)t));
        ws.featbuf.resize(p);
        ws.hcnt.assign(256, 0); ws.hsum.assign(256, 0.0);
        ws.touched.reserve(256);

        // per-tree hyperparameters
        std::uniform_real_distribution<double> U(0.0, 1.0);
        TreeParams tp;
        int d = depth_choices[(int)(U(ws.rng) * n_depth) % n_depth];
        tp.max_depth = (d <= 0) ? 64 : d;
        tp.min_leaf = (int)std::lround(std::exp(
            std::log((double)leaf_lo) + U(ws.rng) * (std::log((double)leaf_hi) - std::log((double)leaf_lo))));
        tp.K = std::max(1, (int)std::lround((feat_lo + U(ws.rng) * (feat_hi - feat_lo)) * p));
        tp.n_thresh = n_thresh;

        // subsample without replacement; complement = holdout
        const double rate = sub_lo + U(ws.rng) * (sub_hi - sub_lo);
        const int64_t msub = (int64_t)std::llround(rate * (double)n);
        std::vector<int64_t> all(n);
        for (int64_t i = 0; i < n; ++i) all[i] = i;
        for (int64_t i = 0; i < msub; ++i) {
            std::uniform_int_distribution<int64_t> pick(i, n - 1);
            std::swap(all[i], all[pick(ws.rng)]);
        }
        ws.idx.assign(all.begin(), all.begin() + msub);
        std::vector<int64_t> hold(all.begin() + msub, all.end());

        GrownTree G;
        G.feat.reserve(2048);
        ws.hybrid_min = hybrid_min;
        if (msub >= (int64_t)hybrid_min) {
            Hist* H = ws.acquire(p);
            fill_hist(*H, X, y, n, p, ws.idx, 0, msub, ws.ybuf);
            grow_node_full(G, X, y, n, p, ws, 0, msub, 0, tp, H);
            ws.release(H);
        } else {
            grow_node_legacy(G, X, y, n, p, ws, 0, msub, 0, tp);
        }
        route_holdout(G, X, y, n, hold);

        const double nh = std::max(1.0, G.h_cnt.empty() ? 1.0 : G.h_cnt[0]);
        const double varh = std::max(1e-12,
            (G.h_ssq[0] - G.h_sum[0] * G.h_sum[0] / nh) / nh);

        std::vector<uint8_t> collapsed; std::vector<int32_t> leaves;
        int v = 0;
        if (emit_raw) {
            export_member_safe(G, nullptr, H->members[offset + (size_t)t * emit + v], t, 0); ++v;
        }
        if (emit_zero) {
            prune_flags(G, 0.0, varh, collapsed, leaves);
            export_member_safe(G, &collapsed, H->members[offset + (size_t)t * emit + v], t, 1); ++v;
        }
        if (emit_rand) {
            const double lam = std::exp(std::log(lam_lo) + U(ws.rng) * (std::log(lam_hi) - std::log(lam_lo)));
            prune_flags(G, lam, varh, collapsed, leaves);
            export_member_safe(G, &collapsed, H->members[offset + (size_t)t * emit + v], t, 2); ++v;
        }
    }
    return (int)H->members.size();
}

// out: column-major n x m (member j's predictions at out + j*n)
void sxe_predict(void* hv, const uint8_t* X, int64_t n, int p,
                 const int32_t* mem, int m, double* out)
{
    (void)p;
    Handle* H = (Handle*)hv;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int j = 0; j < m; ++j) {
        const Tree& T = H->members[mem[j]];
        double* o = out + (int64_t)j * n;
        for (int64_t r = 0; r < n; ++r) {
            int32_t id = 0;
            while (T.feat[id] >= 0)
                id = (X[(int64_t)T.feat[id] * n + r] <= T.thr[id]) ? T.left[id] : T.right[id];
            o[r] = T.val[id];
        }
    }
}

void sxe_predict_weighted(void* hv, const uint8_t* X, int64_t n, int p,
                          const int32_t* mem, const double* w, int m,
                          double intercept, double* out)
{
    (void)p;
    Handle* H = (Handle*)hv;
    for (int64_t r = 0; r < n; ++r) out[r] = intercept;
    for (int j = 0; j < m; ++j) {
        const Tree& T = H->members[mem[j]];
        const double wj = w[j];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int64_t r = 0; r < n; ++r) {
            int32_t id = 0;
            while (T.feat[id] >= 0)
                id = (X[(int64_t)T.feat[id] * n + r] <= T.thr[id]) ? T.left[id] : T.right[id];
            out[r] += wj * T.val[id];
        }
    }
}

// info[0]=n_nodes, info[1]=n_leaves, info[2]=base_id, info[3]=variant
void sxe_member_info(void* hv, int i, int32_t* info) {
    const Tree& T = ((Handle*)hv)->members[i];
    info[0] = (int32_t)T.feat.size(); info[1] = T.n_leaves;
    info[2] = T.base_id; info[3] = T.variant;
}

} // extern "C"

// ---------------------------------------------------------------------------
// batch-boosted dictionary generation (v3):
// grow small batches of shallow trees in parallel against the current
// residual, update the residual with a shrunk batch average, early-stop on
// validation MSE. Each tree emits ONE member: itself pruned on its holdout
// at a per-tree random strength. The sparse weight fit happens once, in
// Python, over the whole dictionary afterwards.
// ---------------------------------------------------------------------------
extern "C" int sxe_boost(void* hv,
             const uint8_t* X, const double* y, int64_t n, int p,
             const uint8_t* Xv, int64_t nv, const double* yv,
             int batch, int max_batches, double shrink, int patience, double tol,
             int n_thresh, int argmax_bins, int max_leaves,
             double sub_lo, double sub_hi,
             const int* depth_choices, int n_depth,
             int leaf_lo, int leaf_hi,
             double feat_lo, double feat_hi,
             double lam_lo, double lam_hi,
             int hybrid_min, uint64_t seed)
{
    Handle* H = (Handle*)hv;
    std::vector<double> r(y, y + n);          // residual (training target)
    std::vector<double> Fv(nv, 0.0);          // generation-ensemble val preds
    std::vector<double> bt((size_t)batch * n);   // per-tree train predictions
    std::vector<double> bv((size_t)batch * nv);  // per-tree val predictions

    double best = 1e300; int bad = 0; int made = 0;

    for (int b = 0; b < max_batches; ++b) {
        const size_t offset = H->members.size();
        H->members.resize(offset + batch);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int t = 0; t < batch; ++t) {
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
                std::log((double)leaf_lo) + U(ws.rng) * (std::log((double)leaf_hi) - std::log((double)leaf_lo))));
            tp.K = std::max(1, (int)std::lround((feat_lo + U(ws.rng) * (feat_hi - feat_lo)) * p));
            tp.n_thresh = n_thresh;
            tp.argmax_bins = argmax_bins;

            const double rate = sub_lo + U(ws.rng) * (sub_hi - sub_lo);
            const int64_t msub = (int64_t)std::llround(rate * (double)n);
            std::vector<int64_t> all(n);
            for (int64_t i = 0; i < n; ++i) all[i] = i;
            for (int64_t i = 0; i < msub; ++i) {
                std::uniform_int_distribution<int64_t> pick(i, n - 1);
                std::swap(all[i], all[pick(ws.rng)]);
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
            Tree& M = H->members[offset + t];
            if ((int64_t)hold.size() >= std::max<int64_t>(50, n / 50)) {
                const double lam = std::exp(std::log(lam_lo) + U(ws.rng) * (std::log(lam_hi) - std::log(lam_lo)));
                prune_flags(G, lam, varh, collapsed, leaves);
                export_member_safe(G, &collapsed, M, b * batch + t, 2);
            } else {
                export_member_safe(G, nullptr, M, b * batch + t, 0);  // holdout too small to prune reliably
            }

            // this tree's predictions on train and val rows
            double* pt = bt.data() + (size_t)t * n;
            for (int64_t i = 0; i < n; ++i) {
                int32_t id = 0;
                while (M.feat[id] >= 0)
                    id = (X[(int64_t)M.feat[id] * n + i] <= M.thr[id]) ? M.left[id] : M.right[id];
                pt[i] = M.val[id];
            }
            double* pv = bv.data() + (size_t)t * nv;
            for (int64_t i = 0; i < nv; ++i) {
                int32_t id = 0;
                while (M.feat[id] >= 0)
                    id = (Xv[(int64_t)M.feat[id] * nv + i] <= M.thr[id]) ? M.left[id] : M.right[id];
                pv[i] = M.val[id];
            }
        }
        made += batch;

        const double step = shrink / (double)batch;
        for (int t = 0; t < batch; ++t) {
            const double* pt = bt.data() + (size_t)t * n;
            for (int64_t i = 0; i < n; ++i) r[i] -= step * pt[i];
            const double* pv = bv.data() + (size_t)t * nv;
            for (int64_t i = 0; i < nv; ++i) Fv[i] += step * pv[i];
        }
        double mse = 0.0;
        for (int64_t i = 0; i < nv; ++i) { const double e = yv[i] - Fv[i]; mse += e * e; }
        mse /= (double)nv;
        if (mse < best - tol) { best = mse; bad = 0; }
        else if (++bad >= patience) break;
    }
    return (int)H->members.size();
}
