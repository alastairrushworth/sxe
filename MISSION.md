# SXE — mission

*Draft for review. This states what the project is for and how it decides things,
so that the answer does not have to be reconstructed from the code each time.*

## What SXE is

A gradient-boosting-adjacent tree ensemble with a scikit-learn API and a C++ core.
Small batches of randomized, subsampled trees are grown **in parallel** against one
shared residual, each pruned on its own out-of-subsample holdout, and a non-negative
lasso then deletes most of the resulting trajectory.

## Who it is for

People already using LightGBM or XGBoost through scikit-learn who have a constraint
those libraries do not serve well:

- the model has to be **small** — it ships to a device, into a container image, or
  through a serving path where load time is the bottleneck;
- the result has to be **reproducible** — the same seed must give the same model,
  regardless of how many threads happened to be available;
- there is **no budget to tune** — the model has to be good on the first fit.

SXE is a bolt-on, not a platform. It should drop into an existing scikit-learn
pipeline and be removable just as easily.

## The bet

That a fundamentally different ensemble construction — parallel batches, randomized
members, aggressive post-hoc sparsification — buys a materially smaller model and a
reproducible one, at accuracy close enough to a tuned LightGBM that the trade is
worth making. Everything else follows from testing that bet honestly.

## What we optimise for, in order

1. **Model size at comparable accuracy.** The headline claim. It must be true of the
   bytes actually written to disk, not of a hypothetical format.
2. **Accuracy without tuning.** SXE should not need per-dataset hyperparameter search
   to be useful. A user's first fit is the product; a tuned fit is a bonus. Where a
   default is wrong on a whole class of data, that is a bug in the default.
3. **Reproducibility.** Same seed, same model — across thread counts, across runs,
   across `hybrid_min` and any other knob that exists for speed. A performance knob
   that changes the fitted model is a defect, not a feature.
4. **Handling the data people actually have.** Missing values and categorical columns
   are not edge cases; they are most real tabular data.
5. **Training speed.** Real, and the parallel batch structure is a genuine advantage,
   but it is the thing we trade away first.

## Non-goals

- **Beating LightGBM on accuracy everywhere.** It will not. The goal is to be close
  enough that the size and reproducibility wins are worth having, and to be honest
  about where the gap is.
- **Feature parity.** No ranking objectives, no distributed training, no GPU (for
  now), no plugin system.
- **Being the fastest library.** Speed is a constraint to stay within, not a target.
- **A general research framework.** SXE is one method, packaged well.

## How we work

- **One accounting rule, applied to both sides.** Every comparative number is
  measured the same way for SXE and for the baseline, and the raw output is committed
  alongside the claim. If a metric flatters us because of how it is computed, it is
  not a result.
- **Baselines get to be at their best.** The comparison is against a *tuned*
  LightGBM at its best thread count, because a win over a misconfigured baseline
  tells us nothing.
- **Claims track evidence.** A claim is stated at the strength the measurement
  supports — with the seed-to-seed spread, not just the mean — and it is weakened or
  withdrawn the moment better measurement says so. The project has done this twice
  already (a 5–20× size claim that was really 1.5–2×; benchmark figures that turned
  out not to be reproducible across C++ standard libraries). That is the norm, not an
  incident.
- **Caveats live next to the claim,** not in a footnote at the bottom of a report.
- **Defaults are a feature.** Changing a default is a product decision and gets
  measured across the whole benchmark suite, not on one dataset.

## What "good" looks like

Concrete, checkable targets rather than adjectives:

| | Target |
|---|---|
| Size | The serialized model is smaller than a tuned LightGBM's on a clear majority of the benchmark suite, and the size written to disk matches the size `summary()` reports. |
| Accuracy | SXE at defaults is within seed-to-seed noise of a tuned LightGBM on most of the suite, and never catastrophically behind on any dataset class. |
| Tuning | The gap between SXE-default and SXE-tuned is smaller than the gap between LightGBM-default and LightGBM-tuned. That is the measurable form of "does not need tuning". |
| Coverage | Fits data with missing values and categorical columns without the user pre-processing them. |
| Reproducibility | Bit-identical across thread counts and speed knobs, on one platform, and this is tested rather than asserted. |
| Trust | Every number in the README and the technical paper traces to a committed CSV produced by a script in the repo. |

## Open questions

Things the project has not decided, recorded here so they are not silently decided
by accident:

- Whether cross-platform reproducibility (same seed, same model on Linux and macOS)
  is worth replacing the C++ standard library's random distributions to get.
- Whether the squared-error loss stays, or second-order/Newton leaf values are worth
  the complexity to close the accuracy gap on smooth signal.
- Multiclass now shares one tree structure across classes (K-vector leaves,
  squared error on one-hot targets); whether a multinomial loss is worth adding on
  top of that is still open.
- Whether SXE should ever add a tuning helper, or whether shipping one would be an
  admission that goal 2 has failed.
