# Regularised Trees: feature-reuse penalty

This file tracks the design and implementation of this fork's modification to
LightGBM's split-gain calculation. It is not upstream LightGBM behavior — see
`CLAUDE.md` for general repo orientation.

## Goal

Make it less attractive to introduce a feature that hasn't been split on yet,
unless its raw gain is good enough to overcome a penalty. The intent is to
encourage the model to reuse a small set of features rather than spreading
splits thinly across many features.

This implements the framework of Deng & Runger, *Feature Selection via
Regularized Trees* ([arXiv:1201.1587](https://arxiv.org/abs/1201.1587)).

## Mechanism

Two config parameters in `include/LightGBM/config.h`:

- `unused_feature_penalty` — a double defaulting to `1.0` (no effect), checked
  to lie in `[0.0, 1.0]`. This is the paper's λ.
- `unused_feature_penalty_scope` — `"tree"` (default) or `"ensemble"`,
  controlling the scope over which the used-feature set F is accumulated.

For a candidate split on feature `i` at any leaf:

```
gain[i] -> unused_feature_penalty * gain[i]   if i is not in F
gain[i] -> gain[i]                            otherwise (unchanged)
```

### The two scopes, and why the choice matters

| scope | F is cleared | does \|F\| converge? |
| --- | --- | --- |
| `tree` | at the start of every tree | **no** |
| `ensemble` | never (only on `Init` / new training data) | **yes** |

`ensemble` is the paper's semantics: *"F now represents the feature set used in
previous splits not only from the current tree, but also from the previous
built trees."* Once F stops growing, F **is** the selected feature subset —
that convergence is the whole mechanism by which regularized trees perform
feature selection.

`tree` scope does not have that property. Because F is rebuilt from scratch
each tree, different trees are free to pick different representatives of the
same correlated group, and the union of features used across the model keeps
growing with `num_iterations` until it saturates at "all of them". λ only
changes the *rate*. Measured on a synthetic 5-block / 16-relevant / 30-noise
design (ideal |F| = 5), counting features used anywhere in the model:

```
  |F| vs rounds:             1     5    20   100   500  1000  2000
  tree     pen=1.0          13    20    41    46    46    46    46
  tree     pen=0.01          1     3     7    16    36    41    44   <- still climbing
  ensemble pen=1.0          13    20    41    46    46    46    46
  ensemble pen=0.01          1     2     2     5     5     5     5   <- converged, and correct
```

At `ensemble` / λ=0.01 the selection is exactly right: one member per block,
zero noise features, and held-out accuracy (0.784) matching both an oracle
model given only the 5 true drivers (0.784) and the all-46-feature model
(0.786).

`tree` remains the default so existing results stay reproducible, but
**`ensemble` is the scope to use for feature selection**.

### Implementation

All in `src/treelearner/serial_tree_learner.{h,cpp}`:

- `feature_used_for_penalty_` — a `std::vector<int8_t>` holding F, indexed by
  the same "real" feature index space as `feature_contri` /
  `monotone_constraints` (`Dataset::RealFeatureIndex`).
- `ResetUnusedFeaturePenalty()` sizes and clears it. Called from `Init()` and
  `ResetTrainingDataInner()` — the only points at which starting the
  accumulation over is correct.
- `BeforeTrain()` clears it **only under `tree` scope**, alongside the existing
  `col_sampler_.ResetByTree()` / `constraints_->Reset()` per-tree resets.
- `ResetConfig()` calls only `ResolveUnusedFeaturePenaltyScope()`, which
  re-reads the scope without touching F. This matters: the `reset_parameter`
  callback (commonly used for learning-rate decay) drives `ResetConfig()` on
  *every* iteration, and clearing F there would silently degrade `ensemble`
  scope back into `tree` scope.
- Set to `true` in `SplitInner()` the moment a split is actually committed
  to the tree (`best_split_info.feature`, already a real index).
- Applied in `ComputeBestSplitForFeature()`, immediately after the existing
  monotone-constraint penalty block, following the same
  `new_split.gain *= penalty` pattern already used there and for
  `feature_contri` (see `src/treelearner/feature_histogram.hpp`).

An unrecognised scope string is a `Log::Fatal`, so a typo fails loudly rather
than silently falling back to a default.

## Guided penalty (GRRF)

`unused_feature_penalty_gamma` (γ) and `unused_feature_penalty_guide` (a per-feature vector)
implement the follow-up paper, [Gene Selection with Guided Regularized Random
Forest](https://arxiv.org/abs/1209.6425). The single λ becomes per-feature:

```
lambda_i = (1 - gamma) * unused_feature_penalty + gamma * guide[i]
```

where `guide[i]` is intended to be a normalised importance `Imp'_i = Imp_i / max_j Imp_j` from a
preliminary ordinary model. A feature the preliminary model rated highly ends up with `lambda_i`
near 1 and is barely penalized; a probe gets `lambda_i` near λ₀ and is suppressed.

The point is to stop an arbitrary early split from deciding which features become penalty-exempt.
The paper's own motivation is the same failure — at a node with few instances gains tie often
(Theorem 1 bounds the number of distinct gains at N instances to `N(N+2)/4 - 1`), so the choice
among tied features is close to arbitrary.

Implementation: `ResolveUnusedFeaturePenaltyLambdas()` folds λ₀, γ and the guide into one
`lambda_i` per feature in `unused_feature_penalty_per_feature_`, computed once per config rather
than per split. `unused_feature_penalty_active_` short-circuits the split loop entirely when every
`lambda_i == 1.0`. A guide of the wrong length is a `Log::Fatal` — a silently misaligned guide would
penalize feature `i` by feature `j`'s score.

## Evaluation outcome

Measured in the companion repo (`../trees_testing`), on MADELON with matched cardinality and paired
per-seed comparisons over 10 seeds:

- GRRF **fixes the mechanism it targets**: against the unguided penalty it wins 10/10 seeds
  (+0.0598 AUC), and probes among the selected drop from 23.7 to 4.3.
- It **still loses** to plain top-k by gain importance at the same cardinality (−0.0199, SE 0.0032,
  0/10 seeds) — using the very importance scores that guide it, at double the training cost.
- Swapping the base learner to a random forest (`boosting=rf`, `feature_fraction_bynode=sqrt(p)/p`)
  **flips the verdict**: the unguided penalty then beats its matched RF baseline 10/10 (+0.0120).

Conclusion: this is a random-forest method. It depends on candidate re-randomisation at every node,
which gradient boosting does not have — under boosting, `F` is fixed early by a few noisy splits and
the penalty can only reorder gains, never evict a feature already admitted. See the companion repo's
`README.md`.

## Known limitation: degenerate trees at small λ

Under `tree` scope with λ ≲ 0.01, trees collapse to a **single feature each**
(measured: mean distinct features per tree = 1.0, min 1, max 1). Once the first
split lands, no other feature can clear the multiplicative bar within that
tree. Under `ensemble` scope this is far less severe, since F is already
populated when each tree starts, but it remains the thing to watch when tuning
λ down. Any evaluation should report distinct-features-per-tree alongside |F| —
metrics computed on the union across trees cannot see this failure.

## Build

`rebuild-dev.sh` builds only the C++ shared library and drops it into a
consuming project's venv (default: `trees_testing`), skipping the wheel build
entirely — seconds instead of minutes for an incremental change. It also
re-signs the dylib: on Apple Silicon, overwriting a dylib in place invalidates
its signature and the kernel SIGKILLs any process that loads it (Python exits
137 producing no output at all, which is otherwise baffling to debug).

`unused_feature_penalty` was regenerated through the existing config
machinery: `.ci/parameter-generator.py` reads the doc comments above the
field in `config.h` and regenerates `src/io/config_auto.cpp` and
`docs/Parameters.rst`. Re-run that script after changing the field's
comments or default.

## How this differs from the existing `feature_contri` parameter

`feature_contri` (upstream LightGBM) is a **static**, per-feature multiplier
fixed for the entire training run via config — the same scalar applies to a
given feature at every leaf, in every tree. `unused_feature_penalty` is
**dynamic and per-tree**: which features are "penalized" resets at the start
of every tree and changes as splits are committed within that tree. It also
applies uniformly to *all* unused features rather than letting the user pick
per-feature multipliers.

## Known limitation: staleness from incremental split-finding

LightGBM's tree-growth loop (`SerialTreeLearner::Train`) is incremental: when
a leaf is split, only its two new children get a freshly computed candidate
split in the next round. Every other pending leaf keeps whatever gain was
cached for it earlier, *unless* something explicitly invalidates and
recomputes it (this is exactly what the monotone-constraints machinery does
via `constraints_->Update()` + `RecomputeBestSplitForLeaf()` in
`SplitInner()`).

Our scheme does not currently do this. Consequence: if leaf `l1`'s cached
best split was computed before some feature `F` became "used" elsewhere in
the tree (e.g. via a split at a sibling leaf), `l1`'s cached number for `F`
still reflects the old (penalized) state and won't be reconsidered until
`l1` itself is chosen and split. This can make already-pending leaves
under-value a feature that has since become penalty-free, while leaves
created after that point correctly see it as unpenalized.

**Current status: accepted for the first round of evaluation.** The
constant-`lambda` (`unused_feature_penalty`) version above is what's
implemented now, without any invalidation/recompute pass, so that it can be
evaluated cheaply first. If staleness turns out to matter empirically, the
next step is to add a targeted invalidation pass modeled on
`RecomputeBestSplitForLeaf`, but cheaper than the monotone-constraints
version in one respect: since "a feature just became used" only ever
*increases* that one feature's own gain (no other feature's gain changes),
a leaf's cached best split only needs to be re-examined against that one
feature's freshly recomputed gain — not a full per-leaf, all-feature
recompute.

## Open questions for follow-up research

- Is a single global constant `lambda` sufficient, or should the penalty
  decay with the number of distinct features already used in the tree
  (to avoid an unbounded "rich get richer" snowball where early-used
  features stay permanently favoured)?
- Does the staleness above measurably affect results, or is it negligible
  in practice (as it apparently is treated for column sampling, which has
  no staleness concern, by contrast — see prior discussion)?
