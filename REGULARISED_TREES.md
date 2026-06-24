# Regularised Trees: per-tree feature-reuse penalty

This file tracks the design and implementation of this fork's modification to
LightGBM's split-gain calculation. It is not upstream LightGBM behavior — see
`CLAUDE.md` for general repo orientation.

## Goal

Within a single tree, make it less attractive to introduce a feature that
hasn't been used yet in that tree, unless its raw gain is good enough to
overcome a penalty. The intent is to encourage trees to reuse a small set of
features rather than spreading splits thinly across many features.

## Mechanism

A new scalar config parameter, `unused_feature_penalty` (`include/LightGBM/config.h`),
defaults to `1.0` (no effect) and is checked to lie in `[0.0, 1.0]`.

For a candidate split on feature `i` at any leaf:

```
gain[i] -> unused_feature_penalty * gain[i]   if i has not yet been used for
                                               a split anywhere in the current
                                               tree
gain[i] -> gain[i]                            otherwise (unchanged)
```

This is implemented in `src/treelearner/serial_tree_learner.{h,cpp}`:

- `feature_used_in_cur_tree_` — a `std::vector<int8_t>`, indexed by the same
  "real" feature index space as `feature_contri`/`monotone_constraints`
  (`Dataset::RealFeatureIndex`), tracking which features have been split on
  in the tree currently being built.
- Reset to all-`false` in `BeforeTrain()`, alongside the existing
  `col_sampler_.ResetByTree()` / `constraints_->Reset()` per-tree resets.
- Set to `true` in `SplitInner()` the moment a split is actually committed
  to the tree (`best_split_info.feature`).
- Applied in `ComputeBestSplitForFeature()`, immediately after the existing
  monotone-constraint penalty block, following the same
  `new_split.gain *= penalty` pattern already used there and for
  `feature_contri` (see `src/treelearner/feature_histogram.hpp`).

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
