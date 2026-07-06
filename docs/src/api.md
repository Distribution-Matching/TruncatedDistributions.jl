# API reference

## Distribution types

```@docs
TruncatedMvDistribution
TruncatedMvDistributionState
TruncatedMvDistributionSecondOrderState
TruncationRegion
BoxTruncationRegion
EllipticalTruncationRegion
BasicBoxTruncatedMvNormal
RecursiveMomentsBoxTruncatedMvNormal
TruncatedMvNormal
```

## Distribution queries

```@docs
insupport
pdf
logpdf
rand
mean
cov
var
std
cor
tp
params
moment
moments
```

## Raw moments and the recursion tree

```@docs
raw_moment
raw_moment_dict
raw_moment_from_indices
compute_moments
update_distribution!
outer_dist_from_state
set_kr_base_backend!
get_kr_base_backend
set_kr_base_rng!
get_kr_base_rng
```

## High-dimensional Monte-Carlo moments

At high `n` the Kan–Robotti tree is infeasible even to build, and adaptive
cubature is exponential; `tp`, `mean`, and `cov` then fall back to
rejection sampling. `mc_moments` exposes that estimate directly, and
`set_moment_mc!` controls the threshold and sample count.

```@docs
mc_moments
set_moment_mc!
```

The internal cached state lives in
`TruncatedDistributions.BoxTruncatedMvNormalRecursiveMomentsState` and is
not part of the public API; users construct
[`TruncatedMvNormal`](@ref) / [`RecursiveMomentsBoxTruncatedMvNormal`](@ref)
and reach the state via `d.state`.

## Numerical helpers

```@docs
hcubature_inf
```

## Moment matching

```@docs
fit_mvnormal
warm_start_diagonal
block_coord_descent
moment_loss
vector_moment_loss
set_loss_gamma!
get_loss_gamma
```

## Univariate tools — dynamic moment matching

```@docs
dynamic_fit_locationscale
dynamic_fit_exponential
DynamicMomentMatch
solution
```

## Bundled examples

```@docs
get_example
get_num_examples
get_example_sizes
dist_from_example
```

## Index

```@index
```
