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
