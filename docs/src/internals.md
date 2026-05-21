# Internals

This page sketches the moving parts behind the public API.

## Type hierarchy

```
TruncatedMvDistribution{D, R, S}
  • D :: MultivariateDistribution      # e.g. MvNormal
  • R :: TruncationRegion              # e.g. BoxTruncationRegion
  • S :: TruncatedMvDistributionState  # cache of computed moments
```

Two states are provided for the box-truncated MvNormal:

* [`TruncatedMvDistributionSecondOrderState`](@ref) caches only mean and
  covariance and computes them by direct cubature
  ([`BasicBoxTruncatedMvNormal`](@ref)).
* `TruncatedDistributions.BoxTruncatedMvNormalRecursiveMomentsState` (used by
  [`TruncatedMvNormal`](@ref) /
  [`RecursiveMomentsBoxTruncatedMvNormal`](@ref)) caches every primitive
  moment up to a configurable order, computed via the recursive moment
  formula of Kan and Robotti (2017).

## Recursive moment computation

The Kan-Robotti recursion expresses a `p`-th order primitive moment of a
truncated multivariate normal in terms of `(p-1)`-th order moments of
lower-dimensional conditional truncated normals at each box face. We
build a children tree once at construction; each `(μ, Σ)` refresh walks
the tree via [`update_distribution!`](@ref) so the topology is reused.

The base case of the recursion is the multivariate-normal box CDF.
Switch backends with [`set_kr_base_backend!`](@ref):

* `:hcubature`   — generic adaptive cubature; works with any element type
                   (e.g. ForwardDiff `Dual`).
* `:mvnormalcdf` — `MvNormalCDF.mvnormcdf` (Genz and Bretz, separation of
                   variables + QMC); Float64-only; typically 10–100×
                   faster on `n ≥ 3` Gaussian box probabilities.

## Lazy cache and `worst_tol`

Every accessor (`mean`, `cov`, `tp`) checks the cached error estimate
against the keyword argument `worst_tol` (default `1e-3`). If the cache is
"good enough" the cached value is returned; otherwise the package
recomputes. The cache is invalidated automatically by
[`update_distribution!`](@ref).

## Moment matching

* `warm_start_diagonal(μ̂, Σ̂, a, b)` — `n` independent 1-D LBFGS runs on
  the closed-form 1-D truncated-normal moment recurrence. Cheap.
* `vector_fg_true_loss(F, G, p, …)` — combined-evaluation `fg!` for
  Optim's `only_fg!` interface, returning both the scalar loss and the
  analytic gradient of `L(μ, U)` from a single Kan-Robotti recursion.
* `block_coord_descent(μ̂, Σ̂, a, b)` — block updates on size-1, 2, 3
  marginals, with marginal- or full-loss acceptance and greedy or
  softmax block selection. Returns `(μ, Σ, hist, picks)`.

## Parameter packing

LBFGS optimises an unconstrained parameter vector that interleaves `μ`
with the upper triangle of `U`, where `U Uᵀ = Σ^{-1}`. This guarantees
positive-definiteness of `Σ` for any `U`.

```julia
p = make_param_vec_from_μ_Σ(μ, Σ)   # pack
μ, Σ = make_μ_Σ_from_param_vec(p)   # unpack
```
