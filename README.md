# TruncatedDistributions.jl

[![CI](https://github.com/Distribution-Matching/TruncatedDistributions.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Distribution-Matching/TruncatedDistributions.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://Distribution-Matching.github.io/TruncatedDistributions.jl/stable/)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://Distribution-Matching.github.io/TruncatedDistributions.jl/dev/)

A Julia package for truncated multivariate distributions. The current
functionality focuses on the **box-truncated multivariate normal**: a
distribution object that exposes the usual `mean`, `cov`, `pdf`, `rand`,
plus arbitrary raw moments via the recursive moment formula of
[Kan and Robotti (2017)](https://doi.org/10.1080/10618600.2017.1322092).
On top of that sits an optional moment-matching parameter-fitting layer.

The type hierarchy (`TruncatedMvDistribution{D, R, S}`) is generic in the
underlying distribution `D`, the truncation region `R`, and the cached
state `S`, so the package is designed to grow to other multivariate
families and other region types.

For univariate truncation use `Distributions.truncated` from
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl); this
package complements it with the multivariate case.

## Features

- Box-truncated multivariate normal as a first-class distribution object —
  `mean`, `cov`, `pdf`, `rand`, `length`, `size`, `insupport`, truncation
  probability `tp(d)`, and arbitrary raw moments `raw_moment(d, κ)` via
  the recursion of Kan and Robotti (2017).
- ±Inf entries in the box bounds are handled natively (half-infinite and
  doubly-infinite faces both work).
- Lazy, cached, mutable state. Moments are computed on demand and stored;
  later queries reuse the cache, governed by a tolerance argument
  (`worst_tol`). The state of an existing distribution can be refreshed
  in place with new `(μ, Σ)` via `update_distribution!` — used to keep
  the O(n!) recursion tree across inner-loop optimiser iterations.
- Two interchangeable backends for the base case of the recursion:
  `:hcubature` (default, works with any element type) and `:mvnormalcdf`
  (Genz and Bretz separation-of-variables + QMC; Float64-only, much
  faster).
- Optional moment-matching layer: `fit_mvnormal(μ̂, Σ̂, a, b)` recovers
  `(μ, Σ)` whose box-truncation has the requested moments. Picks joint
  LBFGS + warm-start for small problems and a hybrid block-coordinate
  solver for larger ones.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Distribution-Matching/TruncatedDistributions.jl")
```

Julia 1.10 or newer.

## Quick start

### Build a distribution and read off its moments

```julia
using TruncatedDistributions, PDMats

μ = [0.5, 0.5]
Σ = PDMat([1.0 1.2; 1.2 2.0])
a = [-1.0, -Inf]      # ±Inf box faces are fine
b = [ 0.5,  1.0]

d = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b)

length(d)             # 2
insupport(d, [0, 0])  # true
tp(d)                 # probability mass inside the box (under the untruncated MvNormal)
mean(d)               # truncated mean
cov(d)                # truncated covariance
pdf(d, [0.0, 0.0])    # truncated density
rand(d)               # one sample via rejection from the untruncated MvNormal
```

The first call to `mean(d)` / `cov(d)` runs the recursive moment formula
of Kan and Robotti (2017) and caches the result. Repeated calls reuse the
cache.

### Tolerance and the cache

Each query carries a `worst_tol` keyword (default `1e-3`). If the cached
estimate's error bound is below this, the cached value is returned;
otherwise the package re-integrates. Tighten the tolerance to force a
higher-accuracy recompute:

```julia
mean(d; worst_tol = 1e-9)
```

### Arbitrary raw moments

`raw_moment(d, κ)` returns the unnormalised moment integral

```
∫_{[a,b]} x_1^{κ_1} … x_n^{κ_n}  φ(x; μ, Σ)  dx
```

for any multi-index `κ`. Divide by `raw_moment(d, zeros(Int, n))` to get
the corresponding truncated-distribution moment.

```julia
d  = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 4)
m0 = raw_moment(d, [0, 0])              # = tp(d) up to integration error
m4 = raw_moment(d, [2, 2])              # truncated E[x₁² x₂²] · tp(d)
```

`max_moment_levels` (default `2`, set higher when you need them) controls
which orders are pre-allocated in the cache.

### Refresh an existing distribution in place

For inner-loop optimisation it is worth reusing the recursion-tree state
across `(μ, Σ)` updates rather than allocating a fresh tree each time:

```julia
update_distribution!(d.state, new_μ, new_Σ)   # rewrite (μ, Σ) in place
d_new = outer_dist_from_state(d.state)        # cheap wrapper around the refreshed state
mean(d_new)                                    # recomputed; cache was invalidated
```

This matters a lot at `n ≥ 5`: constructing the tree alone takes hundreds
of milliseconds and hundreds of megabytes at `n = 6`.

### Choosing the base-case backend

Two backends bottom out the recursion at the multivariate-normal box CDF:

```julia
set_kr_base_backend!(:hcubature)     # default; type-flexible (Dual numbers, BigFloat, …)
set_kr_base_backend!(:mvnormalcdf)   # Genz–Bretz QMC; Float64 only; much faster
```

Stick with `:hcubature` when you differentiate through the distribution
via ForwardDiff. Switch to `:mvnormalcdf` for plain Float64 work.

## Moment-matching: `fit_mvnormal`

Given target moments `(μ̂, Σ̂)`, find `(μ, Σ)` whose box-truncation to
`[a, b]` reproduces them.

```julia
μ̂ = [0.12, -0.12]
Σ̂ = [0.41 0.05; 0.05 0.41]
a = [-1.0, -1.5]
b = [ 1.5,  1.0]

μ_fit, Σ_fit, info = fit_mvnormal(μ̂, Σ̂, a, b)
@show info.method   # :lbfgs at this size; :bcd for n ≥ 7
@show info.loss     # ½‖μA − μ̂‖² + ½‖ΣA − Σ̂‖²_F
@show info.time
```

Force a particular method:

```julia
fit_mvnormal(μ̂, Σ̂, a, b; method = :lbfgs)   # joint LBFGS + warm-start
fit_mvnormal(μ̂, Σ̂, a, b; method = :bcd)     # hybrid block coordinate descent
fit_mvnormal(μ̂, Σ̂, a, b; method = :auto)    # default (dimension-based)
```

Useful keyword arguments:

| Keyword | Default | What it does |
| --- | --- | --- |
| `method` | `:auto` | `:auto`, `:lbfgs`, or `:bcd` |
| `n_threshold` | `6` | dimension cutoff used by `:auto` |
| `ftarget` | `1e-3` | stop when loss drops below this |
| `iterations` | `50` / `30` | outer-iteration cap (LBFGS / BCD) |
| `time_limit` | `60.0` | seconds (LBFGS only) |
| `μ_init`, `Σ_init` | warm-start | override starting point |
| `verbose` | `false` | print one line per iteration |

### Why two algorithms

| Regime | Cost driver | Recommended method |
| --- | --- | --- |
| `n ≤ 6` | one full-`n` moment recursion is fast | `:lbfgs` (joint warm-start + LBFGS on the analytic gradient) |
| `n ≥ 7` | the recursion grows roughly factorially in `n` | `:bcd` (block updates on size-1 / 2 / 3 marginals) |

The block coordinate descent works on 2-D and 3-D *marginals* of the
truncated normal, choosing the block that has the largest per-target
residual. This keeps the per-iteration cost flat in `n` while still
correcting joint correlation structure.

### Lower-level fitting API

```julia
warm_start_diagonal(μ̂, Σ̂, a, b)   # coordinate-wise 1D LBFGS warm-start
block_coord_descent(μ̂, Σ̂, a, b)   # underlying BCD; returns (μ, Σ, hist, picks)
moment_loss(d, μ̂, Σ̂)              # scalar loss read off the cached moments
```

## Bundled examples

A small library of pre-defined cases for testing and benchmarking:

```julia
get_example_sizes()                 # which n are available
get_num_examples(2)                 # how many at that n
ne = get_example(n = 2, index = 4)  # the example as a NormalExample
d  = dist_from_example(ne)          # built as a RecursiveMomentsBoxTruncatedMvNormal
```

## Tests

```julia
] test
```

The unit suite covers regions, the 1D moment recurrence, the multivariate
recursion cross-checked against direct cubature, gradient correctness, and
the `fit_mvnormal` end-to-end recovery.

Long-running experiment scripts (the data behind the companion paper)
live under `test/experiments/`; see the README there.

## References

- Kan, R. and Robotti, C. (2017). "On Moments of Folded and Truncated
  Multivariate Normal Distributions." *Journal of Computational and
  Graphical Statistics*, 26(4), 930–934.
  https://doi.org/10.1080/10618600.2017.1322092
- Genz, A. and Bretz, F. (2009). *Computation of Multivariate Normal and
  t Probabilities*. Lecture Notes in Statistics 195, Springer. (Algorithm
  used by the optional `:mvnormalcdf` backend.)
- The joint-LBFGS and hybrid-BCD fitting algorithms are described in the
  companion paper *Moment Matching of Box Truncated Multivariate Normal
  Distributions* (Carrizo, Draidi, Nazarathy).
