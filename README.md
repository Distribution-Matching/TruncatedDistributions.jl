# TruncatedDistributions.jl

A Julia package for truncated multivariate distributions. The current
functionality focuses on the **box-truncated multivariate normal**: fast
recursive moments via the Kan–Robotti recursion, and parameter fitting by
moment matching. The package is designed to grow to other multivariate
families and truncation region types; the type hierarchy
(`TruncatedMvDistribution{D, R, S}`) is generic in the underlying
distribution, the region, and the cached state.

For univariate truncation use `Distributions.truncated` from
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl);
this package complements it with the multivariate case.

## Features

- Box-truncated multivariate normal with the Kan–Robotti recursive moment
  computation (much faster than direct cubature beyond a few dimensions).
- Plug-in backend for the multivariate-normal box CDF at the recursion's
  base case: `:hcubature` (default, works for any element type) or
  `:mvnormalcdf` (Genz–Bretz QMC, much faster on Float64).
- One-line moment-matching front door, `fit_mvnormal(μ̂, Σ̂, a, b)`, which
  picks a joint LBFGS solver with explicit analytic gradient for small
  problems (`n ≤ 6`) and a hybrid block-coordinate-descent solver for
  larger ones.
- Lower-level building blocks (coordinate warm-start, block coordinate
  descent, raw KR moments) exposed for users who want to compose them
  directly.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/yoninazarathy/TruncatedDistributions.jl")
```

Julia 1.10 or newer.

## Quick start

### Build a truncated MvNormal and read off its moments

```julia
using TruncatedDistributions, PDMats

μ = [0.5, 0.5]
Σ = PDMat([1.0 1.2; 1.2 2.0])
a = [-1.0, -Inf]    # half-infinite lower box face is fine
b = [ 0.5,  1.0]

d = RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b)

tp(d)      # truncation probability
mean(d)    # truncated mean
cov(d)     # truncated covariance
```

Switch to the much faster MvNormalCDF base case if you do not need
ForwardDiff-Dual-typed parameters:

```julia
set_kr_base_backend!(:mvnormalcdf)
```

### Fit a truncated MvNormal to target moments

Given `(μ̂, Σ̂)`, find `(μ, Σ)` whose box truncation to `[a, b]` reproduces
them:

```julia
μ̂ = [0.12, -0.12]
Σ̂ = [0.41 0.05; 0.05 0.41]
a = [-1.0, -1.5]
b = [ 1.5,  1.0]

μ_fit, Σ_fit, info = fit_mvnormal(μ̂, Σ̂, a, b)
@show info.method   # :lbfgs at this size; :bcd for n ≥ 7
@show info.loss
@show info.time
```

The `info` NamedTuple carries the final loss
`L = ½‖μA − μ̂‖² + ½‖ΣA − Σ̂‖²_F`, wall time, the algorithm actually used,
and algorithm-specific trace data.

Force a particular method:

```julia
fit_mvnormal(μ̂, Σ̂, a, b; method = :lbfgs)   # joint LBFGS + warm-start
fit_mvnormal(μ̂, Σ̂, a, b; method = :bcd)     # hybrid block coord descent
fit_mvnormal(μ̂, Σ̂, a, b; method = :auto)    # default (size-based)
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

## Why two algorithms

| Regime | Cost driver | Recommended method |
| --- | --- | --- |
| `n ≤ 6` | one full-`n` Kan–Robotti call is fast | `:lbfgs` (joint warm-start + LBFGS on the analytic gradient) |
| `n ≥ 7` | full-`n` KR call grows roughly factorially | `:bcd` (block updates on size-1 / 2 / 3 marginals) |

The block coordinate descent works on 2-D and 3-D *marginals* of the
truncated normal, choosing the block that has the largest per-target
residual. This keeps the per-iteration cost flat in `n` while still
correcting joint correlation structure.

## Lower-level API (sketch)

```julia
warm_start_diagonal(μ̂, Σ̂, a, b)   # coordinate-wise 1D LBFGS warm-start
block_coord_descent(μ̂, Σ̂, a, b)   # underlying BCD; returns (μ, Σ, hist, picks)
moment_loss(d, μ̂, Σ̂)              # scalar loss read off cached KR moments
raw_moment(d, [1, 0, 2])           # raw multivariate moment via KR recursion
```

Bundled examples for testing live in `get_example(; n = 2, index = 1)` etc.
and the lists are enumerated by `get_example_sizes()` /
`get_num_examples(n)`.

## Tests

```julia
] test
```

The unit suite covers regions, the 1D moment recurrence, KR moments
cross-checked against direct cubature, gradient correctness, and the
`fit_mvnormal` end-to-end recovery.

Long-running experiment scripts (the data behind the companion paper)
live under `test/experiments/`; see the README there.

## Reference

The recursive moment computation follows Kan & Robotti (2017),
"On Moments of Folded and Truncated Multivariate Normal Distributions".
The joint-LBFGS and hybrid-BCD algorithms are described in the companion
paper *Moment Matching of Box Truncated Multivariate Normal Distributions*
(Carrizo, Draidi, Nazarathy).
