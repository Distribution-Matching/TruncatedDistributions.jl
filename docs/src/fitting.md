# Moment matching

Given target moments ``(\hat\mu, \hat\Sigma)`` and box bounds ``[a, b]``,
find an underlying ``(\mu, \Sigma)`` whose box-truncation to ``[a, b]``
reproduces the targets.

```julia
using TruncatedDistributions

μ̂ = [0.12, -0.12]
Σ̂ = [0.41 0.05; 0.05 0.41]
a = [-1.0, -1.5]
b = [ 1.5,  1.0]

μ_fit, Σ_fit, info = fit_mvnormal(μ̂, Σ̂, a, b)
@show info.method   # :lbfgs at this size; :bcd for n ≥ 7
@show info.loss     # ½‖μA − μ̂‖² + ½‖ΣA − Σ̂‖²_F
@show info.time
```

## Two algorithms

| Regime | Cost driver | Recommended method |
| --- | --- | --- |
| `n ≤ 6` | one full-`n` moment recursion is fast | `:lbfgs` (joint warm-start + LBFGS on the analytic gradient) |
| `n ≥ 7` | the recursion grows roughly factorially in `n` | `:bcd` (block updates on size-1 / 2 / 3 marginals) |

The block coordinate descent works on 2-D and 3-D *marginals* of the
truncated normal, choosing the block that has the largest per-target
residual. This keeps the per-iteration cost flat in `n` while still
correcting joint correlation structure.

Force a particular method:

```julia
fit_mvnormal(μ̂, Σ̂, a, b; method = :lbfgs)
fit_mvnormal(μ̂, Σ̂, a, b; method = :bcd)
fit_mvnormal(μ̂, Σ̂, a, b; method = :auto)   # default — picks by dimension
```

## Useful keyword arguments

| Keyword | Default | What it does |
| --- | --- | --- |
| `method` | `:auto` | `:auto`, `:lbfgs`, or `:bcd` |
| `n_threshold` | `6` | dimension cutoff used by `:auto` |
| `ftarget` | `1e-3` | stop when loss drops below this |
| `iterations` | `50` / `30` | outer-iteration cap (LBFGS / BCD) |
| `time_limit` | `60.0` | seconds (LBFGS only) |
| `μ_init`, `Σ_init` | warm-start | override starting point |
| `verbose` | `false` | print one line per iteration |

## Lower-level building blocks

```julia
warm_start_diagonal(μ̂, Σ̂, a, b)   # coordinate-wise 1D LBFGS warm-start
block_coord_descent(μ̂, Σ̂, a, b)   # underlying BCD; returns (μ, Σ, hist, picks)
moment_loss(d, μ̂, Σ̂)              # scalar loss read off the cached moments
```

See [Moment Matching of Box Truncated Multivariate Normal
Distributions](https://github.com/Distribution-Matching/paper_truncated_mv_normal)
for the derivation of the algorithms.
