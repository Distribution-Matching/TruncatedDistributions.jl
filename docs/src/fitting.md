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
@show info.loss     # ½‖μA − μ̂‖² + γ·r(n)·½‖ΣA − Σ̂‖²_F
@show info.time
```

The loss is ``L = L_1 + \gamma\, r(n)\, L_2`` with
``L_1 = \tfrac12\|\mu_A - \hat\mu\|^2``,
``L_2 = \tfrac12\|\Sigma_A - \hat\Sigma\|_F^2``, and
``r(n) = 2/(n+1)`` (the ratio of mean to covariance parameter counts).
The single weight ``\gamma`` (keyword `gamma`, default ``0.2``) is
scale-free across dimensions: ``\gamma = 1`` balances the two residual
blocks for every ``n``, and the default ``\gamma = 0.2`` weights the mean
five times as heavily as the covariance. See
[`set_loss_gamma!`](@ref).

For a reproducible fit under the randomised (`:mvnormalcdf`) base case,
pass a `seed`.

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
| `gamma` | `0.2` | covariance weight in the loss (`γ = 1` balances mean and covariance) |
| `ftarget` | `1e-3` | stop when loss drops below this |
| `seed` | `nothing` | seed the randomised base case for a reproducible fit |
| `iterations` | `50` / `30` | outer-iteration cap (LBFGS / BCD) |
| `time_limit` | `60.0` | seconds (LBFGS only) |
| `μ_init`, `Σ_init` | warm-start | override starting point |
| `verbose` | `false` | print one line per iteration |

The `:bcd` path additionally takes `bcd_selection` (`:softmax` default),
`bcd_accept_by` (`:marginal` default, or `:full` for monotone descent at
small `n`), and `bcd_polish` (default `false`).

## Lower-level building blocks

```julia
warm_start_diagonal(μ̂, Σ̂, a, b)   # coordinate-wise 1D LBFGS warm-start
block_coord_descent(μ̂, Σ̂, a, b)   # underlying BCD; returns (μ, Σ, hist, picks)
moment_loss(d, μ̂, Σ̂)              # scalar loss read off the cached moments
```

See [Moment Matching of Box Truncated Multivariate Normal
Distributions](https://github.com/Distribution-Matching/paper-truncated-mv-normal)
for the derivation of the algorithms.
