# Quick start

## Build a distribution and read off its moments

```julia
using TruncatedDistributions

μ = [0.5, 0.5]
Σ = [1.0 1.2;        # plain Matrix — auto-wrapped in a PDMat
     1.2 2.0]
a = [-1.0, -Inf]     # ±Inf box faces are fine
b = [ 0.5,  1.0]

d = TruncatedMvNormal(μ, Σ, a, b)

length(d)             # 2
insupport(d, [0, 0])  # true
tp(d)                 # probability mass inside the box, under the untruncated MvNormal
mean(d)               # truncated mean
cov(d)                # truncated covariance
var(d)                # diag(cov(d))
std(d)                # sqrt.(var(d))
cor(d)                # correlation matrix
pdf(d, [0.0, 0.0])    # truncated density (0 outside the box)
logpdf(d, [0.0, 0.0]) # truncated log-density (−Inf outside the box)
rand(d)               # one sample via rejection
rand(d, 100)          # 2 × 100 batch
```

`TruncatedMvNormal` is the friendly alias for the recommended
[`RecursiveMomentsBoxTruncatedMvNormal`](@ref). Both `μ`, `Σ`, and the
box bounds may be supplied as plain `Vector` / `Matrix` — they are
converted internally.

The underlying `MvNormal` is on `d.untruncated` and the truncation
region on `d.region`:

```julia
d.untruncated      # the MvNormal(μ, Σ)
d.untruncated.μ
d.untruncated.Σ    # a PDMat
d.region           # the BoxTruncationRegion
d.region.a; d.region.b
```

The first call to `mean(d)` / `cov(d)` runs the moment recursion and
caches the result. Subsequent calls reuse the cache.

## Tolerance and the cache

Each query carries a `worst_tol` keyword (default `1e-3`). If the cached
estimate's error bound is below this, the cached value is returned;
otherwise the package re-integrates. Tighten the tolerance to force a
higher-accuracy recompute:

```julia
mean(d; worst_tol = 1e-9)
```

## Arbitrary raw moments

`raw_moment(d, κ)` returns the unnormalised moment integral

```math
\int_{[a,b]} x_1^{\kappa_1} \cdots x_n^{\kappa_n}\, \phi(x;\mu,\Sigma)\, dx,
```

for any multi-index `κ`. Divide by `raw_moment(d, zeros(Int, n))` to get
the corresponding moment of the truncated distribution.

```julia
d  = TruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 4)
m0 = raw_moment(d, [0, 0])     # = tp(d) up to integration error
m4 = raw_moment(d, [2, 2])     # truncated E[x₁² x₂²] · tp(d)
```

`max_moment_levels` controls which orders are pre-allocated in the cache.

## Refresh an existing distribution in place

For inner-loop optimisation it is worth reusing the recursion-tree state
across `(μ, Σ)` updates rather than allocating a fresh tree each time:

```julia
update_distribution!(d.state, new_μ, new_Σ)
d_new = outer_dist_from_state(d.state)
mean(d_new)
```

At `n ≥ 5` this saves hundreds of milliseconds and hundreds of megabytes
per refresh compared to reconstructing.

## Choosing the base-case backend

```julia
set_kr_base_backend!(:hcubature)     # default; type-flexible
set_kr_base_backend!(:mvnormalcdf)   # Genz–Bretz QMC; Float64 only; faster
```

Keep `:hcubature` when you need ForwardDiff `Dual` numbers; otherwise
`:mvnormalcdf` is dramatically faster.

## Bundled examples

```julia
get_example_sizes()                   # available n
get_num_examples(2)                   # how many at that n
ne = get_example(n = 2, index = 4)
d  = dist_from_example(ne)
```
