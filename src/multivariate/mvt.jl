"""
    TruncatedMvT(ν, μ, Σ, a, b; max_moment_levels = 2)

Box-truncated multivariate Student-t with `ν` degrees of freedom,
location `μ` and scale matrix `Σ` (the untruncated covariance is
`ν/(ν-2) Σ` for `ν > 2`). Box faces in `a` / `b` may be `±Inf`; for an
unbounded box, moments of order `max_moment_levels` require
`ν > max_moment_levels`. Moments come from the Student-t analogue of the
Kan–Robotti recursion (see `boxTruncatedMvTRecursiveMomentsState.jl`).

```julia
d = TruncatedMvT(5.0, [0.0, 0.0], [1.0 0.3; 0.3 1.0], [-1.0, -1.5], [1.5, Inf])
mean(d)
cov(d)
```
"""
const TruncatedMvT = TruncatedMvDistribution{D, BoxTruncationRegion,
                                             BoxTruncatedMvTRecursiveMomentsState} where D

function TruncatedMvT(ν::Real, μₑ::AbstractVector, Σₑ::AbstractMatrix,
                      a::AbstractVector, b::AbstractVector;
                      max_moment_levels::Int = 2)
    r = BoxTruncationRegion(Vector{Float64}(a), Vector{Float64}(b))
    s = BoxTruncatedMvTRecursiveMomentsState(ν, μₑ, Σₑ, r, max_moment_levels)
    return TruncatedMvDistribution(s.d, r, s)
end

function raw_moment(d::TruncatedMvT, k::Vector{Int})
    n = length(d)
    length(k) == n ||
        throw(DimensionMismatch("multi-index k has length $(length(k)); expected n = $n"))
    any(<(0), k) &&
        throw(ArgumentError("multi-index k must have non-negative entries; got k = $k"))
    sum(k) ≤ d.state.max_moment_levels || throw(ArgumentError(
        "raw_moment: total order $(sum(k)) exceeds max_moment_levels = $(d.state.max_moment_levels)"))
    raw_moment(d.state, k)
end

raw_moment_dict(d::TruncatedMvT) = raw_moment_dict(d.state)

function compute_tp(d::TruncatedMvT; kwargs...)
    s = d.state
    s.tp = raw_moment(s, zeros(Int, s.n))
    s.tp_err = _kr_characteristic_err()
    nothing
end

function compute_mean(d::TruncatedMvT; kwargs...)
    s = d.state
    s.max_moment_levels >= 1 || throw(ArgumentError(
        "mean(d) needs max_moment_levels ≥ 1; got $(s.max_moment_levels)"))
    m0 = raw_moment(s, zeros(Int, s.n))
    κ = zeros(Int, s.n)
    s.μ = map(1:s.n) do i
        κ[i] = 1; v = raw_moment(s, κ) / m0; κ[i] = 0; v
    end
    s.μ_err = _kr_characteristic_err()
    nothing
end

function compute_cov(d::TruncatedMvT; kwargs...)
    s = d.state
    s.max_moment_levels >= 2 || throw(ArgumentError(
        "cov(d) needs max_moment_levels ≥ 2; got $(s.max_moment_levels)"))
    n = s.n
    m0 = raw_moment(s, zeros(Int, n))
    μ = mean(d)
    κ = zeros(Int, n)
    Σ = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in i:n
        κ[i] += 1; κ[j] += 1
        Σ[i, j] = Σ[j, i] = raw_moment(s, κ) / m0 - μ[i] * μ[j]
        κ[i] = 0; κ[j] = 0
    end
    s.Σ = PDMat(0.5 .* (Σ .+ Σ'))
    s.Σ_err = _kr_characteristic_err()
    nothing
end

params(d::TruncatedMvT) = (d.state.ν, d.untruncated.μ, d.untruncated.Σ, d.region.a, d.region.b)
Base.minimum(d::TruncatedMvT) = d.region.a
Base.maximum(d::TruncatedMvT) = d.region.b
