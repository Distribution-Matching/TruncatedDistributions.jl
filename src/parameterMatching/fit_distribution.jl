# `fit(D, ...)` entry points in the Distributions.jl style: name the
# family, get back a distribution object. They wrap `fit_mvnormal` and
# `fit_mvt`, which remain available when the diagnostics in `info` are
# wanted.

# The fitted law is returned even when the loss target is not reached; the
# target may then be unattainable, so say so.
function _warn_unmet(info, kwargs)
    ftarget = get(kwargs, :ftarget, 1e-4)
    info.loss < ftarget ||
        @warn "Loss target not reached: the moments are matched only approximately, " *
              "and the target may be unattainable." loss = info.loss ftarget
    return nothing
end

"""
    fit(TruncatedMvNormal, μ̃, Σ̃, a, b; kwargs...) -> TruncatedMvNormal

Moment matching: return the box-truncated normal on `[a, b]` whose mean
and covariance match the targets `μ̃` and `Σ̃` (to the loss target
`ftarget`). Keyword arguments are those of [`fit_mvnormal`](@ref), which
also returns the fitting diagnostics. If the warm start finds an axis
with no univariate solution, the fit starts from the naive point
`(μ̃, Σ̃)` instead, with a warning. Warns if the loss target is not
reached.

```julia
d = fit(TruncatedMvNormal, μ̃, Σ̃, a, b)
mean(d), cov(d)          # ≈ (μ̃, Σ̃)
```
"""
function Distributions.fit(::Type{<:TruncatedMvNormal}, μ̂::AbstractVector,
                           Σ̂::AbstractMatrix, a::AbstractVector,
                           b::AbstractVector; kwargs...)
    μ, Σ, info = fit_mvnormal(μ̂, Σ̂, a, b; kwargs...)
    _warn_unmet(info, kwargs)
    return TruncatedMvNormal(collect(μ), Matrix(Σ), a, b)
end

"""
    fit(TruncatedMvT, ν, μ̃, Σ̃, a, b; kwargs...) -> TruncatedMvT

Moment matching for the box-truncated Student-t with fixed degrees of
freedom `ν`: return the law on `[a, b]` whose mean and covariance match
`μ̃` and `Σ̃`. Keyword arguments are those of [`fit_mvt`](@ref), which
also returns the fitting diagnostics. If the warm start finds an axis
with no univariate solution, the fit starts from the naive point instead,
with a warning. Warns if the loss target is not reached.
"""
function Distributions.fit(::Type{<:TruncatedMvT}, ν::Real, μ̂::AbstractVector,
                           Σ̂::AbstractMatrix, a::AbstractVector,
                           b::AbstractVector; kwargs...)
    μ, Σ, info = fit_mvt(ν, μ̂, Σ̂, a, b; kwargs...)
    _warn_unmet(info, kwargs)
    return TruncatedMvT(ν, collect(μ), Matrix(Σ), a, b)
end
