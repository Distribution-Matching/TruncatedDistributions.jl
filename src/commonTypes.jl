"""
A truncation region defines a subset of space to which the distribution is
truncated. Concrete subtypes implement `intruncationregion(region, x)`,
returning `true` if `x` lies inside the region.
"""
abstract type TruncationRegion end

"""
Abstract state object holding computed quantities for a truncated
multivariate distribution. Every concrete subtype exposes at least

* `n::Int`           - dimension
* `tp::Float64`      - probability mass of the truncated region under the
                       untruncated distribution
* `tp_err::Float64`  - absolute error estimate for `tp`

Subtypes carrying mean/covariance state additionally expose

* `μ::Vector{Float64}` and `μ_err::Float64`
* `Σ::PDMat`           and `Σ_err::Float64`

Higher-moment subtypes expose `moment_dict::Dict{Vector{Int},Float64}`.
"""
abstract type TruncatedMvDistributionState end

"""
A truncated multivariate distribution. Combines an untruncated distribution
`D`, a truncation region `R`, and a state object `S` that caches computed
quantities (truncation probability, moments, etc.).
"""
struct TruncatedMvDistribution{D <: MultivariateDistribution,
                                R <: TruncationRegion,
                                S <: TruncatedMvDistributionState}
    untruncated::D
    region::R
    state::S
end

# Two-argument convenience constructor: build the state from the distribution
# using `S(d)` (states that need extra config — like `max_moment_levels` —
# expose richer constructors instead). The explicit `{D,R,S}` on the inner
# call preserves the requested type parameters; without it, Julia would
# narrow `D` to `typeof(d)` and the resulting object would no longer match
# the `const` aliases (e.g. `BasicBoxTruncatedMvNormal`) that pin `D` at
# the abstract `MvNormal`.
function TruncatedMvDistribution{D,R,S}(d::D, r::R) where
        {D <: MultivariateDistribution, R <: TruncationRegion, S <: TruncatedMvDistributionState}
    return TruncatedMvDistribution{D,R,S}(d, r, S(d))
end

"""
    TruncatedMvDistributionSecondOrderState <: TruncatedMvDistributionState

Minimal cached state for a truncated multivariate distribution: stores
the truncation probability, mean, covariance, and their error estimates.
Used by [`BasicBoxTruncatedMvNormal`](@ref), where moments are computed
by direct cubature on demand.
"""
mutable struct TruncatedMvDistributionSecondOrderState <: TruncatedMvDistributionState
    n::Int              # dimension
    tp::Float64         # truncation probability
    μ::Vector{Float64}  # mean vector
    Σ::PDMat            # covariance matrix
    tp_err::Float64
    μ_err::Float64
    Σ_err::Float64
    function TruncatedMvDistributionSecondOrderState(d::MultivariateDistribution)
        n = length(d)
        new(n,
            NaN,
            Vector{Float64}(undef, 0),
            PDMat(Array{Float64,2}(I, n, n)),
            Inf, Inf, Inf)
    end
end
