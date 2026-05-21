const _TrMv = TruncatedMvDistribution{D,R,S} where
    {D <: MultivariateDistribution, R <: TruncationRegion, S <: TruncatedMvDistributionState}

"""
    insupport(d::TruncatedMvDistribution, x)

True iff `x` is in the support of the untruncated distribution and inside
the truncation region.
"""
function insupport(d::_TrMv, x::AbstractArray)
    insupport(d.untruncated, x) && intruncationregion(d.region, x)
end

"""
    rand(d::TruncatedMvDistribution)
    rand(d::TruncatedMvDistribution, n::Integer)

Draw one sample (or `n` samples returned as an `n_dim × n` matrix) via
naive rejection sampling against the untruncated distribution. Efficient
when the truncation probability is not vanishingly small.
"""
rand(d::_TrMv) = rand_naive(d)

function rand(d::_TrMv, n::Integer)
    X = Matrix{Float64}(undef, length(d), n)
    for k in 1:n
        X[:, k] .= rand_naive(d)
    end
    return X
end

function rand_naive(d::_TrMv)
    while true
        candidate = rand(d.untruncated)
        intruncationregion(d.region, candidate) && return candidate
    end
end

length(d::_TrMv) = length(d.untruncated)
size(d::_TrMv)   = size(d.untruncated)
Base.eltype(::_TrMv) = Float64

"""
    pdf(d::TruncatedMvDistribution, x)

Density of the truncated distribution at `x`: `pdf(untruncated, x) / tp(d)`
when `x` lies inside the truncation region, and `0.0` otherwise.
"""
function pdf(d::_TrMv, x::AbstractArray; worst_tol = 1e-3)
    d.state.tp_err < worst_tol || compute_tp(d)
    if intruncationregion(d.region, x)
        return pdf(d.untruncated, x) / d.state.tp
    else
        return 0.0
    end
end

"""
    logpdf(d::TruncatedMvDistribution, x)

Log-density of the truncated distribution at `x`. Returns `-Inf` outside
the truncation region.
"""
function logpdf(d::_TrMv, x::AbstractArray; worst_tol = 1e-3)
    d.state.tp_err < worst_tol || compute_tp(d)
    if intruncationregion(d.region, x)
        return logpdf(d.untruncated, x) - log(d.state.tp)
    else
        return -Inf
    end
end

"""
    mean(d::TruncatedMvDistribution; worst_tol = 1e-3)

Truncated mean. The first call runs the moment computation and caches the
result; later calls reuse it as long as the cached error bound is below
`worst_tol`.
"""
function mean(d::_TrMv; worst_tol = 1e-3)
    d.state.μ_err < worst_tol || compute_mean(d)
    return d.state.μ
end

"""
    cov(d::TruncatedMvDistribution; worst_tol = 1e-3)

Truncated covariance matrix. Cached after the first call.
"""
function cov(d::_TrMv; worst_tol = 1e-3)
    d.state.Σ_err < worst_tol || compute_cov(d)
    return d.state.Σ
end

"""
    var(d::TruncatedMvDistribution)

Per-coordinate variance vector (diagonal of `cov(d)`).
"""
var(d::_TrMv; kwargs...) = diag(Matrix(cov(d; kwargs...)))

"""
    std(d::TruncatedMvDistribution)

Per-coordinate standard deviation vector.
"""
std(d::_TrMv; kwargs...) = sqrt.(var(d; kwargs...))

"""
    cor(d::TruncatedMvDistribution)

Correlation matrix `cov(d) ./ (std(d) * std(d)')`.
"""
function cor(d::_TrMv; kwargs...)
    σ = std(d; kwargs...)
    return Matrix(cov(d; kwargs...)) ./ (σ * σ')
end

"""
    moment(d::TruncatedMvDistribution, k)

Multivariate truncated moment for multi-index `k`. Heavy: integrates via
`hcubature_inf`. For the box-truncated MvNormal prefer
`raw_moment(d, k) / raw_moment(d, zeros(Int, length(d)))`, which goes
through the cached Kan & Robotti recursion.
"""
function moment(d::_TrMv, k::Vector{Int})
    compute_moment(d, k)
end

"""
    tp(d::TruncatedMvDistribution; worst_tol = 1e-3)

Truncation probability — the mass of the untruncated distribution that
falls inside the truncation region.
"""
function tp(d::_TrMv; worst_tol = 1e-3)
    d.state.tp_err < worst_tol || compute_tp(d)
    d.state.tp
end
