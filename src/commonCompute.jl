# Truncated-moment computation for a box-truncated distribution, shared by
# the lightweight `BasicBoxTruncatedMvNormal` (cubature / Monte Carlo) and
# used as a Monte-Carlo fallback for the recursive-moment type at high n,
# where the Kan–Robotti tree is infeasible to even build.

# --- Monte-Carlo moments -------------------------------------------------
#
# `tp(d)`, `mean(d)` and `cov(d)` fall back to rejection sampling for
# dimension n > `_MOMENT_MC_ABOVE[]`: adaptive cubature is exponential in n
# and the exact Kan–Robotti recursion cannot be built past n ≈ 7. Draws
# use a caller-supplied RNG so a seeded pipeline stays reproducible.

const _MOMENT_MC_ABOVE   = Ref{Int}(6)        # use MC for n > this
const _MOMENT_MC_SAMPLES = Ref{Int}(200_000)  # accepted draws for MC moments

"""
    set_moment_mc!(; above = 6, samples = 200_000)

Configure when and how `tp(d)`, `mean(d)`, `cov(d)` fall back to Monte
Carlo: for dimension `n > above` they use rejection sampling with
`samples` accepted draws instead of adaptive cubature (or the
infeasible-at-high-`n` Kan–Robotti recursion). Returns the previous
`(above, samples)`.
"""
function set_moment_mc!(; above::Integer = _MOMENT_MC_ABOVE[],
                          samples::Integer = _MOMENT_MC_SAMPLES[])
    prev = (_MOMENT_MC_ABOVE[], _MOMENT_MC_SAMPLES[])
    _MOMENT_MC_ABOVE[] = above
    _MOMENT_MC_SAMPLES[] = samples
    return prev
end

# Rejection-sampling estimate of (tp, μ, Σ) for N(μ0, Σ0) truncated to the
# box [a, b]. Samples x = μ0 + L z (L the Cholesky factor, z standard
# normal) in place — the per-draw `rand!(::MvNormal)` allocates, which
# dominates at the millions of draws high-n needs.
function _mc_box_moments(μ0::AbstractVector, Σ0::AbstractMatrix,
                         a::AbstractVector, b::AbstractVector,
                         n_samples::Integer, rng::AbstractRNG)
    n = length(μ0)
    μc = collect(Float64, μ0)
    L  = Matrix(cholesky(Symmetric(Matrix(Σ0))).L)
    sums  = zeros(n)
    cross = zeros(n, n)
    z = Vector{Float64}(undef, n)
    x = Vector{Float64}(undef, n)
    kept = 0
    tried = 0
    maxtries = max(50 * n_samples, 10^7)
    while kept < n_samples && tried < maxtries
        randn!(rng, z)
        mul!(x, L, z)
        x .+= μc
        tried += 1
        inside = true
        @inbounds for i in 1:n
            if x[i] < a[i] || x[i] > b[i]
                inside = false
                break
            end
        end
        inside || continue
        kept += 1
        @inbounds for i in 1:n
            xi = x[i]
            sums[i] += xi
            for j in i:n
                cross[i, j] += xi * x[j]
            end
        end
    end
    kept == 0 && error("MC moments: no draws landed in the box after " *
        "$tried tries; the truncation probability is too small for " *
        "rejection sampling")
    @inbounds for i in 1:n, j in 1:(i - 1)
        cross[i, j] = cross[j, i]
    end
    μ = sums ./ kept
    Σ = cross ./ kept .- μ * μ'
    return kept / tried, μ, 0.5 .* (Σ .+ Σ'), kept
end

"""
    mc_moments(d; n_samples = 200_000, rng = Random.default_rng())
        -> (tp, μ, Σ)

Rejection-sampling estimate of the truncation probability, mean, and
covariance of the box-truncated distribution `d`: draw from the
untruncated law and keep the draws in the box. Works at any dimension
(no moment recursion, no cubature); the Monte-Carlo standard error on
each entry scales as `1/sqrt(n_samples)`. For a reproducible estimate
pass a seeded `rng`.
"""
function mc_moments(d::TruncatedMvDistribution;
                    n_samples::Integer = _MOMENT_MC_SAMPLES[],
                    rng::AbstractRNG = Random.default_rng())
    tp, μ, Σ, _ = _mc_box_moments(d.untruncated.μ, d.untruncated.Σ,
                                  d.region.a, d.region.b, n_samples, rng)
    return tp, μ, Σ
end

# Fill the state's (tp, μ, Σ) cache from one Monte-Carlo pass.
function _fill_moments_mc!(d::TruncatedMvDistribution, n_samples, rng)
    tp, μ, Σ, kept = _mc_box_moments(d.untruncated.μ, d.untruncated.Σ,
                                     d.region.a, d.region.b, n_samples, rng)
    se = 1.0 / sqrt(max(kept, 1))
    d.state.tp = tp;  d.state.tp_err = se
    d.state.μ  = μ;   d.state.μ_err  = se
    d.state.Σ  = PDMat(Σ + (1e-10 * (tr(Σ) + 1)) * I)
    d.state.Σ_err = se
    return nothing
end

_moment_use_mc(d, alg) =
    alg === :mc || (alg === :auto && length(d.region.a) > _MOMENT_MC_ABOVE[])

# --- Dispatchers ---------------------------------------------------------

function compute_tp(d::TruncatedMvDistribution{D,R};
                    tol::Float64 = 10e-4,
                    tol_step::Int = 10^5,
                    alg::Symbol = :auto,
                    n_samples::Integer = _MOMENT_MC_SAMPLES[],
                    rng::AbstractRNG = Random.default_rng()) where {D,R}
    if _moment_use_mc(d, alg)
        _fill_moments_mc!(d, n_samples, rng)
    else
        d.state.tp, d.state.tp_err =
            hcubature_inf((x)->pdf(d.untruncated,x), d.region.a, d.region.b)
    end
    nothing
end

function compute_mean(d::TruncatedMvDistribution{D,R};
                    tol::Float64 = 10e-4,
                    tol_step::Int = 10^5,
                    alg::Symbol = :auto,
                    n_samples::Integer = _MOMENT_MC_SAMPLES[],
                    rng::AbstractRNG = Random.default_rng()) where {D,R}
    if _moment_use_mc(d, alg)
        _fill_moments_mc!(d, n_samples, rng)
    else
        d.state.μ, d.state.μ_err =
            hcubature_inf((x)->pdf(d,x)*x, d.region.a, d.region.b; maxevals = 10^6)
    end
    nothing
end

function compute_cov(d::TruncatedMvDistribution{D,R};
                    tol::Float64 = 10e-4,
                    tol_step::Int = 10^5,
                    alg::Symbol = :auto,
                    n_samples::Integer = _MOMENT_MC_SAMPLES[],
                    rng::AbstractRNG = Random.default_rng()) where {D,R}
    if _moment_use_mc(d, alg)
        _fill_moments_mc!(d, n_samples, rng)
    else
        μ = mean(d)
        res = hcubature_inf((x)->pdf(d,x)*(x-μ)*(x-μ)', d.region.a, d.region.b; maxevals = 10^6)
        d.state.Σ, d.state.Σ_err = PDMat(0.5*(res[1] + res[1]')), res[2]
    end
    nothing
end

function compute_moment(d::TruncatedMvDistribution{D,R}, k::Vector{Int};
                        alg::Symbol = :hc) where {D,R}
    if alg == :mc
        @error("compute_moment via MC not implemented; use mc_moments for (tp, μ, Σ)")
    elseif alg == :hc
        return hcubature_inf((x)->pdf(d,x)*prod(x.^k),d.region.a, d.region.b; maxevals = 10^6)[1]
    else
        error("Unknown algorithm $(alg)")
    end
    nothing
end
