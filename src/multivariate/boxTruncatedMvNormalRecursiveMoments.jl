function raw_moment(d::RecursiveMomentsBoxTruncatedMvNormal, k::Vector{Int})
    # Validate against the cache contract before the lookup so callers get an
    # actionable message instead of a bare `KeyError`. The cache only holds
    # multi-indices with non-negative entries, length n, and total order
    # ≤ max_moment_levels (see `init_dicts`). This guard is on the user-facing
    # distribution method only; the hot `c_vector` path calls `raw_moment` on
    # child *states* directly and is left untouched.
    n = length(d)
    length(k) == n ||
        throw(DimensionMismatch("multi-index k has length $(length(k)); expected n = $n"))
    any(<(0), k) &&
        throw(ArgumentError("multi-index k must have non-negative entries; got k = $k"))
    L = d.state.max_moment_levels
    sum(k) ≤ L ||
        throw(ArgumentError(
            "raw_moment: requested multi-index k = $k has total order $(sum(k)), " *
            "which exceeds max_moment_levels = $L. Reconstruct the distribution " *
            "with `max_moment_levels = $(sum(k))` (or higher) to cache it via the " *
            "Kan–Robotti recursion, or call `moment(d, k)` for a direct (uncached) " *
            "cubature integral of E[∏ xᵢ^kᵢ]."))
    raw_moment(d.state, k)
end

function raw_moment_dict(d::RecursiveMomentsBoxTruncatedMvNormal)
    raw_moment_dict(d.state)
end

"""
    raw_moment_from_indices(d, indices)

Convenience accessor: convert a list of axis indices into a multi-index
`κ` (counting repeats) and return the corresponding raw moment.
`raw_moment_from_indices(d, [1, 2])` is equivalent to
`raw_moment(d, [1, 1, 0, …])` at `n ≥ 2`.
"""
function raw_moment_from_indices(d::RecursiveMomentsBoxTruncatedMvNormal, indices::Vector{Int})
    kappa = zeros(Int, length(d))
    for i in indices
        kappa[i] += 1
    end
    raw_moment(d, kappa)
end

# ---------------------------------------------------------------------------
# Kan–Robotti-backed tp / mean / cov
#
# The generic `compute_tp` / `compute_mean` / `compute_cov` of
# commonCompute.jl fall back to n-dimensional adaptive cubature, which is
# exponentially slow in n (minutes at n = 5). For the recursive-moments
# type the order-≤2 primitive moments are already in the Kan–Robotti cache
# (or one tree walk away), so `tp(d)`, `mean(d)` and `cov(d)` read them
# from the cache instead. The recursion is exact arithmetic on top of the
# base-case box probabilities, so the cached values inherit the base
# integrator's error scale; we record that characteristic scale (not a
# certified per-moment bound) in the state's error fields.
# ---------------------------------------------------------------------------

_kr_characteristic_err() = _KR_BASE_BACKEND[] === :mvnormalcdf ? 1e-5 : 1e-6

# The recursive-moment type reads its cached order-≤2 moments from the
# Kan–Robotti tree. Above `_MOMENT_MC_ABOVE[]` — where even walking the
# tree is expensive — it delegates to the shared Monte-Carlo estimator
# (`_fill_moments_mc!`, commonCompute.jl), using the seeded base-case RNG.
# (This type cannot be *constructed* much past n ≈ 8 anyway, since the
# constructor builds the full 2^{n-1}(n-1)! child tree; high-n callers
# should use `BasicBoxTruncatedMvNormal`, whose moments are pure MC.)

function compute_tp(d::RecursiveMomentsBoxTruncatedMvNormal; kwargs...)
    s = d.state
    if s.n > _MOMENT_MC_ABOVE[]
        _fill_moments_mc!(d, _MOMENT_MC_SAMPLES[], _KR_BASE_RNG[])
        return nothing
    end
    s.tp = raw_moment(s, zeros(Int, s.n))
    s.tp_err = _kr_characteristic_err()
    nothing
end

function compute_mean(d::RecursiveMomentsBoxTruncatedMvNormal; kwargs...)
    s = d.state
    if s.n > _MOMENT_MC_ABOVE[]
        _fill_moments_mc!(d, _MOMENT_MC_SAMPLES[], _KR_BASE_RNG[])
        return nothing
    end
    s.max_moment_levels >= 1 || throw(ArgumentError(
        "mean(d) via the Kan–Robotti cache needs max_moment_levels ≥ 1; " *
        "got $(s.max_moment_levels)"))
    n = s.n
    m0 = raw_moment(s, zeros(Int, n))
    κ = zeros(Int, n)
    μ = Vector{Float64}(undef, n)
    for i in 1:n
        κ[i] = 1
        μ[i] = raw_moment(s, κ) / m0
        κ[i] = 0
    end
    s.μ = μ
    s.μ_err = _kr_characteristic_err()
    nothing
end

function compute_cov(d::RecursiveMomentsBoxTruncatedMvNormal; kwargs...)
    s = d.state
    if s.n > _MOMENT_MC_ABOVE[]
        _fill_moments_mc!(d, _MOMENT_MC_SAMPLES[], _KR_BASE_RNG[])
        return nothing
    end
    s.max_moment_levels >= 2 || throw(ArgumentError(
        "cov(d) via the Kan–Robotti cache needs max_moment_levels ≥ 2; " *
        "got $(s.max_moment_levels)"))
    n = s.n
    m0 = raw_moment(s, zeros(Int, n))
    μ = mean(d)                       # cached by compute_mean above
    κ = zeros(Int, n)
    Σ = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in i:n
        κ[i] += 1
        κ[j] += 1
        Σ[i, j] = raw_moment(s, κ) / m0 - μ[i] * μ[j]
        Σ[j, i] = Σ[i, j]
        κ[i] = 0
        κ[j] = 0
    end
    s.Σ = PDMat(0.5 .* (Σ .+ Σ'))
    s.Σ_err = _kr_characteristic_err()
    nothing
end