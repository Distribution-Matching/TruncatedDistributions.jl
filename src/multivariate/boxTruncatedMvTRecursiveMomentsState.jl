# Recursive raw moments of the box-truncated multivariate Student-t.
#
# The Kan–Robotti step for the normal rests on ∇φ = -Σ⁻¹(x-μ)φ. For a
# target law t_ν(μ, S) the analogous identity uses a *helper* law
#
#     H = t_{ν-2}(μ, Σ_H),   Σ_H = ν S / (ν - 2),
#
# for which ∇f_H(x) = -Σ_H⁻¹ (x - μ) f_T(x) (the degrees-of-freedom shift,
# Lemma 1 of the paper, read with ν ↦ ν - 2). Integrating x^κ against it
# over the box gives
#
#     m_{κ+e_i} = μ_i m_κ + Σ_j (Σ_H)_{ij} c_j,
#     c_j = κ_j m^H_{κ-e_j} + a_j^{κ_j} f^H_j(a_j) F^{a}_j - b_j^{κ_j} f^H_j(b_j) F^{b}_j,
#
# with f^H_j the univariate t_{ν-2}(μ_j, (Σ_H)_{jj}) density and F^{a,b}_j
# the raw moments of the conditional law of H given x_j = a_j (resp. b_j),
# truncated to the remaining box. That conditional law is again a
# Student-t: dof ν - 1, location μ_{-j} + Σ_H[-j,j](a_j - μ_j)/Σ_H[j,j],
# scale (ν - 2 + δ²)/(ν - 1) · (Σ_H[-j,-j] - Σ_H[-j,j]Σ_H[j,-j]/Σ_H[j,j]),
# δ² = (a_j - μ_j)²/Σ_H[j,j]. As ν → ∞ every law involved tends to the
# Gaussian of the parent and the recursion tends to Kan–Robotti.
#
# Each node therefore owns a helper node (dof ν-2, order L-2) and 2n face
# children (dof ν-1, dimension n-1, order L-1). A node whose helper would
# have dof ≤ 0 cannot use the identity; such a node falls back to direct
# cubature of all its moments (it only arises at low dof deep in the tree).

"""
    BoxTruncatedMvTRecursiveMomentsState <: TruncatedMvDistributionState

Cached state for the recursive moment computation of a box-truncated
multivariate Student-t with `ν` degrees of freedom, location `μ` and scale
matrix `Σ` (not the covariance). Holds the helper node and face children
of the recursion, a dictionary of raw moments up to total order
`max_moment_levels`, and the mean / covariance / truncation probability
cache.

Used by [`TruncatedMvT`](@ref).
"""
mutable struct BoxTruncatedMvTRecursiveMomentsState <: TruncatedMvDistributionState
    d::Distributions.GenericMvTDist{Float64}   # not `MvTDist`: its PDMat parameter is version-dependent
    ν::Float64
    r::BoxTruncationRegion
    n::Int
    max_moment_levels::Int

    # true when the dof-shift identity is unavailable at this node (helper
    # dof ≤ 0); all moments are then computed by direct cubature.
    direct::Bool
    helper::Union{BoxTruncatedMvTRecursiveMomentsState,Nothing}
    children_a::Vector{BoxTruncatedMvTRecursiveMomentsState}
    children_b::Vector{BoxTruncatedMvTRecursiveMomentsState}

    rawMomentDict::Dict{Vector{Int},Float64}
    treeDict::Dict{Vector{Int},Children}
    rawMomentsComputed::Bool

    ΣH::Matrix{Float64}                   # helper scale ν S / (ν - 2)
    fH_a::Vector{Float64}                 # univariate helper density at the lower face
    fH_b::Vector{Float64}                 # univariate helper density at the upper face
    complement_idx::Vector{Vector{Int}}
    kbuf::Vector{Int}

    tp::Float64
    μ::Vector{Float64}
    Σ::PDMat
    tp_err::Float64
    μ_err::Float64
    Σ_err::Float64

    function BoxTruncatedMvTRecursiveMomentsState(ν::Real, μₑ::AbstractVector, Σₑ::AbstractMatrix,
                                                  r::BoxTruncationRegion, max_moment_levels::Int;
                                                  _cache = nothing,
                                                  _orig::Union{Nothing,Vector{Int}} = nothing,
                                                  _key::Tuple{Vector{Tuple{Int,Int}},Int} = (Tuple{Int,Int}[], 0))
        a, b = r.a, r.b
        n = length(μₑ)
        length(a) == n ||
            throw(DimensionMismatch("a has length $(length(a)); expected $n"))
        length(b) == n ||
            throw(DimensionMismatch("b has length $(length(b)); expected $n"))
        any(a .>= b) &&
            throw(ArgumentError("each a[i] must be strictly less than b[i]; got a=$a, b=$b"))
        ν > 0 || throw(ArgumentError("degrees of freedom must be positive; got ν = $ν"))
        # A moment of total order L along an unbounded coordinate exists only
        # for ν > L.
        L = max_moment_levels
        if any(isinf, a) || any(isinf, b)
            ν > L || throw(ArgumentError(
                "moments of order $L of a t_ν truncated to an unbounded box need ν > $L; got ν = $ν"))
        end

        Σm = 0.5 .* (Matrix{Float64}(Σₑ) .+ Matrix{Float64}(Σₑ)')
        d = MvTDist(Float64(ν), Vector{Float64}(μₑ), PDMat(Σm))
        complement_idx = [setdiff(1:n, j) for j in 1:n]
        rawMomentDict, treeDict = init_dicts(n, L)
        if L == 0   # init_dicts only seeds the root key for L ≥ 1
            rawMomentDict[zeros(Int, n)] = NaN
            treeDict[zeros(Int, n)] = nothing
        end

        direct = L ≥ 1 && ν ≤ 2
        helper = nothing
        children_a = BoxTruncatedMvTRecursiveMomentsState[]
        children_b = BoxTruncatedMvTRecursiveMomentsState[]
        ΣH = Matrix{Float64}(undef, 0, 0)
        fH_a = Float64[]
        fH_b = Float64[]

        # States are shared across the recursion: a state's law is fixed by
        # the set of fixed (original coordinate, side) faces and by the
        # number of degrees-of-freedom decrements h along the path (helper
        # steps and face steps commute; see the paper), so `cache` maps
        # (faces, number of helper steps) to the state.
        cache = _cache === nothing ?
            Dict{Tuple{Vector{Tuple{Int,Int}},Int},BoxTruncatedMvTRecursiveMomentsState}() : _cache
        orig = _orig === nothing ? collect(1:n) : _orig
        fixed, nh = _key
        if L ≥ 1 && !direct
            νH = ν - 2
            ΣH = (ν / νH) .* Σm
            fH_a = [_tdens1(νH, μₑ[j], ΣH[j, j], a[j]) for j in 1:n]
            fH_b = [_tdens1(νH, μₑ[j], ΣH[j, j], b[j]) for j in 1:n]
            if L ≥ 2
                hkey = (fixed, nh + 1)
                helper = get!(cache, hkey) do
                    BoxTruncatedMvTRecursiveMomentsState(νH, μₑ, ΣH, r, L - 2;
                        _cache = cache, _orig = orig, _key = hkey)
                end
            end
            if n ≥ 2
                for j in 1:n
                    comp = complement_idx[j]
                    rc = BoxTruncationRegion(a[comp], b[comp])
                    for (side, face, store) in ((1, a[j], children_a), (2, b[j], children_b))
                        fkey = (sort(vcat(fixed, [(orig[j], side)])), nh)
                        push!(store, get!(cache, fkey) do
                            μc, Σc = _t_conditional(νH, μₑ, ΣH, j, comp, face)
                            BoxTruncatedMvTRecursiveMomentsState(νH + 1, μc, Σc, rc, L - 1;
                                _cache = cache, _orig = orig[comp], _key = fkey)
                        end)
                    end
                end
            end
        end

        new(d, Float64(ν), r, n, L, direct, helper, children_a, children_b,
            rawMomentDict, treeDict, false,
            ΣH, fH_a, fH_b, complement_idx, zeros(Int, n),
            NaN, Vector{Float64}(undef, 0), PDMat(Array{Float64,2}(I, n, n)),
            Inf, Inf, Inf)
    end
end

# Univariate t_ν(m, s2) density at x; 0 at ±Inf.
function _tdens1(ν, m, s2, x)
    isinf(x) && return 0.0
    σ = sqrt(s2)
    return pdf(TDist(ν), (x - m) / σ) / σ
end

# Location and scale of x_{-j} given x_j = face under t_ν(μ, Σ). At an
# infinite face the conditional law is never used (its weight is 0), so a
# finite placeholder is returned.
function _t_conditional(ν, μ, Σ, j, comp, face)
    Σcc = Σ[comp, comp] .- (Σ[comp, j] * Σ[j, comp]') ./ Σ[j, j]
    Σcc = 0.5 .* (Σcc .+ Σcc')
    isinf(face) && return (μ[comp], Σcc)
    δ2 = (face - μ[j])^2 / Σ[j, j]
    μc = μ[comp] .+ Σ[comp, j] .* ((face - μ[j]) / Σ[j, j])
    return (μc, ((ν + δ2) / (ν + 1)) .* Σcc)
end

function compute_moments(d::BoxTruncatedMvTRecursiveMomentsState)
    if d.direct
        _t_direct_moments!(d)
        d.rawMomentsComputed = true
        return
    end

    function c_vector(d::BoxTruncatedMvTRecursiveMomentsState, k::Vector{Int})
        c = Vector{Float64}(undef, d.n)
        kbuf = d.kbuf
        copyto!(kbuf, k)
        for j in 1:d.n
            if k[j] > 0
                kbuf[j] = k[j] - 1
                F0 = raw_moment(d.helper, kbuf)
                kbuf[j] = k[j]
            else
                F0 = 0.0
            end
            comp = d.complement_idx[j]
            # n == 1: the face is a point and its "raw moment" is 1.
            F1 = d.n == 1 ? 1.0 : raw_moment(d.children_a[j], @view k[comp])
            F2 = d.n == 1 ? 1.0 : raw_moment(d.children_b[j], @view k[comp])
            # At an infinite face fH = 0 and the term vanishes whenever the
            # moment exists (checked at construction); skip to avoid Inf * 0.
            a_term = iszero(d.fH_a[j]) ? 0.0 : d.r.a[j]^k[j] * d.fH_a[j] * F1
            b_term = iszero(d.fH_b[j]) ? 0.0 : d.r.b[j]^k[j] * d.fH_b[j] * F2
            c[j] = k[j] * F0 + a_term - b_term
        end
        c
    end

    function compute_children_moments(d::BoxTruncatedMvTRecursiveMomentsState, baseKey::Vector{Int})
        isnothing(d.treeDict[baseKey]) && return
        Σc = d.ΣH * c_vector(d, baseKey)
        for k in d.treeDict[baseKey]
            i = 0
            @inbounds for s in eachindex(k)
                if k[s] != baseKey[s]
                    i = s
                    break
                end
            end
            d.rawMomentDict[k] = d.d.μ[i] * d.rawMomentDict[baseKey] + Σc[i]
            compute_children_moments(d, k)
        end
    end

    baseKey = zeros(Int, d.n)
    m0 = LL(d)
    if m0 > 0.0 && isfinite(m0)
        d.rawMomentDict[baseKey] = m0
        compute_children_moments(d, baseKey)
    else
        for k in keys(d.rawMomentDict)
            d.rawMomentDict[k] = 0.0
        end
    end
    d.rawMomentsComputed = true
end

function _t_direct_moments!(d::BoxTruncatedMvTRecursiveMomentsState)
    f(x) = pdf(d.d, x)
    for κ in keys(d.rawMomentDict)
        g(x) = prod(x[i]^κ[i] for i in 1:d.n) * f(x)
        d.rawMomentDict[κ] = hcubature_inf(g, d.r.a, d.r.b; maxevals = 10^6)[1]
    end
end

function raw_moment(d::BoxTruncatedMvTRecursiveMomentsState, k::AbstractVector{Int})
    !d.rawMomentsComputed && compute_moments(d)
    return d.rawMomentDict[k]
end

function raw_moment_dict(d::BoxTruncatedMvTRecursiveMomentsState)
    !d.rawMomentsComputed && compute_moments(d)
    return copy(d.rawMomentDict)
end

# QMC sample count for the t box probability under the `:mvnormalcdf`
# backend. The t QMC rule is noisier per sample than MvNormalCDF's
# Gaussian one, and helper-chain coefficients amplify base-case error, so
# the default is larger than the 10_000 used for the normal.
const _KR_T_QMC_M = Ref{Int}(40_000)

"""
    set_kr_t_qmc_samples!(m::Integer)

Set the number of QMC samples per multivariate-t box probability used by
[`TruncatedMvT`](@ref) under the `:mvnormalcdf` backend. Returns the
previous value.
"""
function set_kr_t_qmc_samples!(m::Integer)
    m ≥ 100 || throw(ArgumentError("need at least 100 QMC samples; got $m"))
    prev = _KR_T_QMC_M[]
    _KR_T_QMC_M[] = m
    return prev
end

# Base case: the t box probability. Shares the backend switch of the normal
# recursion: `:mvnormalcdf` selects Genz-style randomized QMC (here from
# AdditionalDistributions.jl), `:hcubature` adaptive cubature.
function LL(d::BoxTruncatedMvTRecursiveMomentsState)
    if d.n == 1
        σ = sqrt(d.d.Σ[1, 1])
        T = TDist(d.ν)
        return cdf(T, (d.r.b[1] - d.d.μ[1]) / σ) - cdf(T, (d.r.a[1] - d.d.μ[1]) / σ)
    elseif _KR_BASE_BACKEND[] === :mvnormalcdf && d.n ≥ 3
        return AdditionalDistributions.cdf_result(d.d, d.r.a, d.r.b;
                                                  m = _KR_T_QMC_M[], rng = _KR_BASE_RNG[]).value
    else
        # Also used at n = 2 under the QMC backend: 2-D cubature of the t
        # density is faster than the t QMC routine and accurate to ~1e-8,
        # which matters because helper-chain coefficients amplify base-case
        # error in the order-4 moments.
        return hcubature_inf((x) -> pdf(d.d, x), d.r.a, d.r.b; rtol = 1e-8, maxevals = 10^6)[1]
    end
end
