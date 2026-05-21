const Children = Union{Vector{Vector{Int}},Nothing}

"""
    BoxTruncatedMvNormalRecursiveMomentsState <: TruncatedMvDistributionState

Cached state for the recursive moment computation of a box-truncated
multivariate normal. Holds the children-trees needed by the Kan and
Robotti recursion, a dictionary of pre-allocated raw moments up to total
order `max_moment_levels`, and the standard mean / covariance / truncation
probability cache.

Used by [`RecursiveMomentsBoxTruncatedMvNormal`](@ref) / [`TruncatedMvNormal`](@ref).
"""
mutable struct BoxTruncatedMvNormalRecursiveMomentsState <: TruncatedMvDistributionState
    d::MvNormal
    r::BoxTruncationRegion
    n::Int                  #dimension
    max_moment_levels::Int

    # Children of type 'a' or 'b' are lower-dimensional distributions used
    # by the Kan–Robotti recursive moment computation.
    children_a::Vector{BoxTruncatedMvNormalRecursiveMomentsState}
    children_b::Vector{BoxTruncatedMvNormalRecursiveMomentsState}

    # Values are non-normalised (raw) moment integrals; divide by the m^{(0)}
    # entry to get the truncated-distribution moment.
    rawMomentDict::Dict{Vector{Int},Float64}
    treeDict::Dict{Vector{Int},Children}
    rawMomentsComputed::Bool

    # --- Precomputed scratch, set once at construction. Recomputing these on
    # --- every call to c_vector was the dominant per-call allocation cost
    # --- before this change.
    phi_a::Vector{Float64}                # 1D Gaussian pdf at the lower box face
    phi_b::Vector{Float64}                # 1D Gaussian pdf at the upper box face
    complement_idx::Vector{Vector{Int}}   # complement_idx[j] = setdiff(1:n, j)
    kbuf::Vector{Int}                     # reusable scratch for kappa-with-one-index-decremented

    tp::Float64
    μ::Vector{Float64}
    Σ::PDMat
    tp_err::Float64
    μ_err::Float64
    Σ_err::Float64

    function BoxTruncatedMvNormalRecursiveMomentsState(d::MvNormal, r::BoxTruncationRegion, max_moment_levels::Int)
        μₑ, Σₑ = d.μ, d.Σ
        a, b = r.a, r.b
        n = length(d)
        length(a) != n && error("The length of a does not match the length")
        length(b) != n && error("The length of b does not match the length")
        a > b && error("The a vector must be less than the b vector")

        if n ≥ 2
            μᵃ = [μₑ[setdiff(1:n,j)]  +  Σₑ[setdiff(1:n,j),j] * (a[j]-μₑ[j])/Σₑ[j,j]
                    for j in 1:n]
            μᵇ = [μₑ[setdiff(1:n,j)]  +  Σₑ[setdiff(1:n,j),j] * (b[j]-μₑ[j])/Σₑ[j,j]
                    for j in 1:n]
            Σ̃ = [Σₑ[setdiff(1:n,j),setdiff(1:n,j)] - (1/Σₑ[j,j])*Σₑ[setdiff(1:n,j),j]*Σₑ[j,setdiff(1:n,j)]' for j in 1:n]
            children_a = [BoxTruncatedMvNormalRecursiveMomentsState(
                            MvNormal(μᵃ[j],0.5*(Σ̃[j] + Σ̃[j]')),
                            BoxTruncationRegion(a[setdiff(1:n,j)],b[setdiff(1:n,j)]),
                            max_moment_levels) for j in 1:n]
            children_b = [BoxTruncatedMvNormalRecursiveMomentsState(
                            MvNormal(μᵇ[j],0.5*(Σ̃[j] + Σ̃[j]')),
                            BoxTruncationRegion(a[setdiff(1:n,j)],b[setdiff(1:n,j)]),
                            max_moment_levels) for j in 1:n]
        else #n==1
            children_a = Array{BoxTruncatedMvNormalRecursiveMomentsState,1}[] #no children
            children_b = Array{BoxTruncatedMvNormalRecursiveMomentsState,1}[] #no children
        end
        rawMomentDict, treeDict = init_dicts(n,max_moment_levels)

        # Precompute the 1D Gaussian pdf at each box face and the per-axis
        # complement index. pdf(Normal, ±Inf) is 0, which is what we want.
        phi_a = [pdf(Normal(μₑ[j], sqrt(Σₑ[j,j])), a[j]) for j in 1:n]
        phi_b = [pdf(Normal(μₑ[j], sqrt(Σₑ[j,j])), b[j]) for j in 1:n]
        complement_idx = [setdiff(1:n, j) for j in 1:n]

        new(d,r,n,max_moment_levels,children_a,children_b,rawMomentDict,treeDict,false,
            phi_a, phi_b, complement_idx, zeros(Int, n),
            NaN,
            Vector{Float64}(undef,0),
            PDMat(Array{Float64,2}(I,n,n)),
            Inf, Inf, Inf)
    end
end

"""
    update_distribution!(s::BoxTruncatedMvNormalRecursiveMomentsState, μ, Σ)

In-place refresh of an existing recursion-tree state with a new
`(μ, Σ)`. The topology (children trees, dictionary keys, complement
indices, box bounds) is invariant across `(μ, Σ)` updates on the same
`n` / `a` / `b` / `max_moment_levels`, so reusing the state and just
walking the tree to rewrite the distribution-dependent data avoids the
`O(n!)` reconstruction cost — significant from `n ≥ 5`.

Invalidates the cached `mean`, `cov`, `tp`, and raw moments; the next
query recomputes from scratch. Pair with [`outer_dist_from_state`](@ref)
to obtain a fresh outer distribution wrapper.
"""
function update_distribution!(s::BoxTruncatedMvNormalRecursiveMomentsState,
                              μₑ::AbstractVector{Float64},
                              Σₑ::AbstractMatrix{Float64})
    Σpd = Σₑ isa PDMat ? Σₑ : PDMat(0.5 .* (Matrix(Σₑ) .+ Matrix(Σₑ)'))
    s.d = MvNormal(Vector{Float64}(μₑ), Σpd)

    @inbounds for j in 1:s.n
        σj = sqrt(Σpd[j, j])
        s.phi_a[j] = pdf(Normal(μₑ[j], σj), s.r.a[j])
        s.phi_b[j] = pdf(Normal(μₑ[j], σj), s.r.b[j])
    end

    if s.n ≥ 2
        Σmat = Matrix(Σpd)
        for j in 1:s.n
            comp = s.complement_idx[j]
            inv_diag = 1.0 / Σmat[j, j]
            μᵃj = μₑ[comp] .+ Σmat[comp, j] .* ((s.r.a[j] - μₑ[j]) * inv_diag)
            μᵇj = μₑ[comp] .+ Σmat[comp, j] .* ((s.r.b[j] - μₑ[j]) * inv_diag)
            Σ̃j = Σmat[comp, comp] .- inv_diag .* (Σmat[comp, j] * Σmat[j, comp]')
            Σ̃j_sym = 0.5 .* (Σ̃j .+ Σ̃j')
            update_distribution!(s.children_a[j], μᵃj, Σ̃j_sym)
            update_distribution!(s.children_b[j], μᵇj, Σ̃j_sym)
        end
    end

    # Invalidate cached moments and their error bounds. Callers reading
    # `mean(d)` / `cov(d)` / `tp(d)` will now trigger a fresh compute.
    for k in keys(s.rawMomentDict)
        s.rawMomentDict[k] = NaN
    end
    s.rawMomentsComputed = false
    s.tp     = NaN
    s.tp_err = Inf
    s.μ_err  = Inf
    s.Σ_err  = Inf
    return s
end

"""
    outer_dist_from_state(state::BoxTruncatedMvNormalRecursiveMomentsState)

Build a lightweight outer [`TruncatedMvDistribution`](@ref) wrapper around
an already-refreshed state. The outer struct is immutable and cheap (three
field references), so callers that hold a long-lived workspace state
rebuild this thin wrapper on each call; the expensive recursion tree is
reused.
"""
function outer_dist_from_state(state::BoxTruncatedMvNormalRecursiveMomentsState)
    return TruncatedMvDistribution(state.d, state.r, state)
end

function init_dicts(n::Int,max_moment_levels::Int)
    function addToBaseKey(  baseKey::Vector{Int},
                            n::Int,
                            md::Dict{Vector{Int},Float64},
                            td::Dict{Vector{Int},Children})
        keys = Vector{Vector{Int}}(undef,n)
        for i in 1:n
            key = copy(baseKey)
            key[i] += 1
            md[key] = NaN
            td[key] = nothing
            keys[i] = key
        end
        md[baseKey] = NaN
        td[baseKey] = keys
        keys
    end

    md = Dict{Vector{Int},Float64}() #rawMomentDict
    td = Dict{Vector{Int},Children}() #treeDict
    rootKey = zeros(Int,n)
    key_vals = [rootKey]
    for _ = 1:max_moment_levels
        levelKeys = Vector{Int}[]
        for key in key_vals
            newKeys = addToBaseKey(key,n,md,td)
            append!(levelKeys,newKeys)
        end
        key_vals = levelKeys
    end
    md,td
end

"""
    compute_moments(d::BoxTruncatedMvNormalRecursiveMomentsState)

Walk the recursion tree and fill `d.rawMomentDict` with every primitive
moment up to total order `d.max_moment_levels`. Cached afterwards;
[`raw_moment`](@ref) reuses the result.
"""
function compute_moments(d::BoxTruncatedMvNormalRecursiveMomentsState)
    function compute_children_moments(d::BoxTruncatedMvNormalRecursiveMomentsState,baseKey::Vector{Int})
        isnothing(d.treeDict[baseKey]) && return  # recursion stopping criterion
        c = c_vector(d,baseKey)
        Σc = d.d.Σ * c                              # hoist out of the inner loop:
                                                    # c depends only on baseKey, so Σ*c does too.
        for k in d.treeDict[baseKey]
            # k and baseKey differ in exactly one coordinate, where k is one
            # larger. A scalar scan finds it without allocating k - baseKey.
            i = 0
            @inbounds for s in eachindex(k)
                if k[s] != baseKey[s]
                    i = s
                    break
                end
            end
            d.rawMomentDict[k] = d.d.μ[i]*d.rawMomentDict[baseKey] + Σc[i]
            compute_children_moments(d,k)
        end
    end

    function c_vector(d::BoxTruncatedMvNormalRecursiveMomentsState,k::Vector{Int})
        c = Vector{Float64}(undef,d.n)
        kbuf = d.kbuf
        copyto!(kbuf, k)
        for j in 1:d.n
            # F0 = m^{(p-e_j)} if k[j] > 0, else 0. Use a reusable buffer
            # rather than `copy(k); kbuf[j] -= 1`, then restore in place.
            if k[j] > 0
                kbuf[j] = k[j] - 1
                F0 = d.rawMomentDict[kbuf]
                kbuf[j] = k[j]              # restore
            else
                F0 = 0.0
            end
            # F1, F2: lower-dimensional truncated moments at the j-th box face.
            comp = d.complement_idx[j]
            F1 = isempty(d.children_a) ? 0.0 : raw_moment(d.children_a[j], @view k[comp])
            F2 = isempty(d.children_b) ? 0.0 : raw_moment(d.children_b[j], @view k[comp])
            # 1D Gaussian pdf at the j-th lower / upper box face — precomputed
            # at construction; constants in the inner loop. pdf(N(μ,σ), ±Inf) = 0.
            # Guard against the indeterminate form: when the box face is at
            # ±∞ and k[j] ≥ 1, naïvely a^k * phi(a) evaluates as Inf * 0 = NaN,
            # but the true limit is 0 because the Gaussian decays
            # super-polynomially. Skip the term when phi vanishes.
            a_term = iszero(d.phi_a[j]) ? 0.0 : d.r.a[j]^k[j] * d.phi_a[j] * F1
            b_term = iszero(d.phi_b[j]) ? 0.0 : d.r.b[j]^k[j] * d.phi_b[j] * F2
            c[j] = k[j]*F0 + a_term - b_term
        end
        c
    end

    if d.n > 1
        baseKey = zeros(Int,d.n) #[0,0,....,0]
        d.rawMomentDict[baseKey] = LL(d)
        compute_children_moments(d,baseKey) #start recursion
    else  #n==1
        @assert d.n == 1
        distTruncated = truncated(Normal(d.d.μ[1],sqrt(d.d.Σ[1])),d.r.a[1],d.r.b[1])
        d.rawMomentDict[[0]] = distTruncated.tp
        m = moments(distTruncated, d.max_moment_levels)
        for i in 1:d.max_moment_levels
            d.rawMomentDict[[i]] = m[i]*distTruncated.tp
        end
    end
    d.rawMomentsComputed = true
end

"""
    raw_moment(d, κ)

Return the unnormalised moment integral
`∫_{[a,b]} x_1^{κ_1} … x_n^{κ_n} φ(x; μ, Σ) dx` for the multi-index `κ`.
Triggers [`compute_moments`](@ref) on the first call; subsequent calls hit
the cache. Divide by `raw_moment(d, zeros(Int, length(d)))` to obtain the
corresponding moment of the truncated distribution.
"""
function raw_moment(d::BoxTruncatedMvNormalRecursiveMomentsState, k::AbstractVector{Int})
    !d.rawMomentsComputed && compute_moments(d)
    return d.rawMomentDict[k]
end

"""
    raw_moment_dict(d)

Copy of the full raw-moment cache, keyed by multi-index `κ`. Useful when
you need many moments at once and want to avoid repeated lookups.
"""
function raw_moment_dict(d::BoxTruncatedMvNormalRecursiveMomentsState)
    !d.rawMomentsComputed && compute_moments(d)
    return copy(d.rawMomentDict)
end

# Backend selector for the multivariate-Gaussian box probability that
# bottoms out the Kan–Robotti recursion. `:hcubature` uses the
# general-purpose adaptive cubature wrapper; `:mvnormalcdf` uses
# `MvNormalCDF.mvnormcdf` (Genz–Bretz separation-of-variables + QMC),
# which is dramatically faster on this specific integrand but Float64-only.
# Switch with `set_kr_base_backend!(:mvnormalcdf)`.
const _KR_BASE_BACKEND = Ref{Symbol}(:hcubature)

"""
    set_kr_base_backend!(backend::Symbol)

Choose the integrator used for the multivariate-Gaussian box probability
that bottoms out the Kan–Robotti recursion. Accepted values:

  * `:hcubature`   — `hcubature_inf` (default; works with any element type
                     compatible with HCubature).
  * `:mvnormalcdf` — `MvNormalCDF.mvnormcdf` (Genz–Bretz, Float64 only;
                     typically 10–100× faster on n ≥ 3 Gaussian box
                     probabilities and natively handles ±Inf bounds).

Returns the previous backend.
"""
function set_kr_base_backend!(backend::Symbol)
    backend in (:hcubature, :mvnormalcdf) ||
        throw(ArgumentError("backend must be :hcubature or :mvnormalcdf, got $backend"))
    prev = _KR_BASE_BACKEND[]
    _KR_BASE_BACKEND[] = backend
    return prev
end

"""
    get_kr_base_backend()

Return the currently-selected base-case backend for the multivariate-normal
box probability — either `:hcubature` or `:mvnormalcdf`. See
[`set_kr_base_backend!`](@ref).
"""
get_kr_base_backend() = _KR_BASE_BACKEND[]

function LL(d::BoxTruncatedMvNormalRecursiveMomentsState)
    # @info "doing base numerical integral on dimension $(d.n)."
    if _KR_BASE_BACKEND[] === :mvnormalcdf
        # mvnormcdf returns (probability, error_estimate); we only need the
        # value. Float64-only — for Dual-typed parameters the caller should
        # leave the backend at :hcubature. m=10_000 quasi-Monte Carlo
        # samples typically gives ~1e-5 error in well under a millisecond
        # after JIT warm-up; that error is small enough not to perturb
        # downstream moments at the test cross-check tolerance.
        return mvnormcdf(d.d, d.r.a, d.r.b; m = 10_000)[1]
    else
        return hcubature_inf((x)->pdf(d.d,x), d.r.a, d.r.b, maxevals = 10^6)[1]
    end
end
