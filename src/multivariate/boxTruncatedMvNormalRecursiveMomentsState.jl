# const max_moment_levels = 2 #Just for mean and covariance matrix
const Children = Union{Vector{Vector{Int}},Nothing}

mutable struct BoxTruncatedMvNormalRecursiveMomentsState <: TruncatedMvDistributionState
    d::MvNormal
    r::BoxTruncationRegion
    n::Int                  #dimension
    max_moment_levels::Int

    # # #Children of type 'a' or 'b' are lower dimensional distributions used to for recursive computation
    children_a::Vector{BoxTruncatedMvNormalRecursiveMomentsState}
    children_b::Vector{BoxTruncatedMvNormalRecursiveMomentsState}

    # # #for each moment vector e.g. [0,1,0,1] or [0,0,2,0] has the tuple which is the computed (non-normalized) moment integral
    # # #of that vector and a list of children vectors
    rawMomentDict::Dict{Vector{Int},Float64} #note that the values are non-normalized moment integrals
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

function compute_moments(d::BoxTruncatedMvNormalRecursiveMomentsState)
    function compute_children_moments(d::BoxTruncatedMvNormalRecursiveMomentsState,baseKey::Vector{Int})
        d.treeDict[baseKey] == nothing && return #recursion stopping criteria
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

function raw_moment(d::BoxTruncatedMvNormalRecursiveMomentsState,k::AbstractVector{Int})
    !d.rawMomentsComputed && compute_moments(d)
    return d.rawMomentDict[k]
end

function raw_moment_dict(d::BoxTruncatedMvNormalRecursiveMomentsState)
    !d.rawMomentsComputed && compute_moments(d)
    return copy(d.rawMomentDict)
end


# moment(d::BoxTruncatedMvNormalRecursiveMomentsState,k::Vector{Int}) = raw_moment(d,k) / alpha(d)

# alpha(d::BoxTruncatedMvNormalRecursiveMomentsState) = raw_moment(d,zeros(Int,d.n))

# function mean(d::BoxTruncatedMvNormalRecursiveMomentsState)
#     μ = Vector{Float64}(undef,d.n)
#     for i in 1:d.n
#         ee = zeros(Int,d.n)
#         ee[i] = 1
#         μ[i] = moment(d,ee)
#     end
#     μ
# end

# function cov(d::BoxTruncatedMvNormalRecursiveMomentsState)
#     Σ = zeros(Float64,d.n,d.n)
#     for i in 1:d.n, j in 1:d.n
#         ee = zeros(Int,d.n)
#         if i == j
#             ee[i] = 2
#         else
#             ee[i], ee[j] = 1, 1
#         end
#         Σ[i,j] = moment(d,ee)
#     end
#     μ = mean(d)
#     Σ-μ*μ'
# end

# function rand(d::BoxTruncatedMvNormalRecursiveMomentsState)
#     rand(MvNormal(d.μₑ,d.Σₑ)) #TODO QQQQ
# end


# function pdf_nontruncated(d::BoxTruncatedMvNormalRecursiveMomentsState,x)
#     d_nontruncated = MvNormal(d.μₑ,d.Σₑ)
#     pdf(d_nontruncated,x)
# end

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
