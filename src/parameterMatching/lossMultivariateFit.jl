# Helpers shared across the moment-matching pipeline:
#
#   * `moment_loss`      - the scalar averaged loss
#                          L(μ, Σ) = ½·(1/n)‖μA − μ̂‖² + γ·½·(1/n²)‖ΣA − Σ̂‖²_F,
#                          read directly off the cached Kan–Robotti primitive
#                          moments of `dist`.
#   * `vector_moment_loss` - the same loss as a function of the packed (μ,V)
#                          parameter vector, building a fresh `dist` per call.
#                          Useful for finite-difference cross-checks and for
#                          callers that don't hold a reusable KR-tree state.
#   * Pack/unpack utilities (`make_param_vec_*`, `make_μ_Σ_from_param_vec`,
#     `n_from_param_size`) that move between the natural (μ, Σ) representation
#     and the unconstrained (μ, V) parameter vector LBFGS optimises over.
#
# The unconstrained parameterisation stores μ followed by the upper triangle
# of U where U U^T = Σ^{-1}. Σ is therefore positive-definite by construction
# for any choice of vec(U).

# Covariance-term weight in the moment-matching loss. The loss is the
# averaged Frobenius loss
#
#   L = ½·(1/n)‖μA − μ̂‖² + γ·½·(1/n²)‖ΣA − Σ̂‖²_F,
#
# i.e. half the mean squared mean-residual entry plus γ times half the
# mean squared covariance-residual entry (over all n² entries). γ is
# therefore exactly the per-entry weight of the covariance relative to the
# mean, for every n: the default γ = 1 weights them equally.
# L < tol bounds the RMS mean error by
# √(2 tol) and the RMS covariance error by √(2 tol / γ), independently of
# n. Set γ with `set_loss_gamma!`.
const _LOSS_GAMMA = Ref{Float64}(1.0)

"""
    set_loss_gamma!(γ::Real)

Set the covariance-term weight parameter γ of the averaged moment-matching
loss `L = ½·(1/n)‖μA − μ̂‖² + γ·½·(1/n²)‖ΣA − Σ̂‖²_F`. γ is the per-entry
weight of the covariance relative to the mean: γ = 1 (the default) weights
them equally; γ < 1 favours the mean; γ > 1 favours the covariance.
Returns the previous value.
"""
set_loss_gamma!(γ::Real) = (prev = _LOSS_GAMMA[]; _LOSS_GAMMA[] = Float64(γ); prev)

"""
    get_loss_gamma()

Return the current covariance-term weight parameter γ. See
[`set_loss_gamma!`](@ref).
"""
get_loss_gamma() = _LOSS_GAMMA[]

# Weights applied to ½‖μA − μ̂‖² and ½‖ΣA − Σ̂‖²_F for a `dim`-dimensional
# problem: 1/dim (average over the dim mean entries) and γ/dim² (average
# over the dim² covariance entries). BCD sub-blocks of size k thus use the
# k-dimensional averages automatically, the full problem uses n.
_mean_weight(dim::Int) = 1.0 / dim
_cov_weight(dim::Int)  = _LOSS_GAMMA[] / dim^2

"""
    moment_loss(d::RecursiveMomentsBoxTruncatedMvNormal, μ̂, Σ̂)

Scalar averaged moment-matching loss
`L = ½·(1/n)‖μA − μ̂‖² + γ·½·(1/n²)‖ΣA − Σ̂‖²_F`, with the covariance
weight γ from [`get_loss_gamma`](@ref) (default 1),
evaluated against the cached primitive moments of `d` (so much cheaper
than going through `mean(d)` / `cov(d)`, which would re-integrate).
Returns a large finite penalty if the cache indicates `m^{(0)} → 0`, so
an LBFGS line search backs off rather than freezing at `L = Inf`.
"""
function moment_loss(dist::RecursiveMomentsBoxTruncatedMvNormal,
                     μ̂::AbstractVector{Float64},
                     Σ̂::AbstractMatrix{Float64})
    # Use the cached Kan–Robotti primitive moments directly. Calling
    # mean(dist) / cov(dist) instead would re-integrate via HCubature, which
    # at n=3 alone dominated ~96% of runtime and 1.4 GiB per call.
    n  = length(dist)
    m0 = raw_moment_from_indices(dist, Int[])
    # Guard against the line search wandering into a region where m^{(0)} → 0;
    # μA = m^{(1)}/m^{(0)} then overflows. Return a large finite penalty so the
    # line search backs off instead of freezing at L = Inf.
    if !isfinite(m0) || m0 < eps(Float64)
        return _LOSS_PENALTY
    end
    m1 = [raw_moment_from_indices(dist, [i])    for i in 1:n]
    m2 = [raw_moment_from_indices(dist, [i, j]) for i in 1:n, j in 1:n]
    μA = m1 ./ m0
    ΣA = m2 ./ m0 .- μA * μA'
    L = 0.5 * (_mean_weight(n) * sum(abs2, μA .- μ̂) +
               _cov_weight(n)  * sum(abs2, ΣA .- Σ̂))
    return isfinite(L) ? L : _LOSS_PENALTY
end

"""
    vector_moment_loss(param_vec, a, b, μ̂, Σ̂)

The moment-matching loss as a function of the packed `(μ, V)` parameter
vector. Useful for finite-difference cross-checks and for callers that
don't hold a reusable Kan and Robotti tree state. Allocates a fresh
distribution per call; for inner-loop use prefer the workspace-aware
`vector_fg_true_loss!` (advanced internals).
"""
function vector_moment_loss(param_vec::Vector{Float64}, a, b,
                            μ̂::AbstractVector{Float64},
                            Σ̂::AbstractMatrix{Float64})
    _param_vec_admissible(param_vec) || return _LOSS_PENALTY
    μ, Σ = make_μ_Σ_from_param_vec(param_vec)
    # Σ comes from U^{-1} U^{-T}; line-search round-off can make it slightly
    # non-symmetric. Symmetrize and add a tiny jitter so the PDMat Cholesky
    # does not throw.
    Σsym = 0.5 .* (Σ .+ Σ')
    Σsym .+= eps(Float64) * (tr(Σsym) + 1.0) * Matrix{Float64}(I, size(Σsym))
    dist = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σsym), a, b)
    return moment_loss(dist, μ̂, Σ̂)
end

# The packed vector is (μ, V): V is upper triangular with V_kk = log U_kk and
# V_kl = U_kl for k < l, where UᵀU = Σ⁻¹ (the log-Cholesky parameterisation
# of the precision matrix). Every real vector is a valid parameter.

# Diagonal entries of V beyond this size put U_kk outside about 1e-9 to 5e8.
# A line search that gets there has overshot: the objective returns
# `_LOSS_PENALTY` so that the search backs off.
const _MAX_ABS_LOG_DIAG = 20.0
const _LOSS_PENALTY = prevfloat(Inf) / 4

function _param_vec_admissible(p)
    n = n_from_param_size(length(p))
    k = n
    for i in 1:n
        k += 1
        abs(p[k]) <= _MAX_ABS_LOG_DIAG || return false
        k += n - i
    end
    return all(isfinite, p)
end

# U (dense, upper triangular, positive diagonal) from the packed (μ, V) vector.
function _U_from_param_vec(p)
    n = n_from_param_size(length(p))
    inds = [CartesianIndex(i, j) for i = 1:n for j = i:n]
    U = zeros(n, n)
    U[inds] = p[(n + 1):end]
    for i in 1:n
        U[i, i] = exp(U[i, i])
    end
    return U
end

function make_μ_Σ_from_param_vec(param_vec)
    n = n_from_param_size(length(param_vec))
    μ = param_vec[1:n]
    Ui = inv(UpperTriangular(_U_from_param_vec(param_vec)))
    Σ = Ui * Ui'
    return μ, Σ
end

function make_param_vec_from_μ_Σ(μ, Σ)
    Σi = inv(Σ)
    F = cholesky(0.5 .* (Σi .+ Σi'))
    return make_param_vec_from_μ_U(μ, F.U)
end

function make_param_vec_from_μ_U(μ, U)
    n = size(U, 1)
    V = Matrix{Float64}(U)
    for i in 1:n
        V[i, i] = log(V[i, i])
    end
    return _pack_μ_V(μ, V)
end

# Pack (μ, V) as they stand; also used for the gradient (g_μ, g_V).
function _pack_μ_V(μ, V)
    n = size(V, 1)
    inds = [CartesianIndex(i, j) for i = 1:n for j = i:n]
    return vcat(μ, V[inds])
end

# Invert n + n(n+1)/2 = length(param_vec) for n.
function n_from_param_size(param_size::Integer)
    return Int((-3 + sqrt(9 + 8 * param_size)) / 2)
end
