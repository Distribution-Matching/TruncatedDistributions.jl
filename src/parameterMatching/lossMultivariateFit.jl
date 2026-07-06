# Helpers shared across the moment-matching pipeline:
#
#   * `moment_loss`      - the scalar loss L(μ, Σ) = ½‖μA − μ̂‖² + ½‖ΣA − Σ̂‖²_F,
#                          read directly off the cached Kan–Robotti primitive
#                          moments of `dist`.
#   * `vector_moment_loss` - the same loss as a function of the packed (μ,U)
#                          parameter vector, building a fresh `dist` per call.
#                          Useful for finite-difference cross-checks and for
#                          callers that don't hold a reusable KR-tree state.
#   * Pack/unpack utilities (`make_param_vec_*`, `make_μ_Σ_from_param_vec`,
#     `n_from_param_size`) that move between the natural (μ, Σ) representation
#     and the unconstrained (μ, U) parameter vector LBFGS optimises over.
#
# The unconstrained parameterisation stores μ followed by the upper triangle
# of U where U U^T = Σ^{-1}. Σ is therefore positive-definite by construction
# for any choice of vec(U).

# Covariance-term weight in the moment-matching loss. The loss is
#
#   L = L1 + γ · r(n) · L2,   L1 = ½‖μA − μ̂‖²,  L2 = ½‖ΣA − Σ̂‖²_F,
#
# with r(n) = (#mean parameters)/(#covariance parameters)
#           = n / (n(n+1)/2) = 2/(n+1).
#
# r(n) makes γ scale-free across dimensions: at γ = 1 the mean- and
# covariance-residual blocks carry equal total weight, and the default
# γ = 0.2 makes the mean five times more important than the covariance,
# for every n. Set with `set_loss_gamma!`.
const _LOSS_GAMMA = Ref{Float64}(0.2)

"""
    set_loss_gamma!(γ::Real)

Set the covariance-term weight parameter γ of the moment-matching loss
`L = L1 + γ·r(n)·L2` with `r(n) = 2/(n+1)`. γ = 1 balances the mean- and
covariance-residual budgets; γ < 1 favours the mean (default γ = 0.2,
i.e. the mean is 5× the covariance); γ > 1 favours the covariance.
Returns the previous value.
"""
set_loss_gamma!(γ::Real) = (prev = _LOSS_GAMMA[]; _LOSS_GAMMA[] = Float64(γ); prev)

"""
    get_loss_gamma()

Return the current covariance-term weight parameter γ. See
[`set_loss_gamma!`](@ref).
"""
get_loss_gamma() = _LOSS_GAMMA[]

# Weight applied to the covariance residual L2 for a `dim`-dimensional
# problem: γ · r(dim) = γ · 2/(dim+1). BCD sub-blocks of size k thus use
# r(k) automatically, the full problem uses r(n).
_cov_weight(dim::Int) = _LOSS_GAMMA[] * 2.0 / (dim + 1)

"""
    moment_loss(d::RecursiveMomentsBoxTruncatedMvNormal, μ̂, Σ̂)

Scalar moment-matching loss
`L = ½‖μA − μ̂‖² + γ·r(n)·½‖ΣA − Σ̂‖²_F`, with `r(n) = 2/(n+1)` and the
covariance weight γ from [`get_loss_gamma`](@ref) (default 0.2),
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
        return prevfloat(Inf) / 4
    end
    m1 = [raw_moment_from_indices(dist, [i])    for i in 1:n]
    m2 = [raw_moment_from_indices(dist, [i, j]) for i in 1:n, j in 1:n]
    μA = m1 ./ m0
    ΣA = m2 ./ m0 .- μA * μA'
    w2 = _cov_weight(n)
    L = 0.5 * (sum(abs2, μA .- μ̂) + w2 * sum(abs2, ΣA .- Σ̂))
    return isfinite(L) ? L : prevfloat(Inf) / 4
end

"""
    vector_moment_loss(param_vec, a, b, μ̂, Σ̂)

The moment-matching loss as a function of the packed `(μ, U)` parameter
vector. Useful for finite-difference cross-checks and for callers that
don't hold a reusable Kan and Robotti tree state. Allocates a fresh
distribution per call; for inner-loop use prefer the workspace-aware
`vector_fg_true_loss!` (advanced internals).
"""
function vector_moment_loss(param_vec::Vector{Float64}, a, b,
                            μ̂::AbstractVector{Float64},
                            Σ̂::AbstractMatrix{Float64})
    μ, Σ = make_μ_Σ_from_param_vec(param_vec)
    # Σ comes from U^{-1} U^{-T}; line-search round-off can make it slightly
    # non-symmetric. Symmetrize and add a tiny jitter so the PDMat Cholesky
    # does not throw.
    Σsym = 0.5 .* (Σ .+ Σ')
    Σsym .+= eps(Float64) * (tr(Σsym) + 1.0) * Matrix{Float64}(I, size(Σsym))
    dist = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σsym), a, b)
    return moment_loss(dist, μ̂, Σ̂)
end

function make_μ_Σ_from_param_vec(param_vec)
    n = n_from_param_size(length(param_vec))
    μ = param_vec[1:n]
    inds = [CartesianIndex(i, j) for i = 1:n for j = i:n]
    U = zeros(n, n)
    U[inds] = param_vec[(n + 1):end]
    U = UpperTriangular(U)
    Ui = inv(U)
    Σ = Ui * Ui'
    return μ, Σ
end

function make_param_vec_from_μ_Σ(μ, Σ)
    Σi = inv(Σ)
    F = cholesky(0.5 .* (Σi .+ Σi'))
    U = F.U
    n = size(U, 1)
    inds = [CartesianIndex(i, j) for i = 1:n for j = i:n]
    vcat(μ, U[inds])
end

function make_param_vec_from_μ_U(μ, U)
    n = size(U, 1)
    inds = [CartesianIndex(i, j) for i = 1:n for j = i:n]
    vcat(μ, U[inds])
end

# Invert n + n(n+1)/2 = length(param_vec) for n.
function n_from_param_size(param_size::Integer)
    return Int((-3 + sqrt(9 + 8 * param_size)) / 2)
end
