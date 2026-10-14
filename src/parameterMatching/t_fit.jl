# Moment matching for the box-truncated multivariate Student-t with fixed
# degrees of freedom ν (proof of concept; paper Section "Extension to the
# truncated multivariate t", Theorem t-gradient).
#
# Same loss and (μ, U) chart as the Gaussian, Σ⁻¹ = UᵀU. The t score is
# the Gaussian one with the quadratic-form terms weighted by
# w(Q) = (n+ν)/(ν+Q), and w·f_ν(·; μ, U) = f_{ν+2}(·; μ, U♯) with
# Σ♯ = ν Σ / (ν+2). So
#
#   ∂m_α/∂μ_k   = Σ_j [UᵀU]_{kj} (m♯_{α,j} − μ_j m♯_α),
#   ∂m_α/∂U_kl  = δ_kl/U_kk · m_α − Σ_{s≥k} U_ks J♯(α, s, l),
#
# with m♯ the raw moments of the truncated t_{ν+2}(μ, Σ♯) and J♯ the
# corresponding centred second-order combination. Everything downstream
# (quotient and chain rule into L) is identical to `grad_true_loss`.

# Raw moment ∫_A x_{inds} f dx of a TruncatedMvT, indexed by a list of
# coordinates (repetitions allowed) rather than a multi-index.
function _mt(d::TruncatedMvT, inds::Vector{Int})
    κ = zeros(Int, length(d))
    for i in inds
        κ[i] += 1
    end
    return raw_moment(d, κ)
end

function _Jt(d::TruncatedMvT, μ, inds::Vector{Int}, s::Int, l::Int)
    return _mt(d, vcat(inds, [s, l])) -
           μ[s] * _mt(d, vcat(inds, [l])) -
           μ[l] * _mt(d, vcat(inds, [s])) +
           μ[s] * μ[l] * _mt(d, inds)
end

# ∂m_α/∂μ (length n) and ∂m_α/∂U (upper triangular n×n) for the t_ν law
# `d`, using the t_{ν+2} law `d♯` for the score-generated terms.
function _t_moment_grad(d::TruncatedMvT, d♯::TruncatedMvT, μ, U, inds::Vector{Int})
    n = length(d)
    P = U' * U
    mp♯ = _mt(d♯, inds)
    g_μ = P * [_mt(d♯, vcat(inds, [j])) - μ[j] * mp♯ for j in 1:n]
    g_U = zeros(n, n)
    mp = _mt(d, inds)
    for k in 1:n, l in k:n
        term1 = k == l ? mp / U[k, k] : 0.0
        g_U[k, l] = term1 - sum(U[k, s] * _Jt(d♯, μ, inds, s, l) for s in k:n)
    end
    return g_μ, g_U
end

"""
    t_moment_matching_pair(ν, μ, Σ, a, b)

The two truncated-t laws the Theorem-2 gradient needs: `t_ν(μ, Σ)` up to
order 2 and `t_{ν+2}(μ, νΣ/(ν+2))` up to order 4, both on the box `[a, b]`.
"""
function t_moment_matching_pair(ν, μ, Σ, a, b)
    d  = TruncatedMvT(ν, μ, Σ, a, b; max_moment_levels = 2)
    d♯ = TruncatedMvT(ν + 2, μ, (ν / (ν + 2)) .* Σ, a, b; max_moment_levels = 4)
    return d, d♯
end

# Loss L = ½‖μA − μ̂‖² + γ r(n) ½‖ΣA − Σ̂‖²_F for the t_ν law `d`.
function moment_loss_t(d::TruncatedMvT, μ̂, Σ̂)
    n = length(d)
    m0 = _mt(d, Int[])
    (!isfinite(m0) || m0 < eps(Float64)) && return prevfloat(Inf) / 4
    m1 = [_mt(d, [i]) for i in 1:n]
    m2 = [_mt(d, [i, j]) for i in 1:n, j in 1:n]
    μA = m1 ./ m0
    ΣA = m2 ./ m0 .- μA * μA'
    L = 0.5 * (sum(abs2, μA .- μ̂) + _cov_weight(n) * sum(abs2, ΣA .- Σ̂))
    return isfinite(L) ? L : prevfloat(Inf) / 4
end

"""
    grad_true_loss_t(d, d♯, U, μ̂, Σ̂)

Gradient `(g_μ, g_U)` of the moment-matching loss for the truncated t_ν
law `d`, with `d♯` its t_{ν+2} companion from
[`t_moment_matching_pair`](@ref) and `U` the upper-triangular factor with
`UᵀU = Σ⁻¹`.
"""
function grad_true_loss_t(d::TruncatedMvT, d♯::TruncatedMvT, U::Matrix{Float64},
                          μ̂::AbstractVector, Σ̂::AbstractMatrix)
    n = length(d)
    μ = d.untruncated.μ
    m0 = _mt(d, Int[])
    m1 = [_mt(d, [i]) for i in 1:n]
    m2 = [_mt(d, [i, j]) for i in 1:n, j in 1:n]
    μA = m1 ./ m0
    ΣA = m2 ./ m0 .- μA * μA'

    ∂m0_μ, ∂m0_U = _t_moment_grad(d, d♯, μ, U, Int[])
    ∂m1 = [_t_moment_grad(d, d♯, μ, U, [i]) for i in 1:n]
    ∂m2 = [_t_moment_grad(d, d♯, μ, U, [i, j]) for i in 1:n, j in 1:n]

    ∂μA_μ = [(∂m1[i][1] .- μA[i] .* ∂m0_μ) ./ m0 for i in 1:n]
    ∂μA_U = [(∂m1[i][2] .- μA[i] .* ∂m0_U) ./ m0 for i in 1:n]
    ∂ΣA_μ = [(∂m2[i, j][1] ./ m0) .- (m2[i, j] / m0^2) .* ∂m0_μ .-
             μA[j] .* ∂μA_μ[i] .- μA[i] .* ∂μA_μ[j] for i in 1:n, j in 1:n]
    ∂ΣA_U = [(∂m2[i, j][2] ./ m0) .- (m2[i, j] / m0^2) .* ∂m0_U .-
             μA[j] .* ∂μA_U[i] .- μA[i] .* ∂μA_U[j] for i in 1:n, j in 1:n]

    w2 = _cov_weight(n)
    g_μ = zeros(n)
    g_U = zeros(n, n)
    for i in 1:n
        g_μ .+= (μA[i] - μ̂[i]) .* ∂μA_μ[i]
        g_U .+= (μA[i] - μ̂[i]) .* ∂μA_U[i]
    end
    for i in 1:n, j in 1:n
        g_μ .+= w2 .* (ΣA[i, j] - Σ̂[i, j]) .* ∂ΣA_μ[i, j]
        g_U .+= w2 .* (ΣA[i, j] - Σ̂[i, j]) .* ∂ΣA_U[i, j]
    end
    return g_μ, g_U
end

function _unpack_μ_U_Σ(p::Vector{Float64})
    n = n_from_param_size(length(p))
    μ = p[1:n]
    U = zeros(n, n)
    U[[CartesianIndex(i, j) for i = 1:n for j = i:n]] = p[(n+1):end]
    Ui = inv(UpperTriangular(U))
    Σ = Matrix(Ui * Ui')
    Σ = 0.5 .* (Σ .+ Σ')
    return μ, U, Σ
end

"""
    vector_moment_loss_t(p, ν, a, b, μ̂, Σ̂)

The t_ν moment-matching loss at the packed `(μ, U)` vector `p`.
"""
function vector_moment_loss_t(p::Vector{Float64}, ν, a, b, μ̂, Σ̂)
    μ, _, Σ = _unpack_μ_U_Σ(p)
    return moment_loss_t(TruncatedMvT(ν, μ, Σ, a, b; max_moment_levels = 2), μ̂, Σ̂)
end

"""
    vector_fg_true_loss_t(F, G, p, ν, a, b, μ̂, Σ̂)

Optim `only_fg!` entry point for the t_ν moment-matching loss and its
Theorem-2 gradient on the packed `(μ, U)` vector.
"""
function vector_fg_true_loss_t(F, G, p::Vector{Float64}, ν, a, b, μ̂, Σ̂)
    μ, U, Σ = _unpack_μ_U_Σ(p)
    d, d♯ = t_moment_matching_pair(ν, μ, Σ, a, b)
    if G !== nothing
        g_μ, g_U = grad_true_loss_t(d, d♯, U, μ̂, Σ̂)
        G .= make_param_vec_from_μ_U(g_μ, g_U)
    end
    F !== nothing && return moment_loss_t(d, μ̂, Σ̂)
    return nothing
end

"""
    fit_mvt(ν, μ̂, Σ̂, a, b; μ0 = μ̂, Σ0 = Σ̂ * (ν-2)/ν, ftarget = 1e-8,
            iterations = 200, time_limit = 60.0, verbose = false)

Find location `μ` and scale `Σ` such that `t_ν(μ, Σ)` truncated to
`[a, b]` has mean `μ̂` and covariance `Σ̂` (least-squares moment matching
by L-BFGS on `(μ, U)` with the Theorem-2 gradient). Proof of concept.
Returns `(μ, Σ, info)`.
"""
function fit_mvt(ν::Real, μ̂::AbstractVector, Σ̂::AbstractMatrix,
                 a::AbstractVector, b::AbstractVector;
                 μ0 = μ̂, Σ0 = (ν > 2 ? (ν - 2) / ν : 1.0) .* Σ̂,
                 ftarget = 1e-8, iterations = 200, time_limit = 60.0,
                 verbose = false)
    μ̂ = Vector{Float64}(μ̂); Σ̂ = Matrix{Float64}(Σ̂)
    a = Vector{Float64}(a); b = Vector{Float64}(b)
    p0 = make_param_vec_from_μ_Σ(Vector{Float64}(μ0), Matrix{Float64}(Σ0))
    fg!(F, G, p) = vector_fg_true_loss_t(F, G, p, ν, a, b, μ̂, Σ̂)
    opts = Optim.Options(show_trace = verbose, iterations = iterations,
                         time_limit = time_limit, callback = s -> s.f_x < ftarget)
    t = @elapsed res = optimize(only_fg!(fg!), p0, LBFGS(), opts)
    μ, _, Σ = _unpack_μ_U_Σ(res.minimizer)
    info = (; loss = res.minimum, time = t, iterations = res.iterations,
              converged = Optim.converged(res), res = res)
    return μ, Σ, info
end
