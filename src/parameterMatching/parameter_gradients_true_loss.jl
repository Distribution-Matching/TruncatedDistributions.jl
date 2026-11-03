# Direct gradients of the *true* loss
#
#   L(μ, Σ) = ½‖μA(μ,Σ) − μ̂‖²  +  ½‖ΣA(μ,Σ) − Σ̂‖²_F,
#
# computed from primitive truncated moments via the chain rule, *without*
# going through the surrogate L̃. The advantage of this approach is that
# the chain rule is taken directly with respect to (μ, V); we never have
# to argue about whether ∂L̃/∂μA vanishes at μA = m^(1)/m^(0). The cost
# is that we re-derive the building blocks
#     ∂m^(p)/∂μ ,    ∂m^(p)/∂V
# for p = 0, 1, 2 from the score-function identity (max moment order 4).
#
# These functions are written for pedagogical clarity (vectors / matrices
# explicit, no fancy reshaping). They are intended for the verification
# experiment in test/experiment_true_loss_gradient.jl, not for fitting.

# ---------------------------------------------------------------------------
# Primitive-moment building blocks
# ---------------------------------------------------------------------------

# ∫_A x_{i_1} … x_{i_p} f(x; μ, Σ) dx
_m(d, inds::Vector{Int}) = raw_moment_from_indices(d, inds)

# ∫_A x_{i_1} … x_{i_p} (x_s - μ_s)(x_l - μ_l) f dx, expanded into m^(p+2),
# m^(p+1), m^(p) terms.
function _J(d::RecursiveMomentsBoxTruncatedMvNormal,
            inds::Vector{Int}, s::Int, l::Int)
    μ = d.untruncated.μ
    return _m(d, vcat(inds, [s, l])) -
           μ[s] * _m(d, vcat(inds, [l])) -
           μ[l] * _m(d, vcat(inds, [s])) +
           μ[s] * μ[l] * _m(d, inds)
end

# ∂m^(p)_{inds}/∂μ as a length-n row (1×n).
function moment_grad_μ(d::RecursiveMomentsBoxTruncatedMvNormal, inds::Vector{Int})
    μ = d.untruncated.μ
    Σ = Matrix(d.untruncated.Σ)
    n = length(d)
    mp = _m(d, inds)
    diff = [_m(d, vcat(inds, [j])) - μ[j] * mp for j in 1:n]
    return reshape(diff, 1, n) * inv(Σ)
end

# ∂m^(p)_{inds}/∂V as an n×n matrix, zero below the diagonal, where
# V_kk = log U_kk and V_kl = U_kl for k < l:
#   ∂m/∂V_kl = δ_kl m − ρ_kl Σ_{s≥k} U_ks J(s, l),  ρ_kk = U_kk, ρ_kl = 1 (k < l).
function moment_grad_V(d::RecursiveMomentsBoxTruncatedMvNormal,
                       inds::Vector{Int}, U::Matrix{Float64})
    n = length(d)
    G = zeros(n, n)
    for k in 1:n, l in k:n
        term2 = sum(U[k, s] * _J(d, inds, s, l) for s in k:n)
        G[k, l] = k == l ? _m(d, inds) - U[k, k] * term2 : -term2
    end
    return G
end

# ---------------------------------------------------------------------------
# Gradient of the true loss
# ---------------------------------------------------------------------------

# Returns (g_μ, g_V): the gradient of L with respect to μ (as a length-n
# vector) and with respect to V (as an upper-triangular n×n matrix).
function grad_true_loss(d::RecursiveMomentsBoxTruncatedMvNormal,
                        μ̂::Vector{Float64},
                        Σ̂::Matrix{Float64};
                        U::Union{Nothing, Matrix{Float64}} = nothing)
    n = length(d)
    Σ = Matrix(d.untruncated.Σ)
    if isnothing(U)
        U = Matrix(cholesky(0.5 * (inv(Σ) + inv(Σ)')).U)
    end

    # primitive moments
    m0  = _m(d, Int[])
    m1  = [_m(d, [i])    for i in 1:n]
    m2  = [_m(d, [i, j]) for i in 1:n, j in 1:n]
    μA  = m1 ./ m0
    ΣA  = m2 ./ m0 .- μA * μA'

    # ∂m^(0)/∂(μ,V), ∂m^(1)/∂(μ,V), ∂m^(2)/∂(μ,V)
    ∂m0_μ = vec(moment_grad_μ(d, Int[]))
    ∂m0_V = moment_grad_V(d, Int[], U)
    ∂m1_μ = [vec(moment_grad_μ(d, [i])) for i in 1:n]                # ∂m1[i]/∂μ
    ∂m1_V = [moment_grad_V(d, [i], U)   for i in 1:n]                # ∂m1[i]/∂V
    ∂m2_μ = [vec(moment_grad_μ(d, [i, j])) for i in 1:n, j in 1:n]   # ∂m2[i,j]/∂μ
    ∂m2_V = [moment_grad_V(d, [i, j], U)   for i in 1:n, j in 1:n]   # ∂m2[i,j]/∂V

    # ∂μA[i]/∂(μ,V)
    ∂μA_μ = [(∂m1_μ[i] .- μA[i] .* ∂m0_μ) ./ m0 for i in 1:n]
    ∂μA_V = [(∂m1_V[i] .- μA[i] .* ∂m0_V) ./ m0 for i in 1:n]

    # ∂ΣA[i,j]/∂(μ,V)  with ΣA[i,j] = m2[i,j]/m0 − μA[i] μA[j]:
    #   ∂ΣA[i,j] = (1/m0) ∂m2[i,j] − (m2[i,j]/m0²) ∂m0  − μA[j] ∂μA[i] − μA[i] ∂μA[j]
    ∂ΣA_μ = [(∂m2_μ[i, j] ./ m0) .- (m2[i, j] / m0^2) .* ∂m0_μ .-
              μA[j] .* ∂μA_μ[i] .- μA[i] .* ∂μA_μ[j] for i in 1:n, j in 1:n]
    ∂ΣA_V = [(∂m2_V[i, j] ./ m0) .- (m2[i, j] / m0^2) .* ∂m0_V .-
              μA[j] .* ∂μA_V[i] .- μA[i] .* ∂μA_V[j] for i in 1:n, j in 1:n]

    # finally chain into L = w1·½‖μA − μ̂‖² + w2·½‖ΣA − Σ̂‖²_F, with the
    # averaging weights w1 = 1/n, w2 = γ/n² (see `_mean_weight`, `_cov_weight`).
    w1 = _mean_weight(n)
    w2 = _cov_weight(n)
    g_μ = zeros(n)
    g_V = zeros(n, n)
    for i in 1:n
        g_μ .+= w1 .* (μA[i] - μ̂[i]) .* ∂μA_μ[i]
        g_V .+= w1 .* (μA[i] - μ̂[i]) .* ∂μA_V[i]
    end
    for i in 1:n, j in 1:n
        g_μ .+= w2 .* (ΣA[i, j] - Σ̂[i, j]) .* ∂ΣA_μ[i, j]
        g_V .+= w2 .* (ΣA[i, j] - Σ̂[i, j]) .* ∂ΣA_V[i, j]
    end
    return g_μ, g_V
end

# Convenience: gradient packed into a single param vector (matching the
# layout of make_param_vec_from_μ_U, so with V in place of U).
function vector_grad_true_loss(p::Vector{Float64}, a, b,
                               μ̂::Vector{Float64}, Σ̂::Matrix{Float64})
    n = n_from_param_size(length(p))
    μ, Σ = make_μ_Σ_from_param_vec(p)
    U = _U_from_param_vec(p)

    Σsym = 0.5 .* (Σ .+ Σ')
    Σsym .+= eps(Float64) * (tr(Σsym) + 1.0) * Matrix{Float64}(I, size(Σsym))
    d = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σsym), a, b; max_moment_levels = 4)
    g_μ, g_V = grad_true_loss(d, μ̂, Σ̂; U = U)
    return _pack_μ_V(g_μ, g_V)
end

# Combined-evaluation entry point for Optim's only_fg! / fg! interface.
# Returns the loss L(p) when F is not nothing, and writes ∇L(p) into G when
# G is not nothing. Inside an LBFGS iteration the same dist is reused for
# both the loss and the gradient, so a single Kan–Robotti recursion serves
# both — avoiding the doubled work of separate f and g! calls.
function vector_fg_true_loss(F, G, p::Vector{Float64}, a, b,
                             μ̂::Vector{Float64}, Σ̂::Matrix{Float64})
    if !_param_vec_admissible(p)
        G !== nothing && fill!(G, 0.0)
        return F !== nothing ? _LOSS_PENALTY : nothing
    end
    n = n_from_param_size(length(p))
    μ, Σ = make_μ_Σ_from_param_vec(p)
    U = _U_from_param_vec(p)

    Σsym = 0.5 .* (Σ .+ Σ')
    Σsym .+= eps(Float64) * (tr(Σsym) + 1.0) * Matrix{Float64}(I, size(Σsym))
    d = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σsym), a, b; max_moment_levels = 4)

    if G !== nothing
        g_μ, g_V = grad_true_loss(d, μ̂, Σ̂; U = U)
        G .= _pack_μ_V(g_μ, g_V)
    end
    if F !== nothing
        return moment_loss(d, μ̂, Σ̂)
    end
    return nothing
end

# Workspace-aware variant: takes a pre-built KR-tree state and refreshes it
# in place rather than allocating a fresh tree per call. Saves the O(n!)
# tree-construction cost across all LBFGS iterations on the same (n, a, b).
# The state must have been constructed with `max_moment_levels >= 4`.
function vector_fg_true_loss!(F, G, p::Vector{Float64},
                              state::BoxTruncatedMvNormalRecursiveMomentsState,
                              μ̂::Vector{Float64}, Σ̂::Matrix{Float64})
    if !_param_vec_admissible(p)
        G !== nothing && fill!(G, 0.0)
        return F !== nothing ? _LOSS_PENALTY : nothing
    end
    n = state.n
    μ, Σ = make_μ_Σ_from_param_vec(p)
    U = _U_from_param_vec(p)

    Σsym = 0.5 .* (Σ .+ Σ')
    Σsym .+= eps(Float64) * (tr(Σsym) + 1.0) * Matrix{Float64}(I, size(Σsym))
    update_distribution!(state, μ, PDMat(Σsym))
    d = outer_dist_from_state(state)

    if G !== nothing
        g_μ, g_V = grad_true_loss(d, μ̂, Σ̂; U = U)
        G .= _pack_μ_V(g_μ, g_V)
    end
    if F !== nothing
        return moment_loss(d, μ̂, Σ̂)
    end
    return nothing
end
