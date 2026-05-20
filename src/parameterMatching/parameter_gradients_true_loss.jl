# Direct gradients of the *true* loss
#
#   L(μ, Σ) = ½‖μA(μ,Σ) − μ̂‖²  +  ½‖ΣA(μ,Σ) − Σ̂‖²_F,
#
# computed from primitive truncated moments via the chain rule, *without*
# going through the surrogate L̃. The advantage of this approach is that
# the chain rule is taken directly with respect to (μ, U); we never have
# to argue about whether ∂L̃/∂μA vanishes at μA = m^(1)/m^(0). The cost
# is that we re-derive the building blocks
#     ∂m^(p)/∂μ ,    ∂m^(p)/∂U
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

# ∂m^(p)_{inds}/∂U as an n×n matrix, zero below the diagonal (matching the
# convention for upper-triangular U-derivatives).
function moment_grad_U(d::RecursiveMomentsBoxTruncatedMvNormal,
                       inds::Vector{Int}, U::Matrix{Float64})
    n = length(d)
    iU = inv(U)
    G = zeros(n, n)
    for k in 1:n, l in k:n
        if k == l
            term1 = (1/U[k,k]) * _m(d, inds)
        else
            term1 = 0.0
        end
        term2 = sum(U[k, s] * _J(d, inds, s, l) for s in k:n)
        G[k, l] = term1 - term2
    end
    return G
end

# ---------------------------------------------------------------------------
# Gradient of the true loss
# ---------------------------------------------------------------------------

# Returns (g_μ, g_U): the gradient of L with respect to μ (as a length-n
# vector) and with respect to U (as an upper-triangular n×n matrix).
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

    # ∂m^(0)/∂(μ,U), ∂m^(1)/∂(μ,U), ∂m^(2)/∂(μ,U)
    ∂m0_μ = vec(moment_grad_μ(d, Int[]))
    ∂m0_U = moment_grad_U(d, Int[], U)
    ∂m1_μ = [vec(moment_grad_μ(d, [i])) for i in 1:n]                # ∂m1[i]/∂μ
    ∂m1_U = [moment_grad_U(d, [i], U)   for i in 1:n]                # ∂m1[i]/∂U
    ∂m2_μ = [vec(moment_grad_μ(d, [i, j])) for i in 1:n, j in 1:n]   # ∂m2[i,j]/∂μ
    ∂m2_U = [moment_grad_U(d, [i, j], U)   for i in 1:n, j in 1:n]   # ∂m2[i,j]/∂U

    # ∂μA[i]/∂(μ,U)
    ∂μA_μ = [(∂m1_μ[i] .- μA[i] .* ∂m0_μ) ./ m0 for i in 1:n]
    ∂μA_U = [(∂m1_U[i] .- μA[i] .* ∂m0_U) ./ m0 for i in 1:n]

    # ∂ΣA[i,j]/∂(μ,U)  with ΣA[i,j] = m2[i,j]/m0 − μA[i] μA[j]:
    #   ∂ΣA[i,j] = (1/m0) ∂m2[i,j] − (m2[i,j]/m0²) ∂m0  − μA[j] ∂μA[i] − μA[i] ∂μA[j]
    ∂ΣA_μ = [(∂m2_μ[i, j] ./ m0) .- (m2[i, j] / m0^2) .* ∂m0_μ .-
              μA[j] .* ∂μA_μ[i] .- μA[i] .* ∂μA_μ[j] for i in 1:n, j in 1:n]
    ∂ΣA_U = [(∂m2_U[i, j] ./ m0) .- (m2[i, j] / m0^2) .* ∂m0_U .-
              μA[j] .* ∂μA_U[i] .- μA[i] .* ∂μA_U[j] for i in 1:n, j in 1:n]

    # finally chain into L
    g_μ = zeros(n)
    g_U = zeros(n, n)
    for i in 1:n
        g_μ .+= (μA[i] - μ̂[i]) .* ∂μA_μ[i]
        g_U .+= (μA[i] - μ̂[i]) .* ∂μA_U[i]
    end
    for i in 1:n, j in 1:n
        g_μ .+= (ΣA[i, j] - Σ̂[i, j]) .* ∂ΣA_μ[i, j]
        g_U .+= (ΣA[i, j] - Σ̂[i, j]) .* ∂ΣA_U[i, j]
    end
    return g_μ, g_U
end

# Convenience: gradient packed into a single param vector (matching the
# layout of make_param_vec_from_μ_U).
function vector_grad_true_loss(p::Vector{Float64}, a, b,
                               μ̂::Vector{Float64}, Σ̂::Matrix{Float64})
    n = n_from_param_size(length(p))
    μ, Σ = make_μ_Σ_from_param_vec(p)
    inds_upper = [CartesianIndex(i, j) for i = 1:n for j = i:n]
    U = zeros(n, n); U[inds_upper] = p[(n+1):end]
    U = Matrix(UpperTriangular(U))

    Σsym = 0.5 .* (Σ .+ Σ')
    Σsym .+= eps(Float64) * (tr(Σsym) + 1.0) * Matrix{Float64}(I, size(Σsym))
    d = RecursiveMomentsBoxTruncatedMvNormal(μ, PDMat(Σsym), a, b; max_moment_levels = 4)
    g_μ, g_U = grad_true_loss(d, μ̂, Σ̂; U = U)
    return make_param_vec_from_μ_U(g_μ, g_U)
end
