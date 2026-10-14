# Moment matching for the box-truncated multivariate Student-t with fixed
# degrees of freedom ν.
#
# Same loss and (μ, U) chart as the Gaussian, Σ⁻¹ = UᵀU. The t score is
# the Gaussian one with the quadratic-form terms weighted by
# w(Q) = (n+ν)/(ν+Q), and w·f_ν(·; μ, Σ) = f_{ν+2}(·; μ, Σ♯) with
# Σ♯ = ν Σ / (ν+2) (the degrees-of-freedom shift). So
#
#   ∂m_α/∂μ_k   = Σ_j [UᵀU]_{kj} (m♯_{α,j} − μ_j m♯_α),
#   ∂m_α/∂U_kl  = δ_kl/U_kk · m_α − Σ_{s≥k} U_ks J♯(α, s, l),
#
# with m♯ the raw moments of the truncated t_{ν+2}(μ, Σ♯) and J♯ the
# corresponding centred second-order combination. Everything downstream
# (quotient and chain rule into L) is identical to `grad_true_loss`.
#
# One recursion tree serves both laws: the helper node of the t_{ν+2}(μ, Σ♯)
# recursion is t_ν(μ, (ν+2)/ν · Σ♯) = t_ν(μ, Σ) at order 2, which is exactly
# the law whose mean and covariance are being matched.

"""
    TMomentPair

The two box-truncated Student-t laws the moment-matching gradient needs,
sharing one recursion tree: `sharp`, the `t_{ν+2}(μ, νΣ/(ν+2))` law with
raw moments up to order 4, and `base`, the `t_ν(μ, Σ)` law (its helper
node) with raw moments up to order 2. Built by
[`t_moment_matching_pair`](@ref).
"""
struct TMomentPair
    ν::Float64
    μ::Vector{Float64}
    sharp::BoxTruncatedMvTRecursiveMomentsState
    base::BoxTruncatedMvTRecursiveMomentsState
end

"""
    t_moment_matching_pair(ν, μ, Σ, a, b) -> TMomentPair

Build the shared recursion for the truncated `t_ν(μ, Σ)` law on the box
`[a, b]` (moments to order 2) and its degrees-of-freedom-shifted companion
`t_{ν+2}(μ, νΣ/(ν+2))` (moments to order 4). On a box that is unbounded
in some coordinate this requires `ν > 2`.
"""
function t_moment_matching_pair(ν::Real, μ::AbstractVector, Σ::AbstractMatrix,
                                a::AbstractVector, b::AbstractVector)
    ν > 0 || throw(ArgumentError("degrees of freedom must be positive; got ν = $ν"))
    if any(isinf, a) || any(isinf, b)
        ν > 2 || throw(ArgumentError(
            "moment matching a t_ν on an unbounded box needs ν > 2; got ν = $ν"))
    end
    r = BoxTruncationRegion(Vector{Float64}(a), Vector{Float64}(b))
    Σs = (ν / (ν + 2)) .* Matrix{Float64}(Σ)
    sharp = BoxTruncatedMvTRecursiveMomentsState(ν + 2, μ, Σs, r, 4)
    base = sharp.helper
    return TMomentPair(Float64(ν), Vector{Float64}(μ), sharp, base)
end

# Raw moment ∫_A x_{inds} f dx of a recursion state, indexed by a list of
# coordinates (repetitions allowed) rather than a multi-index.
function _mt(s::BoxTruncatedMvTRecursiveMomentsState, inds)
    κ = zeros(Int, s.n)
    for i in inds
        κ[i] += 1
    end
    return raw_moment(s, κ)
end

# ∂m_α/∂μ (length n) and ∂m_α/∂U (upper triangular n×n) for the base law,
# using the shifted law for the score-generated terms.
function _t_moment_grad(P::TMomentPair, U, UᵀU, inds::Vector{Int})
    n = P.base.n
    μ = P.μ
    s♯ = P.sharp
    m♯ = _mt(s♯, inds)
    m♯1 = [_mt(s♯, vcat(inds, j)) for j in 1:n]
    g_μ = UᵀU * (m♯1 .- μ .* m♯)
    # J♯(s, l) = ∫ x^α (x_s − μ_s)(x_l − μ_l) f♯
    J = [_mt(s♯, vcat(inds, [s, l])) - μ[s] * m♯1[l] - μ[l] * m♯1[s] + μ[s] * μ[l] * m♯
         for s in 1:n, l in 1:n]
    mp = _mt(P.base, inds)
    g_U = zeros(n, n)
    for k in 1:n, l in k:n
        acc = k == l ? mp / U[k, k] : 0.0
        for s in k:n
            acc -= U[k, s] * J[s, l]
        end
        g_U[k, l] = acc
    end
    return g_μ, g_U
end

"""
    moment_loss_t(P::TMomentPair, μ̂, Σ̂)

The moment-matching loss `L = ½‖μA − μ̂‖² + γ·r(n)·½‖ΣA − Σ̂‖²_F` for the
truncated `t_ν` law held in `P`.
"""
function moment_loss_t(P::TMomentPair, μ̂::AbstractVector, Σ̂::AbstractMatrix)
    s = P.base
    n = s.n
    m0 = _mt(s, Int[])
    (!isfinite(m0) || m0 < eps(Float64)) && return prevfloat(Inf) / 4
    m1 = [_mt(s, [i]) for i in 1:n]
    m2 = [_mt(s, [i, j]) for i in 1:n, j in 1:n]
    μA = m1 ./ m0
    ΣA = m2 ./ m0 .- μA * μA'
    L = 0.5 * (sum(abs2, μA .- μ̂) + _cov_weight(n) * sum(abs2, ΣA .- Σ̂))
    return isfinite(L) ? L : prevfloat(Inf) / 4
end

"""
    grad_true_loss_t(P::TMomentPair, U, μ̂, Σ̂) -> (g_μ, g_U)

Gradient of the truncated-t moment-matching loss with respect to `μ` and
the upper-triangular `U` (`UᵀU = Σ⁻¹`), assembled in closed form from the
truncated `t_ν` moments of order ≤ 2 and the truncated `t_{ν+2}` moments
of order ≤ 4 held in `P` (see [`t_moment_matching_pair`](@ref)).
"""
function grad_true_loss_t(P::TMomentPair, U::AbstractMatrix,
                          μ̂::AbstractVector, Σ̂::AbstractMatrix)
    s = P.base
    n = s.n
    U = Matrix{Float64}(U)
    UᵀU = U' * U
    m0 = _mt(s, Int[])
    m1 = [_mt(s, [i]) for i in 1:n]
    m2 = [_mt(s, [i, j]) for i in 1:n, j in 1:n]
    μA = m1 ./ m0
    ΣA = m2 ./ m0 .- μA * μA'

    ∂m0_μ, ∂m0_U = _t_moment_grad(P, U, UᵀU, Int[])
    ∂m1 = [_t_moment_grad(P, U, UᵀU, [i]) for i in 1:n]
    ∂m2 = Matrix{Tuple{Vector{Float64},Matrix{Float64}}}(undef, n, n)
    for i in 1:n, j in i:n
        ∂m2[i, j] = _t_moment_grad(P, U, UᵀU, [i, j])
        ∂m2[j, i] = ∂m2[i, j]
    end

    ∂μA_μ = [(∂m1[i][1] .- μA[i] .* ∂m0_μ) ./ m0 for i in 1:n]
    ∂μA_U = [(∂m1[i][2] .- μA[i] .* ∂m0_U) ./ m0 for i in 1:n]

    w2 = _cov_weight(n)
    g_μ = zeros(n)
    g_U = zeros(n, n)
    for i in 1:n
        g_μ .+= (μA[i] - μ̂[i]) .* ∂μA_μ[i]
        g_U .+= (μA[i] - μ̂[i]) .* ∂μA_U[i]
    end
    for i in 1:n, j in 1:n
        r = w2 * (ΣA[i, j] - Σ̂[i, j])
        iszero(r) && continue
        ∂ΣA_μ = (∂m2[i, j][1] ./ m0) .- (m2[i, j] / m0^2) .* ∂m0_μ .-
                μA[j] .* ∂μA_μ[i] .- μA[i] .* ∂μA_μ[j]
        ∂ΣA_U = (∂m2[i, j][2] ./ m0) .- (m2[i, j] / m0^2) .* ∂m0_U .-
                μA[j] .* ∂μA_U[i] .- μA[i] .* ∂μA_U[j]
        g_μ .+= r .* ∂ΣA_μ
        g_U .+= r .* ∂ΣA_U
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
    Σ[diagind(Σ)] .+= eps(Float64) * (tr(Σ) + 1.0)
    return μ, U, Σ
end

"""
    vector_moment_loss_t(p, ν, a, b, μ̂, Σ̂)

The truncated-t moment-matching loss at the packed `(μ, U)` vector `p`
(layout of `make_param_vec_from_μ_U`).
"""
function vector_moment_loss_t(p::Vector{Float64}, ν, a, b, μ̂, Σ̂)
    μ, _, Σ = _unpack_μ_U_Σ(p)
    s = BoxTruncatedMvTRecursiveMomentsState(ν, μ, Σ,
            BoxTruncationRegion(Vector{Float64}(a), Vector{Float64}(b)), 2)
    P = TMomentPair(Float64(ν), μ, s, s)
    return moment_loss_t(P, μ̂, Σ̂)
end

"""
    vector_fg_true_loss_t(F, G, p, ν, a, b, μ̂, Σ̂)

Optim `only_fg!` entry point for the truncated-t moment-matching loss and
its closed-form gradient on the packed `(μ, U)` vector. One shared
recursion serves the loss and the gradient.
"""
function vector_fg_true_loss_t(F, G, p::Vector{Float64}, ν, a, b, μ̂, Σ̂)
    μ, U, Σ = _unpack_μ_U_Σ(p)
    P = t_moment_matching_pair(ν, μ, Σ, a, b)
    if G !== nothing
        g_μ, g_U = grad_true_loss_t(P, U, μ̂, Σ̂)
        G .= make_param_vec_from_μ_U(g_μ, g_U)
    end
    F !== nothing && return moment_loss_t(P, μ̂, Σ̂)
    return nothing
end

"""
    warm_start_diagonal_t(ν, μ̂, Σ̂, a, b; iters = 50) -> (μ_ws, Σ_ws)

Coordinate-wise warm start for the truncated multivariate-t
moment-matching problem: the analogue of [`warm_start_diagonal`](@ref)
with each axis fitted as a univariate truncated `t_ν(μ_i, σ_i²)`, and the
correlation structure of `Σ̂` preserved exactly.
"""
function warm_start_diagonal_t(ν::Real, μ̂::AbstractVector, Σ̂::AbstractMatrix,
                               a::AbstractVector, b::AbstractVector; iters::Int = 50)
    n = length(μ̂)
    μ_ws = Vector{Float64}(undef, n)
    σ²_ws = Vector{Float64}(undef, n)
    scale0 = ν > 2 ? (ν - 2) / ν : 1.0
    for i in 1:n
        ai, bi, μ̂i, Σ̂ii = Float64(a[i]), Float64(b[i]), μ̂[i], Σ̂[i, i]
        r = BoxTruncationRegion([ai], [bi])
        function f(p)
            s = BoxTruncatedMvTRecursiveMomentsState(ν, [p[1]], fill(exp(p[2]), 1, 1), r, 2)
            m0 = raw_moment(s, [0])
            (isfinite(m0) && m0 > eps()) || return 1e10
            m1 = raw_moment(s, [1]) / m0
            m2 = raw_moment(s, [2]) / m0
            return (m1 - μ̂i)^2 + (m2 - m1^2 - Σ̂ii)^2
        end
        p0 = [μ̂i, log(max(scale0 * Σ̂ii, eps()))]
        res = optimize(f, p0, NelderMead(), Optim.Options(iterations = 20 * iters))
        μ_ws[i] = res.minimizer[1]
        σ²_ws[i] = exp(res.minimizer[2])
    end
    σ̂ = sqrt.(diag(Σ̂))
    C = Σ̂ ./ (σ̂ * σ̂')
    σ = sqrt.(σ²_ws)
    return μ_ws, (σ * σ') .* C
end

"""
    fit_mvt(ν, μ̂, Σ̂, a, b; ...)

Find a location `μ` and scale matrix `Σ` such that the multivariate
Student-t `t_ν(μ, Σ)` (fixed degrees of freedom `ν`) truncated to the box
`[a, b]` has mean `μ̂` and covariance `Σ̂`. Returns `(μ, Σ, info)`.

The solver is joint LBFGS over the packed `(μ, U)` vector (`UᵀU = Σ⁻¹`)
with the closed-form gradient of the paper: the truncated `t_ν` moments of
order ≤ 2 and the truncated `t_{ν+2}(μ, νΣ/(ν+2))` moments of order ≤ 4,
both from one Kan–Robotti-type recursion tree. The loss and the meaning of
`gamma` are as in [`fit_mvnormal`](@ref). Unlike the Gaussian case, the
matching equations are not likelihood equations here.

On a box that is unbounded in some coordinate, `ν > 2` is required (the
covariance must exist).

# Keyword arguments

* `μ_init`, `Σ_init` — starting point; default is the coordinate warm start
  [`warm_start_diagonal_t`](@ref).
* `ftarget`    — stop when the loss falls below this (default `1e-3`).
* `gamma`      — covariance-term weight (default `0.2`).
* `seed`       — seed the base-case RNG for a reproducible fit.
* `time_limit` — seconds (default 60).
* `iterations` — LBFGS iteration cap (default 50).
* `verbose`    — print the LBFGS trace.

# Example

```julia
μ, Σ, info = fit_mvt(5.0, [0.1, -0.1], [0.4 0.05; 0.05 0.4],
                     [-1.0, -1.5], [1.5, Inf])
```
"""
function fit_mvt(ν::Real, μ̂::AbstractVector, Σ̂::AbstractMatrix,
                 a::AbstractVector, b::AbstractVector;
                 μ_init::Union{Nothing,AbstractVector} = nothing,
                 Σ_init::Union{Nothing,AbstractMatrix} = nothing,
                 ftarget::Float64 = 1e-3,
                 time_limit::Float64 = 60.0,
                 iterations::Int = 50,
                 verbose::Bool = false,
                 gamma::Real = 0.2,
                 seed::Union{Nothing,Integer} = nothing)
    n = length(μ̂)
    size(Σ̂) == (n, n) ||
        throw(DimensionMismatch("Σ̂ must be $(n)×$(n); got $(size(Σ̂))"))
    length(a) == n == length(b) ||
        throw(DimensionMismatch("a, b must have length $(n)"))
    all(a .< b) || throw(ArgumentError("each a[i] must be strictly less than b[i]"))
    ν > 0 || throw(ArgumentError("degrees of freedom must be positive; got ν = $ν"))
    if any(isinf, a) || any(isinf, b)
        ν > 2 || throw(ArgumentError(
            "moment matching a t_ν on an unbounded box needs ν > 2; got ν = $ν"))
    end
    a_v = collect(Float64, a); b_v = collect(Float64, b)
    μ̂_v = collect(Float64, μ̂); Σ̂_m = Matrix{Float64}(Σ̂)

    prev_gamma = set_loss_gamma!(gamma)
    prev_rng = seed === nothing ? nothing : set_kr_base_rng!(MersenneTwister(seed))
    try
        t_ws = @elapsed if μ_init === nothing || Σ_init === nothing
            μ_ws, Σ_ws = warm_start_diagonal_t(ν, μ̂_v, Σ̂_m, a_v, b_v)
            μ_init === nothing && (μ_init = μ_ws)
            Σ_init === nothing && (Σ_init = Σ_ws)
        end
        p0 = make_param_vec_from_μ_Σ(collect(Float64, μ_init), Matrix{Float64}(Σ_init))
        fg!(F, G, p) = vector_fg_true_loss_t(F, G, p, ν, a_v, b_v, μ̂_v, Σ̂_m)
        opts = Optim.Options(show_trace = verbose, iterations = iterations,
                             time_limit = time_limit,
                             callback = s -> s.f_x < ftarget)
        t = @elapsed res = optimize(only_fg!(fg!), p0, LBFGS(), opts)
        μ_fit, _, Σ_fit = _unpack_μ_U_Σ(res.minimizer)
        info = (; method = :lbfgs, loss = res.minimum, time = t + t_ws,
                  iterations = res.iterations, converged = Optim.converged(res),
                  res = res)
        return μ_fit, Symmetric(Σ_fit), info
    finally
        set_loss_gamma!(prev_gamma)
        prev_rng === nothing || set_kr_base_rng!(prev_rng)
    end
end
