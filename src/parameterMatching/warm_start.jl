"""
    warm_start_diagonal(μ̂, Σ̂, a, b; iters = 50) -> (μ_ws, Σ_ws)

Coordinate-wise warm-start for the truncated multivariate-Gaussian
moment-matching problem. For each axis ``i = 1, \\ldots, n`` we solve the
1D problem

    find (μ, σ²) such that the truncated 𝒩(μ, σ²) on [a_i, b_i]
    has mean μ̂_i and variance Σ̂_{i,i},

via a 2-parameter LBFGS over (μ, log σ²) on the closed-form 1D
truncated-Normal moment recurrence (`moments(::Truncated{Normal})`).
The fitted per-axis (μ, σ²) values are then combined with the
correlation structure of `Σ̂` (preserved exactly: C = Σ̂ ⊘ σ̂σ̂',
Σ_ws = σσ' ⊙ C) to produce a full warm-started `(μ_ws, Σ_ws)`.

The joint LBFGS on the n-dimensional problem can then start from
`(μ_ws, Σ_ws)` rather than from `(μ̂, Σ̂)`; the diagonal moments are
already matched (to numerical precision of the 1D fit), so the
joint solver only has to correct off-diagonal Σ entries.

Cost: n independent 1D LBFGS runs of ~10–20 iterations each, each
calling the closed-form 1D moment recurrence ~5 times per iteration.
Total work is O(n) cheap operations — negligible compared to a single
iteration of the joint n-dimensional LBFGS at n ≥ 3.
"""
function warm_start_diagonal(μ̂::AbstractVector, Σ̂::AbstractMatrix,
                              a::AbstractVector, b::AbstractVector;
                              iters::Int = 50)
    n = length(μ̂)
    μ_ws  = Vector{Float64}(undef, n)
    σ²_ws = Vector{Float64}(undef, n)
    for i in 1:n
        ai, bi, μ̂i, Σ̂ii = a[i], b[i], μ̂[i], Σ̂[i, i]
        function f(p)
            μ_i  = p[1]
            σ²_i = exp(p[2])
            d_i  = truncated(Normal(μ_i, sqrt(σ²_i)), ai, bi)
            m    = moments(d_i, 2)
            return (m[1] - μ̂i)^2 + (m[2] - m[1]^2 - Σ̂ii)^2
        end
        p0  = [μ̂i, log(max(Σ̂ii, eps()))]
        res = optimize(f, p0, LBFGS(),
                       Optim.Options(iterations = iters);
                       autodiff = :forward)
        μ_ws[i]  = res.minimizer[1]
        σ²_ws[i] = exp(res.minimizer[2])
    end
    # Preserve the correlation structure of Σ̂ exactly: Σ_ws and Σ̂ have
    # the same correlation matrix, only the diagonal scales differ.
    σ̂   = sqrt.(diag(Σ̂))
    C    = Σ̂ ./ (σ̂ * σ̂')
    σ    = sqrt.(σ²_ws)
    Σ_ws = (σ * σ') .* C
    return μ_ws, Σ_ws
end
