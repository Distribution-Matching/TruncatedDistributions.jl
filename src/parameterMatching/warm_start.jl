"""
    warm_start_diagonal(μ̂, Σ̂, a, b) -> (μ_ws, Σ_ws)

Coordinate-wise warm start for the truncated multivariate-normal
moment-matching problem. For each axis ``i`` it finds the location and
scale ``(μ_i, σ_i)`` of the univariate normal whose truncation to
``[a_i, b_i]`` has mean `μ̂[i]` and variance `Σ̂[i, i]`, by
`DistributionsFactories.truncated_locscale_params` (damped Newton on the
closed-form truncated moments, about a microsecond per axis). The
correlation matrix of `Σ̂` is kept, `Σ_ws = D C̃ D` with
`D = Diagonal(σ)`, so `Σ_ws` is positive definite.

Throws `DistributionsFactories.InfeasibleMomentsError` if some axis has no
univariate solution.
"""
function warm_start_diagonal(μ̂::AbstractVector, Σ̂::AbstractMatrix,
                             a::AbstractVector, b::AbstractVector)
    return _warm_start_diagonal(Normal, μ̂, Σ̂, a, b)
end

function _warm_start_diagonal(family, μ̂::AbstractVector, Σ̂::AbstractMatrix,
                              a::AbstractVector, b::AbstractVector)
    n = length(μ̂)
    μ_ws = Vector{Float64}(undef, n)
    σ_ws = Vector{Float64}(undef, n)
    for i in 1:n
        μ_ws[i], σ_ws[i] = truncated_locscale_params(family, Float64(a[i]), Float64(b[i]),
                                                     Float64(μ̂[i]), Float64(Σ̂[i, i]))
    end
    # Keep the correlation matrix of Σ̂; only the diagonal scales change.
    σ̂ = sqrt.(diag(Σ̂))
    C = Σ̂ ./ (σ̂ * σ̂')
    return μ_ws, (σ_ws * σ_ws') .* C
end

# Warm start, or the naive start when some axis has no univariate solution.
# The fit then still minimises the loss; the warning says why it starts
# where it does (the target may be unattainable).
function _warm_start_or_naive(ws::Function, μ_naive, Σ_naive)
    try
        return ws()
    catch e
        e isa InfeasibleMomentsError || rethrow()
        @warn "Warm start failed, starting from the naive point instead; " *
              "the target may be unattainable. " * sprint(showerror, e)
        return collect(Float64, μ_naive), Matrix{Float64}(Σ_naive)
    end
end
