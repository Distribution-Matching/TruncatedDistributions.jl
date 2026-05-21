"""
    BasicBoxTruncatedMvNormal(μ, Σ, a, b)

Box-truncated multivariate normal with a minimal cached state (mean and
covariance only). Computes everything via direct cubature over the box —
fine at very low dimensions (n = 2, 3) but slow beyond that.

Prefer [`TruncatedMvNormal`](@ref) (the alias for the recursive-moment
implementation) for anything beyond n = 3.
"""
const BasicBoxTruncatedMvNormal =
    TruncatedMvDistribution{MvNormal, BoxTruncationRegion, TruncatedMvDistributionSecondOrderState}

function BasicBoxTruncatedMvNormal(μₑ::Vector{Float64},
                                   Σₑ::PDMat,
                                   a::Vector{Float64},
                                   b::Vector{Float64})
    d = MvNormal(μₑ, Σₑ)
    r = BoxTruncationRegion(a, b)
    TruncatedMvDistribution{MvNormal, BoxTruncationRegion,
                            TruncatedMvDistributionSecondOrderState}(d, r)
end

function BasicBoxTruncatedMvNormal(μₑ::AbstractVector,
                                   Σₑ::AbstractMatrix,
                                   a::AbstractVector,
                                   b::AbstractVector)
    Σm  = Matrix{Float64}(Σₑ)
    Σpd = PDMat(0.5 .* (Σm .+ Σm'))
    return BasicBoxTruncatedMvNormal(Vector{Float64}(μₑ), Σpd,
                                     Vector{Float64}(a),
                                     Vector{Float64}(b))
end

"""
    RecursiveMomentsBoxTruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 2)

Box-truncated multivariate normal with the cached state needed for the
Kan & Robotti recursive moment computation. The cache pre-allocates every
moment up to total order `max_moment_levels`. See [`TruncatedMvNormal`](@ref).
"""
const RecursiveMomentsBoxTruncatedMvNormal =
    TruncatedMvDistribution{MvNormal, BoxTruncationRegion,
                            BoxTruncatedMvNormalRecursiveMomentsState}

function RecursiveMomentsBoxTruncatedMvNormal(μₑ::Vector{Float64},
                                              Σₑ::PDMat,
                                              a::Vector{Float64},
                                              b::Vector{Float64};
                                              max_moment_levels::Int = 2)
    d = MvNormal(μₑ, Σₑ)
    r = BoxTruncationRegion(a, b)
    s = BoxTruncatedMvNormalRecursiveMomentsState(d, r, max_moment_levels)
    TruncatedMvDistribution{MvNormal, BoxTruncationRegion,
                            BoxTruncatedMvNormalRecursiveMomentsState}(d, r, s)
end

# Convenience constructor: accept Σ as any AbstractMatrix and box bounds /
# means as any AbstractVector. Performs the symmetrize-and-wrap step.
function RecursiveMomentsBoxTruncatedMvNormal(μₑ::AbstractVector,
                                              Σₑ::AbstractMatrix,
                                              a::AbstractVector,
                                              b::AbstractVector;
                                              max_moment_levels::Int = 2)
    Σm  = Matrix{Float64}(Σₑ)
    Σpd = PDMat(0.5 .* (Σm .+ Σm'))
    return RecursiveMomentsBoxTruncatedMvNormal(
        Vector{Float64}(μₑ), Σpd,
        Vector{Float64}(a), Vector{Float64}(b);
        max_moment_levels = max_moment_levels)
end

"""
    TruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 2)

Friendly alias for [`RecursiveMomentsBoxTruncatedMvNormal`](@ref) — the
recommended box-truncated multivariate normal distribution. `Σ` can be a
`Matrix`, `Symmetric`, or `PDMat`; box faces in `a` / `b` may be `±Inf`.

```julia
d = TruncatedMvNormal([0.0, 0.0], [1.0 0.3; 0.3 1.0], [-1.0, -1.5], [1.5, 1.0])
mean(d)
cov(d)
pdf(d, [0.0, 0.0])
```
"""
const TruncatedMvNormal = RecursiveMomentsBoxTruncatedMvNormal

"""
    params(d::TruncatedMvNormal)

Return `(μ, Σ, a, b)` of the untruncated normal and the box bounds.
"""
function params(d::RecursiveMomentsBoxTruncatedMvNormal)
    return (d.untruncated.μ, d.untruncated.Σ, d.region.a, d.region.b)
end

function params(d::BasicBoxTruncatedMvNormal)
    return (d.untruncated.μ, d.untruncated.Σ, d.region.a, d.region.b)
end

"""
    minimum(d::TruncatedMvNormal)
    maximum(d::TruncatedMvNormal)

Lower / upper bounds of the box truncation region (possibly `±Inf`).
"""
Base.minimum(d::RecursiveMomentsBoxTruncatedMvNormal) = d.region.a
Base.maximum(d::RecursiveMomentsBoxTruncatedMvNormal) = d.region.b
Base.minimum(d::BasicBoxTruncatedMvNormal) = d.region.a
Base.maximum(d::BasicBoxTruncatedMvNormal) = d.region.b

function Base.show(io::IO, d::RecursiveMomentsBoxTruncatedMvNormal)
    println(io, "TruncatedMvNormal, n = $(length(d))")
    println(io, "μ:\n $(d.untruncated.μ)")
    println(io, "Σ:")
        show(io, "text/plain", d.untruncated.Σ)
    println(io)
    println(io, "a:\n $(d.region.a)")
    println(io, "b:\n $(d.region.b)")
    if isfinite(d.state.tp)
        println(io, "tp: $(round(d.state.tp; digits = 5))")
    end
    if !isempty(d.state.μ)
        println(io, "mean: $(round.(d.state.μ; digits = 5))")
    end
end

function Base.show(io::IO, d::BasicBoxTruncatedMvNormal)
    println(io, "BasicBoxTruncatedMvNormal, n = $(length(d))")
    println(io, "μ:\n $(d.untruncated.μ)")
    println(io, "Σ:")
        show(io, "text/plain", d.untruncated.Σ)
    println(io)
    println(io, "a:\n $(d.region.a)")
    println(io, "b:\n $(d.region.b)")
end
