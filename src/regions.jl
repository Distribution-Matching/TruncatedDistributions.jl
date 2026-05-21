"""
    BoxTruncationRegion(a, b)

Axis-aligned box `{x : a ≤ x ≤ b}`. Either bound may be `±Inf` on any
coordinate (half-infinite and doubly-infinite faces are supported).
"""
struct BoxTruncationRegion <: TruncationRegion
    a::Vector{Float64}
    b::Vector{Float64}
end

intruncationregion(r::BoxTruncationRegion, x::AbstractArray) = all(r.a .<= x) && all(x .<= r.b) 

"""
    EllipticalTruncationRegion(H, h, c)

Elliptical truncation region: the set `{x : (x − h)ᵀ H (x − h) ≤ c}`,
where `H` is a positive-definite matrix.

Currently the package exposes the region type and its membership predicate
only; no full multivariate distribution is wired up against elliptical
regions yet.
"""
struct EllipticalTruncationRegion <: TruncationRegion
    H::PDMat
    h::Vector{Float64}
    c::Float64
end

intruncationregion(r::EllipticalTruncationRegion, x::AbstractArray) = (x - r.h)' * r.H * (x - r.h) <= r.c

"""
    intruncationregion(region, x)

`true` iff `x` lies inside the truncation region.
"""
intruncationregion