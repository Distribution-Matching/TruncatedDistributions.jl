# Dynamic (ODE-based) moment matching for *univariate* truncated distributions.
#
# Implements the method of
#
#   B. Liquet and Y. Nazarathy, "A dynamic view to moment matching of truncated
#   distributions", Statistics & Probability Letters 104 (2015) 87–93.
#   https://doi.org/10.1016/j.spl.2015.05.006
#
# Idea: to match the moments of a distribution truncated to a target interval
# [a, b] we follow a homotopy. With z ∈ (0, 1] define the expanding interval
#
#     (l_z, h_z) = ( a - (1-z)/z ,  b + (1-z)/z ).
#
# As z → 0 the interval → (-∞, ∞) (no truncation), where the moment-matching
# solution is explicit; at z = 1 it is the target [a, b]. Differentiating the
# moment-matching conditions in z gives an ODE for the parameter path θ(z) that
# keeps the desired moments fixed throughout. Solving it from z = ε to z = 1
# yields the parameters of the untruncated distribution that, once truncated to
# [a, b], has the requested moments — and, as a bonus, the whole trajectory
# solves the family of problems for every intermediate truncation interval.
#
# This is a *tool* operating on univariate location-scale and exponential
# families; it is independent of the multivariate truncated-normal machinery in
# the rest of the package.
#
# ─────────────────────────────────────────────────────────────────────────────
# Implementation notes
#
# * We integrate in the reparameterisation s = (1-z)/z, so that l_z = a - s and
#   h_z = b + s. This cancels the 1/z² factor that appears in the z-form of the
#   ODE (that factor is the bound velocity dh_z/dz = -1/z²), removing the
#   stiffness near z = 0 and letting a plain fixed-step RK4 reach the published
#   trajectories to ~1e-9. No external ODE solver is required.
#
# * NB — typo in the paper. The closed-form coefficients p_i are given correctly
#   by the *general* Eq. (11), p_i = θ₂²(h_i - l_i - i·n_{i-1}), but the explicit
#   list printed at the bottom of p. 91 drops the integer factor i: it shows
#   p₂ = θ₂²(h₂ - l₂ - n₀m₁*) and p₃ = θ₂²(h₃ - l₃ - n₀m₂*), which should read
#   -2n₀m₁* and -3n₀m₂* respectively. The code below uses the correct Eq. (11)
#   form (the `i` multiplier in `_p`); with the printed coefficients the scale
#   ODE produces NaNs and does not reproduce the paper's own Fig. 2.

"""
    DynamicMomentMatch

Result of a dynamic (ODE-based) moment-matching solve. Holds the full parameter
trajectory `θ(z)` along the homotopy as well as the endpoint solution at `z = 1`.

Fields:
- `family::Symbol`         — `:locationscale` or `:exponential`.
- `a::Float64`, `b::Float64` — the target truncation interval `[a, b]`.
- `z::Vector{Float64}`     — homotopy values where the path was recorded (ascending; last is `1.0`).
- `params::Vector{Vector{Float64}}` — parameter vector at each `z`. For
  `:locationscale` each entry is `[location, scale]`; for `:exponential` each is `[rate]`.

Use [`solution`](@ref) for the endpoint parameters at `z = 1`.
"""
struct DynamicMomentMatch
    family::Symbol
    a::Float64
    b::Float64
    z::Vector{Float64}
    params::Vector{Vector{Float64}}
end

"""
    solution(r::DynamicMomentMatch)

Return the fitted parameters at `z = 1` (the target interval). A `[location, scale]`
vector for a location-scale fit, or `[rate]` for an exponential fit.
"""
solution(r::DynamicMomentMatch) = r.params[end]

function Base.show(io::IO, r::DynamicMomentMatch)
    p = solution(r)
    if r.family === :locationscale
        print(io, "DynamicMomentMatch(location-scale on [", r.a, ", ", r.b,
                  "]: location=", p[1], ", scale=", p[2], ")")
    else
        print(io, "DynamicMomentMatch(exponential on [", r.a, ", ", r.b,
                  "]: rate=", p[1], ")")
    end
end

# Fixed-step RK4 in s, integrating from s = S0 (z = ε) down to s = 0 (z = 1).
# `rhs(θ, s)` returns dθ/ds. Records the parameter vector at each requested z in
# `z_save` (ascending), plus always the endpoint z = 1.
function _integrate_homotopy(rhs, θ0::Vector{Float64}, ε::Float64, nsteps::Int,
                             z_save::AbstractVector{<:Real})
    S0 = (1 - ε) / ε
    h  = S0 / nsteps
    # checkpoints as s-values, visited in descending order as s decreases
    zs = sort!(unique(vcat(Float64.(z_save), 1.0)))      # ascending z
    s_targets = [(1 - z) / z for z in zs]                # descending? z asc → s desc
    order = sortperm(s_targets; rev = true)              # visit largest s first
    rec_params = Vector{Vector{Float64}}(undef, length(zs))
    θ = copy(θ0)
    s = S0
    ci = 1
    for _ in 1:nsteps
        while ci <= length(order) && s <= s_targets[order[ci]] + 1e-12
            rec_params[order[ci]] = copy(θ)
            ci += 1
        end
        k1 = rhs(θ,                 s)
        k2 = rhs(θ .- (h/2) .* k1,  s - h/2)
        k3 = rhs(θ .- (h/2) .* k2,  s - h/2)
        k4 = rhs(θ .- h     .* k3,  s - h)
        θ  = θ .- (h/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        s -= h
    end
    while ci <= length(order)
        rec_params[order[ci]] = copy(θ)
        ci += 1
    end
    return zs, rec_params
end

"""
    dynamic_fit_locationscale(mean, std, a, b; kernel=Normal(),
                              epsilon=0.01, nsteps=200_000, save_z=Float64[])

Find the location and scale of a location-scale distribution
`f(x) = (1/scale)·φ((x-location)/scale)`, with symmetric base density `φ` given by
`kernel`, such that **truncating it to `[a, b]` yields the requested `mean` and
standard deviation `std`**. Uses the dynamic (ODE) method of Liquet & Nazarathy (2015).

`kernel` may be any symmetric continuous `Distributions.UnivariateDistribution`
(default `Normal()`); the Normal case is the one validated against the paper.
`a`/`b` may be `-Inf`/`Inf` for one-sided truncation. `epsilon` is the small
starting value of the homotopy parameter `z`; `nsteps` the number of RK4 steps;
`save_z` an optional list of intermediate `z`-values at which to record the path.

Returns a [`DynamicMomentMatch`](@ref); call [`solution`](@ref) for `[location, scale]`.
"""
function dynamic_fit_locationscale(mean::Real, std::Real, a::Real, b::Real;
                                   kernel = Normal(),
                                   epsilon::Real = 0.01,
                                   nsteps::Int = 200_000,
                                   save_z = Float64[])
    a < b || throw(ArgumentError("require a < b, got a=$a, b=$b"))
    std > 0 || throw(ArgumentError("std must be positive, got $std"))
    μ  = float(mean)
    m1 = μ                       # target first raw moment   (m₁*)
    m2 = std^2 + μ^2             # target second raw moment  (m₂*)
    af = float(a); bf = float(b)

    F(x, t1, t2) = cdf(kernel, (x - t1) / t2)
    f(x, t1, t2) = pdf(kernel, (x - t1) / t2) / t2

    # dθ/ds for θ = [location, scale]; see header note (Eq. 11 form, with the i factor).
    function rhs(u, s)
        ll = af - s
        hh = bf + s
        n0 = F(hh, u[1], u[2]) - F(ll, u[1], u[2])
        # boundary terms x^i f(x); vanish at ±∞
        _l(i) = isinf(ll) ? 0.0 : ll^i * f(ll, u[1], u[2])
        _h(i) = isinf(hh) ? 0.0 : hh^i * f(hh, u[1], u[2])
        l0, l1, l2, l3 = _l(0), _l(1), _l(2), _l(3)
        h0, h1, h2, h3 = _h(0), _h(1), _h(2), _h(3)
        _p(i, hi, li) = u[2]^2 * (hi - li - i * (i == 0 ? 0.0 : (i == 1 ? 1.0 : (i == 2 ? m1 : m2))) * n0)
        p0 = _p(0, h0, l0); p1 = _p(1, h1, l1); p2 = _p(2, h2, l2); p3 = _p(3, h3, l3)
        _c(i, hi, li, mi) = hi + li - mi * (h0 + l0)
        c1 = _c(1, h1, l1, m1); c2 = _c(2, h2, l2, m2)
        Δ  = (p2 * (p1*m1 + p0*m2) - p1^2*m2 - p0*p3*m1 + p3*p1 - p2^2) / u[2]^5
        K  = -1 / (u[2]^3 * Δ)        # the 1/z² has been cancelled by the s-reparam
        L1 = c2 * (p0*m1*u[1] - p1*(m1 + u[1]) + p2) - c1 * ((p0*m2 - p2)*u[1] - p1*m2 + p3)
        L2 = u[2] * (c2 * (p0*m1 - p1) - c1 * (p0*m2 - p2))
        return [K * L1, K * L2]
    end

    θ0 = [μ, std / Distributions.std(kernel)]   # untruncated solution at z → 0
    zs, params = _integrate_homotopy(rhs, θ0, float(epsilon), nsteps, save_z)
    return DynamicMomentMatch(:locationscale, af, bf, zs, params)
end

"""
    dynamic_fit_exponential(mean, b; epsilon=0.01, nsteps=200_000, save_z=Float64[])

Find the `rate` of an exponential distribution `f(x) = rate·exp(-rate·x)`, `x ≥ 0`,
such that **truncating it to `[0, b]` yields the requested `mean`**. Uses the
dynamic (ODE) method of Liquet & Nazarathy (2015), §2.

The exponential is a one-parameter, *non* location-scale family, so it follows a
separate scalar ODE `dθ/dz = (1/z²)·c₁/B₁₁` with closed-form coefficients. A
target is feasible only if `mean < b/2`. Returns a [`DynamicMomentMatch`](@ref);
call [`solution`](@ref) for `[rate]`.
"""
function dynamic_fit_exponential(mean::Real, b::Real;
                                 epsilon::Real = 0.01,
                                 nsteps::Int = 200_000,
                                 save_z = Float64[])
    m = float(mean)
    bf = float(b)
    (0 < m < bf / 2) || throw(ArgumentError(
        "infeasible target: need 0 < mean < b/2, got mean=$m, b=$bf"))

    # B₁₁ = ∫₀ᴴ (x - m)·e^{-θx}·(1 - θx) dx, closed form via the standard
    # antiderivatives J₀, J₁, J₂ of x^k e^{-θx}.
    function B11(θ, H)
        e  = exp(-θ * H)
        J0 = (1 - e) / θ
        J1 = (1 - (1 + θ*H) * e) / θ^2
        J2 = (2 - (2 + 2θ*H + θ^2*H^2) * e) / θ^3
        return -θ*J2 + (1 + m*θ)*J1 - m*J0
    end
    # dθ/ds; lower bound is the support edge 0 (density there does not move with z),
    # so only the upper boundary h_z = b + s contributes to c₁.
    function rhs(u, s)
        θ = u[1]
        H = bf + s
        c1 = (H - m) * θ * exp(-θ * H)
        return [-c1 / B11(θ, H)]
    end

    θ0 = [1 / m]                  # untruncated solution: rate = 1/mean
    zs, params = _integrate_homotopy(rhs, θ0, float(epsilon), nsteps, save_z)
    return DynamicMomentMatch(:exponential, 0.0, bf, zs, params)
end
