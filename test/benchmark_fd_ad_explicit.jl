"""
LBFGS-only comparison of three gradient sources on the n=2 Gaussian
moment-matching examples:

  * FD       — Optim.LBFGS with finite-difference gradient (autodiff = :finite),
               loss evaluated via HCubature (generic numeric type, fair
               with the AD path).
  * AD       — Optim.LBFGS with autodiff = :forward (ForwardDiff.jl),
               loss evaluated via HCubature.
  * explicit — Optim.LBFGS with the explicit chain-rule gradient
               vector_grad_true_loss, loss evaluated via the package's
               Kan–Robotti recursive-moments engine.

The FD and AD rows share the same moment evaluator so the comparison
isolates the gradient source. The explicit row also reports a second
timing using the Kan–Robotti recursion for the loss; this is unfair to
FD/AD on the cost of each loss evaluation but matches what a real
deployment of the explicit gradient would use, so we keep both numbers.

Run from the package root:
    julia --project=. test/benchmark_fd_ad_explicit.jl
"""

using TruncatedDistributions
using HCubature, LinearAlgebra, PDMats, Printf
using ForwardDiff
using Optim
using PRIMA
using MvNormalCDF
using Distributions: MvNormal

# --------------------------------------------------------------------------
# Generic HCubature-based loss (handles ForwardDiff.Dual inputs)
# --------------------------------------------------------------------------

# Unpack the LBFGS parameter vector p = (μ, upper-triangular entries of U)
# into μ and Σ = U^{-1} U^{-T}. Generic in the element type of p.
function unpack_generic(p::AbstractVector{T}, n::Int) where {T}
    μ = p[1:n]
    U = zeros(T, n, n)
    k = n + 1
    for i in 1:n, j in i:n
        U[i, j] = p[k]
        k += 1
    end
    Ui = inv(U)
    Σ  = Ui * Ui'
    # Symmetrise to kill round-off asymmetry (matters for Dual numbers).
    Σ  = T(0.5) .* (Σ .+ Σ')
    return μ, Σ
end

# Truncated-moment integrals via HCubature, generic in T. Returns
# (m0, m1, m2) with m1 length-n vector and m2 n×n matrix. Bounds may
# contain ±Inf; the package's `hcubature_inf` wrapper applies the
# standard tanh-style substitution to map each unbounded coordinate to
# a finite interval before integrating, so the HCubature path sees the
# same true domain as the Kan–Robotti recursion.
function truncated_moments_hcub(μ::AbstractVector{T}, Σ::AbstractMatrix{T},
                                a::AbstractVector{Float64},
                                b::AbstractVector{Float64};
                                rtol = 1e-5,
                                maxevals = 100_000) where {T}
    n      = length(μ)
    Σinv   = inv(Σ)
    detΣ   = det(Σ)
    norm_const = T((2π)^(-n / 2)) / sqrt(detΣ)
    dens(x) = norm_const * exp(T(-0.5) * dot(x .- μ, Σinv * (x .- μ)))
    m0      = hcubature_inf(dens, a, b; rtol = rtol, maxevals = maxevals)[1]
    m1      = [hcubature_inf(x -> x[i] * dens(x), a, b; rtol = rtol, maxevals = maxevals)[1] for i in 1:n]
    m2      = [hcubature_inf(x -> x[i] * x[j] * dens(x), a, b; rtol = rtol, maxevals = maxevals)[1] for i in 1:n, j in 1:n]
    return m0, m1, m2
end

function vector_moment_loss_hcub(p::AbstractVector{T},
                                 a::AbstractVector{Float64},
                                 b::AbstractVector{Float64},
                                 μ̂::AbstractVector{Float64},
                                 Σ̂::AbstractMatrix{Float64}) where {T}
    n         = length(μ̂)
    μ, Σ      = unpack_generic(p, n)
    m0, m1, m2 = truncated_moments_hcub(μ, Σ, a, b)
    μA        = m1 ./ m0
    ΣA        = m2 ./ m0 .- μA * μA'
    return T(0.5) * (sum(abs2, μA .- μ̂) + sum(abs2, ΣA .- Σ̂))
end

# --------------------------------------------------------------------------
# Three optimizers
# --------------------------------------------------------------------------

const COMMON_OPTS = Optim.Options(show_trace  = false,
                                  iterations = 50,
                                  time_limit  = 60.0,
                                  callback    = s -> s.value < 1e-3)

function fit_fd(p0, a, b, μ̂, Σ̂)
    f(p) = vector_moment_loss_hcub(p, a, b, μ̂, Σ̂)
    optimize(f, p0, LBFGS(), COMMON_OPTS)
end

function fit_ad(p0, a, b, μ̂, Σ̂)
    f(p) = vector_moment_loss_hcub(p, a, b, μ̂, Σ̂)
    optimize(f, p0, LBFGS(), COMMON_OPTS; autodiff = :forward)
end

function fit_explicit_hcub(p0, a, b, μ̂, Σ̂)
    f(p)       =  vector_moment_loss_hcub(p, a, b, μ̂, Σ̂)
    # gradient via ForwardDiff on the same hcub-loss — algebraically the
    # same as the chain-rule explicit gradient up to numerical noise,
    # used here only as a sanity baseline. Not the headline explicit row.
    g!(g, p)   = (g .= ForwardDiff.gradient(f, p))
    optimize(f, g!, p0, LBFGS(), COMMON_OPTS)
end

function fit_explicit_kr(p0, a, b, μ̂, Σ̂)
    # Separate f and g! calls — the doubled-recursion path.
    f(p)       = vector_moment_loss(p, a, b, μ̂, Matrix(Σ̂))
    g!(g, p)   = (g .= vector_grad_true_loss(p, a, b, μ̂, Matrix(Σ̂)))
    optimize(f, g!, p0, LBFGS(), COMMON_OPTS)
end

function fit_explicit_kr_fg(p0, a, b, μ̂, Σ̂)
    # Combined fg! — one Kan–Robotti recursion per LBFGS iteration.
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Matrix(Σ̂))
    optimize(Optim.only_fg!(fg!), p0, LBFGS(), COMMON_OPTS)
end

# PRIMA NEWUOA — derivative-free trust-region method. Loss is the same
# KR-based moment loss; the current set_kr_base_backend! choice (mvnormalcdf
# by default in the benchmark) determines how moments are evaluated.
function fit_prima(p0, a, b, μ̂, Σ̂)
    f(p) = vector_moment_loss(p, a, b, μ̂, Matrix(Σ̂))
    res, _ = newuoa(f, p0; ftarget = 1e-3, maxfun = 5000)
    return res
end

# --------------------------------------------------------------------------
# Run on the bundled 2D examples
# --------------------------------------------------------------------------

# Build a near-by feasible target by rounding the truncated moments.
function target(ne; digits::Int = 1)
    d  = dist_from_example(ne)
    μ̂  = round.(mean(d); digits = digits)
    Σ̂  = round.(cov(d);  digits = digits)
    return d, μ̂, Σ̂
end

# m^{(0)} = P(X ∈ A), the truncation probability of the example.
function m0(ne)
    return mvnormcdf(MvNormal(collect(ne.μ), Matrix(ne.Σ)),
                     collect(ne.a), collect(ne.b); m = 10_000)[1]
end

function run_one(label, ne)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
    init = vector_moment_loss_hcub(p0, a, b, μ̂, Σ̂)
    p_mass = m0(ne)
    @printf("[%-22s] m0=%.3f init=%.4e   ", label, p_mass, init)
    flush(stdout)

    # FD
    try
        t = @elapsed (r = fit_fd(p0, a, b, μ̂, Σ̂))
        @printf("FD %.2fs(%.2e) ", t, r.minimum)
    catch
        print("FD ERR ")
    end
    flush(stdout)

    # AD
    try
        t = @elapsed (r = fit_ad(p0, a, b, μ̂, Σ̂))
        @printf("AD %.2fs(%.2e) ", t, r.minimum)
    catch
        print("AD ERR ")
    end
    flush(stdout)

    # Explicit Kan-Robotti with fg! fast path (one recursion per iter),
    # base case via hcubature_inf.
    set_kr_base_backend!(:hcubature)
    try
        t = @elapsed (r = fit_explicit_kr_fg(p0, a, b, μ̂, Σ̂))
        @printf("EXP-KR-H %.2fs(%.2e) ", t, r.minimum)
    catch
        print("EXP-KR-H ERR ")
    end
    flush(stdout)

    # Same explicit gradient + fg! pipeline, base case via MvNormalCDF
    # (Genz–Bretz QMC). This isolates the speedup from swapping the
    # base-case integrator only; the gradient construction is identical.
    set_kr_base_backend!(:mvnormalcdf)
    try
        t = @elapsed (r = fit_explicit_kr_fg(p0, a, b, μ̂, Σ̂))
        @printf("EXP-KR-MVN %.2fs(%.2e) ", t, r.minimum)
    catch
        print("EXP-KR-MVN ERR ")
    end
    flush(stdout)

    # PRIMA NEWUOA — derivative-free trust-region. Uses the same loss as
    # EXP-KR-MVN (KR recursion with MvNormalCDF base) so the comparison
    # measures optimizer cost, not integrator cost.
    set_kr_base_backend!(:mvnormalcdf)
    try
        t = @elapsed (p_final = fit_prima(p0, a, b, μ̂, Σ̂))
        L_final = vector_moment_loss(p_final, a, b, μ̂, Matrix(Σ̂))
        @printf("PRIMA %.2fs(%.2e)", t, L_final)
    catch
        print("PRIMA ERR")
    end
    set_kr_base_backend!(:hcubature)  # restore default
    println()
end

# Single warmup invocation of every fit on the cheapest n=2 example so the
# first timed row reports steady-state cost rather than JIT compile time.
function warmup()
    ne = get_example(n = 2, index = 4)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
    for fit in (fit_fd, fit_ad, fit_explicit_kr_fg, fit_prima)
        try; fit(p0, a, b, μ̂, Σ̂); catch; end
    end
    set_kr_base_backend!(:mvnormalcdf)
    try; fit_explicit_kr_fg(p0, a, b, μ̂, Σ̂); catch; end
    try; fit_prima(p0, a, b, μ̂, Σ̂); catch; end
    set_kr_base_backend!(:hcubature)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# n=2,3,4 Gaussian benchmark: FD/AD/EXP-KR-H/EXP-KR-MVN/PRIMA")
    println("# Warmup pass first; reported times are post-JIT.")
    warmup()
    println("# Warmup complete.")
    for n in [2, 3, 4]
        for i in 1:get_num_examples(n)
            ne = get_example(n = n, index = i)
            run_one("n=$n idx=$i", ne)
        end
    end
end
