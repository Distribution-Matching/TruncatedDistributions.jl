"""
High-n benchmark of EXP-KR-MVN (explicit gradient with the
MvNormalCDF.jl base case) on n = 5, 6, 7. The other gradient sources
(FD, AD, EXP-KR-H) are all dominated by general-purpose adaptive
cubature, which scales exponentially with n and is not informative
above n = 4 (they all time out without converging); we run only the
fast path here.

Each row also reports m^{(0)} = P(X ∈ A), the truncation probability,
as a measure of "how non-truncated" each example is: values near 1
mean the box barely truncates the underlying Gaussian, values near 0
mean almost all the mass lies outside.

Run from the package root:
    julia --project=. test/benchmark_kr_mvn_high_n.jl
"""

using TruncatedDistributions
using MvNormalCDF
using Distributions, PDMats, LinearAlgebra, Printf
using Optim

# Same fg! pipeline as benchmark_fd_ad_explicit.jl, but the base case
# is MvNormalCDF.jl (Genz–Bretz QMC). PRIMA is omitted at high n —
# derivative-free trust-region methods do not scale with parameter
# dimension and would simply time out without converging.
const COMMON_OPTS = Optim.Options(show_trace = false,
                                  iterations = 50,
                                  time_limit = 60.0,
                                  callback   = s -> s.value < 1e-3)

function fit_explicit_kr_fg(p0, a, b, μ̂, Σ̂)
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Matrix(Σ̂))
    optimize(Optim.only_fg!(fg!), p0, LBFGS(), COMMON_OPTS)
end

function target(ne; digits::Int = 1)
    d  = dist_from_example(ne)
    μ̂  = round.(mean(d); digits = digits)
    Σ̂  = round.(cov(d);  digits = digits)
    return d, μ̂, Σ̂
end

# m^{(0)} = P(X ∈ A) computed directly via MvNormalCDF (fast at any n).
function m0(ne)
    return mvnormcdf(MvNormal(collect(ne.μ), Matrix(ne.Σ)),
                     collect(ne.a), collect(ne.b); m = 10_000)[1]
end

function run_one(label, ne)
    p_mass = m0(ne)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
    @printf("[%-14s] m0=%.4f  ", label, p_mass)
    flush(stdout)

    set_kr_base_backend!(:mvnormalcdf)
    try
        t = @elapsed (r = fit_explicit_kr_fg(p0, a, b, μ̂, Σ̂))
        @printf("EXP-KR-MVN  t=%6.2fs  L=%.2e\n", t, r.minimum)
    catch e
        println("EXP-KR-MVN ERR: ", sprint(showerror, e))
    end
    flush(stdout)
end

function warmup()
    ne = get_example(n = 2, index = 4)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf)
    try; fit_explicit_kr_fg(p0, a, b, μ̂, Σ̂); catch; end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# High-n Gaussian benchmark: EXP-KR-MVN only")
    println("# m0 = P(X ∈ A), the truncation probability (1 = no truncation, 0 = all mass excluded)")
    println("# Warmup pass first; reported times are post-JIT.")
    warmup()
    println("# Warmup complete.")
    println()
    for n in [5, 6, 7]
        for i in 1:get_num_examples(n)
            ne = get_example(n = n, index = i)
            label = "n=$n idx=$i"
            run_one(label, ne)
        end
    end
end
