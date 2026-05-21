"""
Side-by-side comparison of EXP-KR-MVN with and without the coordinate-wise
warm-start (`warm_start_diagonal`). The §6.1 baseline rows used the simple
starting point `p0 = (μ̂, Σ̂)`; this script measures whether the cheap 1D
moment-matching warm-start meaningfully reduces wall-clock for the joint
LBFGS on the n-dim problem.

Reported columns per row:
  * m^{(0)}          — truncation probability of the example
  * EXP-KR-MVN       — joint LBFGS from `p0 = (μ̂, Σ̂)`
  * EXP-KR-MVN+WS    — coordinate warm-start, then joint LBFGS from the
                       resulting `(μ_ws, Σ_ws)`. Wall-clock includes
                       the warm-start cost.

Both runs use the MvNormalCDF base case for the Kan–Robotti recursion.
This script does NOT re-run FD / AD / EXP-KR-H / PRIMA — those are
unchanged from §6.1.

Run from the package root:
    julia --project=. test/benchmark_kr_mvn_improvements.jl
"""

using TruncatedDistributions
using MvNormalCDF
using Distributions, PDMats, LinearAlgebra, Printf
using Optim

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

function m0(ne)
    return mvnormcdf(MvNormal(collect(ne.μ), Matrix(ne.Σ)),
                     collect(ne.a), collect(ne.b); m = 10_000)[1]
end

function run_one(label, ne)
    p_mass = m0(ne)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    @printf("[%-14s] m0=%.4f  ", label, p_mass)
    flush(stdout)

    set_kr_base_backend!(:mvnormalcdf)

    # Plain EXP-KR-MVN: start from (μ̂, Σ̂).
    try
        p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
        t  = @elapsed (r = fit_explicit_kr_fg(p0, a, b, μ̂, Σ̂))
        @printf("MVN %.2fs(%.2e)  ", t, r.minimum)
    catch
        print("MVN ERR  ")
    end
    flush(stdout)

    # EXP-KR-MVN with coordinate warm-start. Wall-clock includes the
    # warm-start cost itself.
    try
        t = @elapsed begin
            μ_ws, Σ_ws = warm_start_diagonal(μ̂, Σ̂, a, b)
            p0_ws      = make_param_vec_from_μ_Σ(μ_ws, Σ_ws)
            r          = fit_explicit_kr_fg(p0_ws, a, b, μ̂, Σ̂)
        end
        @printf("MVN+WS %.2fs(%.2e)", t, r.minimum)
    catch e
        @printf("MVN+WS ERR (%s)", sprint(showerror, e))
    end
    println()
end

function warmup()
    ne = get_example(n = 2, index = 4)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    p0 = make_param_vec_from_μ_Σ(μ̂, Σ̂)
    set_kr_base_backend!(:mvnormalcdf)
    try; fit_explicit_kr_fg(p0, a, b, μ̂, Σ̂); catch; end
    try; warm_start_diagonal(μ̂, Σ̂, a, b); catch; end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# EXP-KR-MVN vs EXP-KR-MVN+WS (coordinate warm-start) — n = 2,3,4,5")
    println("# Warmup pass first; reported times are post-JIT and INCLUDE the warm-start cost.")
    warmup()
    println("# Warmup complete.")
    println()
    for n in [2, 3, 4, 5]
        for i in 1:get_num_examples(n)
            ne = get_example(n = n, index = i)
            run_one("n=$n idx=$i", ne)
        end
    end
end
