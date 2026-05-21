# Sweep over inner_iters and selection mode on n=3,4 to find a good default.
# Each row: (inner_iters, selection, T) -> (wall-clock, BCD loss, polished loss)
#
# Run: julia --project=. test/quick_bcd_sweep.jl

using TruncatedDistributions
using Distributions, PDMats, LinearAlgebra, Printf
using Optim, Random

const OPTS = Optim.Options(show_trace = false, iterations = 50,
                           time_limit = 60.0,
                           callback   = s -> s.value < 1e-3)

function target(ne; digits = 1)
    d = dist_from_example(ne)
    return d, round.(mean(d); digits = digits), round.(cov(d); digits = digits)
end

function joint_lbfgs(p0, a, b, μ̂, Σ̂)
    fg!(F, G, p) = vector_fg_true_loss(F, G, p, a, b, μ̂, Matrix(Σ̂))
    optimize(Optim.only_fg!(fg!), p0, LBFGS(), OPTS)
end

function run_one(label, ne; inner::Int, sel::Symbol, T::Float64,
                 iters::Int = 25, seed::Int = 1, λ::Float64 = 0.0)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    set_kr_base_backend!(:mvnormalcdf)
    rng = MersenneTwister(seed)

    local L_bcd, n_it, L_pol
    t = @elapsed begin
        μ_ws, Σ_ws = warm_start_diagonal(μ̂, Σ̂, a, b)
        μ_b, Σ_b, hist = block_coord_descent(
            μ̂, Σ̂, a, b;
            μ_init = μ_ws, Σ_init = Σ_ws,
            block_sizes = [1, 2, 3],
            max_iters = iters,
            inner_iters = inner,
            accept_by = :full,
            selection = sel,
            softmax_T = T,
            proximal_λ = λ,
            rng = rng,
            ftarget = 1e-3,
            verbose = false)
        L_bcd = hist[end]; n_it = length(hist) - 1
        if L_bcd < 1e-3
            L_pol = L_bcd
        else
            p_b = make_param_vec_from_μ_Σ(μ_b, Σ_b)
            r = joint_lbfgs(p_b, a, b, μ̂, Σ̂)
            L_pol = r.minimum
        end
    end
    @printf("[%-10s] inner=%2d  sel=%-8s T=%.2f  λ=%.2g  iters=%2d  t=%5.2fs  BCD=%.2e  pol=%.2e\n",
            label, inner, sel, T, λ, n_it, t, L_bcd, L_pol)
end

function warmup()
    ne = get_example(n = 2, index = 4)
    d, μ̂, Σ̂ = target(ne)
    a = collect(d.region.a); b = collect(d.region.b)
    set_kr_base_backend!(:mvnormalcdf)
    try; joint_lbfgs(make_param_vec_from_μ_Σ(μ̂, Σ̂), a, b, μ̂, Σ̂); catch; end
    try; block_coord_descent(μ̂, Σ̂, a, b;
                              block_sizes = [1, 2],
                              max_iters = 2, inner_iters = 3,
                              accept_by = :full, verbose = false); catch; end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# BCD sweep: inner_iters × selection on n=3, n=4")
    warmup()
    println("# Warmup complete\n")

    examples = [("n=3 idx=2", get_example(n = 3, index = 2)),
                ("n=4 idx=1", get_example(n = 4, index = 1))]

    for (lab, ne) in examples
        println("--- $lab ---")
        for inner in [5, 8, 15]
            run_one(lab, ne; inner = inner, sel = :greedy, T = 1.0)
        end
        for T in [0.5, 1.0]
            run_one(lab, ne; inner = 8, sel = :softmax, T = T)
        end
        for λ in [1e-3, 1e-2, 1e-1, 1.0]
            run_one(lab, ne; inner = 15, sel = :greedy, T = 1.0, λ = λ)
        end
        println()
    end
end
